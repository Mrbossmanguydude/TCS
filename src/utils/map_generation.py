"""
Procedural map generation for setup/train previews.

Generation order:
1. Build a maze-like base using a reverse-BFS carving tree.
2. Overlay roads for usable traffic flow.
3. Place road structures when location rules are satisfied.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Dict, List, Set, Tuple


GridPoint = Tuple[int, int]


@dataclass(frozen=True)
class PhaseSpec:
    """
    Curriculum phase generation profile.
    """

    phase_id: int
    name: str
    continuous: bool
    map_levels: Tuple[int, ...]


@dataclass
class GeneratedMap:
    """
    Render-ready map payload for setup preview.
    """

    width: int
    height: int
    roads: Set[GridPoint]
    node_types: Dict[GridPoint, str]
    structure_counts: Dict[str, int]
    phase: int
    level_index: int
    level_size: int
    continuous: bool
    roundabouts_enabled: bool
    tile_pixels: int
    vehicle_scale: float
    road_density: float
    structure_density: float
    vehicles: List["PreviewVehicle"]


@dataclass
class PreviewVehicle:
    """
    Use:
    Store preview-only vehicle routing points for setup visualisation.

    Inputs:
    - vehicle_id: Zero-based identifier.
    - spawn: Road cell where the vehicle starts.
    - destination: Target road cell for the route objective.
    - continuous: Whether this vehicle belongs to a continuous phase preview.

    Output:
    A value object used by the setup renderer and VN feature preparation.
    """

    vehicle_id: int
    spawn: GridPoint
    destination: GridPoint
    continuous: bool


PHASE_LIBRARY: Tuple[PhaseSpec, ...] = (
    PhaseSpec(1, "Single-Vehicle Routing", False, (8, 10, 12, 14)),
    PhaseSpec(2, "Multi-Vehicle Routing", False, (8, 10, 12, 14)),
    PhaseSpec(3, "Multi-Vehicle With Setbacks", False, (8, 10, 12, 14)),
    PhaseSpec(4, "Single-Vehicle Continuous", True, (10, 12, 14, 16)),
    PhaseSpec(5, "Multi-Vehicle Continuous", True, (10, 12, 14, 16)),
    PhaseSpec(6, "Final Continuous + Acceleration", True, (12, 14, 16, 18)),
)

STRUCTURE_KEYS: Tuple[str, ...] = (
    "roundabout",
    "junction_turn_one_lane",
    "junction_turn_two_lane",
    "junction_t",
    "junction_cross",
    "road_two_lane",
)


def clamp_phase(phase: int) -> int:
    """
    Clamp user phase into valid curriculum range (1..6).
    """
    return max(1, min(6, int(phase)))


def phase_spec(phase: int) -> PhaseSpec:
    """
    Get curriculum spec for the requested phase.
    """
    return PHASE_LIBRARY[clamp_phase(phase) - 1]


def map_level_size(phase: int, level_index: int) -> tuple[int, int]:
    """
    Resolve square map dimensions for a phase level index.
    """
    spec = phase_spec(phase)
    idx = max(0, min(int(level_index), len(spec.map_levels) - 1))
    size = int(spec.map_levels[idx])
    return size, size


def map_level_count(phase: int) -> int:
    """
    Return number of complexity levels configured for this phase.
    """
    return len(phase_spec(phase).map_levels)


def _clamp_density(value: float, minimum: float, maximum: float) -> float:
    """
    Normalise density parameter.
    """
    return max(minimum, min(maximum, float(value)))


def _in_bounds(x_pos: int, y_pos: int, width: int, height: int) -> bool:
    return 0 <= x_pos < width and 0 <= y_pos < height


def _interior_bounds(x_pos: int, y_pos: int, width: int, height: int) -> bool:
    return 1 <= x_pos < (width - 1) and 1 <= y_pos < (height - 1)


def _add_line(roads: Set[GridPoint], x0: int, y0: int, x1: int, y1: int) -> None:
    """
    Add axis aligned segment to road set.
    """
    if x0 == x1:
        lo, hi = sorted((y0, y1))
        for y_val in range(lo, hi + 1):
            roads.add((x0, y_val))
        return
    if y0 == y1:
        lo, hi = sorted((x0, x1))
        for x_val in range(lo, hi + 1):
            roads.add((x_val, y0))
        return


def _build_graph(roads: Set[GridPoint]) -> Dict[GridPoint, List[GridPoint]]:
    """
    Build 4 neighbour adjacency graph from road cells.
    """
    graph: Dict[GridPoint, List[GridPoint]] = {node: [] for node in roads}
    for x_val, y_val in roads:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbour = (x_val + dx, y_val + dy)
            if neighbour in roads:
                graph[(x_val, y_val)].append(neighbour)
    return graph


def _connected_component(start: GridPoint, graph: Dict[GridPoint, List[GridPoint]]) -> Set[GridPoint]:
    """
    Return nodes reachable from start.
    """
    seen: Set[GridPoint] = {start}
    queue: deque[GridPoint] = deque([start])
    while queue:
        node = queue.popleft()
        for nxt in graph[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def _largest_connected_roads(roads: Set[GridPoint]) -> Set[GridPoint]:
    """
    Keep only the largest connected road component.
    """
    if not roads:
        return set()
    graph = _build_graph(roads)
    best: Set[GridPoint] = set()
    visited: Set[GridPoint] = set()
    for node in graph:
        if node in visited:
            continue
        comp = _connected_component(node, graph)
        visited.update(comp)
        if len(comp) > len(best):
            best = comp
    return best


def _odd_interior_values(limit: int) -> List[int]:
    """
    Return odd coordinates inside boundary walls.
    """
    return [val for val in range(1, max(1, limit - 1)) if val % 2 == 1]


def _build_reverse_bfs_maze(
    size: tuple[int, int],
    seed: int,
    level_index: int,
    level_count: int,
    road_density: float,
) -> Set[GridPoint]:
    """
    Build maze basis using reverse-BFS tree carving.

    The queue starts from a seeded target node, then expands outward to connect
    every odd cell via parent links. This is a reverse BFS maze concept.
    """
    width, height = size
    roads: Set[GridPoint] = set()

    if width < 7 or height < 7:
        for y_val in range(height):
            for x_val in range(width):
                if _interior_bounds(x_val, y_val, width, height):
                    roads.add((x_val, y_val))
        return roads

    max_levels = max(1, int(level_count) - 1)
    complexity = max(0.0, min(1.0, float(level_index) / float(max_levels)))
    rng = random.Random(int(seed))

    odd_x = _odd_interior_values(width)
    odd_y = _odd_interior_values(height)
    if not odd_x or not odd_y:
        return roads

    target = (rng.choice(odd_x), rng.choice(odd_y))
    roads.add(target)

    queue: deque[GridPoint] = deque([target])
    visited: Set[GridPoint] = {target}
    all_cells = {(x_val, y_val) for x_val in odd_x for y_val in odd_y}
    density = _clamp_density(road_density, 0.35, 1.35)
    target_ratio = max(0.22, min(0.78, (0.30 + (0.30 * complexity)) * density))
    target_cells = max(1, int(round(len(all_cells) * target_ratio)))

    while queue:
        if len(visited) >= target_cells:
            break
        # Higher density uses deeper pops to encourage winding corridor growth.
        if density >= 1.0 and rng.random() < 0.7:
            cx, cy = queue.pop()
        else:
            cx, cy = queue.popleft()
        candidates: List[GridPoint] = []
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in all_cells and (nx, ny) not in visited:
                candidates.append((nx, ny))

        # Prefer far-away cells first so lanes extend outward before filling gaps.
        candidates.sort(key=lambda node: _manhattan(node, target), reverse=True)
        if rng.random() < 0.55:
            rng.shuffle(candidates)
        for nx, ny in candidates:
            visited.add((nx, ny))
            queue.append((nx, ny))
            roads.add((nx, ny))
            roads.add(((cx + nx) // 2, (cy + ny) // 2))

    # Add limited loops to avoid a pure tree and create richer intersections.
    # Higher complexity levels receive more optional loop connectors.
    loop_candidates: List[GridPoint] = []
    for x_val in odd_x:
        for y_val in odd_y:
            for dx, dy in ((2, 0), (0, 2)):
                nx, ny = x_val + dx, y_val + dy
                if (nx, ny) not in all_cells:
                    continue
                wall = (x_val + (dx // 2), y_val + (dy // 2))
                if wall not in roads:
                    loop_candidates.append(wall)

    rng.shuffle(loop_candidates)
    loop_ratio = (0.02 + (0.10 * complexity)) * density
    extra_loops = int(round(len(loop_candidates) * max(0.0, min(0.18, loop_ratio))))
    for wall in loop_candidates[:extra_loops]:
        roads.add(wall)

    return roads


def _overlay_arterials(
    roads: Set[GridPoint],
    size: tuple[int, int],
    seed: int,
    level_index: int,
    level_count: int,
    road_density: float,
) -> tuple[List[int], List[int]]:
    """
    Overlay deterministic primary roads on top of maze basis.
    """
    width, height = size
    max_levels = max(1, int(level_count) - 1)
    complexity = max(0.0, min(1.0, float(level_index) / float(max_levels)))
    rng = random.Random((int(seed) * 7) + 13)
    density = _clamp_density(road_density, 0.35, 1.35)

    verticals: List[int] = []
    horizontals: List[int] = []
    centre_v = max(1, min(width - 2, width // 2))
    centre_h = max(1, min(height - 2, height // 2))

    if density < 0.8:
        if rng.random() < 0.5:
            verticals.append(centre_v)
        else:
            horizontals.append(centre_h)
    else:
        verticals.append(centre_v)
        horizontals.append(centre_h)

    if density >= 0.95 and complexity >= 0.35 and width >= 12:
        verticals.append(max(1, min(width - 2, (width // 3) + rng.choice((-1, 0, 1)))))
    if density >= 0.95 and complexity >= 0.35 and height >= 12:
        horizontals.append(max(1, min(height - 2, (height // 3) + rng.choice((-1, 0, 1)))))
    if density >= 1.15 and complexity >= 0.7 and width >= 14:
        verticals.append(max(1, min(width - 2, ((2 * width) // 3) + rng.choice((-1, 0, 1)))))
    if density >= 1.15 and complexity >= 0.7 and height >= 14:
        horizontals.append(max(1, min(height - 2, ((2 * height) // 3) + rng.choice((-1, 0, 1)))))

    unique_verticals = sorted(set(verticals))
    unique_horizontals = sorted(set(horizontals))
    for x_val in unique_verticals:
        _add_line(roads, x_val, 1, x_val, height - 2)
    for y_val in unique_horizontals:
        _add_line(roads, 1, y_val, width - 2, y_val)

    return unique_verticals, unique_horizontals


def _expand_two_lane_roads(
    roads: Set[GridPoint],
    size: tuple[int, int],
    verticals: List[int],
    horizontals: List[int],
    continuous: bool,
    structure_density: float,
) -> Set[GridPoint]:
    """
    Expand selected arterials into adjacent parallel lanes.
    """
    width, height = size
    two_lane_cells: Set[GridPoint] = set()
    if not continuous:
        return two_lane_cells
    density = _clamp_density(structure_density, 0.2, 1.5)
    if density < 0.6:
        return two_lane_cells

    for x_val in verticals:
        lane_pair_x = x_val + 1 if x_val + 1 < width - 1 else x_val - 1
        if lane_pair_x < 1 or lane_pair_x >= width - 1:
            continue
        for y_val in range(1, height - 1):
            if (x_val, y_val) in roads:
                roads.add((lane_pair_x, y_val))
                two_lane_cells.add((x_val, y_val))
                two_lane_cells.add((lane_pair_x, y_val))

    for y_val in horizontals:
        lane_pair_y = y_val + 1 if y_val + 1 < height - 1 else y_val - 1
        if lane_pair_y < 1 or lane_pair_y >= height - 1:
            continue
        for x_val in range(1, width - 1):
            if (x_val, y_val) in roads:
                roads.add((x_val, lane_pair_y))
                two_lane_cells.add((x_val, y_val))
                two_lane_cells.add((x_val, lane_pair_y))

    return two_lane_cells


def _straight_run_length(roads: Set[GridPoint], start: GridPoint, dx: int, dy: int) -> int:
    """
    Count contiguous road cells from start in one cardinal direction.
    """
    x_val, y_val = start
    length = 0
    while True:
        x_val += dx
        y_val += dy
        if (x_val, y_val) in roads:
            length += 1
            continue
        break
    return length


def _is_roundabout_candidate(
    roads: Set[GridPoint],
    node: GridPoint,
    size: tuple[int, int],
    min_exit_len: int = 5,
) -> bool:
    """
    Validate a roundabout candidate with exit-length and clearance rules.
    """
    width, height = size
    cx, cy = node
    if not (2 <= cx <= (width - 3) and 2 <= cy <= (height - 3)):
        return False

    # All cardinal exits must sustain minimum length.
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if _straight_run_length(roads, node, dx, dy) < min_exit_len:
            return False
    return True


def _place_roundabout(roads: Set[GridPoint], node: GridPoint, size: tuple[int, int], min_exit_len: int = 5) -> bool:
    """
    Place a simple 3x3 ring-style roundabout at the node.
    """
    if not _is_roundabout_candidate(roads, node, size, min_exit_len=min_exit_len):
        return False

    width, height = size
    cx, cy = node

    for xx in range(cx - 1, cx + 2):
        for yy in range(cy - 1, cy + 2):
            if not _interior_bounds(xx, yy, width, height):
                return False

    # Ring shape: keep perimeter, block centre.
    for xx in range(cx - 1, cx + 2):
        for yy in range(cy - 1, cy + 2):
            if (xx, yy) != (cx, cy):
                roads.add((xx, yy))
    roads.discard((cx, cy))

    # Ensure exits from ring extend along cardinal directions.
    _add_line(roads, cx, cy - 1, cx, cy - min_exit_len)
    _add_line(roads, cx, cy + 1, cx, cy + min_exit_len)
    _add_line(roads, cx - 1, cy, cx - min_exit_len, cy)
    _add_line(roads, cx + 1, cy, cx + min_exit_len, cy)
    return True


def _select_spaced_nodes(
    candidates: List[GridPoint],
    rng: random.Random,
    count: int,
    min_distance: int,
) -> List[GridPoint]:
    """
    Select nodes with Manhattan spacing.
    """
    pool = list(candidates)
    rng.shuffle(pool)
    chosen: List[GridPoint] = []
    for node in pool:
        if all(abs(node[0] - other[0]) + abs(node[1] - other[1]) >= min_distance for other in chosen):
            chosen.append(node)
            if len(chosen) >= count:
                break
    return chosen


def _apply_structures(
    roads: Set[GridPoint],
    size: tuple[int, int],
    seed: int,
    continuous: bool,
    complexity: float,
    two_lane_cells: Set[GridPoint],
    structure_density: float,
) -> Dict[GridPoint, str]:
    """
    Mark and place structure types based on road rules.
    """
    node_types: Dict[GridPoint, str] = {}
    graph = _build_graph(roads)
    rng = random.Random((int(seed) * 19) + 5)
    density = _clamp_density(structure_density, 0.2, 1.5)

    # Base lane markup.
    for cell in two_lane_cells:
        if cell in roads:
            node_types[cell] = "road_two_lane"

    t_candidates = [node for node, neigh in graph.items() if len(neigh) == 3]
    cross_candidates = [node for node, neigh in graph.items() if len(neigh) >= 4]

    t_count = max(0, min(len(t_candidates), int(round(len(t_candidates) * (0.14 + (0.06 * complexity)) * density))))
    cross_count = max(0, min(len(cross_candidates), int(round(len(cross_candidates) * (0.18 + (0.08 * complexity)) * density))))
    for node in _select_spaced_nodes(t_candidates, rng, t_count, min_distance=3):
        node_types[node] = "junction_t"
    for node in _select_spaced_nodes(cross_candidates, rng, cross_count, min_distance=4):
        node_types[node] = "junction_cross"

    # Junctions tied to lane transitions.
    turn_two_candidates: List[GridPoint] = []
    turn_one_candidates: List[GridPoint] = []
    for node, neigh in graph.items():
        if len(neigh) < 3:
            continue
        two_lane_neigh = sum(1 for n in neigh if n in two_lane_cells)
        if two_lane_neigh >= 1 and (len(neigh) - two_lane_neigh) >= 1:
            turn_two_candidates.append(node)
        if two_lane_neigh == 0:
            turn_one_candidates.append(node)

    turn_two_count = max(0, min(len(turn_two_candidates), int(round((0.5 + (complexity * 2.2)) * density))))
    turn_one_count = max(0, min(len(turn_one_candidates), int(round((0.5 + (complexity * 2.0)) * density))))
    for node in _select_spaced_nodes(turn_one_candidates, rng, turn_one_count, min_distance=4):
        node_types[node] = "junction_turn_one_lane"
    for node in _select_spaced_nodes(turn_two_candidates, rng, turn_two_count, min_distance=4):
        node_types[node] = "junction_turn_two_lane"

    # Roundabouts only in continuous phases.
    if continuous:
        graph = _build_graph(roads)
        round_candidates = [node for node, neigh in graph.items() if len(neigh) >= 4]
        round_count = max(0, min(len(round_candidates), int(round((0.4 + (complexity * 1.5)) * density))))
        for node in _select_spaced_nodes(round_candidates, rng, round_count * 3, min_distance=6):
            if _place_roundabout(roads, node, size, min_exit_len=5):
                node_types[node] = "roundabout"
                round_count -= 1
                if round_count <= 0:
                    break

    return node_types


def _structure_counts(node_types: Dict[GridPoint, str]) -> Dict[str, int]:
    """
    Count occurrences of known structure labels.
    """
    counts = {key: 0 for key in STRUCTURE_KEYS}
    for value in node_types.values():
        if value in counts:
            counts[value] += 1
    return counts


def _phase_preview_vehicle_count(phase: int, level_index: int) -> int:
    """
    Use:
    Resolve preview vehicle count from curriculum phase and map complexity level.

    Inputs:
    - phase: Active phase number (1..6).
    - level_index: Zero-based map complexity index.

    Output:
    A deterministic count suitable for setup visualisation.
    """
    idx = max(0, int(level_index))
    if phase in (1, 4):
        return 1
    if phase in (2, 5):
        return min(14, 2 + (idx * 2))
    if phase == 3:
        return min(16, 3 + (idx * 2))
    return min(20, 4 + (idx * 3))


def _manhattan(a_node: GridPoint, b_node: GridPoint) -> int:
    """
    Use:
    Compute Manhattan distance between two grid cells.

    Inputs:
    - a_node: First point (x, y).
    - b_node: Second point (x, y).

    Output:
    Absolute grid distance.
    """
    return abs(a_node[0] - b_node[0]) + abs(a_node[1] - b_node[1])


def _farthest_node(graph: Dict[GridPoint, List[GridPoint]], start: GridPoint) -> GridPoint:
    """
    Use:
    Return the reachable node with maximum BFS distance from `start`.

    Inputs:
    - graph: Road adjacency map.
    - start: BFS origin node.

    Output:
    A road node that is farthest in hop distance.
    """
    seen = {start}
    queue: deque[GridPoint] = deque([start])
    farthest = start
    while queue:
        node = queue.popleft()
        farthest = node
        for nxt in graph.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return farthest


def _initialise_preview_vehicles(
    roads: Set[GridPoint],
    seed: int,
    phase: int,
    level_index: int,
    continuous: bool,
) -> List[PreviewVehicle]:
    """
    Use:
    Create preview vehicles with deterministic spawn and destination nodes.

    Inputs:
    - roads: Connected road cells from generated map.
    - seed: Deterministic seed source.
    - phase: Active curriculum phase.
    - level_index: Zero-based map complexity tier.
    - continuous: Whether this is a continuous movement phase.

    Output:
    A list of preview vehicles used by Setup visualisation.
    """
    if not roads:
        return []

    graph = _build_graph(roads)
    nodes = [node for node, neigh in graph.items() if len(neigh) > 0]
    if not nodes:
        return []

    rng = random.Random((int(seed) * 31) + (phase * 101) + (level_index * 17))
    count = min(len(nodes), _phase_preview_vehicle_count(phase, level_index))
    selected_spawns: List[GridPoint] = []

    pool = list(nodes)
    rng.shuffle(pool)
    for node in pool:
        if all(_manhattan(node, other) >= 3 for other in selected_spawns):
            selected_spawns.append(node)
            if len(selected_spawns) >= count:
                break
    if len(selected_spawns) < count:
        selected_spawns.extend(pool[: count - len(selected_spawns)])

    vehicles: List[PreviewVehicle] = []
    for vehicle_id, spawn in enumerate(selected_spawns):
        destination = _farthest_node(graph, spawn)
        if destination == spawn:
            alternatives = [n for n in nodes if n != spawn]
            if alternatives:
                destination = rng.choice(alternatives)
        vehicles.append(
            PreviewVehicle(
                vehicle_id=vehicle_id,
                spawn=spawn,
                destination=destination,
                continuous=continuous,
            )
        )

    return vehicles


def generate_phase_map(
    seed: int,
    phase: int,
    level_index: int,
    road_density: float = 0.72,
    structure_density: float = 0.62,
) -> GeneratedMap:
    """
    Generate deterministic preview map for a curriculum phase/level.
    """
    spec = phase_spec(phase)
    levels = spec.map_levels
    idx = max(0, min(int(level_index), len(levels) - 1))
    level_size = int(levels[idx])
    size = (level_size, level_size)

    max_levels = max(1, len(levels) - 1)
    complexity = max(0.0, min(1.0, float(idx) / float(max_levels)))
    road_density_norm = _clamp_density(road_density, 0.35, 1.35)
    structure_density_norm = _clamp_density(structure_density, 0.2, 1.5)

    roads = _build_reverse_bfs_maze(size, int(seed), idx, len(levels), road_density=road_density_norm)
    verticals, horizontals = _overlay_arterials(
        roads,
        size,
        int(seed),
        idx,
        len(levels),
        road_density=road_density_norm,
    )
    two_lane_cells = _expand_two_lane_roads(
        roads,
        size,
        verticals,
        horizontals,
        continuous=bool(spec.continuous),
        structure_density=structure_density_norm,
    )

    node_types = _apply_structures(
        roads=roads,
        size=size,
        seed=int(seed),
        continuous=bool(spec.continuous),
        complexity=complexity,
        two_lane_cells=two_lane_cells,
        structure_density=structure_density_norm,
    )

    roads = _largest_connected_roads(roads)
    node_types = {node: typ for node, typ in node_types.items() if node in roads or typ == "roundabout"}
    structure_counts = _structure_counts(node_types)
    vehicles = _initialise_preview_vehicles(
        roads=roads,
        seed=int(seed),
        phase=spec.phase_id,
        level_index=idx,
        continuous=bool(spec.continuous),
    )

    tile_pixels = 50 if spec.continuous else 1
    vehicle_scale = 0.68 if spec.continuous else 1.0

    return GeneratedMap(
        width=size[0],
        height=size[1],
        roads=roads,
        node_types=node_types,
        structure_counts=structure_counts,
        phase=spec.phase_id,
        level_index=idx,
        level_size=level_size,
        continuous=bool(spec.continuous),
        roundabouts_enabled=bool(spec.continuous),
        tile_pixels=tile_pixels,
        vehicle_scale=vehicle_scale,
        road_density=road_density_norm,
        structure_density=structure_density_norm,
        vehicles=vehicles,
    )

