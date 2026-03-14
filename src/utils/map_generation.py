"""
Procedural map generation for setup/train previews.

Generation order:
1. Build a maze-like base using a reverse-BFS carving tree.
2. Overlay roads for usable traffic flow.
3. Place road structures when location rules are satisfied.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# INITIALISATION / STARTUP MATERIAL                                           #
# --------------------------------------------------------------------------- #

from collections import deque
from dataclasses import dataclass
import math
import random
from typing import Dict, List, Set, Tuple


# --------------------------------------------------------------------------- #
# CONSTANTS AND SHARED TYPES                                                  #
# --------------------------------------------------------------------------- #

GridPoint = Tuple[int, int]


@dataclass(frozen=True)
class PhaseSpec:
    """
    Purpose:
    Store immutable generation settings for one curriculum phase.

    This configuration is intentionally compact and declarative so curriculum
    complexity can be tuned without changing generation algorithms.

    Attributes:
    - phase_id: Numeric phase identifier (1..6).
    - name: Human-readable phase name shown in UI/status text.
    - continuous: Whether movement/rendering in this phase is continuous.
    - map_levels: Ordered map sizes used as complexity levels for the phase.
    """

    phase_id: int
    name: str
    continuous: bool
    map_levels: Tuple[int, ...]


@dataclass
class GeneratedMap:
    """
    Purpose:
    Bundle all generated map data needed by Setup and Train rendering and by
    backend episode initialisation logic.

    This object is the canonical hand-off payload between procedural generation
    and every downstream consumer (preview rendering, reset logic, VN features,
    and training episode state creation).

    Attributes:
    - width: Grid width in tiles.
    - height: Grid height in tiles.
    - roads: Set of driveable road grid nodes.
    - node_types: Per-node road/structure type tags.
    - structure_counts: Structure occurrence summary for UI metrics.
    - phase: Phase used for generation.
    - level_index: Complexity level index used for generation.
    - level_size: Convenience size value for UI display.
    - continuous: Whether this map is in continuous motion mode.
    - roundabouts_enabled: Whether roundabout placement was enabled.
    - tile_pixels: Suggested render tile size.
    - vehicle_scale: Suggested vehicle sprite scale factor.
    - road_density: Effective road density used for this map.
    - structure_density: Effective structure density used for this map.
    - vehicles: Preview vehicle spawn/destination pairs for this map.
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

    The same spawn/destination tuples are also used to derive initial VN
    feature vectors and to seed episode vehicle state in TRAIN resets.

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
    PhaseSpec(1, "Single-Vehicle Routing", False, (16, 20, 24, 28)),
    PhaseSpec(2, "Multi-Vehicle Routing", False, (16, 20, 24, 28)),
    PhaseSpec(3, "Multi-Vehicle With Setbacks", False, (16, 20, 24, 28)),
    # Continuous phases use larger maps to support wider lanes and roundabout spacing.
    PhaseSpec(4, "Single-Vehicle Continuous", True, (24, 28, 32, 36)),
    PhaseSpec(5, "Multi-Vehicle Continuous", True, (28, 32, 36, 40)),
    PhaseSpec(6, "Final Continuous + Acceleration", True, (32, 36, 40, 44)),
)

STRUCTURE_KEYS: Tuple[str, ...] = (
    "roundabout",
    "junction_turn_one_lane",
    "junction_turn_two_lane",
    "junction_t",
    "junction_cross",
    "road_two_lane",
)


# --------------------------------------------------------------------------- #
# FUNCTIONS AND HELPERS                                                       #
# --------------------------------------------------------------------------- #

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
    normalise density parameter.
    """
    return max(minimum, min(maximum, float(value)))


def _in_bounds(x_pos: int, y_pos: int, width: int, height: int) -> bool:
    """
    Type:
    FUNCTION.

    Purpose:
    Check whether a coordinate lies inside inclusive map bounds.

    Inputs:
    - x_pos: Candidate x-coordinate.
    - y_pos: Candidate y-coordinate.
    - width: Map width in tiles.
    - height: Map height in tiles.

    Outputs:
    - Returnvalue: `True` when `(x_pos, y_pos)` is inside the map rectangle.
    """
    return 0 <= x_pos < width and 0 <= y_pos < height


def _interior_bounds(x_pos: int, y_pos: int, width: int, height: int) -> bool:
    """
    Type:
    FUNCTION.

    Purpose:
    Check whether a coordinate lies strictly inside the border wall ring.

    Inputs:
    - x_pos: Candidate x-coordinate.
    - y_pos: Candidate y-coordinate.
    - width: Map width in tiles.
    - height: Map height in tiles.

    Outputs:
    - Returnvalue: `True` when node is inside the non-border interior area.
    """
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
    # Keep higher-density maps corridor-heavy instead of over-filling all cells.
    target_ratio = max(0.18, min(0.62, (0.24 + (0.22 * complexity)) * (0.72 + (0.28 * density))))
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
            # Reject candidates that would immediately create crowded clusters.
            neighbourhood = 0
            for odx, ody in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (nx + odx, ny + ody) in roads:
                    neighbourhood += 1
            if neighbourhood >= 3 and rng.random() < 0.8:
                continue
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
    loop_ratio = (0.01 + (0.06 * complexity)) * density
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
    center_v = max(1, min(width - 2, width // 2))
    center_h = max(1, min(height - 2, height // 2))

    if density < 0.8:
        if rng.random() < 0.5:
            verticals.append(center_v)
        else:
            horizontals.append(center_h)
    else:
        verticals.append(center_v)
        horizontals.append(center_h)

    if density >= 1.00 and complexity >= 0.45 and width >= 12:
        verticals.append(max(1, min(width - 2, (width // 3) + rng.choice((-1, 0, 1)))))
    if density >= 1.00 and complexity >= 0.45 and height >= 12:
        horizontals.append(max(1, min(height - 2, (height // 3) + rng.choice((-1, 0, 1)))))
    # Keep arterial overlays capped so complex maps remain maze-like.
    if density >= 1.22 and complexity >= 0.78 and width >= 15:
        verticals.append(max(1, min(width - 2, ((2 * width) // 3) + rng.choice((-1, 0, 1)))))
    if density >= 1.22 and complexity >= 0.78 and height >= 15:
        horizontals.append(max(1, min(height - 2, ((2 * height) // 3) + rng.choice((-1, 0, 1)))))

    unique_verticals = sorted(set(verticals))[:2]
    unique_horizontals = sorted(set(horizontals))[:2]
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


def _roundabout_ring_nodes(center: GridPoint) -> List[GridPoint]:
    """
    Return the 8 perimeter cells of a 3x3 roundabout footprint.
    """
    cx, cy = center
    return [
        (cx - 1, cy - 1),
        (cx, cy - 1),
        (cx + 1, cy - 1),
        (cx - 1, cy),
        (cx + 1, cy),
        (cx - 1, cy + 1),
        (cx, cy + 1),
        (cx + 1, cy + 1),
    ]


def _roundabout_clearance_distance(structure_density: float) -> float:
    """
    Density-scaled minimum Euclidean spacing between roundabout centers.

    Low structure density keeps roundabouts farther apart (~6 tiles),
    while high density allows tighter placement (~5 tiles).
    """
    density = _clamp_density(structure_density, 0.2, 1.5)
    ratio = (density - 0.2) / 1.3
    ratio = max(0.0, min(1.0, ratio))
    return 6.0 - ratio


def _roundabout_base_exit_length(structure_density: float) -> int:
    """
    Density-scaled base exit length for roundabout cardinal exits.

    Low density: minimum 5 cells. High density: minimum 3 cells.
    """
    density = _clamp_density(structure_density, 0.2, 1.5)
    ratio = (density - 0.2) / 1.3
    ratio = max(0.0, min(1.0, ratio))
    return int(round(5.0 - (2.0 * ratio)))


def _structure_spacing_distance(structure_density: float, rng: random.Random) -> float:
    """
    Density-scaled spacing for non-roundabout structures.

    Rules:
    - Min density: base spacing 10 with random chance to allow as low as 7.
    - Max density: fixed spacing 5 (no random reduction).
    """
    density = _clamp_density(structure_density, 0.2, 1.5)
    if density >= 1.5:
        return 5.0

    ratio = (density - 0.2) / 1.3
    ratio = max(0.0, min(1.0, ratio))
    base_distance = 10.0 - (5.0 * ratio)

    # Keep requested variance at lower densities only.
    if rng.random() < 0.35:
        base_distance = max(5.0, base_distance - float(rng.randint(1, 3)))
    return max(5.0, base_distance)


def _is_roundabout_candidate(
    roads: Set[GridPoint],
    node: GridPoint,
    size: tuple[int, int],
    exit_lengths: tuple[int, int, int, int],
) -> bool:
    """
    Validate a roundabout candidate with per-direction exit-length bounds.
    """
    width, height = size
    cx, cy = node
    if not (2 <= cx <= (width - 3) and 2 <= cy <= (height - 3)):
        return False

    # Validate full 3x3 placement footprint is interior.
    for xx in range(cx - 1, cx + 2):
        for yy in range(cy - 1, cy + 2):
            if not _interior_bounds(xx, yy, width, height):
                return False

    # Validate each cardinal exit endpoint remains within interior bounds.
    north_len, east_len, south_len, west_len = exit_lengths
    exit_checks = (
        (cx, cy - 1 - north_len),
        (cx + 1 + east_len, cy),
        (cx, cy + 1 + south_len),
        (cx - 1 - west_len, cy),
    )
    for x_pos, y_pos in exit_checks:
        if not _interior_bounds(x_pos, y_pos, width, height):
            return False
    return True


def _place_roundabout(
    roads: Set[GridPoint],
    node: GridPoint,
    size: tuple[int, int],
    exit_lengths: tuple[int, int, int, int],
) -> bool:
    """
    Place a 3x3 roundabout:
    - centre cell is blocked,
    - surrounding 8 cells become ring road,
    - exits are extended in 4 cardinal directions.
    """
    if not _is_roundabout_candidate(roads, node, size, exit_lengths=exit_lengths):
        return False

    cx, cy = node

    # Ring shape: keep perimeter as road and centre as obstacle.
    for ring_cell in _roundabout_ring_nodes(node):
        roads.add(ring_cell)
    roads.discard((cx, cy))

    north_len, east_len, south_len, west_len = exit_lengths
    _add_line(roads, cx, cy - 1, cx, cy - 1 - north_len)
    _add_line(roads, cx + 1, cy, cx + 1 + east_len, cy)
    _add_line(roads, cx, cy + 1, cx, cy + 1 + south_len)
    _add_line(roads, cx - 1, cy, cx - 1 - west_len, cy)
    return True


def _select_spaced_nodes(
    candidates: List[GridPoint],
    rng: random.Random,
    count: int,
    min_distance: float,
    blocked_nodes: List[GridPoint] | None = None,
) -> List[GridPoint]:
    """
    Select nodes with Euclidean spacing.
    """
    pool = list(candidates)
    rng.shuffle(pool)
    chosen: List[GridPoint] = []
    blocked = list(blocked_nodes or [])
    for node in pool:
        if all(_euclidean(node, other) >= min_distance for other in chosen) and all(
            _euclidean(node, other) >= min_distance for other in blocked
        ):
            chosen.append(node)
            if len(chosen) >= count:
                break
    return chosen


def _tag_turn_and_junction_centres(
    roads: Set[GridPoint],
    node_types: Dict[GridPoint, str],
    protected_centres: List[GridPoint] | None = None,
    clearance_distance: float = 0.0,
) -> None:
    """
    Add visual tags for:
    - generic junction centres (degree >= 3),
    - turning road tiles (degree == 2 and orthogonal).
    """
    graph = _build_graph(roads)
    protected = list(protected_centres or [])
    for node, neighbours in graph.items():
        if node_types.get(node) == "roundabout_center":
            continue
        if protected and any(_euclidean(node, center) < clearance_distance for center in protected):
            continue

        if len(neighbours) >= 3:
            node_types.setdefault(node, "junction_center")
            continue

        if len(neighbours) != 2:
            continue

        a_node, b_node = neighbours
        # A corner turn has one horizontal and one vertical connection.
        is_turn = (a_node[0] != b_node[0]) and (a_node[1] != b_node[1])
        if is_turn:
            node_types.setdefault(node, "road_turn")


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
    rng = random.Random((int(seed) * 19) + 5)
    density = _clamp_density(structure_density, 0.2, 1.5)

    # Base lane markup.
    for cell in two_lane_cells:
        if cell in roads:
            node_types[cell] = "road_two_lane"

    roundabout_centres: List[GridPoint] = []

    # Roundabouts are generated first in continuous phases.
    if continuous:
        graph = _build_graph(roads)
        round_candidates = [node for node, neigh in graph.items() if len(neigh) >= 2]
        round_count = max(0, min(len(round_candidates), int(round((0.35 + (complexity * 1.4)) * density))))
        if complexity >= 0.45 and round_candidates:
            round_count = max(1, round_count)
        round_spacing = _roundabout_clearance_distance(density)
        base_exit_len = _roundabout_base_exit_length(density)

        for node in _select_spaced_nodes(round_candidates, rng, round_count * 4, min_distance=round_spacing):
            exit_lengths = (
                base_exit_len + rng.randint(0, 2),
                base_exit_len + rng.randint(0, 2),
                base_exit_len + rng.randint(0, 2),
                base_exit_len + rng.randint(0, 2),
            )
            if not _place_roundabout(roads, node, size, exit_lengths=exit_lengths):
                continue

            roundabout_centres.append(node)
            node_types[node] = "roundabout_center"
            for ring_cell in _roundabout_ring_nodes(node):
                node_types[ring_cell] = "roundabout_ring"

            if len(roundabout_centres) >= round_count:
                break

    graph = _build_graph(roads)
    structure_spacing = _structure_spacing_distance(density, rng)
    occupied_structure_nodes: List[GridPoint] = list(roundabout_centres)

    t_candidates = [node for node, neigh in graph.items() if len(neigh) == 3]
    cross_candidates = [node for node, neigh in graph.items() if len(neigh) >= 4]

    t_count = max(0, min(len(t_candidates), int(round(len(t_candidates) * (0.14 + (0.06 * complexity)) * density))))
    cross_count = max(0, min(len(cross_candidates), int(round(len(cross_candidates) * (0.18 + (0.08 * complexity)) * density))))
    for node in _select_spaced_nodes(
        t_candidates,
        rng,
        t_count,
        min_distance=structure_spacing,
        blocked_nodes=occupied_structure_nodes,
    ):
        node_types[node] = "junction_t"
        occupied_structure_nodes.append(node)
    for node in _select_spaced_nodes(
        cross_candidates,
        rng,
        cross_count,
        min_distance=structure_spacing,
        blocked_nodes=occupied_structure_nodes,
    ):
        node_types[node] = "junction_cross"
        occupied_structure_nodes.append(node)

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
    for node in _select_spaced_nodes(
        turn_one_candidates,
        rng,
        turn_one_count,
        min_distance=structure_spacing,
        blocked_nodes=occupied_structure_nodes,
    ):
        node_types[node] = "junction_turn_one_lane"
        occupied_structure_nodes.append(node)
    for node in _select_spaced_nodes(
        turn_two_candidates,
        rng,
        turn_two_count,
        min_distance=structure_spacing,
        blocked_nodes=occupied_structure_nodes,
    ):
        node_types[node] = "junction_turn_two_lane"
        occupied_structure_nodes.append(node)

    _tag_turn_and_junction_centres(
        roads,
        node_types,
        protected_centres=roundabout_centres,
        clearance_distance=_roundabout_clearance_distance(density),
    )

    return node_types


def _structure_counts(node_types: Dict[GridPoint, str]) -> Dict[str, int]:
    """
    Count occurrences of known structure labels.
    """
    counts = {key: 0 for key in STRUCTURE_KEYS}
    for value in node_types.values():
        if value == "roundabout_center":
            counts["roundabout"] += 1
            continue
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


def _euclidean(a_node: GridPoint, b_node: GridPoint) -> float:
    """
    Compute Euclidean distance between two grid cells.
    """
    dx = float(a_node[0] - b_node[0])
    dy = float(a_node[1] - b_node[1])
    return math.sqrt((dx * dx) + (dy * dy))


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


def _bfs_distance_map(graph: Dict[GridPoint, List[GridPoint]], start: GridPoint) -> Dict[GridPoint, int]:
    """
    Use:
    Build a BFS distance map from a start road node.

    Inputs:
    - graph: Road adjacency map.
    - start: BFS origin node.

    Output:
    Mapping of reachable node -> hop distance from `start`.
    """
    distances: Dict[GridPoint, int] = {start: 0}
    queue: deque[GridPoint] = deque([start])
    while queue:
        node = queue.popleft()
        base = distances[node]
        for nxt in graph.get(node, []):
            if nxt not in distances:
                distances[nxt] = base + 1
                queue.append(nxt)
    return distances


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
    selected_destinations: List[GridPoint] = []
    # Keep at least one empty tile between destination cells.
    destination_min_gap = 2

    pool = list(nodes)
    rng.shuffle(pool)
    for node in pool:
        if all(_manhattan(node, other) >= 3 for other in selected_spawns):
            selected_spawns.append(node)
            if len(selected_spawns) >= count:
                break
    if len(selected_spawns) < count:
        for node in pool:
            if node not in selected_spawns:
                selected_spawns.append(node)
                if len(selected_spawns) >= count:
                    break

    vehicles: List[PreviewVehicle] = []
    for vehicle_id, spawn in enumerate(selected_spawns):
        # Destinations are chosen after map generation from driveable road nodes.
        distance_map = _bfs_distance_map(graph, spawn)
        candidates = [node for node in distance_map.keys() if node != spawn and node in roads]

        # Prefer farther nodes first so preview routes are visibly meaningful.
        candidates.sort(key=lambda node: distance_map[node], reverse=True)
        if candidates:
            # Shuffle equal-distance groups deterministically so multiple vehicles
            # do not always pick the same branch when a tie exists.
            grouped: Dict[int, List[GridPoint]] = {}
            for node in candidates:
                grouped.setdefault(distance_map[node], []).append(node)
            candidates = []
            for distance in sorted(grouped.keys(), reverse=True):
                group_nodes = grouped[distance]
                rng.shuffle(group_nodes)
                candidates.extend(group_nodes)

        destination = next(
            (
                node
                for node in candidates
                if all(_manhattan(node, existing) >= destination_min_gap for existing in selected_destinations)
            ),
            None,
        )

        if destination is None:
            # Graceful fallback for very dense multi-vehicle previews on sparse maps:
            # keep destination on-road even if spacing cannot be fully satisfied.
            if candidates:
                destination = candidates[0]
            else:
                fallback = [node for node in nodes if node != spawn and node in roads]
                destination = rng.choice(fallback) if fallback else spawn

        selected_destinations.append(destination)
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
    node_types = {
        node: typ
        for node, typ in node_types.items()
        if node in roads or typ in ("roundabout_center", "roundabout_ring")
    }
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
