"""
Backend helper functions for TRAIN simulation/runtime logic.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# INITIALISATION / STARTUP MATERIAL                                           #
# --------------------------------------------------------------------------- #

from collections import deque
import math
import random
from typing import Dict, List, Optional, Sequence, Set

from src.utils.controller_prep import build_vn_feature_vector
from src.utils.map_generation import GeneratedMap, PreviewVehicle
from src.utils.train_types import EpisodeState, EpisodeVehicle


# --------------------------------------------------------------------------- #
# FUNCTIONS AND HELPERS                                                       #
# --------------------------------------------------------------------------- #

def manhattan_distance(a_node: tuple[int, int], b_node: tuple[int, int]) -> int:
    """
    Type:
    FUNCTION.
    
    Purpose:
    Return Manhattan distance on the tile grid.
    
    Inputs:
    - a_node: Input parameter used by this routine.
    - b_node: Input parameter used by this routine.
    
    Outputs:
    - Returnvalue: Computed result returned to the caller.
    """
    return abs(a_node[0] - b_node[0]) + abs(a_node[1] - b_node[1])


def cell_centre(cell: tuple[int, int]) -> tuple[float, float]:
    """
    Type:
    FUNCTION.
    
    Purpose:
    Return cell centre in grid-space float coordinates.
    
    Inputs:
    - cell: Input parameter used by this routine.
    
    Outputs:
    - Returnvalue: Computed result returned to the caller.
    """
    return float(cell[0]) + 0.5, float(cell[1]) + 0.5


def world_distance(a_pos: tuple[float, float], b_pos: tuple[float, float]) -> float:
    """
    Type:
    FUNCTION.
    
    Purpose:
    Return Euclidean distance between two float grid-space positions.
    
    Inputs:
    - a_pos: Input parameter used by this routine.
    - b_pos: Input parameter used by this routine.
    
    Outputs:
    - Returnvalue: Computed result returned to the caller.
    """
    dx = float(a_pos[0] - b_pos[0])
    dy = float(a_pos[1] - b_pos[1])
    return math.sqrt((dx * dx) + (dy * dy))


def position_to_cell(position: tuple[float, float]) -> tuple[int, int]:
    """
    Type:
    FUNCTION.
    
    Purpose:
    Map float grid-space position to owning tile cell.
    
    Inputs:
    - position: Input parameter used by this routine.
    
    Outputs:
    - Returnvalue: Computed result returned to the caller.
    """
    return int(position[0]), int(position[1])


def heading_from_vector(dx: float, dy: float, fallback: float = 0.0) -> float:
    """
    Type:
    FUNCTION.
    
    Purpose:
    Convert movement vector into sprite heading degrees.
    
    Inputs:
    - dx: Input parameter used by this routine.
    - dy: Input parameter used by this routine.
    - fallback: Input parameter used by this routine.
    
    Outputs:
    - Returnvalue: Computed result returned to the caller.
    """
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return float(fallback)
    return float(math.degrees(math.atan2(dx, dy)))


def phase_reward_weights(phase: int) -> Dict[str, float]:
    """
    Return phase-aware reward weights tuned to curriculum goals.

    Emphasis profile:
    - Phase 1: routing/efficiency dominant.
    - Phase 2: collision avoidance + congestion dominant, routing minor.
    - Phase 3: balanced routing/collision/congestion.
    - Phase 4: heavy routing emphasis.
    - Phase 5: heavy routing + collision emphasis.
    - Phase 6: near-equal emphasis across all components.
    """
    phase_id = max(1, min(6, int(phase)))
    reward_table: Dict[int, Dict[str, float]] = {
        1: {
            "arrival_reward": 4.0,
            "progress_reward": 0.42,
            "idle_penalty": 0.010,
            "wait_penalty": 0.018,
            "collision_penalty": 4.0,
        },
        2: {
            "arrival_reward": 1.0,
            "progress_reward": 0.05,
            "idle_penalty": 0.060,
            "wait_penalty": 0.090,
            "collision_penalty": 13.0,
        },
        3: {
            "arrival_reward": 2.4,
            "progress_reward": 0.22,
            "idle_penalty": 0.042,
            "wait_penalty": 0.048,
            "collision_penalty": 9.0,
        },
        4: {
            "arrival_reward": 4.2,
            "progress_reward": 0.46,
            "idle_penalty": 0.008,
            "wait_penalty": 0.012,
            "collision_penalty": 4.5,
        },
        5: {
            "arrival_reward": 3.2,
            "progress_reward": 0.28,
            "idle_penalty": 0.020,
            "wait_penalty": 0.030,
            "collision_penalty": 11.0,
        },
        6: {
            "arrival_reward": 2.8,
            "progress_reward": 0.24,
            "idle_penalty": 0.030,
            "wait_penalty": 0.032,
            "collision_penalty": 10.0,
        },
    }
    return dict(reward_table[phase_id])


def phase_pass_thresholds(phase: int) -> tuple[float, float]:
    """
    Return phase-specific pass thresholds.

    Output tuple:
    - success threshold (minimum success_rate to pass),
    - collision threshold (maximum collision_rate to pass).
    """
    phase_id = max(1, min(6, int(phase)))
    threshold_table: Dict[int, tuple[float, float]] = {
        1: (0.72, 0.14),  # Moderate
        2: (0.84, 0.06),  # High
        3: (0.86, 0.05),  # High
        4: (0.60, 0.24),  # Low
        5: (0.76, 0.12),  # Moderate
        6: (0.92, 0.04),  # Much higher
    }
    return threshold_table[phase_id]


def phase_step_limit(base_steps: int, phase: int, map_width: int, map_height: int) -> int:
    """
    Type:
    FUNCTION.
    
    Purpose:
    Return effective episode step cap for one phase/map.
    
    Inputs:
    - base_steps: Input parameter used by this routine.
    - phase: Input parameter used by this routine.
    - map_width: Input parameter used by this routine.
    - map_height: Input parameter used by this routine.
    
    Outputs:
    - Returnvalue: Computed result returned to the caller.
    """
    base = float(max(30, int(base_steps)))
    map_area = float(max(1, int(map_width) * int(map_height)))
    reference_area = 24.0 * 24.0
    size_scalar = max(1.0, math.sqrt(map_area / reference_area))

    phase_clamped = max(1, min(6, int(phase)))
    scale_damp = max(0.60, 1.0 - (0.07 * float(phase_clamped - 1)))
    phase_bonus = 1.0 + (0.03 * float(phase_clamped - 1))

    scaled_steps = base * (1.0 + ((size_scalar - 1.0) * scale_damp)) * phase_bonus
    return int(max(30, round(scaled_steps)))


def collision_loss_multiplier(phase: int) -> float:
    """
    Type:
    FUNCTION.
    
    Purpose:
    Return phase-aware collision loss multiplier.
    
    Inputs:
    - phase: Input parameter used by this routine.
    
    Outputs:
    - Returnvalue: Computed result returned to the caller.
    """
    phase_id = max(1, min(6, int(phase)))
    multiplier_table: Dict[int, float] = {
        1: 6.0,
        2: 20.0,
        3: 14.0,
        4: 6.5,
        5: 15.0,
        6: 12.0,
    }
    return multiplier_table[phase_id]


def road_neighbours(
    roads: Set[tuple[int, int]],
    node_types: Dict[tuple[int, int], str],
    cell: tuple[int, int],
) -> List[tuple[int, int]]:
    """
    Type:
    FUNCTION.
    
    Purpose:
    Return cardinal road neighbours for one road cell.
    
    Inputs:
    - roads: Input parameter used by this routine.
    - node_types: Input parameter used by this routine.
    - cell: Input parameter used by this routine.
    
    Outputs:
    - Returnvalue: Computed result returned to the caller.
    """
    x_pos, y_pos = cell
    neighbours: List[tuple[int, int]] = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nxt = (x_pos + dx, y_pos + dy)
        if nxt in roads and node_types.get(nxt, "") != "roundabout_centre":
            neighbours.append(nxt)
    return neighbours


def next_path_cell(
    start: tuple[int, int],
    destination: tuple[int, int],
    roads: Set[tuple[int, int]],
    node_types: Dict[tuple[int, int], str],
) -> Optional[tuple[int, int]]:
    """
    Type:
    FUNCTION.
    
    Purpose:
    Resolve next road cell on shortest path from start to destination.
    
    Inputs:
    - start: Input parameter used by this routine.
    - destination: Input parameter used by this routine.
    - roads: Input parameter used by this routine.
    - node_types: Input parameter used by this routine.
    
    Outputs:
    - Returnvalue: Computed result returned to the caller.
    """
    if start == destination:
        return start
    if start not in roads or destination not in roads:
        return None

    queue: deque[tuple[int, int]] = deque([start])
    parent: Dict[tuple[int, int], Optional[tuple[int, int]]] = {start: None}

    while queue:
        node = queue.popleft()
        if node == destination:
            break
        for nxt in road_neighbours(roads=roads, node_types=node_types, cell=node):
            if nxt in parent:
                continue
            parent[nxt] = node
            queue.append(nxt)

    if destination not in parent:
        return None

    cursor = destination
    while parent[cursor] is not None and parent[cursor] != start:
        cursor = parent[cursor]  # type: ignore[index]
    if parent[cursor] is None:
        return cursor
    return cursor


def is_driveable_position(
    position: tuple[float, float],
    roads: Set[tuple[int, int]],
    node_types: Dict[tuple[int, int], str],
    width: int,
    height: int,
) -> bool:
    """
    Type:
    FUNCTION.
    
    Purpose:
    Check whether one float position belongs to a valid drivable tile.
    
    Inputs:
    - position: Input parameter used by this routine.
    - roads: Input parameter used by this routine.
    - node_types: Input parameter used by this routine.
    - width: Input parameter used by this routine.
    - height: Input parameter used by this routine.
    
    Outputs:
    - Returnvalue: Computed result returned to the caller.
    """
    x_pos, y_pos = position
    if x_pos < 0.0 or y_pos < 0.0 or x_pos >= float(width) or y_pos >= float(height):
        return False
    cell = position_to_cell(position)
    if node_types.get(cell, "") == "roundabout_centre":
        return False
    return cell in roads


def target_cell_from_action(
    vehicle: EpisodeVehicle,
    action: int,
    roads: Set[tuple[int, int]],
    node_types: Dict[tuple[int, int], str],
) -> Optional[tuple[int, int]]:
    """
    Type:
    FUNCTION.
    
    Purpose:
    Map raw policy action id to one direct neighbouring road target cell.
    
    Inputs:
    - vehicle: Input parameter used by this routine.
    - action: Input parameter used by this routine.
    - roads: Input parameter used by this routine.
    - node_types: Input parameter used by this routine.
    Outputs:
    - Returnvalue: Computed result returned to the caller.
    """
    current_cell = position_to_cell(vehicle.position)
    action_idx = int(action) % 5
    if action_idx == 0:
        return None

    # Action mapping is intentionally fixed and heuristic-free:
    # 0=stay, 1=up, 2=right, 3=down, 4=left
    deltas = {
        1: (0, -1),
        2: (1, 0),
        3: (0, 1),
        4: (-1, 0),
    }
    dx, dy = deltas.get(action_idx, (0, 0))
    if dx == 0 and dy == 0:
        return None
    target = (current_cell[0] + dx, current_cell[1] + dy)
    if target not in roads:
        return None
    if node_types.get(target, "") == "roundabout_centre":
        return None
    return target


def reset_vehicle_to_spawn(vehicle: EpisodeVehicle) -> None:
    """
    Type:
    PROCEDURE.
    
    Purpose:
    Reset one vehicle to spawn state (setback behaviour).
    
    Inputs:
    - vehicle: Input parameter used by this routine.
    
    Outputs:
    - Updates object/runtime state in place; no return value.
    """
    vehicle.position = cell_centre(vehicle.spawn)
    destination_centre = cell_centre(vehicle.destination)
    vehicle.remaining_distance = (
        world_distance(vehicle.position, destination_centre)
        if vehicle.continuous
        else float(max(1, manhattan_distance(vehicle.spawn, vehicle.destination)))
    )
    vehicle.heading_deg = heading_from_vector(
        destination_centre[0] - vehicle.position[0],
        destination_centre[1] - vehicle.position[1],
        fallback=vehicle.heading_deg,
    )
    vehicle.arrived = False


def nearest_vehicle_nodes(
    source: EpisodeVehicle,
    candidates: Sequence[EpisodeVehicle],
    limit: int = 2,
) -> List[tuple[int, int]]:
    """
    Type:
    FUNCTION.
    
    Purpose:
    Return nearest vehicle cells around one source vehicle.
    
    Inputs:
    - source: Input parameter used by this routine.
    - candidates: Input parameter used by this routine.
    - limit: Input parameter used by this routine.
    
    Outputs:
    - Returnvalue: Computed result returned to the caller.
    """
    source_cell = position_to_cell(source.position)
    neighbours: List[tuple[int, tuple[int, int]]] = []
    for other in candidates:
        if other.vehicle_id == source.vehicle_id or other.arrived:
            continue
        other_cell = position_to_cell(other.position)
        dist = abs(source_cell[0] - other_cell[0]) + abs(source_cell[1] - other_cell[1])
        neighbours.append((dist, other_cell))
    neighbours.sort(key=lambda item: item[0])
    return [node for _, node in neighbours[:limit]]


def build_observation_batch(
    state: EpisodeState,
    vehicles: Sequence[EpisodeVehicle],
    generated_map: GeneratedMap,
    step_limit: int,
) -> List[List[float]]:
    """
    Type:
    FUNCTION.
    
    Purpose:
    Build policy observation batch for active vehicles.
    
    Inputs:
    - state: Input parameter used by this routine.
    - vehicles: Input parameter used by this routine.
    - generated_map: Input parameter used by this routine.
    - step_limit: Input parameter used by this routine.
    
    Outputs:
    - Returnvalue: Computed result returned to the caller.
    """
    observations: List[List[float]] = []
    safe_step_limit = max(1, int(step_limit))
    for vehicle in vehicles:
        current_cell = position_to_cell(vehicle.position)
        pseudo_vehicle = PreviewVehicle(
            vehicle_id=int(vehicle.vehicle_id),
            spawn=(int(current_cell[0]), int(current_cell[1])),
            destination=(int(vehicle.destination[0]), int(vehicle.destination[1])),
            continuous=bool(vehicle.continuous),
        )
        neighbour_nodes = nearest_vehicle_nodes(vehicle, vehicles, limit=2)
        base_features = build_vn_feature_vector(
            vehicle=pseudo_vehicle,
            generated_map=generated_map,
            neighbour_nodes=neighbour_nodes,
        )
        dynamic_features = [
            float(state.step_count) / float(safe_step_limit),
            float(vehicle.collisions) / float(max(1, state.step_count)),
            float(state.metrics.get("congestion", 0.0)),
            float(state.metrics.get("throughput", 0.0)),
        ]
        observations.append(base_features + dynamic_features)
    return observations

