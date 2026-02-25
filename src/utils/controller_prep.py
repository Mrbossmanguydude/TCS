"""
Controller preparation helpers for VN + PN setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from src.utils.map_generation import GeneratedMap, GridPoint, PreviewVehicle


VN_FEATURE_NAMES: Tuple[str, ...] = (
    "spawn_x_norm",
    "spawn_y_norm",
    "dest_x_norm",
    "dest_y_norm",
    "goal_dx_norm",
    "goal_dy_norm",
    "manhattan_to_goal_norm",
    "is_continuous",
    "local_degree_norm",
    "nearby_road_density",
    "neighbour_1_dist_norm",
    "neighbour_2_dist_norm",
)


@dataclass(frozen=True)
class VN:
    """
    Use:
    Describe the Vehicle Network input contract.

    Inputs:
    - feature_names: Ordered VN feature labels.
    - input_size: Length of each VN feature vector.

    Output:
    Immutable VN interface description.
    """

    feature_names: Tuple[str, ...]
    input_size: int


@dataclass(frozen=True)
class PN:
    """
    Use:
    Describe the Policy Network input/output contract.

    Inputs:
    - input_size: Expected PN input width.
    - action_size: Number of actions emitted by policy head.
    - uses_previous_model: Indicates if a previous model checkpoint was found.
    - previous_model_path: Resolved model path when available.

    Output:
    Immutable PN interface description.
    """

    input_size: int
    action_size: int
    uses_previous_model: bool
    previous_model_path: str


@dataclass(frozen=True)
class PreparedControllers:
    """
    Use:
    Aggregate VN and PN preparation metadata.

    Inputs:
    - vn: VN preparation details.
    - pn: PN preparation details.

    Output:
    Single object returned by setup initialisation.
    """

    vn: VN
    pn: PN


def _norm(value: float, lower: float, upper: float) -> float:
    """
    Use:
    Normalize a scalar into [0, 1].

    Inputs:
    - value: Raw numeric value.
    - lower: Lower bound.
    - upper: Upper bound.

    Output:
    Normalized float in [0, 1].
    """
    if upper <= lower:
        return 0.0
    clipped = max(lower, min(upper, float(value)))
    return (clipped - lower) / (upper - lower)


def _neighbour_distance(source: GridPoint, neighbours: Sequence[GridPoint], index: int) -> float:
    """
    Use:
    Return Manhattan distance to neighbour by index.

    Inputs:
    - source: Source grid point.
    - neighbours: Candidate neighbour points.
    - index: Desired neighbour slot.

    Output:
    Manhattan distance or 0.0 when unavailable.
    """
    if index >= len(neighbours):
        return 0.0
    node = neighbours[index]
    return float(abs(source[0] - node[0]) + abs(source[1] - node[1]))


def build_vn_feature_vector(
    vehicle: PreviewVehicle,
    generated_map: GeneratedMap,
    neighbour_nodes: Sequence[GridPoint] | None = None,
) -> List[float]:
    """
    Use:
    Build VN feature vector from preview vehicle/map state.

    Inputs:
    - vehicle: Vehicle preview state with spawn + destination.
    - generated_map: Map payload containing roads and phase flags.
    - neighbour_nodes: Optional neighbouring vehicle positions.

    Output:
    Ordered feature vector matching `VN_FEATURE_NAMES`.
    """
    neighbours = list(neighbour_nodes or [])
    width = max(1, generated_map.width - 1)
    height = max(1, generated_map.height - 1)

    sx, sy = vehicle.spawn
    dx, dy = vehicle.destination
    goal_dx = dx - sx
    goal_dy = dy - sy
    manhattan = abs(goal_dx) + abs(goal_dy)

    # Estimate local road density in a 3x3 window around spawn.
    local_cells = 0
    local_roads = 0
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            nx, ny = sx + ox, sy + oy
            if 0 <= nx < generated_map.width and 0 <= ny < generated_map.height:
                local_cells += 1
                if (nx, ny) in generated_map.roads:
                    local_roads += 1
    local_density = float(local_roads) / float(max(1, local_cells))

    # Degree approximation from cardinal neighbours.
    degree = 0
    for cx, cy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if (sx + cx, sy + cy) in generated_map.roads:
            degree += 1

    return [
        _norm(sx, 0, width),
        _norm(sy, 0, height),
        _norm(dx, 0, width),
        _norm(dy, 0, height),
        _norm(goal_dx, -width, width),
        _norm(goal_dy, -height, height),
        _norm(manhattan, 0, width + height),
        1.0 if generated_map.continuous else 0.0,
        _norm(degree, 0, 4),
        max(0.0, min(1.0, local_density)),
        _norm(_neighbour_distance(vehicle.spawn, neighbours, 0), 0, width + height),
        _norm(_neighbour_distance(vehicle.spawn, neighbours, 1), 0, width + height),
    ]


def prepare_vn_pn(config: Dict[str, Any], base_dir: Path) -> PreparedControllers:
    """
    Use:
    Prepare VN + PN metadata from runtime configuration.

    Inputs:
    - config: Runtime configuration dictionary.
    - base_dir: Project root used to resolve relative checkpoint paths.

    Output:
    `PreparedControllers` object for setup/train usage.
    """
    train_cfg = config.get("train", {})
    action_size = int(max(3, min(9, train_cfg.get("action_size", 5))))

    raw_prev = str(train_cfg.get("prev_model_path", "")).strip()
    resolved_path = ""
    uses_previous_model = False
    if raw_prev:
        path = Path(raw_prev)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        resolved_path = str(path)
        uses_previous_model = path.exists()

    vn = VN(feature_names=VN_FEATURE_NAMES, input_size=len(VN_FEATURE_NAMES))
    pn = PN(
        input_size=vn.input_size,
        action_size=action_size,
        uses_previous_model=uses_previous_model,
        previous_model_path=resolved_path,
    )
    return PreparedControllers(vn=vn, pn=pn)

