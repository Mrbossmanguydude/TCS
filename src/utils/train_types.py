"""
Typed data structures shared by TRAIN backend and GUI.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# INITIALISATION / STARTUP MATERIAL                                           #
# --------------------------------------------------------------------------- #

from dataclasses import dataclass
from typing import Dict, List


# --------------------------------------------------------------------------- #
# CLASSES                                                                     #
# --------------------------------------------------------------------------- #

@dataclass
class EpisodeVehicle:
    """
    Purpose:
    Represent mutable state for one vehicle during an active episode.

    Attributes:
    - vehicle_id: Stable identifier inside the episode.
    - spawn: Initial spawn cell.
    - destination: Target destination cell.
    - position: Current float-grid position.
    - heading_deg: Current heading angle for rendering/motion logic.
    - remaining_distance: Distance estimate to destination.
    - continuous: Whether movement uses continuous motion rules.
    - arrived: Arrival completion flag.
    - travel_steps: Number of active movement steps taken.
    - wait_steps: Number of blocked/idle steps.
    - collisions: Collision count for this vehicle in current episode.
    """

    vehicle_id: int
    spawn: tuple[int, int]
    destination: tuple[int, int]
    position: tuple[float, float]
    heading_deg: float
    remaining_distance: float
    continuous: bool
    arrived: bool
    travel_steps: int
    wait_steps: int
    collisions: int


@dataclass
class EpisodeState:
    """
    Purpose:
    Store all runtime episode state required for stepping, rendering, and metrics.

    Attributes:
    - episode_index: Episode counter within current run.
    - seed: Deterministic seed for the episode environment reset.
    - phase: Active curriculum phase.
    - level_index: Active map complexity level index.
    - vehicles: Live vehicle states for the episode.
    - step_count: Current simulation step count.
    - elapsed_seconds: Wall-clock elapsed time for this episode.
    - metrics: Per-episode metric values accumulated so far.
    - done: Terminal flag for episode completion.
    - passed: Whether this episode met progression thresholds.
    """

    episode_index: int
    seed: int
    phase: int
    level_index: int
    vehicles: List[EpisodeVehicle]
    step_count: int
    elapsed_seconds: float
    metrics: Dict[str, float]
    done: bool
    passed: bool


@dataclass
class EpisodeSummary:
    """
    Purpose:
    Provide compact post-episode metrics shown in UI and logs.

    Attributes:
    - passed: Whether progression thresholds were satisfied.
    - success_rate: Fraction of vehicles that reached destinations.
    - collision_rate: Collision frequency metric.
    - throughput: Throughput score for completed journeys.
    - avg_journey_time: Average travel duration metric.
    - reward: Aggregate reward for the episode.
    - loss: PPO update loss value associated with the episode.
    """

    passed: bool
    success_rate: float
    collision_rate: float
    throughput: float
    avg_journey_time: float
    reward: float
    loss: float


@dataclass
class VehicleRollout:
    """
    Purpose:
    Capture one vehicle rollout trajectory for PPO training updates.

    Attributes:
    - observations: Ordered observation vectors per step.
    - actions: Sampled action per step.
    - log_probs: Policy log-probability of each selected action.
    - values: Critic value estimate per step.
    - rewards: Reward signal per step.
    - dones: Terminal markers aligned with step entries.
    """

    observations: List[List[float]]
    actions: List[int]
    log_probs: List[float]
    values: List[float]
    rewards: List[float]
    dones: List[float]
