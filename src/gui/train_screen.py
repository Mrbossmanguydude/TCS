"""
Train screen template for environment reset and initial episode-condition setup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import pygame

from src.utils.controller_prep import prepare_vn_pn
from src.utils.hold_repeat import HoldRepeatController
from src.utils.map_generation import PreviewVehicle, clamp_phase, generate_phase_map, map_level_count
from src.utils.network_slots import first_empty_slot, load_slots, mark_slot_saved, save_slots, slot_checkpoint_path
from src.utils.ppo_controller import PPOConfig, PPOController, PPOUpdateStats
from src.utils.replay import save_episode_replay_to_slots
from src.utils.run_init import create_episode_record, insert_metric_record, reseed_all, write_rolling_config
from src.utils.train_backend_helpers import (
    build_observation_batch,
    cell_centre,
    collision_loss_multiplier,
    heading_from_vector,
    is_driveable_position,
    manhattan_distance,
    phase_pass_thresholds,
    phase_reward_weights,
    phase_step_limit,
    target_cell_from_action,
    world_distance,
)
from src.utils.train_types import EpisodeState as TrainEpisodeState
from src.utils.train_types import EpisodeSummary
from src.utils.train_types import EpisodeVehicle as TrainEpisodeVehicle
from src.utils.train_types import VehicleRollout


BG = (53, 62, 67)
FG = (235, 235, 235)
ACCENT = (180, 180, 180)
HILITE = (255, 215, 0)
PANEL_BG = (46, 53, 58)
PANEL_BORDER = (78, 91, 99)
MAP_EMPTY_COLOUR = (34, 39, 44)
MAP_GRID_COLOUR = (56, 63, 70)
ROAD_COLOUR = (145, 162, 182)
ROAD_STRUCTURE_LIGHT = (166, 166, 166)
SPAWN_COLOUR = (112, 216, 104)
DEST_COLOUR = (86, 198, 92)


@dataclass
class EpisodeVehicle:
    """
    Use:
    Represent one instantiated episode vehicle for the current environment reset.

    Attributes:
    - vehicle_id: Unique identifier within current episode.
    - spawn: Initial spawn tile.
    - destination: Destination tile.
    - current: Current tile position (derived from `position` each step).
    - position: Current float-grid position (cell-centre based).
    - heading_deg: Current heading used for car rotation.
    - speed: Current speed in cells per simulation step.
    - max_speed: Per-vehicle max speed cap in cells per step.
    - accel: Per-step acceleration/deceleration amount.
    """

    vehicle_id: int
    spawn: tuple[int, int]
    destination: tuple[int, int]
    current: tuple[int, int]
    position: tuple[float, float]
    heading_deg: float
    speed: float
    max_speed: float
    accel: float


@dataclass
class EpisodeState:
    """
    Use:
    Store runtime state for a single episode attempt.

    Attributes:
    - episode_index: Zero-based counter for reset cycles.
    - seed: Deterministic seed used for current reset.
    - vehicles: Instantiated vehicle list.
    - step_count: Number of environment steps elapsed.
    - elapsed_seconds: Elapsed wall time for current episode.
    - metrics: Mutable metric dictionary reset on every environment reset.
    """

    episode_index: int
    seed: int
    vehicles: List[EpisodeVehicle]
    step_count: int
    elapsed_seconds: float
    metrics: Dict[str, float]


class TrainScreen:
    """
    Use:
    Provide a TRAIN template screen with explicit environment reset logic
    and deterministic episode initialisation.

    Attributes:
    - run_ctx: Shared run context used for seed/config access and persistence metadata.
    - phase: Active curriculum phase for TRAIN map initialisation.
    - level_index: Active map complexity level.
    - road_density: Road-density parameter forwarded to map generation.
    - structure_density: Structure-density parameter forwarded to map generation.
    - preview_map: Current generated map used to instantiate episode vehicles.
    - episode_state: Current episode runtime state.
    - episode_running: Whether an episode timer is currently active.
    """

    def __init__(
        self,
        screen_rect: pygame.Rect,
        font_path: Path | None = None,
        run_ctx: Any = None,
        ui_offsets: Dict[str, tuple[int, int]] | None = None,
    ) -> None:
        """
        Use:
        Construct TRAIN screen controls and create initial environment state.

        Inputs:
        - screen_rect: Full display bounds.
        - font_path: Optional custom font path.
        - run_ctx: Shared startup context.

        Output:
        None.
        """
        self.screen_rect = screen_rect
        self.font_path = font_path
        self.run_ctx = run_ctx
        self.ui_offsets = ui_offsets or {}

        scenario_cfg = run_ctx.config.get("scenario", {}) if run_ctx is not None else {}
        self.seed = int(run_ctx.seed) if run_ctx is not None else 0
        self.phase = clamp_phase(int(scenario_cfg.get("phase", 0)) + 1)
        self.level_index = int(max(0, scenario_cfg.get("level_index", 0)))
        self.road_density = float(scenario_cfg.get("preview_road_density", 0.72))
        self.structure_density = float(scenario_cfg.get("preview_structure_density", 0.62))

        self.status_message = "TRAIN ready. Reset environment before beginning episode."
        self.episode_running = False
        self._episode_start_t = 0.0
        self._reset_counter = 0
        self.controllers = None
        self.ppo: Optional[PPOController] = None
        self.preview_obs_batch: List[List[float]] = []
        self.preview_actions: List[int] = []
        self.preview_values: List[float] = []
        self.active_episode_id: Optional[int] = None
        self._next_auto_start_ms = 0
        self.total_episodes_completed = 0
        self.completed_curriculum = False
        self.level_passes = 0
        self.last_summary: Optional[EpisodeSummary] = None
        self._scaled_cache: Dict[tuple[str, int], pygame.Surface] = {}
        self.road_h_img: Optional[pygame.Surface] = None
        self.road_v_img: Optional[pygame.Surface] = None
        self.car_img: Optional[pygame.Surface] = None
        self.training_view = False
        self.training_paused = False
        self.auto_continue_training = True
        self._sim_step_interval_s = 0.18
        self._last_sim_step_t = 0.0
        self._sim_accumulator_s = 0.0
        self._runtime_prev_t = 0.0
        self._max_steps_per_frame = 240
        self._headless_max_steps_per_frame = 4000
        self._headless_frame_budget_s = 0.012
        self.sim_speed = 1.0
        self._speed_hold = HoldRepeatController()
        self._arrow_speed_direction = 0
        self._arrow_speed_last_ms = 0
        self._arrow_speed_interval_ms = 80
        self.last_episode_failed: Optional[bool] = None
        self.last_episode_success: float = 0.0
        self.loaded_network_name: str = "Current policy"
        self.network_save_path = self._project_root() / "data" / "models" / "current_network.pt"
        self.network_meta_path = self._project_root() / "data" / "models" / "current_network_meta.json"
        self.screenshot_dir = self._project_root() / "data" / "screenshots"
        self._pending_screenshot = False
        self._pending_replay_save = False
        self._episode_replay_steps: List[Dict[str, Any]] = []
        self._last_replay_path: Optional[Path] = None
        self._last_replay_save_message: str = ""
        self._phase_hold = HoldRepeatController()
        self._level_hold = HoldRepeatController()
        self._episode_rollouts: Dict[int, VehicleRollout] = {}

        train_cfg = run_ctx.config.get("train", {}) if run_ctx is not None else {}
        options_cfg = run_ctx.config.get("options", {}) if run_ctx is not None else {}
        self.episodes_per_level = max(1, int(train_cfg.get("episodes_per_level", 3)))
        self.success_threshold = float(train_cfg.get("success_threshold", 0.72))
        self.collision_threshold = float(train_cfg.get("collision_threshold", 0.14))
        self.auto_training_delay_ms = max(0, int(train_cfg.get("auto_training_delay_ms", 240)))
        self.phase2_collision_setbacks = bool(train_cfg.get("phase2_collision_setbacks", False))
        self.training_visualised = bool(options_cfg.get("visualise_training", True))
        self.requested_device = str(options_cfg.get("device", "auto")).lower()
        self.active_device = "cpu"
        self.reward_weights = phase_reward_weights(self.phase)

        self.preview_map = generate_phase_map(
            self.seed,
            self.phase,
            self.level_index,
            road_density=self.road_density,
            structure_density=self.structure_density,
        )
        self.episode_state: Optional[EpisodeState] = None

        self._load_preview_assets()
        self._init_backend_bridge()
        self._build_layout()
        self.reset_environment(initial=True)

    def _offset(self, key: str) -> tuple[int, int]:
        """
        Use:
        Resolve an `(x, y)` offset for a named train UI element.

        Inputs:
        - key: Offset key in `ui_offsets`.

        Output:
        Tuple `(dx, dy)`; defaults to `(0, 0)` when unset.
        """
        value = self.ui_offsets.get(key, (0, 0))
        if isinstance(value, tuple) and len(value) == 2:
            return int(value[0]), int(value[1])
        return 0, 0

    def _offset_rect(self, key: str, rect: pygame.Rect) -> pygame.Rect:
        """
        Use:
        Return a moved copy of a base rectangle using an offset key.

        Inputs:
        - key: Offset key in `ui_offsets`.
        - rect: Base rectangle before offset.

        Output:
        Offset-adjusted rectangle.
        """
        dx, dy = self._offset(key)
        return rect.move(dx, dy)

    def _size(self, key: str, default: tuple[int, int]) -> tuple[int, int]:
        """
        Use:
        Resolve a `(width, height)` tuple for a named train UI element.

        Inputs:
        - key: Size key in `ui_offsets`.
        - default: Fallback `(width, height)` when key is unset.

        Output:
        Tuple `(width, height)`.
        """
        value = self.ui_offsets.get(key, default)
        if isinstance(value, tuple) and len(value) == 2:
            return int(value[0]), int(value[1])
        return int(default[0]), int(default[1])

    def _text_pos(self, key: str, base: tuple[int, int]) -> tuple[int, int]:
        """
        Use:
        Return an offset-adjusted text anchor.

        Inputs:
        - key: Offset key in `ui_offsets`.
        - base: Base `(x, y)` coordinate.

        Output:
        Offset-adjusted text position.
        """
        dx, dy = self._offset(key)
        return base[0] + dx, base[1] + dy

    def _build_layout(self) -> None:
        """
        Use:
        Build button and panel rectangles for TRAIN screen.

        Inputs:
        - None.

        Output:
        None.
        """
        back_w, back_h = self._size("back_button_size", (96, 46))
        save_w, save_h = self._size("save_network_button_size", (140, 44))
        load_w, load_h = self._size("load_network_button_size", (140, 44))
        reset_w, reset_h = self._size("reset_button_size", (300, 44))
        begin_w, begin_h = self._size("begin_button_size", (300, 44))
        visualise_w, visualise_h = self._size("visualise_button_size", (300, 44))
        phase_minus_w, phase_minus_h = self._size(
            "phase_minus_button_size",
            self._size("phase_button_size", (52, 44)),
        )
        phase_plus_w, phase_plus_h = self._size(
            "phase_plus_button_size",
            self._size("phase_button_size", (52, 44)),
        )
        level_minus_w, level_minus_h = self._size(
            "level_minus_button_size",
            self._size("level_button_size", (52, 44)),
        )
        level_plus_w, level_plus_h = self._size(
            "level_plus_button_size",
            self._size("level_button_size", (52, 44)),
        )

        self.back_button = self._offset_rect("back_button", pygame.Rect(34, 28, back_w, back_h))
        self.save_network_button = self._offset_rect("save_network_button", pygame.Rect(50, 150, save_w, save_h))
        self.load_network_button = self._offset_rect("load_network_button", pygame.Rect(198, 150, load_w, load_h))
        self.reset_button = self._offset_rect("reset_button", pygame.Rect(50, 204, reset_w, reset_h))
        self.begin_button = self._offset_rect("begin_button", pygame.Rect(50, 258, begin_w, begin_h))
        self.visualise_button = self._offset_rect("visualise_button", pygame.Rect(50, 312, visualise_w, visualise_h))
        self.phase_minus_button = self._offset_rect("phase_minus_button", pygame.Rect(50, 372, phase_minus_w, phase_minus_h))
        self.phase_plus_button = self._offset_rect("phase_plus_button", pygame.Rect(286, 372, phase_plus_w, phase_plus_h))
        self.level_minus_button = self._offset_rect("level_minus_button", pygame.Rect(50, 430, level_minus_w, level_minus_h))
        self.level_plus_button = self._offset_rect("level_plus_button", pygame.Rect(286, 430, level_plus_w, level_plus_h))

        self.left_panel = self._offset_rect("left_panel", pygame.Rect(34, 108, 304, self.screen_rect.height - 132))
        self.right_panel = self._offset_rect(
            "right_panel",
            pygame.Rect(356, 108, self.screen_rect.width - 390, self.screen_rect.height - 132),
        )
        self.training_back_button = self._offset_rect("training_back_button", pygame.Rect(34, 28, 96, 46))
        self.screenshot_button = self._offset_rect("screenshot_button", pygame.Rect(160, 28, 146, 46))
        self.replay_button = self._offset_rect("replay_button", pygame.Rect(316, 28, 146, 46))
        self.play_pause_button = self._offset_rect("play_pause_button", pygame.Rect(472, 28, 130, 46))
        self.speed_minus_button = self._offset_rect("speed_minus_button", pygame.Rect(612, 28, 48, 46))
        self.speed_plus_button = self._offset_rect("speed_plus_button", pygame.Rect(788, 28, 48, 46))
        self.speed_label_rect = self._offset_rect("speed_label_rect", pygame.Rect(670, 28, 108, 46))

    def _font(self, size: int) -> pygame.font.Font:
        """
        Use:
        Return configured font with fallback.

        Inputs:
        - size: Font size in pixels.

        Output:
        Pygame font object.
        """
        return pygame.font.Font(self.font_path, size) if self.font_path else pygame.font.SysFont(None, size)

    def _draw_button(
        self,
        screen: pygame.Surface,
        rect: pygame.Rect,
        text: str,
        text_size: int,
        border_colour: tuple[int, int, int] = HILITE,
    ) -> None:
        """
        Use:
        Render one reusable gradient button.

        Inputs:
        - screen: Destination surface.
        - rect: Button rectangle.
        - text: Label text.
        - text_size: Label font size.
        - border_colour: RGB border colour.

        Output:
        None.
        """
        gradient = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        for y_pos in range(rect.height):
            blend = y_pos / max(1, rect.height - 1)
            red = int(90 * (1 - blend) + 50 * blend)
            green = int(95 * (1 - blend) + 52 * blend)
            blue = int(100 * (1 - blend) + 55 * blend)
            pygame.draw.line(gradient, (red, green, blue), (0, y_pos), (rect.width, y_pos))

        mask = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=10)
        gradient.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        screen.blit(gradient, rect.topleft)
        pygame.draw.rect(screen, border_colour, rect, 2, border_radius=10)

        font = self._font(text_size)
        label = font.render(text, True, FG)
        shadow = font.render(text, True, (0, 0, 0))
        text_rect = label.get_rect(center=rect.center)
        screen.blit(shadow, text_rect.move(1, 1))
        screen.blit(label, text_rect)

    def _project_root(self) -> Path:
        """
        Use:
        Resolve project root path for shared asset loading.

        Inputs:
        - None.

        Output:
        Project root path.
        """
        if self.run_ctx is not None:
            return Path(self.run_ctx.base_dir)
        return Path(__file__).resolve().parents[2]

    def _load_image(self, candidates: List[Path]) -> Optional[pygame.Surface]:
        """
        Use:
        Load the first existing image path from a candidate list.

        Inputs:
        - candidates: Ordered image path candidates.

        Output:
        Loaded surface or `None` if not found.
        """
        for path in candidates:
            if path.exists():
                try:
                    return pygame.image.load(path.as_posix()).convert_alpha()
                except Exception:
                    return None
        return None

    def _trim_transparent_padding(self, image: Optional[pygame.Surface]) -> Optional[pygame.Surface]:
        """
        Use:
        Crop transparent borders so road sprites fill tiles after scaling.

        Inputs:
        - image: Source sprite.

        Output:
        Cropped sprite when possible; otherwise original image.
        """
        if image is None:
            return None
        bounds = image.get_bounding_rect(min_alpha=1)
        if bounds.width <= 0 or bounds.height <= 0:
            return image
        if bounds.width == image.get_width() and bounds.height == image.get_height():
            return image
        return image.subsurface(bounds).copy()

    def _load_preview_assets(self) -> None:
        """
        Use:
        Load road and vehicle sprites for TRAIN map preview.

        Inputs:
        - None.

        Output:
        None.
        """
        primary_root = self._project_root()
        fallback_root = Path(__file__).resolve().parents[2]
        roots = [primary_root]
        if fallback_root != primary_root:
            roots.append(fallback_root)

        self.road_h_img = self._trim_transparent_padding(
            self._load_image(
                [base / "assets" / "road_imgs" / "road_horizontal.png" for base in roots]
                + [base / "assets" / "road_horizontal.png" for base in roots]
            )
        )
        self.road_v_img = self._trim_transparent_padding(
            self._load_image(
                [base / "assets" / "road_imgs" / "road_vertical.png" for base in roots]
                + [base / "assets" / "road_vertical.png" for base in roots]
            )
        )
        self.car_img = self._load_image(
            [base / "assets" / "road_imgs" / "vehicle_imgs" / "car.png" for base in roots]
            + [base / "assets" / "vehicle_imgs" / "car.png" for base in roots]
        )

    def _scaled_asset(self, key: str, image: Optional[pygame.Surface], tile: int) -> Optional[pygame.Surface]:
        """
        Use:
        Return a cached sprite scaled to current tile size.

        Inputs:
        - key: Cache key.
        - image: Source image.
        - tile: Target tile size.

        Output:
        Scaled surface or `None`.
        """
        if image is None:
            return None
        cache_key = (key, int(tile))
        cached = self._scaled_cache.get(cache_key)
        if cached is not None:
            return cached
        scaled = pygame.transform.smoothscale(image, (int(tile), int(tile)))
        self._scaled_cache[cache_key] = scaled
        return scaled

    def _road_orientation(self, x_pos: int, y_pos: int, node_type: str = "") -> str:
        """
        Use:
        Estimate preferred road sprite orientation for one road cell.

        Inputs:
        - x_pos: Cell x-index.
        - y_pos: Cell y-index.
        - node_type: Optional structure/node tag.

        Output:
        One of `horizontal`, `vertical`, or `mixed`.
        """
        roads = self.preview_map.roads
        left = (x_pos - 1, y_pos) in roads
        right = (x_pos + 1, y_pos) in roads
        up = (x_pos, y_pos - 1) in roads
        down = (x_pos, y_pos + 1) in roads

        horizontal_links = int(left) + int(right)
        vertical_links = int(up) + int(down)

        if node_type == "road_two_lane":
            if horizontal_links >= vertical_links and horizontal_links > 0:
                return "horizontal"
            if vertical_links > 0:
                return "vertical"

        has_h = horizontal_links > 0
        has_v = vertical_links > 0
        if has_h and not has_v:
            return "horizontal"
        if has_v and not has_h:
            return "vertical"
        return "mixed"

    def _requested_runtime_device(self) -> str:
        """
        Use:
        Read requested device from runtime config (`auto`, `cpu`, `cuda`).

        Inputs:
        - None.

        Output:
        Normalised device token.
        """
        if self.run_ctx is None:
            return "auto"
        options_cfg = self.run_ctx.config.setdefault("options", {})
        token = str(options_cfg.get("device", "auto")).lower().strip()
        if token not in {"auto", "cpu", "cuda"}:
            token = "auto"
        return token

    def _resolve_runtime_device(self) -> str:
        """
        Use:
        Resolve effective torch device from requested config and hardware.

        Inputs:
        - None.

        Output:
        Resolved text (`cpu` or `cuda`).
        """
        requested = self._requested_runtime_device()
        self.requested_device = requested
        if requested == "cpu":
            return "cpu"
        try:
            import torch  # type: ignore
            has_cuda = bool(torch.cuda.is_available())
        except Exception:
            has_cuda = False

        if requested == "cuda":
            return "cuda" if has_cuda else "cpu"
        return "cuda" if has_cuda else "cpu"

    def _set_training_visualised(self, enabled: bool) -> None:
        """
        Use:
        Update visualisation mode and persist it into runtime config.

        Inputs:
        - enabled: True to render training map/vehicles, False for headless loop.

        Output:
        None.
        """
        self.training_visualised = bool(enabled)
        if not self.training_visualised:
            self._speed_hold.stop()
            self._arrow_speed_direction = 0
            self._sim_accumulator_s = 0.0
        if self.run_ctx is None:
            return
        options_cfg = self.run_ctx.config.setdefault("options", {})
        options_cfg["visualise_training"] = bool(self.training_visualised)
        write_rolling_config(self.run_ctx.base_dir, self.run_ctx.config)

    def _init_backend_bridge(self) -> None:
        """
        Use:
        Initialise backend controller metadata and PPO model for TRAIN preview.

        Inputs:
        - None.

        Output:
        None.
        """
        config = self.run_ctx.config if self.run_ctx is not None else {}
        base_dir = getattr(self.run_ctx, "base_dir", Path(__file__).resolve().parents[2])

        try:
            self.controllers = prepare_vn_pn(config=config, base_dir=base_dir)
            train_cfg = config.get("train", {})
            resolved_device = self._resolve_runtime_device()
            ppo_cfg = PPOConfig(
                gamma=float(train_cfg.get("gamma", 0.99)),
                clip_eps=float(train_cfg.get("clip_eps", 0.2)),
                minibatch_size=int(max(32, train_cfg.get("batch_size", 2048))),
                entropy_coef=float(train_cfg.get("entropy_coef", 0.01)),
            )

            # VN features + 4 dynamic features from build_observation_batch.
            obs_dim = int(self.controllers.vn.input_size + 4)
            self.ppo = PPOController(
                obs_dim=obs_dim,
                action_dim=int(self.controllers.pn.action_size),
                config=ppo_cfg,
                device=resolved_device,
            )
            self.active_device = str(self.ppo.device)
            if self.requested_device == "cuda" and self.active_device != "cuda":
                self.status_message = "CUDA unavailable, using CPU."
        except Exception as exc:
            self.ppo = None
            self.active_device = "cpu"
            self.status_message = f"Backend init failed: {exc}"

    def _build_backend_preview_state(self) -> tuple[TrainEpisodeState, List[TrainEpisodeVehicle]]:
        """
        Use:
        Build a backend-compatible episode state for TRAIN preview.

        Inputs:
        - None.

        Output:
        Tuple of `(state, vehicles)` used for observation-batch generation.
        """
        backend_vehicles: List[TrainEpisodeVehicle] = []
        for vehicle in self.preview_map.vehicles:
            spawn = (int(vehicle.spawn[0]), int(vehicle.spawn[1]))
            destination = (int(vehicle.destination[0]), int(vehicle.destination[1]))
            spawn_centre = cell_centre(spawn)
            destination_centre = cell_centre(destination)
            is_continuous = bool(self.preview_map.continuous or vehicle.continuous)

            if is_continuous:
                remaining_distance = world_distance(spawn_centre, destination_centre)
            else:
                remaining_distance = float(max(1, manhattan_distance(spawn, destination)))

            heading = heading_from_vector(
                destination_centre[0] - spawn_centre[0],
                destination_centre[1] - spawn_centre[1],
                fallback=0.0,
            )

            backend_vehicles.append(
                TrainEpisodeVehicle(
                    vehicle_id=int(vehicle.vehicle_id),
                    spawn=spawn,
                    destination=destination,
                    position=spawn_centre,
                    heading_deg=heading,
                    remaining_distance=remaining_distance,
                    continuous=is_continuous,
                    arrived=False,
                    travel_steps=0,
                    wait_steps=0,
                    collisions=0,
                )
            )

        backend_state = TrainEpisodeState(
            episode_index=self._reset_counter,
            seed=int(self.seed),
            phase=int(self.phase),
            level_index=int(self.level_index),
            vehicles=backend_vehicles,
            step_count=0,
            elapsed_seconds=0.0,
            metrics={
                "congestion": 0.0,
                "throughput": 0.0,
            },
            done=False,
            passed=False,
        )
        return backend_state, backend_vehicles

    def _refresh_backend_preview(self) -> None:
        """
        Use:
        Build observation preview and run one policy inference pass.

        Inputs:
        - None.

        Output:
        None.
        """
        if self.ppo is None:
            self.preview_obs_batch = []
            self.preview_actions = []
            self.preview_values = []
            return

        step_limit = self._effective_step_limit()

        backend_state, backend_vehicles = self._build_backend_preview_state()
        self.preview_obs_batch = build_observation_batch(
            state=backend_state,
            vehicles=backend_vehicles,
            generated_map=self.preview_map,
            step_limit=step_limit,
        )
        actions, _, values = self.ppo.select_actions(self.preview_obs_batch, deterministic=False)
        self.preview_actions = actions
        self.preview_values = values

    def _motion_profile(self) -> tuple[float, float, float]:
        """
        Use:
        Resolve per-phase motion profile `(base_speed, max_speed, accel)`.

        Inputs:
        - None.

        Output:
        Tuple of motion parameters in cells per simulation step.
        """
        if not self.preview_map.continuous:
            return 1.0, 1.0, 0.0
        if self.phase == 6:
            return 0.08, 0.34, 0.03
        return 0.14, 0.14, 0.0

    def _sync_reward_profile(self) -> None:
        """
        Use:
        Refresh active reward profile for the current phase and mirror it into
        runtime config so progression logic can reference the same values.

        Inputs:
        - None.

        Output:
        None.
        """
        base_weights = dict(phase_reward_weights(self.phase))
        # Keep pass thresholds phase-specific so curriculum expectations match
        # the intended difficulty profile at each phase.
        self._sync_phase_thresholds()
        if self.run_ctx is None:
            self.reward_weights = base_weights
            return

        train_cfg = self.run_ctx.config.setdefault("train", {})
        collision_key = f"reward_collision_scale_p{int(self.phase)}"
        progress_key = f"reward_progress_scale_p{int(self.phase)}"
        collision_scale = float(train_cfg.get(collision_key, 1.0))
        progress_scale = float(train_cfg.get(progress_key, 1.0))

        # Apply per-phase user scales so reward emphasis can be tuned in Options.
        base_weights["collision_penalty"] = float(base_weights.get("collision_penalty", 0.0)) * max(0.0, collision_scale)
        base_weights["progress_reward"] = float(base_weights.get("progress_reward", 0.0)) * max(0.0, progress_scale)

        self.reward_weights = base_weights
        train_cfg["reward_profile"] = dict(self.reward_weights)
        train_cfg["active_success_threshold"] = float(self.success_threshold)
        train_cfg["active_collision_threshold"] = float(self.collision_threshold)

    def _sync_phase_thresholds(self) -> None:
        """
        Use:
        Apply phase-specific pass thresholds for success and collision metrics.

        Inputs:
        - None.

        Output:
        None.
        """
        success_threshold, collision_threshold = phase_pass_thresholds(int(self.phase))
        self.success_threshold = float(success_threshold)
        self.collision_threshold = float(collision_threshold)

    def _effective_step_limit(self) -> int:
        """
        Use:
        Compute phase/map-aware episode step limit.

        Inputs:
        - None.

        Output:
        Integer step cap for the current episode.
        """
        scenario_cfg = self.run_ctx.config.get("scenario", {}) if self.run_ctx is not None else {}
        base_limit = int(max(30, scenario_cfg.get("episode_len", 300)))
        return int(
            phase_step_limit(
                base_steps=base_limit,
                phase=int(self.phase),
                map_width=int(self.preview_map.width),
                map_height=int(self.preview_map.height),
            )
        )

    def _vehicle_has_arrived(self, vehicle: EpisodeVehicle) -> bool:
        """
        Use:
        Resolve whether one vehicle has reached its destination tile.

        Inputs:
        - vehicle: Runtime episode vehicle.

        Output:
        Boolean arrival status for pass/metric logic.
        """
        destination = (int(vehicle.destination[0]), int(vehicle.destination[1]))
        if not self.preview_map.continuous:
            current = (int(vehicle.current[0]), int(vehicle.current[1]))
            return current == destination

        # Continuous motion may finish slightly off the exact cell centre,
        # so use a small tolerance for arrival detection.
        destination_centre = cell_centre(destination)
        return world_distance(vehicle.position, destination_centre) <= 0.18

    def _apply_curriculum_env_config(self, phase: int, level_index: int) -> None:
        """
        Use:
        Increase environment difficulty when curriculum advances by applying
        denser road/structure settings and persisting them to runtime config.

        Inputs:
        - phase: Target phase after advancement.
        - level_index: Target level after advancement.

        Output:
        None.
        """
        base_road = 0.62
        base_structure = 0.48
        density_boost = (0.03 * float(max(0, level_index))) + (0.02 * float(max(0, phase - 1)))
        self.road_density = max(0.35, min(1.35, round(base_road + density_boost, 2)))
        self.structure_density = max(0.20, min(1.50, round(base_structure + density_boost, 2)))

        if self.run_ctx is not None:
            scenario_cfg = self.run_ctx.config.setdefault("scenario", {})
            scenario_cfg["preview_road_density"] = float(self.road_density)
            scenario_cfg["preview_structure_density"] = float(self.structure_density)

    def _instantiate_vehicles(self, source: List[PreviewVehicle]) -> List[EpisodeVehicle]:
        """
        Use:
        Instantiate fresh episode vehicle objects from preview vehicle definitions.

        Inputs:
        - source: Preview vehicle list from generated map.

        Output:
        New list of `EpisodeVehicle` objects.
        """
        vehicles: List[EpisodeVehicle] = []
        base_speed, max_speed, accel = self._motion_profile()
        for vehicle in source:
            spawn = (int(vehicle.spawn[0]), int(vehicle.spawn[1]))
            destination = (int(vehicle.destination[0]), int(vehicle.destination[1]))
            spawn_position = cell_centre(spawn)
            destination_position = cell_centre(destination)
            heading = heading_from_vector(
                destination_position[0] - spawn_position[0],
                destination_position[1] - spawn_position[1],
                fallback=0.0,
            )
            speed_scale = 1.0
            if self.preview_map.continuous and self.phase == 6:
                speed_scale = 0.85 + (0.07 * float(int(vehicle.vehicle_id) % 5))

            vehicles.append(
                EpisodeVehicle(
                    vehicle_id=int(vehicle.vehicle_id),
                    spawn=spawn,
                    destination=destination,
                    current=spawn,
                    position=spawn_position,
                    heading_deg=float(heading),
                    speed=float(base_speed * speed_scale),
                    max_speed=float(max_speed * speed_scale),
                    accel=float(accel),
                )
            )
        return vehicles

    def _reset_episode_rollouts(self, vehicles: List[EpisodeVehicle]) -> None:
        """
        Use:
        Create fresh per-vehicle rollout buffers for a new episode lifecycle.

        Inputs:
        - vehicles: Current episode vehicle list.

        Output:
        None.
        """
        self._episode_rollouts = {
            int(vehicle.vehicle_id): VehicleRollout(
                observations=[],
                actions=[],
                log_probs=[],
                values=[],
                rewards=[],
                dones=[],
            )
            for vehicle in vehicles
        }

    def _record_rollout_step(
        self,
        vehicle_id: int,
        observation: List[float],
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: float,
    ) -> None:
        """
        Use:
        Append one PPO transition row into the owning vehicle rollout buffer.

        Inputs:
        - vehicle_id: Target vehicle identifier.
        - observation: Observation vector used at step selection time.
        - action: Selected action index.
        - log_prob: Policy log-probability of selected action.
        - value: Critic value estimate for this transition.
        - reward: Per-step reward assigned to this vehicle.
        - done: Terminal marker (`1.0` terminal, `0.0` non-terminal).

        Output:
        None.
        """
        rollout = self._episode_rollouts.setdefault(
            int(vehicle_id),
            VehicleRollout(
                observations=[],
                actions=[],
                log_probs=[],
                values=[],
                rewards=[],
                dones=[],
            ),
        )
        rollout.observations.append(list(observation))
        rollout.actions.append(int(action))
        rollout.log_probs.append(float(log_prob))
        rollout.values.append(float(value))
        rollout.rewards.append(float(reward))
        rollout.dones.append(float(done))

    def _run_ppo_update(self) -> PPOUpdateStats:
        """
        Use:
        Compute returns/advantages from collected rollouts, then run one PPO
        optimisation pass over the episode batch.

        Inputs:
        - None.

        Output:
        PPO update statistics for this episode.
        """
        if self.ppo is None:
            return PPOUpdateStats(0.0, 0.0, 0.0, 0.0, 0.0)

        batch_obs: List[List[float]] = []
        batch_actions: List[int] = []
        batch_log_probs: List[float] = []
        batch_returns: List[float] = []
        batch_advantages: List[float] = []

        gamma = float(getattr(self.ppo.config, "gamma", 0.99))
        gae_lambda = float(getattr(self.ppo.config, "gae_lambda", 0.95))

        for rollout in self._episode_rollouts.values():
            if not rollout.observations:
                continue
            returns, advantages = self.ppo.compute_gae(
                rewards=rollout.rewards,
                values=rollout.values,
                dones=rollout.dones,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
            batch_obs.extend(rollout.observations)
            batch_actions.extend(rollout.actions)
            batch_log_probs.extend(rollout.log_probs)
            batch_returns.extend(returns)
            batch_advantages.extend(advantages)

        if not batch_obs:
            return PPOUpdateStats(0.0, 0.0, 0.0, 0.0, 0.0)

        return self.ppo.update(
            observations=batch_obs,
            actions=batch_actions,
            old_log_probs=batch_log_probs,
            returns=batch_returns,
            advantages=batch_advantages,
        )

    def _sync_train_settings_from_config(self) -> None:
        """
        Use:
        Refresh runtime train thresholds/settings from shared config.

        Inputs:
        - None.

        Output:
        None.
        """
        if self.run_ctx is None:
            return
        # Keep TRAIN seed aligned with shared runtime context so Setup seed
        # updates propagate without requiring screen re-instantiation.
        self.seed = int(self.run_ctx.config.get("seed", self.run_ctx.seed))

        scenario_cfg = self.run_ctx.config.get("scenario", {})
        self.phase = clamp_phase(int(scenario_cfg.get("phase", max(0, self.phase - 1))) + 1)
        max_level_index = max(0, map_level_count(self.phase) - 1)
        self.level_index = max(0, min(int(scenario_cfg.get("level_index", self.level_index)), max_level_index))
        self.road_density = float(scenario_cfg.get("preview_road_density", self.road_density))
        self.structure_density = float(scenario_cfg.get("preview_structure_density", self.structure_density))

        train_cfg = self.run_ctx.config.get("train", {})
        self.episodes_per_level = max(1, int(train_cfg.get("episodes_per_level", self.episodes_per_level)))
        # Phase-specific thresholds are resolved in `_sync_phase_thresholds()`
        # so the curriculum profile remains deterministic.
        self.auto_training_delay_ms = max(0, int(train_cfg.get("auto_training_delay_ms", self.auto_training_delay_ms)))
        self.phase2_collision_setbacks = bool(train_cfg.get("phase2_collision_setbacks", self.phase2_collision_setbacks))
        options_cfg = self.run_ctx.config.get("options", {})
        self.training_visualised = bool(options_cfg.get("visualise_training", self.training_visualised))
        requested = self._requested_runtime_device()
        if requested != self.requested_device:
            self._init_backend_bridge()

    def _capture_replay_step(self) -> None:
        """
        Use:
        Append one replay frame for the current step.

        Inputs:
        - None.

        Output:
        None.
        """
        if self.episode_state is None:
            return
        state = self.episode_state
        frame = {
            "step": int(state.step_count),
            "elapsed_seconds": float(state.elapsed_seconds),
            "metrics": {
                "reward_sum": float(state.metrics.get("reward_sum", 0.0)),
                "collisions": float(state.metrics.get("collisions", 0.0)),
                "arrivals": float(state.metrics.get("arrivals", 0.0)),
                "throughput": float(state.metrics.get("throughput", 0.0)),
                "success_rate": float(state.metrics.get("success_rate", 0.0)),
                "collision_rate": float(state.metrics.get("collision_rate", 0.0)),
            },
            "vehicles": [
                {
                    "vehicle_id": int(vehicle.vehicle_id),
                    "current": [int(vehicle.current[0]), int(vehicle.current[1])],
                    "position": [float(vehicle.position[0]), float(vehicle.position[1])],
                    "heading_deg": float(vehicle.heading_deg),
                    "spawn": [int(vehicle.spawn[0]), int(vehicle.spawn[1])],
                    "destination": [int(vehicle.destination[0]), int(vehicle.destination[1])],
                }
                for vehicle in state.vehicles
            ],
        }
        self._episode_replay_steps.append(frame)

    def _persist_episode_replay(self, state: EpisodeState, passed: bool) -> Optional[Path]:
        """
        Use:
        Save current episode replay payload if user requested replay save.

        Inputs:
        - state: Completed episode state.
        - passed: Episode pass/fail outcome.

        Output:
        Path to replay file when saved, else `None`.
        """
        if not self._pending_replay_save or self.run_ctx is None:
            return None

        payload: Dict[str, Any] = {
            "run_id": self.run_ctx.run_id,
            "seed": int(state.seed),
            "phase": int(self.phase),
            "level_index": int(self.level_index),
            "episode_index": int(state.episode_index),
            "passed": bool(passed),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metrics_final": {key: float(value) for key, value in state.metrics.items()},
            "frames": self._episode_replay_steps,
            # Persist a compact road-grid so Replay can render the same map layout
            # style used during training playback.
            "map_grid": [
                [
                    1 if (x_pos, y_pos) in self.preview_map.roads else 0
                    for x_pos in range(int(self.preview_map.width))
                ]
                for y_pos in range(int(self.preview_map.height))
            ],
            "map_node_types": [
                [
                    str(self.preview_map.node_types.get((x_pos, y_pos), ""))
                    for x_pos in range(int(self.preview_map.width))
                ]
                for y_pos in range(int(self.preview_map.height))
            ],
        }
        path, message = save_episode_replay_to_slots(
            base_dir=self._project_root(),
            replay_data=payload,
            replay_name=f"P{int(self.phase)} L{int(self.level_index) + 1} E{int(state.episode_index)}",
            seed=int(state.seed),
            network_name=str(self.loaded_network_name),
        )
        self._pending_replay_save = False
        if path is None:
            self._last_replay_save_message = str(message)
            self._last_replay_path = None
            return None
        self._last_replay_save_message = str(message)
        self._last_replay_path = path
        return path

    def _save_training_screenshot(self, screen: pygame.Surface) -> None:
        """
        Use:
        Save a screenshot of the current training runtime view.

        Inputs:
        - screen: Current frame surface.

        Output:
        None.
        """
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_token = f"run_{self.run_ctx.run_id}" if (self.run_ctx is not None and self.run_ctx.run_id is not None) else "run_none"
        episode_idx = self.episode_state.episode_index if self.episode_state is not None else 0
        file_name = f"{run_token}_p{self.phase}_l{self.level_index + 1}_e{episode_idx}_{stamp}.png"
        path = self.screenshot_dir / file_name
        pygame.image.save(screen, path.as_posix())
        self.status_message = f"Screenshot saved -> {path.name}"

    def _advance_curriculum_if_ready(self) -> None:
        """
        Use:
        Execute curriculum-step progression using episode metrics and configured
        thresholds, then adjust environment and reward profiles when advancing.

        Inputs:
        - None.

        Output:
        None.
        """
        if self.last_summary is None:
            return

        meets_thresholds = bool(
            float(self.last_summary.success_rate) >= float(self.success_threshold)
            and float(self.last_summary.collision_rate) <= float(self.collision_threshold)
        )
        if not meets_thresholds or self.level_passes < self.episodes_per_level:
            return

        self.level_passes = 0
        next_phase = int(self.phase)
        next_level_index = int(self.level_index)
        max_level_index = max(0, map_level_count(next_phase) - 1)

        advanced_phase = False
        if next_level_index < max_level_index:
            next_level_index += 1
        elif next_phase < 6:
            next_phase += 1
            next_level_index = 0
            advanced_phase = True
        else:
            self.completed_curriculum = True
            self.auto_continue_training = False
            self.status_message = "Curriculum complete across all phases."
            return

        self.phase = int(next_phase)
        self.level_index = int(next_level_index)
        self._apply_curriculum_env_config(self.phase, self.level_index)
        self._sync_reward_profile()
        self.reset_environment(initial=False)

        if advanced_phase:
            self.status_message = f"Curriculum advanced to phase {self.phase}."
        else:
            max_level_index = max(0, map_level_count(self.phase) - 1)
            self.status_message = (
                f"Curriculum advanced to level {self.level_index + 1}/{max_level_index + 1}."
            )

        if self.run_ctx is not None:
            scenario_cfg = self.run_ctx.config.setdefault("scenario", {})
            scenario_cfg["phase"] = int(self.phase - 1)
            scenario_cfg["level_index"] = int(self.level_index)


    def _finalise_episode(self) -> None:
        """
        Use:
        Finalise episode metrics, run logging/persistence, and schedule next
        episode start if auto-training is enabled.

        Inputs:
        - None.

        Output:
        None.
        """
        if self.episode_state is None:
            return

        state = self.episode_state
        vehicle_count = max(1, len(state.vehicles))
        arrivals = float(sum(1 for vehicle in state.vehicles if self._vehicle_has_arrived(vehicle)))
        collisions = float(state.metrics.get("collisions", 0.0))
        # Episode-level pass/fail is derived from aggregate success + collision
        # rates so curriculum logic uses one consistent completion signal.
        success_rate = arrivals / float(vehicle_count)
        collision_rate = collisions / float(max(1, vehicle_count * max(1, state.step_count)))
        state.metrics["arrivals"] = float(arrivals)
        state.metrics["success_rate"] = float(success_rate)
        state.metrics["collision_rate"] = float(collision_rate)

        passed = bool(
            success_rate >= float(self.success_threshold)
            and collision_rate <= float(self.collision_threshold)
        )

        # PPO update is executed once per completed episode using the rollout
        # buffers recorded during `_run_training_step`.
        ppo_stats = self._run_ppo_update()
        state.metrics["loss"] = float(ppo_stats.loss)
        state.metrics["entropy"] = float(ppo_stats.entropy)
        state.metrics["actor_loss"] = float(ppo_stats.actor_loss)
        state.metrics["value_loss"] = float(ppo_stats.value_loss)
        state.metrics["clip_fraction"] = float(ppo_stats.clip_fraction)

        self.episode_running = False
        self.last_episode_success = float(success_rate)
        self.last_episode_failed = not passed
        self.total_episodes_completed += 1
        self.last_summary = EpisodeSummary(
            passed=bool(passed),
            success_rate=float(success_rate),
            collision_rate=float(collision_rate),
            throughput=float(state.metrics.get("throughput", 0.0)),
            avg_journey_time=float(max(1, state.step_count)) / float(max(1.0, arrivals)),
            reward=float(state.metrics.get("reward_sum", 0.0)),
            loss=float(state.metrics.get("loss", 0.0)),
        )

        if passed:
            self.level_passes += 1
        else:
            self.level_passes = max(0, self.level_passes - 1)

        # Persist the finished episode metrics so exports/evaluation views can
        # aggregate TRAIN outcomes without replaying every episode.
        if self.run_ctx is not None and self.run_ctx.run_id is not None and self.active_episode_id is not None:
            try:
                for key, value in state.metrics.items():
                    insert_metric_record(
                        self.run_ctx.db_path,
                        run_id=int(self.run_ctx.run_id),
                        episode_id=int(self.active_episode_id),
                        key=key,
                        value=float(value),
                        step=int(state.step_count),
                    )
                insert_metric_record(
                    self.run_ctx.db_path,
                    run_id=int(self.run_ctx.run_id),
                    episode_id=int(self.active_episode_id),
                    key="phase",
                    value=float(self.phase),
                    step=int(state.step_count),
                )
                insert_metric_record(
                    self.run_ctx.db_path,
                    run_id=int(self.run_ctx.run_id),
                    episode_id=int(self.active_episode_id),
                    key="level_index",
                    value=float(self.level_index),
                    step=int(state.step_count),
                )
                insert_metric_record(
                    self.run_ctx.db_path,
                    run_id=int(self.run_ctx.run_id),
                    episode_id=int(self.active_episode_id),
                    key="passed",
                    value=1.0 if passed else 0.0,
                    step=int(state.step_count),
                )
            except Exception:
                # Persistence failures should not block runtime progression.
                pass

        replay_path = self._persist_episode_replay(state=state, passed=passed)
        self._advance_curriculum_if_ready()

        token = "PASS" if passed else "FAIL"
        base_status = (
            f"Episode {token}: success={success_rate:.2f} "
            f"collision={collision_rate:.3f} throughput={float(state.metrics.get('throughput', 0.0)):.3f} "
            f"loss={ppo_stats.loss:.4f}"
        )
        if replay_path is not None:
            base_status = f"{base_status} | replay={replay_path.name}"
        elif self._last_replay_save_message:
            base_status = f"{base_status} | {self._last_replay_save_message}"
        self.status_message = base_status

        if self.auto_continue_training and self.training_view and not self.completed_curriculum:
            delay_ms = 0 if (not self.training_visualised) else int(self.auto_training_delay_ms)
            self._next_auto_start_ms = pygame.time.get_ticks() + delay_ms

    def _begin_training_session(self) -> None:
        """
        Use:
        Switch to DURING TRAINING view and start the first episode.

        Inputs:
        - None.

        Output:
        None.
        """
        self._init_backend_bridge()
        self.training_view = True
        self.training_paused = False
        self._arrow_speed_direction = 0
        self._next_auto_start_ms = 0
        self.begin_episode()

    def _exit_training_session(self) -> None:
        """
        Use:
        Stop runtime training loop and return to TRAIN setup mode.

        Inputs:
        - None.

        Output:
        None.
        """
        self.training_view = False
        self.training_paused = False
        self.episode_running = False
        self._arrow_speed_direction = 0
        self._next_auto_start_ms = 0
        self.status_message = "Returned to TRAIN setup view."

    def reset_environment(self, initial: bool = False) -> EpisodeState:
        """
        Use:
        Fully reset TRAIN environment state and rebuild episode initial conditions.

        Inputs:
        - initial: True for startup reset; False for user-triggered reset.

        Output:
        New `EpisodeState` object.
        """
        self._sync_train_settings_from_config()
        self._sync_reward_profile()
        if not initial:
            self._reset_counter += 1

        episode_seed = int(self.seed)

        # Sync all known RNG providers so reset remains reproducible.
        reseed_all(episode_seed)

        # Regenerate map using deterministic seed and current setup parameters.
        self.preview_map = generate_phase_map(
            episode_seed,
            self.phase,
            self.level_index,
            road_density=self.road_density,
            structure_density=self.structure_density,
        )
        if not self.preview_map.roads:
            # Fail-safe: regenerate a visible preview map if a sparse config produced no roads.
            self.preview_map = generate_phase_map(
                episode_seed,
                self.phase,
                self.level_index,
                road_density=max(0.60, self.road_density),
                structure_density=max(0.40, self.structure_density),
            )

        vehicles = self._instantiate_vehicles(self.preview_map.vehicles)

        # Timers + metrics are cleared on every reset by creating a fresh state object.
        self.episode_state = EpisodeState(
            episode_index=self._reset_counter,
            seed=episode_seed,
            vehicles=vehicles,
            step_count=0,
            elapsed_seconds=0.0,
            metrics={
                "collisions": 0.0,
                "arrivals": 0.0,
                "reward_sum": 0.0,
                "throughput": 0.0,
                "congestion": 0.0,
                "success_rate": 0.0,
                "collision_rate": 0.0,
                "avg_speed": 0.0,
            },
        )
        self._reset_episode_rollouts(vehicles)
        self._episode_replay_steps = []

        self.episode_running = False
        self._episode_start_t = 0.0
        self.active_episode_id = None
        self._refresh_backend_preview()
        self.status_message = "Environment reset complete. Initial conditions ready."
        return self.episode_state

    def begin_episode(self) -> None:
        """
        Use:
        Start a live episode and enter DURING TRAINING view.

        Inputs:
        - None.

        Output:
        None.
        """
        if self.episode_state is None:
            self.reset_environment(initial=True)

        # Start each training run from a fresh, reproducible environment state.
        self._start_episode_cycle(enter_training_view=True)
        self.status_message = "Episode started. Training running."

    def _start_episode_cycle(self, enter_training_view: bool) -> None:
        """
        Use:
        Start the next episode cycle from a clean reset state.

        Inputs:
        - enter_training_view: When True, force switch to DURING TRAINING view.

        Output:
        None.
        """
        self.reset_environment(initial=False)
        if enter_training_view:
            self.training_view = True
        self.training_paused = False
        self.episode_running = True
        now_t = time.perf_counter()
        self._episode_start_t = now_t
        self._last_sim_step_t = now_t
        # Runtime clocks/accumulators are reset so speed controls do not leak
        # timing state from the previous episode.
        self._runtime_prev_t = now_t
        self._sim_accumulator_s = 0.0
        self._next_auto_start_ms = 0
        self._episode_replay_steps = []
        if self.episode_state is not None:
            self._reset_episode_rollouts(self.episode_state.vehicles)
        self.active_episode_id = None
        if self.run_ctx is not None and self.run_ctx.run_id is not None:
            try:
                # Episode rows are created up front so per-step metrics can
                # always reference a stable episode identifier.
                episode_id, _ = create_episode_record(
                    self.run_ctx.db_path,
                    run_id=int(self.run_ctx.run_id),
                    mode="TRAIN",
                    seed=int(self.seed),
                )
                self.active_episode_id = int(episode_id)
            except Exception:
                self.active_episode_id = None

    def _apply_phase_delta(self, direction: int) -> None:
        """
        Use:
        Increment/decrement active phase in TRAIN config view and rebuild preview.

        Inputs:
        - direction: `-1` for previous phase, `+1` for next phase.

        Output:
        None.
        """
        next_phase = clamp_phase(int(self.phase) + int(direction))
        if next_phase == self.phase:
            return
        self.phase = int(next_phase)
        self.level_index = 0
        if self.run_ctx is not None:
            scenario_cfg = self.run_ctx.config.setdefault("scenario", {})
            scenario_cfg["phase"] = int(self.phase - 1)
            scenario_cfg["level_index"] = int(self.level_index)
        self.reset_environment(initial=False)
        self.status_message = f"Phase set to {self.phase}. Preview updated."

    def _apply_level_delta(self, direction: int) -> None:
        """
        Use:
        Increment/decrement map level for current phase and rebuild preview.

        Inputs:
        - direction: `-1` for previous level, `+1` for next level.

        Output:
        None.
        """
        max_level_index = max(0, map_level_count(self.phase) - 1)
        next_level = max(0, min(max_level_index, int(self.level_index) + int(direction)))
        if next_level == self.level_index:
            return
        self.level_index = int(next_level)
        if self.run_ctx is not None:
            scenario_cfg = self.run_ctx.config.setdefault("scenario", {})
            scenario_cfg["level_index"] = int(self.level_index)
        self.reset_environment(initial=False)
        self.status_message = (
            f"Map level set to {self.level_index + 1}/{max_level_index + 1}. Preview updated."
        )

    def _save_network_snapshot(self) -> None:
        """
        Use:
        Save PPO network into the first free named slot (max 5).

        Inputs:
        - None.

        Output:
        None.
        """
        if self.ppo is None:
            self._init_backend_bridge()
        if self.ppo is None:
            self.status_message = "Unable to save: PPO not initialised."
            return

        base_dir = self._project_root()
        slots = load_slots(base_dir)
        slot_id = first_empty_slot(slots)
        if slot_id is None:
            self.status_message = "No free network slots. Open Replays > Networks to manage slots."
            return

        payload: Dict[str, Any] = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "seed": int(self.seed),
            "phase": int(self.phase),
            "level_index": int(self.level_index),
            "road_density": float(self.road_density),
            "structure_density": float(self.structure_density),
            "slot_id": int(slot_id),
        }
        try:
            checkpoint_path = slot_checkpoint_path(base_dir, slot_id)
            self.ppo.save(checkpoint_path, metadata=payload)
            mark_slot_saved(
                slots,
                slot_id,
                seed=int(self.seed),
                phase=int(self.phase),
                level_index=int(self.level_index),
            )
            save_slots(base_dir, slots)
            slot_name = next((str(s.get("name", f"Network {slot_id}")) for s in slots if int(s.get("slot_id", 0)) == int(slot_id)), f"Network {slot_id}")
            self.loaded_network_name = str(slot_name)
            self.status_message = f"Network saved -> slot {slot_id} ({slot_name})"
        except Exception as exc:
            self.status_message = f"Network save failed: {exc}"

    def load_network_from_path(self, checkpoint_path: Path, slot_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Use:
        Load PPO network from a selected checkpoint and refresh TRAIN preview.

        Inputs:
        - checkpoint_path: Source checkpoint path selected in Replay browser.
        - slot_metadata: Optional slot metadata fallback.

        Output:
        `True` on successful load, otherwise `False`.
        """
        if not checkpoint_path.exists():
            self.status_message = "Selected network file is missing."
            return False

        if self.ppo is None:
            self._init_backend_bridge()
        if self.ppo is None:
            self.status_message = "Unable to load: PPO not initialised."
            return False

        payload: Dict[str, Any] = {}
        try:
            metadata = self.ppo.load(checkpoint_path)
            if isinstance(metadata, dict):
                payload = dict(metadata)
        except Exception as exc:
            self.status_message = f"Network load failed: {exc}"
            return False

        if not payload and isinstance(slot_metadata, dict):
            payload = dict(slot_metadata)

        self.seed = int(payload.get("seed", self.seed))
        self.phase = clamp_phase(int(payload.get("phase", self.phase)))
        max_level_index = max(0, map_level_count(self.phase) - 1)
        self.level_index = max(0, min(int(payload.get("level_index", self.level_index)), max_level_index))
        self.road_density = float(payload.get("road_density", self.road_density))
        self.structure_density = float(payload.get("structure_density", self.structure_density))
        if isinstance(slot_metadata, dict):
            self.loaded_network_name = str(slot_metadata.get("name", self.loaded_network_name))

        if self.run_ctx is not None:
            self.run_ctx.seed = int(self.seed)
            self.run_ctx.config["seed"] = int(self.seed)
            scenario_cfg = self.run_ctx.config.setdefault("scenario", {})
            scenario_cfg["phase"] = int(self.phase - 1)
            scenario_cfg["level_index"] = int(self.level_index)
            scenario_cfg["preview_road_density"] = float(self.road_density)
            scenario_cfg["preview_structure_density"] = float(self.structure_density)

        self.reset_environment(initial=False)
        self.status_message = f"Network loaded -> {checkpoint_path.name}"
        return True

    def _load_network_snapshot(self) -> None:
        """
        Use:
        Backward-compatible rolling-load helper.
        """
        _ = self.load_network_from_path(self.network_save_path, None)

    def _apply_speed_delta(self, direction: int, step: float = 0.01) -> None:
        """
        Use:
        Adjust simulation speed within safe bounds.

        Inputs:
        - direction: `-1` for slower, `+1` for faster.
        - step: Absolute speed increment applied per action.

        Output:
        None.
        """
        next_speed = float(self.sim_speed) + (float(step) * float(direction))
        next_speed = max(0.01, min(1000.0, next_speed))
        self.sim_speed = round(next_speed, 2)
        self.status_message = f"Simulation speed set to {self.sim_speed:.2f}x."

    def _update_arrow_speed_hold(self) -> None:
        """
        Use:
        Apply constant-rate arrow-key speed updates.

        Inputs:
        - None.

        Output:
        None.
        """
        if self._arrow_speed_direction == 0:
            return
        now_ms = pygame.time.get_ticks()
        if now_ms - self._arrow_speed_last_ms < self._arrow_speed_interval_ms:
            return
        self._apply_speed_delta(self._arrow_speed_direction, step=0.01)
        self._arrow_speed_last_ms = now_ms

    def _tick_runtime(self) -> None:
        """
        Use:
        Update runtime clocks and advance training simulation steps while active.

        Inputs:
        - None.

        Output:
        None.
        """
        # Runtime-only controls and auto-continue trigger checks live here so
        # setup-screen mode stays passive.
        if self.training_view:
            self._speed_hold.update()
            self._update_arrow_speed_hold()
            if (
                self.auto_continue_training
                and (not self.training_paused)
                and (not self.episode_running)
                and (not self.completed_curriculum)
                and pygame.time.get_ticks() >= int(self._next_auto_start_ms)
            ):
                self._start_episode_cycle(enter_training_view=False)

        if not self.episode_running or self.episode_state is None:
            return
        now_t = time.perf_counter()
        self.episode_state.elapsed_seconds = max(0.0, now_t - self._episode_start_t)

        # Keep runtime timestamps stable when the user is outside training view.
        if not self.training_view:
            self._runtime_prev_t = now_t
            return

        if self.training_paused:
            self._runtime_prev_t = now_t
            return

        if not self.training_visualised:
            # Headless mode prioritises throughput: run as many steps as possible
            # within this frame budget and skip render-timed throttling.
            self._runtime_prev_t = now_t
            started = time.perf_counter()
            steps_done = 0
            while self.episode_running and steps_done < int(self._headless_max_steps_per_frame):
                self._run_training_step()
                steps_done += 1
                if (time.perf_counter() - started) >= float(self._headless_frame_budget_s):
                    break
            if not self.episode_running:
                self._sim_accumulator_s = 0.0
            return

        if self._runtime_prev_t <= 0.0:
            self._runtime_prev_t = now_t
            return

        # Scale simulation time by user speed multiplier so 1x/1000x has visible effect.
        real_dt = max(0.0, now_t - self._runtime_prev_t)
        self._runtime_prev_t = now_t
        scaled_dt = real_dt * max(0.01, float(self.sim_speed))
        self._sim_accumulator_s += scaled_dt

        if self._sim_step_interval_s <= 0:
            return

        # Convert accumulated scaled time into an integer step budget.
        steps_budget = int(self._sim_accumulator_s // self._sim_step_interval_s)
        if steps_budget <= 0:
            return

        if steps_budget > self._max_steps_per_frame:
            steps_budget = self._max_steps_per_frame

        # Execute a bounded number of logical simulation steps this frame to
        # avoid starving rendering/event handling at high speed multipliers.
        for _ in range(steps_budget):
            self._run_training_step()
            self._sim_accumulator_s = max(0.0, self._sim_accumulator_s - self._sim_step_interval_s)
            if not self.episode_running:
                self._sim_accumulator_s = 0.0
                break

    def _run_training_step(self) -> None:
        """
        Use:
        Execute one visible training-step update so vehicles move in the training
        runtime screen.

        Inputs:
        - None.

        Output:
        None.
        """
        if self.episode_state is None:
            return

        state = self.episode_state
        state.step_count += 1
        roads = self.preview_map.roads
        node_types = self.preview_map.node_types
        total_vehicles = max(1, len(state.vehicles))
        step_limit = self._effective_step_limit()
        reward_weights = self.reward_weights

        backend_vehicles: List[TrainEpisodeVehicle] = []
        # Mirror GUI vehicle objects into lightweight backend types so
        # observation building and PPO inference stay decoupled from rendering.
        for vehicle in state.vehicles:
            cell = (int(vehicle.position[0]), int(vehicle.position[1]))
            destination = (int(vehicle.destination[0]), int(vehicle.destination[1]))
            position = (float(vehicle.position[0]), float(vehicle.position[1]))
            destination_centre = cell_centre(destination)
            backend_vehicles.append(
                TrainEpisodeVehicle(
                    vehicle_id=int(vehicle.vehicle_id),
                    spawn=(int(vehicle.spawn[0]), int(vehicle.spawn[1])),
                    destination=destination,
                    position=position,
                    heading_deg=heading_from_vector(
                        destination_centre[0] - position[0],
                        destination_centre[1] - position[1],
                        fallback=float(vehicle.heading_deg),
                    ),
                    remaining_distance=(
                        float(world_distance(position, destination_centre))
                        if self.preview_map.continuous
                        else float(max(1, manhattan_distance(cell, destination)))
                    ),
                    continuous=bool(self.preview_map.continuous),
                    arrived=(cell == destination),
                    travel_steps=int(state.step_count),
                    wait_steps=0,
                    collisions=0,
                )
            )

        backend_state = TrainEpisodeState(
            episode_index=int(state.episode_index),
            seed=int(state.seed),
            phase=int(self.phase),
            level_index=int(self.level_index),
            vehicles=backend_vehicles,
            step_count=int(state.step_count),
            elapsed_seconds=float(state.elapsed_seconds),
            metrics={
                "congestion": float(state.metrics.get("congestion", 0.0)),
                "throughput": float(state.metrics.get("throughput", 0.0)),
            },
            done=False,
            passed=False,
        )

        observations = build_observation_batch(
            state=backend_state,
            vehicles=backend_vehicles,
            generated_map=self.preview_map,
            step_limit=step_limit,
        )
        self.preview_obs_batch = observations
        # Policy output drives movement directly; action=0 remains "stay".
        if self.ppo is not None:
            actions, log_probs, values = self.ppo.select_actions(observations, deterministic=False)
        else:
            actions = [0 for _ in observations]
            log_probs = [0.0 for _ in observations]
            values = [0.0 for _ in observations]
        self.preview_actions = actions
        self.preview_values = values

        step_rewards: Dict[int, float] = {}
        transition_rows: List[tuple[int, List[float], int, float, float]] = []
        moved_count = 0
        base_speed, _, _ = self._motion_profile()
        # Phase-dependent collision weighting is combined with the current
        # options profile so users can tune strictness by phase.
        collision_penalty_weight = float(reward_weights.get("collision_penalty", 1.0)) * (
            float(collision_loss_multiplier(self.phase)) / 10.0
        )
        for idx, vehicle in enumerate(state.vehicles):
            current = (int(vehicle.position[0]), int(vehicle.position[1]))
            vehicle.current = current
            destination = (int(vehicle.destination[0]), int(vehicle.destination[1]))
            if self._vehicle_has_arrived(vehicle):
                continue

            observation = observations[idx] if idx < len(observations) else []
            action = int(actions[idx]) if idx < len(actions) else 0
            log_prob = float(log_probs[idx]) if idx < len(log_probs) else 0.0
            value = float(values[idx]) if idx < len(values) else 0.0
            vehicle_id = int(vehicle.vehicle_id)
            transition_rows.append((vehicle_id, observation, action, log_prob, value))
            step_rewards[vehicle_id] = 0.0

            if self.preview_map.continuous:
                remaining_before = world_distance(vehicle.position, cell_centre(destination))
            else:
                remaining_before = float(manhattan_distance(current, destination))

            next_cell = None
            if idx < len(actions) and idx < len(backend_vehicles):
                next_cell = target_cell_from_action(
                    vehicle=backend_vehicles[idx],
                    action=int(actions[idx]),
                    roads=roads,
                    node_types=node_types,
                )
            moved_this_step = False
            previous_position = (float(vehicle.position[0]), float(vehicle.position[1]))
            # Integrate action -> movement update using continuous or discrete
            # rules for the active phase.
            if next_cell is not None:
                target_position = cell_centre(next_cell)
                if self.preview_map.continuous:
                    # Continuous phases move by short vectors each step so cars
                    # can rotate and travel smoothly instead of tile-hopping.
                    vector = pygame.Vector2(
                        float(target_position[0] - vehicle.position[0]),
                        float(target_position[1] - vehicle.position[1]),
                    )
                    distance_to_target = float(vector.length())
                    if distance_to_target > 1e-6:
                        if self.phase == 6:
                            vehicle.speed = min(float(vehicle.max_speed), float(vehicle.speed + vehicle.accel))
                        else:
                            vehicle.speed = float(vehicle.max_speed)

                        step_distance = min(distance_to_target, max(0.01, float(vehicle.speed)))
                        vector.scale_to_length(step_distance)
                        candidate = (
                            float(vehicle.position[0] + float(vector.x)),
                            float(vehicle.position[1] + float(vector.y)),
                        )
                        if is_driveable_position(
                            position=candidate,
                            roads=roads,
                            node_types=node_types,
                            width=int(self.preview_map.width),
                            height=int(self.preview_map.height),
                        ):
                            vehicle.position = candidate
                            moved_this_step = world_distance(previous_position, vehicle.position) > 1e-5
                            if moved_this_step:
                                vehicle.heading_deg = heading_from_vector(
                                    vehicle.position[0] - previous_position[0],
                                    vehicle.position[1] - previous_position[1],
                                    fallback=vehicle.heading_deg,
                                )
                        elif self.phase == 6:
                            vehicle.speed = max(0.01, float(vehicle.speed - vehicle.accel))

                    if world_distance(vehicle.position, target_position) <= max(0.02, float(vehicle.speed)):
                        vehicle.position = target_position
                else:
                    # Discrete phases snap to the next selected tile.
                    vehicle.position = cell_centre(next_cell)
                    moved_this_step = next_cell != current
                    if moved_this_step:
                        vehicle.heading_deg = heading_from_vector(
                            float(next_cell[0] - current[0]),
                            float(next_cell[1] - current[1]),
                            fallback=vehicle.heading_deg,
                        )
            elif self.preview_map.continuous and self.phase == 6:
                vehicle.speed = max(0.01, float(vehicle.speed - vehicle.accel))

            current_after = (int(vehicle.position[0]), int(vehicle.position[1]))
            vehicle.current = current_after
            if moved_this_step:
                moved_count += 1

            if self.preview_map.continuous:
                remaining_after = world_distance(vehicle.position, cell_centre(destination))
            else:
                remaining_after = float(manhattan_distance(current_after, destination))

            # Reward is primarily progress-based, with penalties for idling and
            # a strong collision term applied later from occupancy checks.
            progress_reward = float(max(0.0, remaining_before - remaining_after)) * float(
                reward_weights.get("progress_reward", 0.0)
            )
            step_reward = progress_reward
            if self._vehicle_has_arrived(vehicle):
                step_reward += float(reward_weights.get("arrival_reward", 0.0))
            else:
                if action == 0:
                    step_reward -= float(reward_weights.get("wait_penalty", 0.0))
                elif not moved_this_step:
                    step_reward -= float(reward_weights.get("idle_penalty", 0.0))

            step_rewards[vehicle_id] = float(step_reward)

        occupancy: Dict[tuple[int, int], int] = {}
        # Occupancy map is used for per-step collision counting and penalties.
        for vehicle in state.vehicles:
            cell = (int(vehicle.current[0]), int(vehicle.current[1]))
            occupancy[cell] = occupancy.get(cell, 0) + 1

        step_collisions = sum(count for count in occupancy.values() if count > 1)
        collided_ids = {
            int(vehicle.vehicle_id)
            for vehicle in state.vehicles
            if occupancy.get((int(vehicle.current[0]), int(vehicle.current[1])), 0) > 1
        }
        # Apply collision loss per impacted vehicle for this step.
        for vehicle_id in collided_ids:
            step_rewards[vehicle_id] = step_rewards.get(vehicle_id, 0.0) - float(collision_penalty_weight)

        arrivals = sum(1 for vehicle in state.vehicles if self._vehicle_has_arrived(vehicle))

        state.metrics["collisions"] = float(state.metrics.get("collisions", 0.0) + step_collisions)
        state.metrics["arrivals"] = float(arrivals)
        state.metrics["throughput"] = float(arrivals) / float(total_vehicles)
        state.metrics["congestion"] = float(step_collisions) / float(total_vehicles)
        state.metrics["success_rate"] = float(arrivals) / float(total_vehicles)
        state.metrics["collision_rate"] = float(state.metrics["collisions"]) / float(
            max(1, len(state.vehicles) * max(1, state.step_count))
        )
        state.metrics["avg_speed"] = float(moved_count) / float(total_vehicles)
        step_reward_total = float(sum(step_rewards.values()))
        # Keep reward scale comparable across episodes with different vehicle counts.
        state.metrics["reward_sum"] = float(state.metrics.get("reward_sum", 0.0) + (step_reward_total / float(total_vehicles)))
        if self.phase == 2 and self.phase2_collision_setbacks and collided_ids:
            # Optional phase-2 behaviour: collided vehicles return to spawn.
            for vehicle in state.vehicles:
                if int(vehicle.vehicle_id) in collided_ids:
                    vehicle.current = (int(vehicle.spawn[0]), int(vehicle.spawn[1]))
                    vehicle.position = cell_centre(vehicle.spawn)
                    vehicle.speed = float(base_speed)
                    vehicle.heading_deg = heading_from_vector(
                        float(vehicle.destination[0] - vehicle.spawn[0]),
                        float(vehicle.destination[1] - vehicle.spawn[1]),
                        fallback=vehicle.heading_deg,
                    )

        all_arrived = arrivals >= len(state.vehicles) and len(state.vehicles) > 0
        episode_done = bool(all_arrived or state.step_count >= step_limit)
        # Record one rollout row per controlled vehicle for PPO update later.
        for vehicle_id, observation, action, log_prob, value in transition_rows:
            vehicle = next((item for item in state.vehicles if int(item.vehicle_id) == int(vehicle_id)), None)
            arrived_now = bool(vehicle is not None and self._vehicle_has_arrived(vehicle))
            done_flag = 1.0 if (episode_done or arrived_now) else 0.0
            self._record_rollout_step(
                vehicle_id=vehicle_id,
                observation=observation,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=float(step_rewards.get(vehicle_id, 0.0)),
                done=done_flag,
            )
        self._capture_replay_step()

        if episode_done:
            self._finalise_episode()
            return

    def _draw_map_preview(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        """
        Use:
        Render a compact road/vehicle preview in TRAIN screen.

        Inputs:
        - screen: Destination surface.
        - rect: Map render area.

        Output:
        None.
        """
        preview = self.preview_map
        pygame.draw.rect(screen, MAP_EMPTY_COLOUR, rect, border_radius=8)
        pygame.draw.rect(screen, PANEL_BORDER, rect, 1, border_radius=8)

        if preview.width <= 0 or preview.height <= 0:
            return

        tile = max(4, min(rect.width // preview.width, rect.height // preview.height))
        grid_w = tile * preview.width
        grid_h = tile * preview.height
        ox = rect.x + (rect.width - grid_w) // 2
        oy = rect.y + (rect.height - grid_h) // 2
        road_h_cache: Dict[tuple[int, int], pygame.Surface] = {}
        road_v_cache: Dict[tuple[int, int], pygame.Surface] = {}
        car_sprite = self._scaled_asset("car", self.car_img, max(8, int(tile * 0.80)))

        structure_fill_types = {
            "junction_turn_one_lane",
            "junction_turn_two_lane",
            "junction_t",
            "junction_cross",
            "junction_centre",
            "road_turn",
        }

        def _blit_scaled(
            image: Optional[pygame.Surface],
            cache: Dict[tuple[int, int], pygame.Surface],
            cell_rect: pygame.Rect,
        ) -> bool:
            if image is None:
                return False
            key = (max(1, cell_rect.width), max(1, cell_rect.height))
            sprite = cache.get(key)
            if sprite is None:
                sprite = pygame.transform.smoothscale(image, key)
                cache[key] = sprite
            screen.blit(sprite, cell_rect.topleft)
            return True

        for y_pos in range(preview.height):
            for x_pos in range(preview.width):
                cell = pygame.Rect(ox + x_pos * tile, oy + y_pos * tile, tile, tile)
                point = (x_pos, y_pos)
                node_type = preview.node_types.get(point, "")
                pygame.draw.rect(screen, MAP_EMPTY_COLOUR, cell)
                pygame.draw.rect(screen, MAP_GRID_COLOUR, cell, 1)

                if point not in preview.roads:
                    continue

                orientation = self._road_orientation(x_pos=x_pos, y_pos=y_pos, node_type=node_type)
                # Junctions/turns are rendered as neutral structure tiles to keep
                # intersections visually distinct from straight lanes.
                if node_type in structure_fill_types or orientation == "mixed":
                    pygame.draw.rect(screen, ROAD_STRUCTURE_LIGHT, cell)
                elif orientation == "horizontal":
                    pygame.draw.rect(screen, ROAD_COLOUR, cell)
                    _blit_scaled(self.road_h_img, road_h_cache, cell)
                elif orientation == "vertical":
                    pygame.draw.rect(screen, ROAD_COLOUR, cell)
                    _blit_scaled(self.road_v_img, road_v_cache, cell)
                else:
                    pygame.draw.rect(screen, ROAD_COLOUR, cell)

        if self.episode_state is not None:
            # Draw vehicles and numbered destinations so each pair is traceable.
            for destination_index, vehicle in enumerate(self.episode_state.vehicles, start=1):
                position = (float(vehicle.position[0]), float(vehicle.position[1]))
                current_centre = (
                    int(ox + (position[0] * tile)),
                    int(oy + (position[1] * tile)),
                )
                current_rect = pygame.Rect(0, 0, tile, tile)
                current_rect.center = current_centre
                dest_rect = pygame.Rect(ox + vehicle.destination[0] * tile, oy + vehicle.destination[1] * tile, tile, tile)
                if car_sprite is not None:
                    rotated = pygame.transform.rotozoom(car_sprite, -float(vehicle.heading_deg), 1.0)
                    car_rect = rotated.get_rect(center=current_rect.center)
                    screen.blit(rotated, car_rect)
                else:
                    pygame.draw.circle(screen, SPAWN_COLOUR, current_rect.center, max(2, tile // 4))
                dest_radius = max(3, int(tile * 0.48))
                pygame.draw.circle(screen, DEST_COLOUR, dest_rect.center, dest_radius)
                pygame.draw.circle(screen, (26, 102, 32), dest_rect.center, dest_radius, 1)
                label = self._font(max(8, int(tile * 0.50))).render(str(destination_index), True, (0, 0, 0))
                screen.blit(label, label.get_rect(center=dest_rect.center))

    def handle_events(self, events: List[pygame.event.Event]) -> str:
        """
        Use:
        Process TRAIN interactions and return next screen state.

        Inputs:
        - events: Current frame event list.

        Output:
        Next state token (`TRAIN` or `MENU`).
        """
        if not self.training_view:
            self._phase_hold.update()
            self._level_hold.update()

        for event in events:
            if self.training_view:
                # Runtime-only controls: speed, pause/resume, screenshot, replay save.
                if event.type == pygame.KEYDOWN:
                    if self.training_visualised and event.key == pygame.K_LEFT:
                        self._arrow_speed_direction = -1
                        self._arrow_speed_last_ms = pygame.time.get_ticks()
                        self._apply_speed_delta(-1, step=0.01)
                        continue
                    if self.training_visualised and event.key == pygame.K_RIGHT:
                        self._arrow_speed_direction = +1
                        self._arrow_speed_last_ms = pygame.time.get_ticks()
                        self._apply_speed_delta(+1, step=0.01)
                        continue

                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT and self._arrow_speed_direction == -1:
                        self._arrow_speed_direction = 0
                        continue
                    if event.key == pygame.K_RIGHT and self._arrow_speed_direction == +1:
                        self._arrow_speed_direction = 0
                        continue

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.training_visualised and self.speed_minus_button.collidepoint(event.pos):
                        self._speed_hold.begin(lambda: self._apply_speed_delta(-1, step=1.0))
                        continue
                    if self.training_visualised and self.speed_plus_button.collidepoint(event.pos):
                        self._speed_hold.begin(lambda: self._apply_speed_delta(+1, step=1.0))
                        continue

                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self._speed_hold.stop()
                    if self.training_back_button.collidepoint(event.pos):
                        self._exit_training_session()
                        continue
                    if self.screenshot_button.collidepoint(event.pos):
                        self._pending_screenshot = True
                        continue
                    if self.replay_button.collidepoint(event.pos):
                        self._pending_replay_save = not self._pending_replay_save
                        self.status_message = (
                            "Replay will be saved at episode end."
                            if self._pending_replay_save
                            else "Replay save request cleared."
                        )
                        continue
                    if self.play_pause_button.collidepoint(event.pos):
                        self.training_paused = not self.training_paused
                        if not self.training_paused:
                            resume_t = time.perf_counter()
                            self._last_sim_step_t = resume_t
                            self._runtime_prev_t = resume_t
                        self.status_message = "Training paused." if self.training_paused else "Training resumed."
                        continue
                continue

            # Pre-training configuration controls: phase/level/network/start.
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.phase_minus_button.collidepoint(event.pos):
                    self._phase_hold.begin(lambda: self._apply_phase_delta(-1))
                    continue
                if self.phase_plus_button.collidepoint(event.pos):
                    self._phase_hold.begin(lambda: self._apply_phase_delta(+1))
                    continue
                if self.level_minus_button.collidepoint(event.pos):
                    self._level_hold.begin(lambda: self._apply_level_delta(-1))
                    continue
                if self.level_plus_button.collidepoint(event.pos):
                    self._level_hold.begin(lambda: self._apply_level_delta(+1))
                    continue

            if event.type != pygame.MOUSEBUTTONUP or event.button != 1:
                continue

            self._phase_hold.stop()
            self._level_hold.stop()

            if self.back_button.collidepoint(event.pos):
                return "MENU"
            if self.save_network_button.collidepoint(event.pos):
                self._save_network_snapshot()
                continue
            if self.load_network_button.collidepoint(event.pos):
                self.status_message = "Open Replays > Networks to load a saved network."
                return "REPLAYS_NETWORK_LOAD"
            if self.reset_button.collidepoint(event.pos):
                self.reset_environment(initial=False)
                continue
            if self.begin_button.collidepoint(event.pos):
                self._begin_training_session()
                continue
            if self.visualise_button.collidepoint(event.pos):
                self._set_training_visualised(not self.training_visualised)
                mode = "ON" if self.training_visualised else "OFF (full throttle)"
                self.status_message = f"Training visualisation set to {mode}."
                continue

        return "TRAIN"

    def _draw_training_runtime(self, screen: pygame.Surface) -> None:
        """
        Use:
        Render DURING TRAINING runtime view with top pause/play controls.

        Inputs:
        - screen: Target surface.

        Output:
        None.
        """
        screen.fill(BG)

        pygame.draw.rect(screen, PANEL_BG, self.left_panel, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, self.left_panel, 2, border_radius=12)
        pygame.draw.rect(screen, PANEL_BG, self.right_panel, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, self.right_panel, 2, border_radius=12)

        self._draw_button(screen, self.training_back_button, "BACK", 22)
        self._draw_button(screen, self.screenshot_button, "SCREENSHOT", 18, border_colour=(95, 170, 200))
        self._draw_button(
            screen,
            self.replay_button,
            f"SAVE REPLAY: {'ON' if self._pending_replay_save else 'OFF'}",
            16,
            border_colour=(110, 195, 130) if self._pending_replay_save else PANEL_BORDER,
        )
        self._draw_button(
            screen,
            self.play_pause_button,
            "PLAY" if self.training_paused else "PAUSE",
            22,
            border_colour=(80, 180, 110) if self.training_paused else (220, 140, 92),
        )
        speed_border = (95, 170, 200) if self.training_visualised else PANEL_BORDER
        self._draw_button(screen, self.speed_minus_button, "-", 28, border_colour=speed_border)
        self._draw_button(screen, self.speed_plus_button, "+", 28, border_colour=speed_border)

        pygame.draw.rect(screen, PANEL_BG, self.speed_label_rect, border_radius=10)
        pygame.draw.rect(screen, PANEL_BORDER, self.speed_label_rect, 2, border_radius=10)
        speed_text = f"{self.sim_speed:.2f}x" if self.training_visualised else "MAX"
        speed_label = self._font(22).render(speed_text, True, FG)
        screen.blit(speed_label, speed_label.get_rect(center=self.speed_label_rect.center))

        info_font = self._font(22)
        detail_font = self._font(20)

        lines = [
            f"Run ID: {getattr(self.run_ctx, 'run_id', None)}",
            f"Vehicles: {len(self.preview_map.vehicles)}",
            f"PPO backend: {'READY' if self.ppo is not None else 'NOT READY'}",
        ]
        start_x, start_y = self._text_pos("training_info_start", (self.left_panel.x + 18, self.left_panel.y + 72))
        y_pos = start_y
        for line in lines:
            txt = detail_font.render(line, True, FG)
            screen.blit(txt, (start_x, y_pos))
            y_pos += 30

        if self.episode_state is not None:
            throughput = float(self.episode_state.metrics.get("throughput", 0.0))
            current_success_rate = float(self.episode_state.metrics.get("success_rate", 0.0))
            avg_speed = float(self.episode_state.metrics.get("avg_speed", 0.0))
            last_pass = "N/A" if self.last_summary is None else ("YES" if self.last_summary.passed else "NO")
            last_failure = (
                "N/A"
                if self.last_episode_failed is None
                else ("YES" if self.last_episode_failed else "NO")
            )
            state_lines = [
                f"Episode index: {self.episode_state.episode_index}",
                f"Step count: {self.episode_state.step_count}",
                f"Collisions: {self.episode_state.metrics['collisions']:.0f}",
                f"Throughput: {throughput:.3f}",
                f"Loss: {float(self.episode_state.metrics.get('loss', 0.0)):.4f}",
                f"Current success rate: {current_success_rate:.2f}",
                f"Last episode pass: {last_pass}",
                f"Last episode failure: {last_failure}",
                f"Avg speed: {avg_speed:.3f}",
                (
                    f"Sample action (V1): {self.preview_actions[0]}"
                    if self.preview_actions
                    else "Sample action (V1): n/a"
                ),
            ]
            for line in state_lines:
                txt = detail_font.render(line, True, ACCENT)
                screen.blit(txt, (start_x, y_pos))
                y_pos += 26

        map_rect = self.right_panel.inflate(-26, -84)
        map_rect.y += 28
        if self.training_visualised:
            self._draw_map_preview(screen, map_rect)
        else:
            pygame.draw.rect(screen, MAP_EMPTY_COLOUR, map_rect, border_radius=8)
            pygame.draw.rect(screen, PANEL_BORDER, map_rect, 1, border_radius=8)
            headless_font = self._font(26)
            line_1 = headless_font.render("VISUALISATION OFF", True, FG)
            line_2 = self._font(21).render("Headless training running at maximum throughput.", True, ACCENT)
            screen.blit(line_1, line_1.get_rect(center=(map_rect.centerx, map_rect.centery - 16)))
            screen.blit(line_2, line_2.get_rect(center=(map_rect.centerx, map_rect.centery + 18)))

        phase_line = self._font(21).render(
            f"Phase {self.phase} | Level {self.level_index + 1}/{max(1, map_level_count(self.phase))}",
            True,
            FG,
        )
        phase_pos = self._text_pos("training_status", (self.left_panel.x + 16, self.left_panel.bottom - 60))
        screen.blit(phase_line, phase_pos)

        runtime_line = self._font(20).render(self.status_message, True, ACCENT)
        runtime_pos = self._text_pos("training_detail_status", (self.left_panel.x + 16, self.left_panel.bottom - 34))
        screen.blit(runtime_line, runtime_pos)
        if self._pending_screenshot:
            self._save_training_screenshot(screen)
            self._pending_screenshot = False

    def draw(self, screen: pygame.Surface) -> None:
        """
        Use:
        Render the TRAIN template UI and current environment summary.

        Inputs:
        - screen: Target surface.

        Output:
        None.
        """
        self._tick_runtime()

        if self.training_view:
            self._draw_training_runtime(screen)
            return

        screen.fill(BG)
        title = self._font(96).render("TRAIN", True, FG)
        title_pos = self._text_pos("title", (self.screen_rect.centerx, 8))
        screen.blit(title, title.get_rect(midtop=title_pos))

        pygame.draw.rect(screen, PANEL_BG, self.left_panel, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, self.left_panel, 2, border_radius=12)
        pygame.draw.rect(screen, PANEL_BG, self.right_panel, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, self.right_panel, 2, border_radius=12)

        self._draw_button(screen, self.back_button, "BACK", 22)
        self._draw_button(screen, self.save_network_button, "SAVE NETWORK", 20, border_colour=(80, 170, 110))
        self._draw_button(screen, self.load_network_button, "LOAD NETWORK", 20, border_colour=(100, 155, 220))
        self._draw_button(screen, self.reset_button, "RESET ENV", 22)
        self._draw_button(screen, self.phase_minus_button, "-", 34)
        self._draw_button(screen, self.phase_plus_button, "+", 34)
        self._draw_button(screen, self.level_minus_button, "-", 34, border_colour=(95, 170, 200))
        self._draw_button(screen, self.level_plus_button, "+", 34, border_colour=(95, 170, 200))
        self._draw_button(screen, self.begin_button, "BEGIN TRAINING", 22)
        self._draw_button(
            screen,
            self.visualise_button,
            f"VISUALISE TRAINING: {'ON' if self.training_visualised else 'OFF'}",
            19,
            border_colour=(95, 170, 200) if self.training_visualised else PANEL_BORDER,
        )

        phase_font = self._font(24)
        phase_text = phase_font.render(f"START PHASE: {self.phase}", True, FG)
        phase_pos = self._text_pos("phase_label", (self.left_panel.x + 108, self.phase_minus_button.y - 38))
        screen.blit(phase_text, phase_pos)

        level_font = self._font(24)
        level_text = level_font.render(
            f"MAP LEVEL: {self.level_index + 1}/{max(1, map_level_count(self.phase))}",
            True,
            FG,
        )
        level_pos = self._text_pos("level_label", (self.left_panel.x + 108, self.level_minus_button.y - 38))
        screen.blit(level_text, level_pos)

        info_font = self._font(22)
        detail_font = self._font(20)

        lines = [
            f"Run ID: {getattr(self.run_ctx, 'run_id', None)}",
            f"Vehicles: {len(self.preview_map.vehicles)}",
            f"PPO backend: {'READY' if self.ppo is not None else 'NOT READY'}",
        ]
        info_anchor_y = max(self.level_minus_button.bottom + 16, self.left_panel.y + 330)
        start_x, start_y = self._text_pos("info_start", (self.left_panel.x + 18, info_anchor_y))
        y_pos = start_y
        for line in lines:
            txt = detail_font.render(line, True, FG)
            screen.blit(txt, (start_x, y_pos))
            y_pos += 30

        if self.episode_state is not None:
            state_lines = [
                f"Episode index: {self.episode_state.episode_index}",
                f"Step count: {self.episode_state.step_count}",
                f"Collisions: {self.episode_state.metrics['collisions']:.0f}",
                (
                    f"Sample action (V1): {self.preview_actions[0]}"
                    if self.preview_actions
                    else "Sample action (V1): n/a"
                ),
            ]
            for line in state_lines:
                txt = detail_font.render(line, True, ACCENT)
                screen.blit(txt, (start_x, y_pos))
                y_pos += 26

        map_rect = self.right_panel.inflate(-26, -84)
        map_rect.y += 28
        self._draw_map_preview(screen, map_rect)

        status = info_font.render(self.status_message, True, ACCENT)
        screen.blit(status, self._text_pos("status", (self.left_panel.x + 16, self.left_panel.bottom - 34)))
