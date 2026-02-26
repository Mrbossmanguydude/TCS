"""
Train screen template for environment reset and initial episode-condition setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import pygame

from src.utils.map_generation import PreviewVehicle, clamp_phase, generate_phase_map
from src.utils.run_init import reseed_all


BG = (53, 62, 67)
FG = (235, 235, 235)
ACCENT = (180, 180, 180)
HILITE = (255, 215, 0)
PANEL_BG = (46, 53, 58)
PANEL_BORDER = (78, 91, 99)
ROAD_COLOR = (130, 145, 162)
SPAWN_COLOR = (112, 216, 104)
DEST_COLOR = (238, 124, 124)


@dataclass
class EpisodeVehicle:
    """
    Use:
    Represent one instantiated episode vehicle for the current environment reset.

    Attributes:
    - vehicle_id: Unique identifier within current episode.
    - spawn: Initial spawn tile.
    - destination: Destination tile.
    - current: Current tile position (initially equals spawn).
    """

    vehicle_id: int
    spawn: tuple[int, int]
    destination: tuple[int, int]
    current: tuple[int, int]


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
    Provide a TRAIN template screen with explicit environment reset logic,
    deterministic episode initialisation, and a training sub-window trigger.

    Attributes:
    - run_ctx: Shared run context used for seed/config access and persistence metadata.
    - phase: Active curriculum phase for TRAIN map initialisation.
    - level_index: Active map complexity level.
    - road_density: Road-density parameter forwarded to map generation.
    - structure_density: Structure-density parameter forwarded to map generation.
    - preview_map: Current generated map used to instantiate episode vehicles.
    - episode_state: Current episode runtime state.
    - training_window_visible: Whether the train sub-window is shown.
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
        self.training_window_visible = False
        self.episode_running = False
        self._episode_start_t = 0.0
        self._reset_counter = 0

        self.preview_map = generate_phase_map(
            self.seed,
            self.phase,
            self.level_index,
            road_density=self.road_density,
            structure_density=self.structure_density,
        )
        self.episode_state: Optional[EpisodeState] = None

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
        self.back_button = self._offset_rect("back_button", pygame.Rect(34, 28, 96, 46))
        self.reset_button = self._offset_rect("reset_button", pygame.Rect(50, 150, 230, 44))
        self.begin_button = self._offset_rect("begin_button", pygame.Rect(50, 204, 230, 44))
        self.toggle_sub_button = self._offset_rect("sub_toggle_button", pygame.Rect(50, 258, 230, 44))

        self.left_panel = self._offset_rect("left_panel", pygame.Rect(34, 108, 304, self.screen_rect.height - 132))
        self.right_panel = self._offset_rect(
            "right_panel",
            pygame.Rect(356, 108, self.screen_rect.width - 390, self.screen_rect.height - 132),
        )

        sw = min(760, self.screen_rect.width - 140)
        sh = min(460, self.screen_rect.height - 140)
        self.subwindow_rect = self._offset_rect(
            "subwindow",
            pygame.Rect(
            self.screen_rect.centerx - sw // 2,
            self.screen_rect.centery - sh // 2,
            sw,
            sh,
            ),
        )
        self.subwindow_close = self._offset_rect(
            "subwindow_close",
            pygame.Rect(self.subwindow_rect.right - 118, self.subwindow_rect.y + 14, 96, 36),
        )

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
        border_color: tuple[int, int, int] = HILITE,
    ) -> None:
        """
        Use:
        Render one reusable gradient button.

        Inputs:
        - screen: Destination surface.
        - rect: Button rectangle.
        - text: Label text.
        - text_size: Label font size.
        - border_color: RGB border color.

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
        pygame.draw.rect(screen, border_color, rect, 2, border_radius=10)

        font = self._font(text_size)
        label = font.render(text, True, FG)
        shadow = font.render(text, True, (0, 0, 0))
        text_rect = label.get_rect(center=rect.center)
        screen.blit(shadow, text_rect.move(1, 1))
        screen.blit(label, text_rect)

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
        for vehicle in source:
            vehicles.append(
                EpisodeVehicle(
                    vehicle_id=int(vehicle.vehicle_id),
                    spawn=(int(vehicle.spawn[0]), int(vehicle.spawn[1])),
                    destination=(int(vehicle.destination[0]), int(vehicle.destination[1])),
                    current=(int(vehicle.spawn[0]), int(vehicle.spawn[1])),
                )
            )
        return vehicles

    def reset_environment(self, initial: bool = False) -> EpisodeState:
        """
        Use:
        Fully reset TRAIN environment state and rebuild episode initial conditions.

        Inputs:
        - initial: True for startup reset; False for user-triggered reset.

        Output:
        New `EpisodeState` object.
        """
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
            },
        )

        self.episode_running = False
        self._episode_start_t = 0.0
        self.status_message = "Environment reset complete. Initial conditions ready."
        return self.episode_state

    def begin_episode(self) -> None:
        """
        Use:
        Start runtime timing for the current episode template and open the training
        sub-window for future live visualisation.

        Inputs:
        - None.

        Output:
        None.
        """
        if self.episode_state is None:
            self.reset_environment(initial=True)

        self.episode_running = True
        self._episode_start_t = time.perf_counter()
        self.training_window_visible = True
        self.status_message = "Episode started. Training sub-window opened."

    def _tick_runtime(self) -> None:
        """
        Use:
        Update episode elapsed time while an episode is running.

        Inputs:
        - None.

        Output:
        None.
        """
        if not self.episode_running or self.episode_state is None:
            return
        self.episode_state.elapsed_seconds = max(0.0, time.perf_counter() - self._episode_start_t)

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
        pygame.draw.rect(screen, (28, 32, 36), rect, border_radius=8)
        pygame.draw.rect(screen, PANEL_BORDER, rect, 1, border_radius=8)

        if preview.width <= 0 or preview.height <= 0:
            return

        tile = max(6, min(rect.width // preview.width, rect.height // preview.height))
        grid_w = tile * preview.width
        grid_h = tile * preview.height
        ox = rect.x + (rect.width - grid_w) // 2
        oy = rect.y + (rect.height - grid_h) // 2

        for y_pos in range(preview.height):
            for x_pos in range(preview.width):
                cell = pygame.Rect(ox + x_pos * tile, oy + y_pos * tile, tile, tile)
                if (x_pos, y_pos) in preview.roads:
                    pygame.draw.rect(screen, ROAD_COLOR, cell)

        if self.episode_state is not None:
            for vehicle in self.episode_state.vehicles:
                spawn_rect = pygame.Rect(ox + vehicle.spawn[0] * tile, oy + vehicle.spawn[1] * tile, tile, tile)
                dest_rect = pygame.Rect(ox + vehicle.destination[0] * tile, oy + vehicle.destination[1] * tile, tile, tile)
                pygame.draw.circle(screen, SPAWN_COLOR, spawn_rect.center, max(2, tile // 4))
                pygame.draw.rect(screen, DEST_COLOR, dest_rect.inflate(-max(2, tile // 3), -max(2, tile // 3)), 2)

    def handle_events(self, events: List[pygame.event.Event]) -> str:
        """
        Use:
        Process TRAIN interactions and return next screen state.

        Inputs:
        - events: Current frame event list.

        Output:
        Next state token (`TRAIN` or `MENU`).
        """
        for event in events:
            if event.type != pygame.MOUSEBUTTONUP or event.button != 1:
                continue

            if self.training_window_visible and self.subwindow_close.collidepoint(event.pos):
                self.training_window_visible = False
                continue

            if self.back_button.collidepoint(event.pos):
                return "MENU"
            if self.reset_button.collidepoint(event.pos):
                self.reset_environment(initial=False)
                continue
            if self.begin_button.collidepoint(event.pos):
                self.begin_episode()
                continue
            if self.toggle_sub_button.collidepoint(event.pos):
                self.training_window_visible = not self.training_window_visible
                continue

        return "TRAIN"

    def draw(self, screen: pygame.Surface) -> None:
        """
        Use:
        Render the TRAIN template UI, current environment summary, and optional
        training sub-window overlay.

        Inputs:
        - screen: Target surface.

        Output:
        None.
        """
        self._tick_runtime()

        screen.fill(BG)
        title = self._font(96).render("TRAIN", True, FG)
        title_pos = self._text_pos("title", (self.screen_rect.centerx, 8))
        screen.blit(title, title.get_rect(midtop=title_pos))

        pygame.draw.rect(screen, PANEL_BG, self.left_panel, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, self.left_panel, 2, border_radius=12)
        pygame.draw.rect(screen, PANEL_BG, self.right_panel, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, self.right_panel, 2, border_radius=12)

        self._draw_button(screen, self.back_button, "BACK", 22)
        self._draw_button(screen, self.reset_button, "RESET ENV", 22)
        self._draw_button(screen, self.begin_button, "BEGIN EPISODE", 22)
        self._draw_button(screen, self.toggle_sub_button, "TRAIN SUB-WINDOW", 20)

        info_font = self._font(22)
        detail_font = self._font(20)

        lines = [
            f"Run ID: {getattr(self.run_ctx, 'run_id', None)}",
            f"Seed: {self.seed}",
            f"Phase: {self.phase}",
            f"Map: {self.preview_map.width}x{self.preview_map.height}",
            f"Vehicles: {len(self.preview_map.vehicles)}",
        ]
        start_x, start_y = self._text_pos("info_start", (self.left_panel.x + 18, self.left_panel.y + 330))
        y_pos = start_y
        for line in lines:
            txt = detail_font.render(line, True, FG)
            screen.blit(txt, (start_x, y_pos))
            y_pos += 30

        if self.episode_state is not None:
            state_lines = [
                f"Episode index: {self.episode_state.episode_index}",
                f"Step count: {self.episode_state.step_count}",
                f"Elapsed: {self.episode_state.elapsed_seconds:.2f}s",
                f"Collisions: {self.episode_state.metrics['collisions']:.0f}",
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

        if self.training_window_visible:
            overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 136))
            screen.blit(overlay, (0, 0))

            pygame.draw.rect(screen, PANEL_BG, self.subwindow_rect, border_radius=14)
            pygame.draw.rect(screen, PANEL_BORDER, self.subwindow_rect, 2, border_radius=14)
            self._draw_button(screen, self.subwindow_close, "CLOSE", 18)

            head = self._font(30).render("TRAINING SUB-WINDOW (TEMPLATE)", True, FG)
            screen.blit(head, self._text_pos("subwindow_header", (self.subwindow_rect.x + 24, self.subwindow_rect.y + 20)))

            lines = [
                "This window is the placeholder for live training visuals.",
                "Environment reset + initial conditions are now wired in TRAIN.",
                "Next step: attach simulation stepping and reward/metric streams.",
            ]
            text_x, text_y = self._text_pos("subwindow_text", (self.subwindow_rect.x + 24, self.subwindow_rect.y + 78))
            y = text_y
            for line in lines:
                txt = self._font(22).render(line, True, ACCENT)
                screen.blit(txt, (text_x, y))
                y += 34
