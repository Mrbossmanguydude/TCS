"""
Setup screen for training configuration preview.
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Dict, List, Tuple

import pygame

from src.utils.controller_prep import PreparedControllers, build_vn_feature_vector, prepare_vn_pn
from src.utils.map_generation import (
    clamp_phase,
    generate_phase_map,
    map_level_count,
    phase_spec,
)
from src.utils.run_init import save_manual_config_log, write_rolling_config


BG = (53, 62, 67)
FG = (235, 235, 235)
ACCENT = (180, 180, 180)
HILITE = (255, 215, 0)
PANEL_BG = (46, 53, 58)
PANEL_BORDER = (78, 91, 99)
ROAD_DISCRETE = (142, 155, 165)
ROAD_CONTINUOUS = (125, 145, 176)
ROAD_CENTER_MARK = (230, 230, 230)
BLOCKED = (27, 31, 35)
ROUNDABOUT_OUTER = (204, 172, 92)
ROUNDABOUT_CENTER = (72, 78, 86)
ROAD_TWO_LANE = (90, 150, 210)
JUNCTION_ONE_LANE = (228, 171, 90)
JUNCTION_TWO_LANE = (104, 222, 205)
JUNCTION_T = (122, 195, 122)
JUNCTION_CROSS = (190, 138, 222)
SPAWN_COLOR = (122, 221, 109)
DEST_COLOR = (245, 122, 122)


class SetupScreen:
    """
    Training setup preview screen.
    """

    def __init__(self, screen_rect: pygame.Rect, font_path: Path | None = None, run_ctx: Any = None, ui_offsets: Dict[str, Tuple[int, int]] | None = None) -> None:
        """
        Use:
        Build setup screen state, controls, and preview snapshots.

        Inputs:
        - screen_rect: Full display rectangle for this screen.
        - font_path: Optional custom font path.
        - run_ctx: Shared runtime context from startup initialisation.

        Output:
        None. The object is initialised in-place for the GUI loop.
        """
        self.screen_rect = screen_rect
        self.font_path = font_path
        self.run_ctx = run_ctx
        self.ui_offsets = ui_offsets or {}

        self.seed = int(run_ctx.seed) if run_ctx is not None else random.randint(0, 2**31 - 1)
        scenario_cfg = (run_ctx.config.get("scenario", {}) if run_ctx is not None else {})
        phase_zero = int(scenario_cfg.get("phase", 0))
        self.phase = clamp_phase(phase_zero + 1)
        self.level_index = 0
        self.road_density = float(scenario_cfg.get("preview_road_density", 0.72))
        self.structure_density = float(scenario_cfg.get("preview_structure_density", 0.62))
        self.status_message = "Preview ready."
        self.latest_feature_vector: List[float] = []

        self.back_button = self._offset_rect("back_button", pygame.Rect(40, 30, 96, 48))
        self.seed_button = self._offset_rect("seed_button", pygame.Rect(58, 232, 300, 50))
        self.phase_minus = self._offset_rect("phase_minus", pygame.Rect(58, 338, 56, 50))
        self.phase_plus = self._offset_rect("phase_plus", pygame.Rect(302, 338, 56, 50))
        self.level_minus = self._offset_rect("level_minus", pygame.Rect(58, 418, 56, 50))
        self.level_plus = self._offset_rect("level_plus", pygame.Rect(302, 418, 56, 50))
        self.road_minus = self._offset_rect("road_minus", pygame.Rect(58, 498, 56, 50))
        self.road_plus = self._offset_rect("road_plus", pygame.Rect(302, 498, 56, 50))
        self.struct_minus = self._offset_rect("struct_minus", pygame.Rect(58, 568, 56, 50))
        self.struct_plus = self._offset_rect("struct_plus", pygame.Rect(302, 568, 56, 50))
        self.refresh_button = self._offset_rect("refresh_button", pygame.Rect(58, 624, 145, 30))
        self.save_button = self._offset_rect("save_button", pygame.Rect(213, 624, 145, 30))

        self.left_panel = self._offset_rect("left_panel", pygame.Rect(40, 118, 336, self.screen_rect.height - 158))
        self.preview_panel = self._offset_rect("preview_panel", pygame.Rect(398, 118, self.screen_rect.width - 438, self.screen_rect.height - 158))

        self.preview_map = generate_phase_map(
            self.seed,
            self.phase,
            self.level_index,
            road_density=self.road_density,
            structure_density=self.structure_density,
        )
        self.controllers: PreparedControllers = self._prepare_controllers()
        self._refresh_feature_preview()
        self._print_preview_metrics("init")
        self._sync_run_context()

    def _offset(self, key: str) -> tuple[int, int]:
        """
        Use:
        Get pixel offset tuple for a setup UI element.

        Inputs:
        - key: Offset key in `ui_offsets`.

        Output:
        Tuple `(dx, dy)`.
        """
        value = self.ui_offsets.get(key, (0, 0))
        if isinstance(value, tuple) and len(value) == 2:
            return int(value[0]), int(value[1])
        return 0, 0

    def _offset_rect(self, key: str, rect: pygame.Rect) -> pygame.Rect:
        """
        Use:
        Return a moved copy of `rect` using an offset key.

        Inputs:
        - key: Offset key in `ui_offsets`.
        - rect: Base rectangle before applying offset.

        Output:
        Offset-adjusted rectangle.
        """
        dx, dy = self._offset(key)
        return rect.move(dx, dy)

    def _text_pos(self, key: str, base: tuple[int, int]) -> tuple[int, int]:
        """
        Use:
        Apply offset key to a base text anchor coordinate.

        Inputs:
        - key: Offset key in `ui_offsets`.
        - base: Base `(x, y)` position.

        Output:
        Offset-adjusted `(x, y)` position.
        """
        dx, dy = self._offset(key)
        return base[0] + dx, base[1] + dy
    def _font(self, size: int) -> pygame.font.Font:
        """
        Use:
        Resolve configured font with fallback when custom font is unavailable.

        Inputs:
        - size: Font size in pixels.

        Output:
        Pygame font object.
        """
        return pygame.font.Font(self.font_path, size) if self.font_path else pygame.font.SysFont(None, size)

    def _prepare_controllers(self) -> PreparedControllers:
        """
        Use:
        Prepare VN + PN metadata from current runtime config.

        Inputs:
        - None. Uses `self.run_ctx` and project root.

        Output:
        `PreparedControllers` metadata object.
        """
        base_dir = self.run_ctx.base_dir if self.run_ctx is not None else Path(__file__).resolve().parents[2]
        cfg = self.run_ctx.config if self.run_ctx is not None else {}
        return prepare_vn_pn(cfg, base_dir)

    def _refresh_feature_preview(self) -> None:
        """
        Use:
        Build one VN feature vector from current preview vehicle state.

        Inputs:
        - None. Uses the first preview vehicle and nearby preview spawns.

        Output:
        None. Updates `self.latest_feature_vector`.
        """
        self.latest_feature_vector = []
        if not self.preview_map.vehicles:
            return
        primary = self.preview_map.vehicles[0]
        neighbour_nodes = [vehicle.spawn for vehicle in self.preview_map.vehicles[1:3]]
        self.latest_feature_vector = build_vn_feature_vector(primary, self.preview_map, neighbour_nodes)

    def _draw_gradient_button(
        self,
        screen: pygame.Surface,
        rect: pygame.Rect,
        text: str,
        text_size: int,
        border_color: tuple[int, int, int] = HILITE,
    ) -> None:
        """
        Use:
        Draw a rounded gradient button with centered text.

        Inputs:
        - screen: Target Pygame surface.
        - rect: Button rectangle.
        - text: Button label.
        - text_size: Font size for label.
        - border_color: Border RGB color.

        Output:
        None. Draws directly onto `screen`.
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
        text_surface = font.render(text, True, FG)
        shadow = font.render(text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=rect.center)
        screen.blit(shadow, text_rect.move(1, 1))
        screen.blit(text_surface, text_rect)

    def _set_phase(self, phase: int) -> None:
        """
        Use:
        Update phase and clamp map level to valid range.

        Inputs:
        - phase: Candidate phase number.

        Output:
        None. Regenerates preview and updates status.
        """
        self.phase = clamp_phase(phase)
        max_levels = map_level_count(self.phase)
        self.level_index = max(0, min(self.level_index, max_levels - 1))
        self._rebuild_preview("Phase updated.")

    def _set_level(self, new_level: int) -> None:
        """
        Use:
        Update map complexity tier within active phase bounds.

        Inputs:
        - new_level: Candidate zero-based level index.

        Output:
        None. Regenerates preview and updates status.
        """
        max_levels = map_level_count(self.phase)
        self.level_index = max(0, min(new_level, max_levels - 1))
        self._rebuild_preview("Complexity updated.")

    def _set_road_density(self, delta: float) -> None:
        """
        Use:
        Adjust road-density control and rebuild preview.

        Inputs:
        - delta: Increment/decrement value.

        Output:
        None. Updates control state and map preview.
        """
        self.road_density = max(0.35, min(1.35, round(self.road_density + delta, 2)))
        self._rebuild_preview("Road density updated.")

    def _set_structure_density(self, delta: float) -> None:
        """
        Use:
        Adjust structure-density control and rebuild preview.

        Inputs:
        - delta: Increment/decrement value.

        Output:
        None. Updates control state and map preview.
        """
        self.structure_density = max(0.20, min(1.50, round(self.structure_density + delta, 2)))
        self._rebuild_preview("Structure density updated.")

    def _print_preview_metrics(self, reason: str) -> None:
        """
        Use:
        Print setup preview metrics to terminal for quick validation.

        Inputs:
        - reason: Trigger label for the print call.

        Output:
        None. Writes a formatted line to stdout.
        """
        preview = self.preview_map
        road_cells = len(preview.roads)
        total_cells = max(1, preview.width * preview.height)
        coverage = (road_cells / total_cells) * 100.0
        counts = preview.structure_counts
        print(
            "[SETUP PREVIEW]",
            f"reason={reason}",
            f"seed={self.seed}",
            f"phase={self.phase}",
            f"level={self.level_index + 1}/{map_level_count(self.phase)}",
            f"size={preview.width}x{preview.height}",
            f"mode={'continuous' if preview.continuous else 'discrete'}",
            f"road_density={self.road_density:.2f}",
            f"structure_density={self.structure_density:.2f}",
            f"road_cells={road_cells}",
            f"coverage={coverage:.1f}%",
            f"roundabout={counts['roundabout']}",
            f"j1={counts['junction_turn_one_lane']}",
            f"j2={counts['junction_turn_two_lane']}",
            f"jt={counts['junction_t']}",
            f"jx={counts['junction_cross']}",
            f"two_lane={counts['road_two_lane']}",
            f"vehicles={len(preview.vehicles)}",
            f"vn_in={self.controllers.vn.input_size}",
            f"pn_actions={self.controllers.pn.action_size}",
            f"prev_model={self.controllers.pn.uses_previous_model}",
        )

    def _rebuild_preview(self, status: str | None = None) -> None:
        """
        Use:
        Regenerate map preview and dependent setup metadata.

        Inputs:
        - status: Optional status message after rebuild.

        Output:
        None. Updates preview map, controller metadata, and runtime config sync.
        """
        self.preview_map = generate_phase_map(
            self.seed,
            self.phase,
            self.level_index,
            road_density=self.road_density,
            structure_density=self.structure_density,
        )
        self.controllers = self._prepare_controllers()
        self._refresh_feature_preview()
        self._sync_run_context()
        self._print_preview_metrics("rebuild")
        if status:
            self.status_message = status

    def _sync_run_context(self) -> None:
        """
        Use:
        Keep shared runtime context aligned with setup controls.

        Inputs:
        - None. Uses current local setup values.

        Output:
        None. Mutates `run_ctx` and updates rolling runtime config file.
        """
        if self.run_ctx is None:
            return

        self.run_ctx.seed = int(self.seed)
        self.run_ctx.config["seed"] = int(self.seed)

        scenario = self.run_ctx.config.setdefault("scenario", {})
        scenario["seed"] = int(self.seed)
        scenario["phase"] = int(self.phase - 1)
        scenario["map_size"] = int(self.preview_map.level_size)
        scenario["continuous"] = bool(self.preview_map.continuous)
        scenario["roundabouts_enabled"] = bool(self.preview_map.roundabouts_enabled)
        scenario["preview_road_density"] = float(self.road_density)
        scenario["preview_structure_density"] = float(self.structure_density)
        scenario["preview_vehicle_count"] = int(len(self.preview_map.vehicles))
        write_rolling_config(self.run_ctx.base_dir, self.run_ctx.config)

    def handle_events(self, events: list[pygame.event.Event]) -> str:
        """
        Use:
        Process setup mouse events and return the next GUI state.

        Inputs:
        - events: Pygame event list for current frame.

        Output:
        Next state token (`SETUP` or `MENU`).
        """
        for event in events:
            if event.type != pygame.MOUSEBUTTONUP or event.button != 1:
                continue

            if self.back_button.collidepoint(event.pos):
                return "MENU"
            if self.seed_button.collidepoint(event.pos):
                self.seed = random.randint(0, 2**31 - 1)
                self._rebuild_preview(f"Seed regenerated: {self.seed}")
                continue
            if self.phase_minus.collidepoint(event.pos):
                self._set_phase(self.phase - 1)
                continue
            if self.phase_plus.collidepoint(event.pos):
                self._set_phase(self.phase + 1)
                continue
            if self.level_minus.collidepoint(event.pos):
                self._set_level(self.level_index - 1)
                continue
            if self.level_plus.collidepoint(event.pos):
                self._set_level(self.level_index + 1)
                continue
            if self.road_minus.collidepoint(event.pos):
                self._set_road_density(-0.05)
                continue
            if self.road_plus.collidepoint(event.pos):
                self._set_road_density(+0.05)
                continue
            if self.struct_minus.collidepoint(event.pos):
                self._set_structure_density(-0.05)
                continue
            if self.struct_plus.collidepoint(event.pos):
                self._set_structure_density(+0.05)
                continue
            if self.refresh_button.collidepoint(event.pos):
                self._rebuild_preview("Preview refreshed.")
                continue
            if self.save_button.collidepoint(event.pos):
                if self.run_ctx is not None:
                    path = save_manual_config_log(
                        base_dir=self.run_ctx.base_dir,
                        config=self.run_ctx.config,
                        run_id=self.run_ctx.run_id,
                        label="setup",
                    )
                    self.run_ctx.config_snapshot_path = path
                    self.status_message = f"Config log saved: {path.name}"
                else:
                    self.status_message = "Config save unavailable (no runtime context)."

        return "SETUP"

    def _draw_preview_map(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        """
        Use:
        Draw generated road grid plus preview vehicles inside preview panel.

        Inputs:
        - screen: Target Pygame surface.
        - rect: Preview panel rectangle.

        Output:
        None. Draw operations are applied directly to `screen`.
        """
        preview = self.preview_map
        inner = rect.inflate(-20, -140)
        pygame.draw.rect(screen, BLOCKED, inner, border_radius=8)
        pygame.draw.rect(screen, PANEL_BORDER, inner, 1, border_radius=8)

        if preview.width <= 0 or preview.height <= 0:
            return

        tile = max(4, min(inner.width // preview.width, inner.height // preview.height))
        grid_width = tile * preview.width
        grid_height = tile * preview.height
        ox = inner.x + (inner.width - grid_width) // 2
        oy = inner.y + (inner.height - grid_height) // 2

        default_road_color = ROAD_CONTINUOUS if preview.continuous else ROAD_DISCRETE

        for y_val in range(preview.height):
            for x_val in range(preview.width):
                cell = pygame.Rect(ox + (x_val * tile), oy + (y_val * tile), tile, tile)
                point = (x_val, y_val)
                node_type = preview.node_types.get(point)
                if point in preview.roads:
                    road_color = default_road_color
                    if node_type == "road_two_lane":
                        road_color = ROAD_TWO_LANE
                    elif node_type == "junction_turn_one_lane":
                        road_color = JUNCTION_ONE_LANE
                    elif node_type == "junction_turn_two_lane":
                        road_color = JUNCTION_TWO_LANE
                    elif node_type == "junction_t":
                        road_color = JUNCTION_T
                    elif node_type == "junction_cross":
                        road_color = JUNCTION_CROSS

                    pygame.draw.rect(screen, road_color, cell)
                    if preview.continuous and tile >= 8:
                        pygame.draw.line(
                            screen,
                            ROAD_CENTER_MARK,
                            (cell.centerx, cell.y + 1),
                            (cell.centerx, cell.bottom - 2),
                            1,
                        )
                else:
                    pygame.draw.rect(screen, BLOCKED, cell)

                if node_type == "roundabout":
                    pygame.draw.rect(screen, ROUNDABOUT_OUTER, cell, 2)
                    radius = max(2, int(tile * 0.28))
                    pygame.draw.circle(screen, ROUNDABOUT_CENTER, cell.center, radius)

        # Draw vehicle spawn/destination markers on top of the road layout.
        for vehicle in preview.vehicles:
            spawn_cell = pygame.Rect(ox + (vehicle.spawn[0] * tile), oy + (vehicle.spawn[1] * tile), tile, tile)
            dest_cell = pygame.Rect(ox + (vehicle.destination[0] * tile), oy + (vehicle.destination[1] * tile), tile, tile)
            pygame.draw.circle(screen, SPAWN_COLOR, spawn_cell.center, max(2, tile // 4))
            pygame.draw.rect(screen, DEST_COLOR, dest_cell.inflate(-(tile // 3), -(tile // 3)))

    def draw(self, screen: pygame.Surface) -> None:
        """
        Use:
        Render setup controls, metadata, and map preview.

        Inputs:
        - screen: Target Pygame surface.

        Output:
        None. Draw operations are applied directly to `screen`.
        """
        screen.fill(BG)

        title_font = self._font(92)
        body_font = self._font(24)
        value_font = self._font(30)
        info_font = self._font(22)
        status_font = self._font(22)

        title_text = title_font.render("SETUP", True, FG)
        title_rect = title_text.get_rect(midtop=(self.screen_rect.centerx, 20))
        screen.blit(title_text, title_rect)

        pygame.draw.rect(screen, PANEL_BG, self.left_panel, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, self.left_panel, 2, border_radius=12)
        pygame.draw.rect(screen, PANEL_BG, self.preview_panel, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, self.preview_panel, 2, border_radius=12)

        # Left controls
        seed_label = body_font.render("SEED", True, ACCENT)
        screen.blit(seed_label, self._text_pos("seed_label", (58, 188)))
        seed_value = value_font.render(str(self.seed), True, FG)
        screen.blit(seed_value, self._text_pos("seed_value", (58, 210)))

        phase_name = phase_spec(self.phase).name
        phase_label = body_font.render("PHASE", True, ACCENT)
        phase_value = value_font.render(str(self.phase), True, FG)
        screen.blit(phase_label, self._text_pos("phase_label", (58, 304)))
        screen.blit(phase_value, self._text_pos("phase_value", (174, 344)))

        level_label = body_font.render("MAP LEVEL", True, ACCENT)
        level_value = value_font.render(str(self.level_index + 1), True, FG)
        level_total = info_font.render(f"/ {map_level_count(self.phase)}", True, ACCENT)
        screen.blit(level_label, self._text_pos("level_label", (58, 384)))
        screen.blit(level_value, self._text_pos("level_value", (174, 424)))
        screen.blit(level_total, self._text_pos("level_total", (214, 432)))

        road_label = body_font.render("ROAD DENSITY", True, ACCENT)
        road_value = value_font.render(f"{self.road_density:.2f}", True, FG)
        screen.blit(road_label, self._text_pos("road_label", (58, 464)))
        screen.blit(road_value, self._text_pos("road_value", (174, 504)))

        struct_label = body_font.render("STRUCTURE DENSITY", True, ACCENT)
        struct_value = value_font.render(f"{self.structure_density:.2f}", True, FG)
        screen.blit(struct_label, self._text_pos("struct_label", (58, 534)))
        screen.blit(struct_value, self._text_pos("struct_value", (174, 574)))

        self._draw_gradient_button(screen, self.back_button, "BACK", 24)
        self._draw_gradient_button(screen, self.seed_button, "REGENERATE SEED", 24)
        self._draw_gradient_button(screen, self.phase_minus, "-", 36)
        self._draw_gradient_button(screen, self.phase_plus, "+", 36)
        self._draw_gradient_button(screen, self.level_minus, "-", 36)
        self._draw_gradient_button(screen, self.level_plus, "+", 36)
        self._draw_gradient_button(screen, self.road_minus, "-", 36)
        self._draw_gradient_button(screen, self.road_plus, "+", 36)
        self._draw_gradient_button(screen, self.struct_minus, "-", 36)
        self._draw_gradient_button(screen, self.struct_plus, "+", 36)
        self._draw_gradient_button(screen, self.refresh_button, "REFRESH", 20)
        self._draw_gradient_button(screen, self.save_button, "SAVE LOG", 20)

        # Right information
        info_x, info_y = self._text_pos("info_block", (self.preview_panel.x + 20, self.preview_panel.y + 16))
        info_lines = [
            f"Phase {self.phase} ({'Continuous' if self.preview_map.continuous else 'Discrete'})  |  Map {self.preview_map.width}x{self.preview_map.height}",
            f"Vehicles: {len(self.preview_map.vehicles)}  |  VN {self.controllers.vn.input_size}  PN {self.controllers.pn.action_size}",
            "Preview metrics are printed in terminal.",
        ]
        for line in info_lines:
            rendered = info_font.render(line, True, FG)
            screen.blit(rendered, (info_x, info_y))
            info_y += 26

        self._draw_preview_map(screen, self.preview_panel)

        status_surface = status_font.render(self.status_message, True, ACCENT)
        screen.blit(status_surface, self._text_pos("status", (self.left_panel.x + 18, self.left_panel.bottom - 36)))
