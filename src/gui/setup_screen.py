"""
Setup screen for configuring training parameters and generating a map preview sub-window.
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Dict, List, Optional

import pygame

from src.utils.controller_prep import PreparedControllers, build_vn_feature_vector, prepare_vn_pn
from src.utils.map_generation import clamp_phase, generate_phase_map, map_level_count, phase_spec
from src.utils.run_init import save_manual_config_log, write_rolling_config


BG = (53, 62, 67)
FG = (235, 235, 235)
ACCENT = (180, 180, 180)
HILITE = (255, 215, 0)
PANEL_BG = (46, 53, 58)
PANEL_BORDER = (78, 91, 99)
BLOCKED = (27, 31, 35)
ROAD_DISCRETE = (142, 155, 165)
ROAD_CONTINUOUS = (125, 145, 176)
ROAD_TWO_LANE = (90, 150, 210)
JUNCTION_ONE_LANE = (228, 171, 90)
JUNCTION_TWO_LANE = (104, 222, 205)
JUNCTION_T = (122, 195, 122)
JUNCTION_CROSS = (190, 138, 222)
ROUNDABOUT_OUTER = (204, 172, 92)
ROUNDABOUT_CENTER = (72, 78, 86)
DEST_COLOR = (245, 122, 122)


class SetupScreen:
    """
    Use:
    Provide setup controls for TRAIN configuration and open a generated map preview
    as a dedicated sub-window when requested by the user.

    Attributes:
    - screen_rect: Full display rectangle used for responsive layout.
    - font_path: Optional custom font path.
    - run_ctx: Shared runtime context for seed/config persistence.
    - seed: Active setup seed used for deterministic generation.
    - phase: Curriculum phase in range 1..6.
    - level_index: Zero-based map complexity index for selected phase.
    - road_density: Density multiplier for road generation.
    - structure_density: Density multiplier for road structures.
    - status_message: Latest UI status line shown at the bottom of the control panel.
    - preview_visible: True only while the preview sub-window is open.
    - preview_map: Latest generated map payload for rendering and controller prep.
    - controllers: Prepared VN + PN metadata from current config.
    - latest_feature_vector: Example VN vector from the primary preview vehicle.
    """

    def __init__(
        self,
        screen_rect: pygame.Rect,
        font_path: Path | None = None,
        run_ctx: Any = None,
        ui_offsets: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Use:
        Create setup screen controls, layout rectangles, and initial deterministic
        preview payload while keeping the preview window hidden by default.

        Inputs:
        - screen_rect: Current display bounds.
        - font_path: Optional custom font path.
        - run_ctx: Optional startup run context.

        Output:
        None. Instance state is initialised in place.
        """
        self.screen_rect = screen_rect
        self.font_path = font_path
        self.run_ctx = run_ctx
        self.ui_offsets = ui_offsets or {}

        scenario_cfg = run_ctx.config.get("scenario", {}) if run_ctx is not None else {}
        self.seed = int(run_ctx.seed) if run_ctx is not None else random.randint(0, 2**31 - 1)
        self.phase = clamp_phase(int(scenario_cfg.get("phase", 0)) + 1)
        self.level_index = 0
        self.road_density = float(scenario_cfg.get("preview_road_density", 0.72))
        self.structure_density = float(scenario_cfg.get("preview_structure_density", 0.62))

        self.status_message = "Setup ready. Click GEN PREVIEW to open map preview."
        self.preview_visible = False
        self.latest_feature_vector: List[float] = []

        # Asset cache for image-based preview rendering.
        self.road_h_img: Optional[pygame.Surface] = None
        self.road_v_img: Optional[pygame.Surface] = None
        self.car_img: Optional[pygame.Surface] = None
        self._scaled_cache: Dict[tuple[str, int], pygame.Surface] = {}
        self._load_preview_assets()

        # Build initial deterministic payload without opening the preview window.
        self.preview_map = generate_phase_map(
            self.seed,
            self.phase,
            self.level_index,
            road_density=self.road_density,
            structure_density=self.structure_density,
        )
        self.controllers: PreparedControllers = self._prepare_controllers()
        self._refresh_feature_preview()
        self._sync_run_context()
        self._print_preview_metrics("init")

        self._build_layout()

    def _offset_int(self, name: str, default: int) -> int:
        """
        Use:
        Read one integer UI offset/size value from setup offsets.

        Inputs:
        - name: Offset key in `ui_offsets`.
        - default: Fallback integer value.

        Output:
        Integer value for the requested key.
        """
        value = self.ui_offsets.get(name, default)
        if isinstance(value, (int, float)):
            return int(value)
        return int(default)

    def _offset_xy(self, name: str, default: tuple[int, int]) -> tuple[int, int]:
        """
        Use:
        Read one 2D UI offset tuple from setup offsets.

        Inputs:
        - name: Offset key in `ui_offsets`.
        - default: Fallback `(x, y)` tuple.

        Output:
        Tuple `(x, y)` offset for the requested key.
        """
        value = self.ui_offsets.get(name, default)
        if isinstance(value, tuple) and len(value) == 2:
            return int(value[0]), int(value[1])
        return default

    def _text_pos(self, name: str, base: tuple[int, int]) -> tuple[int, int]:
        """
        Use:
        Shift a text anchor position by configured UI offsets.

        Inputs:
        - name: Offset key in `ui_offsets`.
        - base: Base `(x, y)` anchor.

        Output:
        Offset-adjusted `(x, y)` position.
        """
        dx, dy = self._offset_xy(name, (0, 0))
        return base[0] + dx, base[1] + dy

    def _build_layout(self) -> None:
        """
        Use:
        Build all setup panel and button rectangles with explicit spacing so controls
        do not overlap and remain readable.

        Inputs:
        - None. Uses `self.screen_rect`.

        Output:
        None. Rectangles are stored on the instance.
        """
        back_x, back_y = self._offset_xy("back_pos", (34, 28))
        back_w, back_h = self._offset_xy("back_size", (96, 46))
        self.back_button = pygame.Rect(back_x, back_y, back_w, back_h)

        margin = self._offset_int("panel_margin", 34)
        top = self._offset_int("panel_top", 108)
        bottom_margin = self._offset_int("panel_bottom_margin", 24)
        panel_height = self.screen_rect.height - top - bottom_margin

        left_panel_w = self._offset_int("left_panel_width", 392)
        panel_gap = self._offset_int("panel_gap", 20)
        left_dx, left_dy = self._offset_xy("left_panel_offset", (0, 0))
        right_dx, right_dy = self._offset_xy("right_panel_offset", (0, 0))

        self.left_panel = pygame.Rect(margin + left_dx, top + left_dy, left_panel_w, panel_height)
        right_x = self.left_panel.right + panel_gap + right_dx
        right_y = top + right_dy
        right_w = self.screen_rect.width - right_x - margin
        self.right_panel = pygame.Rect(right_x, right_y, right_w, panel_height)

        cx = self.left_panel.x + self._offset_int("control_pad_x", 22)
        row_y = self.left_panel.y + self._offset_int("control_start_y", 132)
        row_gap = self._offset_int("row_gap", 78)
        row_shift = self._offset_int("row_shift_y", 0)

        seed_dx, seed_dy = self._offset_xy("seed_button_offset", (0, 0))
        pm_dx, pm_dy = self._offset_xy("pm_buttons_offset", (0, 0))
        pm_w, pm_h = self._offset_xy("pm_button_size", (50, 44))
        pm_x_gap = self._offset_int("pm_x_gap", 238)

        seed_w = self.left_panel.width - self._offset_int("seed_button_w_pad", 44)
        seed_h = self._offset_int("seed_button_h", 44)
        self.seed_button = pygame.Rect(cx + seed_dx, row_y - 36 + row_shift + seed_dy, seed_w, seed_h)

        self.phase_minus = pygame.Rect(cx + pm_dx, row_y + row_gap - 10 + row_shift + pm_dy, pm_w, pm_h)
        self.phase_plus = pygame.Rect(cx + pm_x_gap + pm_dx, row_y + row_gap - 10 + row_shift + pm_dy, pm_w, pm_h)

        self.level_minus = pygame.Rect(cx + pm_dx, row_y + row_gap * 2 - 10 + row_shift + pm_dy, pm_w, pm_h)
        self.level_plus = pygame.Rect(cx + pm_x_gap + pm_dx, row_y + row_gap * 2 - 10 + row_shift + pm_dy, pm_w, pm_h)

        self.road_minus = pygame.Rect(cx + pm_dx, row_y + row_gap * 3 - 10 + row_shift + pm_dy, pm_w, pm_h)
        self.road_plus = pygame.Rect(cx + pm_x_gap + pm_dx, row_y + row_gap * 3 - 10 + row_shift + pm_dy, pm_w, pm_h)

        self.struct_minus = pygame.Rect(cx + pm_dx, row_y + row_gap * 4 - 10 + row_shift + pm_dy, pm_w, pm_h)
        self.struct_plus = pygame.Rect(cx + pm_x_gap + pm_dx, row_y + row_gap * 4 - 10 + row_shift + pm_dy, pm_w, pm_h)

        footer_y = self.left_panel.bottom - self._offset_int("footer_from_bottom", 124)
        preview_dx, preview_dy = self._offset_xy("preview_button_offset", (0, 0))
        refresh_dx, refresh_dy = self._offset_xy("refresh_button_offset", (0, 0))
        save_dx, save_dy = self._offset_xy("save_button_offset", (0, 0))

        preview_h = self._offset_int("preview_button_h", 40)
        mini_h = self._offset_int("mini_button_h", 34)
        mini_gap = self._offset_int("mini_buttons_gap", 8)
        preview_to_mini_gap = self._offset_int("preview_to_mini_gap", 48)

        self.preview_button = pygame.Rect(
            cx + preview_dx,
            footer_y + preview_dy,
            self.left_panel.width - self._offset_int("preview_button_w_pad", 44),
            preview_h,
        )
        half_w = (self.left_panel.width - self._offset_int("mini_buttons_w_pad", 52)) // 2
        self.refresh_button = pygame.Rect(
            cx + refresh_dx,
            footer_y + preview_to_mini_gap + refresh_dy,
            half_w,
            mini_h,
        )
        self.save_button = pygame.Rect(
            self.refresh_button.right + mini_gap + save_dx,
            footer_y + preview_to_mini_gap + save_dy,
            half_w,
            mini_h,
        )

        # Preview appears as a sub-window only when `preview_visible` is True.
        preview_max_w = self._offset_int("preview_window_w", 860)
        preview_max_h = self._offset_int("preview_window_h", 560)
        preview_margin_w = self._offset_int("preview_window_margin_w", 120)
        preview_margin_h = self._offset_int("preview_window_margin_h", 120)
        preview_shift_x, preview_shift_y = self._offset_xy("preview_window_offset", (0, 0))

        pw = min(preview_max_w, self.screen_rect.width - preview_margin_w)
        ph = min(preview_max_h, self.screen_rect.height - preview_margin_h)
        self.preview_window = pygame.Rect(
            self.screen_rect.centerx - pw // 2 + preview_shift_x,
            self.screen_rect.centery - ph // 2 + preview_shift_y,
            pw,
            ph,
        )
        close_w, close_h = self._offset_xy("preview_close_size", (96, 36))
        close_inset_x, close_inset_y = self._offset_xy("preview_close_inset", (22, 14))
        self.preview_close_button = pygame.Rect(
            self.preview_window.right - close_w - close_inset_x,
            self.preview_window.y + close_inset_y,
            close_w,
            close_h,
        )

    def _font(self, size: int) -> pygame.font.Font:
        """
        Use:
        Resolve custom or fallback system font for UI text.

        Inputs:
        - size: Font pixel size.

        Output:
        Pygame font object.
        """
        return pygame.font.Font(self.font_path, size) if self.font_path else pygame.font.SysFont(None, size)

    def _project_root(self) -> Path:
        """
        Use:
        Resolve project root for asset lookup and persistence writes.

        Inputs:
        - None.

        Output:
        Absolute project root path.
        """
        if self.run_ctx is not None:
            return Path(self.run_ctx.base_dir)
        return Path(__file__).resolve().parents[2]

    def _load_image(self, candidates: List[Path]) -> Optional[pygame.Surface]:
        """
        Use:
        Load first existing image path from a candidate list.

        Inputs:
        - candidates: Ordered list of candidate filesystem paths.

        Output:
        Loaded pygame surface, or None when all candidates are missing.
        """
        for path in candidates:
            if path.exists():
                try:
                    return pygame.image.load(path.as_posix()).convert_alpha()
                except Exception:
                    return None
        return None

    def _load_preview_assets(self) -> None:
        """
        Use:
        Load road/car sprites so preview rendering uses project image assets for
        primary road types and vehicles.

        Inputs:
        - None.

        Output:
        None. Loaded surfaces are stored on the instance.
        """
        root = self._project_root()
        self.road_h_img = self._load_image(
            [
                root / "assets" / "road_imgs" / "road_horizontal.png",
                root / "assets" / "road_horizontal.png",
            ]
        )
        self.road_v_img = self._load_image(
            [
                root / "assets" / "road_imgs" / "road_vertical.png",
                root / "assets" / "road_vertical.png",
            ]
        )
        self.car_img = self._load_image(
            [
                root / "assets" / "road_imgs" / "vehicle_imgs" / "car.png",
                root / "assets" / "vehicle_imgs" / "car.png",
            ]
        )

    def _scaled_asset(self, key: str, image: Optional[pygame.Surface], tile: int) -> Optional[pygame.Surface]:
        """
        Use:
        Return a cached, tile-sized version of a source sprite.

        Inputs:
        - key: Cache key prefix.
        - image: Source image.
        - tile: Target tile size in pixels.

        Output:
        Scaled surface or None if source image is unavailable.
        """
        if image is None:
            return None
        cache_key = (key, tile)
        cached = self._scaled_cache.get(cache_key)
        if cached is not None:
            return cached
        scaled = pygame.transform.smoothscale(image, (tile, tile))
        self._scaled_cache[cache_key] = scaled
        return scaled

    def _prepare_controllers(self) -> PreparedControllers:
        """
        Use:
        Prepare VN + PN metadata from current runtime config.

        Inputs:
        - None. Uses `self.run_ctx` when available.

        Output:
        Prepared controller metadata object.
        """
        base_dir = self._project_root()
        cfg = self.run_ctx.config if self.run_ctx is not None else {}
        return prepare_vn_pn(cfg, base_dir)

    def _refresh_feature_preview(self) -> None:
        """
        Use:
        Build one representative VN feature vector from current preview vehicles.

        Inputs:
        - None.

        Output:
        None. Updates `self.latest_feature_vector`.
        """
        self.latest_feature_vector = []
        if not self.preview_map.vehicles:
            return
        primary = self.preview_map.vehicles[0]
        nearby = [vehicle.spawn for vehicle in self.preview_map.vehicles[1:3]]
        self.latest_feature_vector = build_vn_feature_vector(primary, self.preview_map, nearby)

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
        Draw a rounded gradient button with centered label text.

        Inputs:
        - screen: Destination surface.
        - rect: Button rectangle.
        - text: Label string.
        - text_size: Font size for button label.
        - border_color: RGB outline color.

        Output:
        None. Draws directly on `screen`.
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

    def _set_phase(self, phase: int) -> None:
        """
        Use:
        Update selected phase and clamp level index to available complexity levels.

        Inputs:
        - phase: Candidate phase number.

        Output:
        None. Triggers preview regeneration and config sync.
        """
        self.phase = clamp_phase(phase)
        self.level_index = max(0, min(self.level_index, map_level_count(self.phase) - 1))
        self._rebuild_preview("Phase updated.")

    def _set_level(self, level: int) -> None:
        """
        Use:
        Update selected map level index within the active phase range.

        Inputs:
        - level: Candidate zero-based level index.

        Output:
        None. Triggers preview regeneration and config sync.
        """
        self.level_index = max(0, min(level, map_level_count(self.phase) - 1))
        self._rebuild_preview("Complexity updated.")

    def _set_road_density(self, delta: float) -> None:
        """
        Use:
        Apply road density adjustment and rebuild preview data.

        Inputs:
        - delta: Signed adjustment value.

        Output:
        None.
        """
        self.road_density = max(0.35, min(1.35, round(self.road_density + delta, 2)))
        self._rebuild_preview("Road density updated.")

    def _set_structure_density(self, delta: float) -> None:
        """
        Use:
        Apply structure density adjustment and rebuild preview data.

        Inputs:
        - delta: Signed adjustment value.

        Output:
        None.
        """
        self.structure_density = max(0.20, min(1.50, round(self.structure_density + delta, 2)))
        self._rebuild_preview("Structure density updated.")

    def _sync_run_context(self) -> None:
        """
        Use:
        Synchronize setup selections back into shared runtime context and write the
        rolling runtime config snapshot.

        Inputs:
        - None. Uses current setup state.

        Output:
        None. Mutates `run_ctx` when available.
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

    def _print_preview_metrics(self, reason: str) -> None:
        """
        Use:
        Print deterministic preview metrics to terminal for quick development checks.

        Inputs:
        - reason: Trigger label for this metrics emission.

        Output:
        None. Writes one formatted log line to stdout.
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

    def _rebuild_preview(self, status: Optional[str] = None) -> None:
        """
        Use:
        Regenerate preview map, recompute controller metadata, and resync runtime
        config using current setup values.

        Inputs:
        - status: Optional status string shown in UI.

        Output:
        None.
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

    def _road_orientation(self, x_pos: int, y_pos: int) -> str:
        """
        Use:
        Determine whether a road tile should prefer horizontal or vertical sprite
        rendering based on neighbouring road connectivity.

        Inputs:
        - x_pos: Tile x coordinate.
        - y_pos: Tile y coordinate.

        Output:
        Orientation token: `horizontal`, `vertical`, or `mixed`.
        """
        roads = self.preview_map.roads
        left = (x_pos - 1, y_pos) in roads
        right = (x_pos + 1, y_pos) in roads
        up = (x_pos, y_pos - 1) in roads
        down = (x_pos, y_pos + 1) in roads

        has_h = left or right
        has_v = up or down
        if has_h and not has_v:
            return "horizontal"
        if has_v and not has_h:
            return "vertical"
        return "mixed"

    def _draw_preview_map(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        """
        Use:
        Render generated preview grid using image sprites for primary road segments
        and vehicles, with color overlays for structures.

        Inputs:
        - screen: Target surface.
        - rect: Available map-render rectangle within preview window.

        Output:
        None.
        """
        preview = self.preview_map
        pygame.draw.rect(screen, BLOCKED, rect, border_radius=8)
        pygame.draw.rect(screen, PANEL_BORDER, rect, 1, border_radius=8)

        if preview.width <= 0 or preview.height <= 0:
            return

        tile = max(6, min(rect.width // preview.width, rect.height // preview.height))
        grid_width = tile * preview.width
        grid_height = tile * preview.height
        ox = rect.x + (rect.width - grid_width) // 2
        oy = rect.y + (rect.height - grid_height) // 2

        road_h = self._scaled_asset("road_h", self.road_h_img, tile)
        road_v = self._scaled_asset("road_v", self.road_v_img, tile)

        default_color = ROAD_CONTINUOUS if preview.continuous else ROAD_DISCRETE

        for y_val in range(preview.height):
            for x_val in range(preview.width):
                cell = pygame.Rect(ox + x_val * tile, oy + y_val * tile, tile, tile)
                point = (x_val, y_val)

                if point not in preview.roads:
                    pygame.draw.rect(screen, BLOCKED, cell)
                    continue

                orientation = self._road_orientation(x_val, y_val)
                node_type = preview.node_types.get(point, "")

                if orientation == "horizontal" and road_h is not None:
                    screen.blit(road_h, cell.topleft)
                elif orientation == "vertical" and road_v is not None:
                    screen.blit(road_v, cell.topleft)
                else:
                    pygame.draw.rect(screen, default_color, cell)

                # Overlay explicit structure types so users can interpret generated roads.
                if node_type == "road_two_lane":
                    pygame.draw.rect(screen, ROAD_TWO_LANE, cell, 2)
                elif node_type == "junction_turn_one_lane":
                    pygame.draw.rect(screen, JUNCTION_ONE_LANE, cell, 2)
                elif node_type == "junction_turn_two_lane":
                    pygame.draw.rect(screen, JUNCTION_TWO_LANE, cell, 2)
                elif node_type == "junction_t":
                    pygame.draw.rect(screen, JUNCTION_T, cell, 2)
                elif node_type == "junction_cross":
                    pygame.draw.rect(screen, JUNCTION_CROSS, cell, 2)
                elif node_type == "roundabout":
                    pygame.draw.rect(screen, ROUNDABOUT_OUTER, cell, 2)
                    pygame.draw.circle(screen, ROUNDABOUT_CENTER, cell.center, max(2, int(tile * 0.26)))

        # Vehicle spawn/destination overlays.
        car_sprite = self._scaled_asset("car", self.car_img, max(8, int(tile * 0.82)))
        for vehicle in preview.vehicles:
            spawn_rect = pygame.Rect(ox + vehicle.spawn[0] * tile, oy + vehicle.spawn[1] * tile, tile, tile)
            dest_rect = pygame.Rect(ox + vehicle.destination[0] * tile, oy + vehicle.destination[1] * tile, tile, tile)

            if car_sprite is not None:
                car_rect = car_sprite.get_rect(center=spawn_rect.center)
                screen.blit(car_sprite, car_rect)
            else:
                pygame.draw.circle(screen, (122, 221, 109), spawn_rect.center, max(2, tile // 4))

            pygame.draw.rect(screen, DEST_COLOR, dest_rect.inflate(-max(2, tile // 3), -max(2, tile // 3)), 2)

    def handle_events(self, events: List[pygame.event.Event]) -> str:
        """
        Use:
        Process setup interactions and return next GUI state.

        Inputs:
        - events: Pygame event list for current frame.

        Output:
        Next state token (`SETUP` or `MENU`).
        """
        for event in events:
            if event.type != pygame.MOUSEBUTTONUP or event.button != 1:
                continue

            if self.preview_visible and self.preview_close_button.collidepoint(event.pos):
                self.preview_visible = False
                self.status_message = "Preview closed."
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
                self._set_road_density(0.05)
                continue

            if self.struct_minus.collidepoint(event.pos):
                self._set_structure_density(-0.05)
                continue
            if self.struct_plus.collidepoint(event.pos):
                self._set_structure_density(0.05)
                continue

            if self.preview_button.collidepoint(event.pos):
                self._rebuild_preview("Preview generated.")
                self.preview_visible = True
                continue
            if self.refresh_button.collidepoint(event.pos):
                self._rebuild_preview("Preview data refreshed.")
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

    def _draw_control_row(self, screen: pygame.Surface, label: str, value: str, row_index: int) -> None:
        """
        Use:
        Draw one labeled control row with centered value text.

        Inputs:
        - screen: Destination surface.
        - label: Control label.
        - value: Current value string.
        - row_index: Zero-based row position.

        Output:
        None.
        """
        body = self._font(22)
        value_font = self._font(30)

        y_base = self.left_panel.y + 210 + (row_index * 78)
        label_surface = body.render(label, True, ACCENT)
        value_surface = value_font.render(value, True, FG)
        label_pos = self._text_pos("control_row_label_offset", (self.left_panel.x + 22, y_base - 26))
        value_center = self._text_pos("control_row_value_offset", (self.left_panel.x + 220, y_base - 2))
        screen.blit(label_surface, label_pos)
        value_rect = value_surface.get_rect(center=value_center)
        screen.blit(value_surface, value_rect)

    def draw(self, screen: pygame.Surface) -> None:
        """
        Use:
        Render setup controls, summary details, and optionally the preview sub-window
        when the user requests `GEN PREVIEW`.

        Inputs:
        - screen: Target Pygame surface.

        Output:
        None.
        """
        screen.fill(BG)

        title_font = self._font(96)
        body_font = self._font(23)
        info_font = self._font(22)
        status_font = self._font(20)

        title = title_font.render("SETUP", True, FG)
        title_midtop = self._text_pos("title_offset", (self.screen_rect.centerx, 8))
        screen.blit(title, title.get_rect(midtop=title_midtop))

        pygame.draw.rect(screen, PANEL_BG, self.left_panel, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, self.left_panel, 2, border_radius=12)

        pygame.draw.rect(screen, PANEL_BG, self.right_panel, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, self.right_panel, 2, border_radius=12)

        self._draw_gradient_button(screen, self.back_button, "BACK", 22)
        self._draw_gradient_button(screen, self.seed_button, "REGENERATE SEED", 22)

        seed_label = body_font.render("SEED", True, ACCENT)
        seed_value = self._font(28).render(str(self.seed), True, FG)
        screen.blit(seed_label, self._text_pos("seed_label_offset", (self.left_panel.x + 22, self.left_panel.y + 78)))
        screen.blit(seed_value, self._text_pos("seed_value_offset", (self.left_panel.x + 22, self.left_panel.y + 102)))

        self._draw_control_row(screen, "PHASE", str(self.phase), 0)
        self._draw_control_row(screen, "MAP LEVEL", f"{self.level_index + 1}/{map_level_count(self.phase)}", 1)
        self._draw_control_row(screen, "ROAD DENSITY", f"{self.road_density:.2f}", 2)
        self._draw_control_row(screen, "STRUCTURE DENSITY", f"{self.structure_density:.2f}", 3)

        self._draw_gradient_button(screen, self.phase_minus, "-", 34)
        self._draw_gradient_button(screen, self.phase_plus, "+", 34)
        self._draw_gradient_button(screen, self.level_minus, "-", 34)
        self._draw_gradient_button(screen, self.level_plus, "+", 34)
        self._draw_gradient_button(screen, self.road_minus, "-", 34)
        self._draw_gradient_button(screen, self.road_plus, "+", 34)
        self._draw_gradient_button(screen, self.struct_minus, "-", 34)
        self._draw_gradient_button(screen, self.struct_plus, "+", 34)

        self._draw_gradient_button(screen, self.preview_button, "GEN PREVIEW", 22)
        self._draw_gradient_button(screen, self.refresh_button, "REFRESH", 18)
        self._draw_gradient_button(screen, self.save_button, "SAVE LOG", 18)

        # Right panel summary only (preview is a dedicated sub-window).
        info_x, info_y = self._text_pos("right_info_offset", (self.right_panel.x + 20, self.right_panel.y + 24))
        info_gap = self._offset_int("right_info_gap", 34)
        phase_name = phase_spec(self.phase).name
        lines = [
            f"Phase {self.phase}: {phase_name}",
            f"Mode: {'Continuous' if self.preview_map.continuous else 'Discrete'}",
            f"Map size: {self.preview_map.width}x{self.preview_map.height}",
            f"Preview vehicles: {len(self.preview_map.vehicles)}",
            f"VN input size: {self.controllers.vn.input_size}",
            f"PN action size: {self.controllers.pn.action_size}",
            "Map preview opens in a sub-window via GEN PREVIEW.",
            "Preview metrics are printed in terminal for logging.",
        ]
        for line in lines:
            text = info_font.render(line, True, FG)
            screen.blit(text, (info_x, info_y))
            info_y += info_gap

        status = status_font.render(self.status_message, True, ACCENT)
        status_pos = self._text_pos("status_offset", (self.left_panel.x + 18, self.left_panel.bottom - 34))
        screen.blit(status, status_pos)

        if self.preview_visible:
            overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            screen.blit(overlay, (0, 0))

            pygame.draw.rect(screen, PANEL_BG, self.preview_window, border_radius=14)
            pygame.draw.rect(screen, PANEL_BORDER, self.preview_window, 2, border_radius=14)

            head = self._font(30).render("MAP PREVIEW", True, FG)
            preview_head_pos = self._text_pos(
                "preview_header_offset",
                (self.preview_window.x + 24, self.preview_window.y + 16),
            )
            screen.blit(head, preview_head_pos)
            mode = self._font(20).render(
                f"Seed {self.seed}  |  Phase {self.phase}  |  Level {self.level_index + 1}",
                True,
                ACCENT,
            )
            preview_mode_pos = self._text_pos(
                "preview_subheader_offset",
                (self.preview_window.x + 24, self.preview_window.y + 50),
            )
            screen.blit(mode, preview_mode_pos)

            self._draw_gradient_button(screen, self.preview_close_button, "CLOSE", 18)

            map_rect = self.preview_window.inflate(-40, -96)
            map_rect.y += 28
            map_rect.height -= 20
            self._draw_preview_map(screen, map_rect)
