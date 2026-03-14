"""
Options screen with General and Advanced configuration tabs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pygame

from src.utils.hold_repeat import HoldRepeatController
from src.utils.run_init import write_rolling_config


BG = (53, 62, 67)
FG = (235, 235, 235)
ACCENT = (180, 180, 180)
HILITE = (255, 215, 0)
PANEL_BG = (46, 53, 58)
PANEL_BORDER = (78, 91, 99)
TAB_ACTIVE = (90, 150, 220)
TAB_INACTIVE = (70, 82, 95)


@dataclass(frozen=True)
class OptionControl:
    """
    Use:
    Define one configurable options-row item for the options screen.

    Attributes:
    - label: Human-readable row title.
    - section: Top-level config section (`scenario`, `options`, or `train`).
    - key: Config key inside the section.
    - kind: Supported value type (`int`, `float`, or `bool`).
    - minimum: Lower numeric bound for int/float controls.
    - maximum: Upper numeric bound for int/float controls.
    - step: Increment/decrement step for int/float controls.
    """

    label: str
    section: str
    key: str
    kind: str
    minimum: float
    maximum: float
    step: float


GENERAL_CONTROLS: Tuple[OptionControl, ...] = (
    OptionControl("FPS CAP", "options", "fps_cap", "int", 30, 165, 5),
    OptionControl("OVERLAYS", "options", "overlays", "bool", 0, 1, 1),
    OptionControl("PHASE 2 SETBACKS", "train", "phase2_collision_setbacks", "bool", 0, 1, 1),
    OptionControl("EPISODE LENGTH", "scenario", "episode_len", "int", 60, 1500, 30),
    OptionControl("ROAD DENSITY", "scenario", "preview_road_density", "float", 0.35, 1.35, 0.05),
    OptionControl("STRUCTURE DENSITY", "scenario", "preview_structure_density", "float", 0.20, 1.50, 0.05),
)

ADVANCED_CONTROLS: Tuple[OptionControl, ...] = (
    OptionControl("GAMMA", "train", "gamma", "float", 0.90, 0.999, 0.005),
    OptionControl("CLIP EPS", "train", "clip_eps", "float", 0.05, 0.40, 0.01),
    OptionControl("BATCH SIZE", "train", "batch_size", "int", 256, 8192, 256),
    OptionControl("EPISODES / LEVEL", "train", "episodes_per_level", "int", 1, 20, 1),
    OptionControl("PASS SUCCESS >=", "train", "success_threshold", "float", 0.40, 1.00, 0.02),
    OptionControl("COLLISION MAX <=", "train", "collision_threshold", "float", 0.00, 0.50, 0.01),
    OptionControl("AUTO DELAY (MS)", "train", "auto_training_delay_ms", "int", 0, 1200, 50),
)

ADVANCED_PAGE_SPLIT_LABEL = "PHASE CAR MIN"


def _build_advanced_pages() -> Tuple[Tuple[OptionControl, ...], Tuple[OptionControl, ...]]:
    """
    Use:
    Split advanced controls into two pages where page 2 starts strictly after
    `PHASE CAR MIN` when that label exists.

    Inputs:
    - None.

    Output:
    Pair of advanced control tuples `(page_1, page_2)`.
    """
    controls = list(ADVANCED_CONTROLS)
    split_index = -1
    for idx, control in enumerate(controls):
        if control.label == ADVANCED_PAGE_SPLIT_LABEL or control.key == "phase_vehicle_min":
            split_index = idx + 1
            break

    if split_index <= 0:
        split_index = max(1, len(controls) // 2)

    page_1 = tuple(controls[:split_index])
    page_2 = tuple(controls[split_index:])
    return page_1, page_2


ADVANCED_CONTROL_PAGES: Tuple[Tuple[OptionControl, ...], Tuple[OptionControl, ...]] = _build_advanced_pages()


class OptionsScreen:
    """
    Use:
    Render and manage configurable project options split into General and
    Advanced tabs, including hold-to-repeat plus/minus controls.

    Attributes:
    - screen_rect: Full display bounds.
    - font_path: Optional font path for custom typography.
    - run_ctx: Shared runtime context carrying mutable config.
    - ui_offsets: Position offset dictionary from GUI main.
    - active_tab: Current tab token (`general` or `advanced`).
    - status_message: Latest operation status shown at screen bottom.
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
        Initialise options screen UI state and configuration defaults.

        Inputs:
        - screen_rect: Display rectangle.
        - font_path: Optional custom font path.
        - run_ctx: Shared runtime context.
        - ui_offsets: Optional offsets for UI tuning.

        Output:
        None.
        """
        self.screen_rect = screen_rect
        self.font_path = font_path
        self.run_ctx = run_ctx
        self.ui_offsets = ui_offsets or {}

        self.active_tab = "general"
        self.advanced_page_index = 0
        self.status_message = "OPTIONS ready. Adjust values with +/-."

        self._hold_repeat = HoldRepeatController()

        self.row_controls_cache: List[Tuple[OptionControl, pygame.Rect, pygame.Rect]] = []

        self._ensure_defaults()
        self._build_layout()

    def _font(self, size: int) -> pygame.font.Font:
        """
        Use:
        Resolve configured font with fallback.

        Inputs:
        - size: Font size.

        Output:
        Pygame font object.
        """
        return pygame.font.Font(self.font_path, size) if self.font_path else pygame.font.SysFont(None, size)

    def _offset_xy(self, key: str, default: tuple[int, int]) -> tuple[int, int]:
        """
        Use:
        Read one `(x, y)` offset tuple from options offsets.

        Inputs:
        - key: Offset key.
        - default: Fallback value.

        Output:
        Offset tuple.
        """
        value = self.ui_offsets.get(key, default)
        if isinstance(value, tuple) and len(value) == 2:
            return int(value[0]), int(value[1])
        return default

    def _offset_int(self, key: str, default: int) -> int:
        """
        Use:
        Read one integer offset/size value from options offsets.

        Inputs:
        - key: Offset key.
        - default: Fallback value.

        Output:
        Integer value.
        """
        value = self.ui_offsets.get(key, default)
        if isinstance(value, (int, float)):
            return int(value)
        return int(default)

    def _build_layout(self) -> None:
        """
        Use:
        Build fixed rectangles for options screen panels and tab/buttons.

        Inputs:
        - None.

        Output:
        None.
        """
        bx, by = self._offset_xy("back_button", (34, 28))
        self.back_button = pygame.Rect(bx, by, 96, 46)

        left_x, left_y = self._offset_xy("left_panel_pos", (34, 108))
        left_w = self._offset_int("left_panel_w", 420)
        left_h = self.screen_rect.height - left_y - self._offset_int("panel_bottom_margin", 24)
        self.left_panel = pygame.Rect(left_x, left_y, left_w, left_h)

        right_x = self.left_panel.right + self._offset_int("panel_gap", 20)
        right_y = left_y
        right_w = self.screen_rect.width - right_x - 34
        self.right_panel = pygame.Rect(right_x, right_y, right_w, left_h)

        tab_w = self._offset_int("tab_w", 192)
        tab_h = self._offset_int("tab_h", 40)
        tx, ty = self._offset_xy("tab_pos", (self.left_panel.x + 22, self.left_panel.y + 16))
        self.tab_general = pygame.Rect(tx, ty, tab_w, tab_h)
        self.tab_advanced = pygame.Rect(tx + tab_w + 10, ty, tab_w, tab_h)

    def _ensure_defaults(self) -> None:
        """
        Use:
        Ensure all option keys used by this screen exist in runtime config.

        Inputs:
        - None.

        Output:
        None.
        """
        if self.run_ctx is None:
            return

        scenario = self.run_ctx.config.setdefault("scenario", {})
        options = self.run_ctx.config.setdefault("options", {})
        train = self.run_ctx.config.setdefault("train", {})

        defaults = {
            ("options", "fps_cap"): 60,
            ("options", "overlays"): True,
            ("scenario", "episode_len"): 300,
            ("scenario", "preview_road_density"): 0.72,
            ("scenario", "preview_structure_density"): 0.62,
            ("train", "gamma"): 0.99,
            ("train", "clip_eps"): 0.2,
            ("train", "batch_size"): 2048,
            ("train", "episodes_per_level"): 3,
            ("train", "success_threshold"): 0.72,
            ("train", "collision_threshold"): 0.14,
            ("train", "phase2_collision_setbacks"): False,
            ("train", "auto_training_delay_ms"): 240,
        }

        for (section, key), value in defaults.items():
            if section == "scenario":
                scenario.setdefault(key, value)
            elif section == "options":
                options.setdefault(key, value)
            elif section == "train":
                train.setdefault(key, value)

    def _controls_for_tab(self) -> Tuple[OptionControl, ...]:
        """
        Use:
        Return control metadata list for active tab.

        Inputs:
        - None.

        Output:
        Tuple of `OptionControl`.
        """
        if self.active_tab == "general":
            return GENERAL_CONTROLS
        return ADVANCED_CONTROL_PAGES[self.advanced_page_index]

    def _toggle_advanced_page(self) -> None:
        """
        Use:
        Switch between advanced page 1 and page 2.

        Inputs:
        - None.

        Output:
        None.
        """
        self.advanced_page_index = 1 - self.advanced_page_index
        self._hold_repeat.stop()
        self.status_message = f"Viewing ADVANCED page {self.advanced_page_index + 1}."

    def _read_value(self, control: OptionControl) -> Any:
        """
        Use:
        Read one control value from runtime config.

        Inputs:
        - control: Row metadata.

        Output:
        Raw config value.
        """
        if self.run_ctx is None:
            return 0
        section = self.run_ctx.config.setdefault(control.section, {})
        return section.get(control.key, 0)

    def _write_value(self, control: OptionControl, value: Any) -> None:
        """
        Use:
        Write one control value back into runtime config and rolling config file.

        Inputs:
        - control: Row metadata.
        - value: New value.

        Output:
        None.
        """
        if self.run_ctx is None:
            return
        section = self.run_ctx.config.setdefault(control.section, {})
        section[control.key] = value
        write_rolling_config(self.run_ctx.base_dir, self.run_ctx.config)

    def _format_value(self, control: OptionControl, value: Any) -> str:
        """
        Use:
        Render one option value for screen display.

        Inputs:
        - control: Row metadata.
        - value: Current row value.

        Output:
        Formatted text.
        """
        if control.kind == "bool":
            return "ON" if bool(value) else "OFF"
        if control.kind == "int":
            return str(int(value))
        if control.key == "gamma":
            return f"{float(value):.3f}"
        return f"{float(value):.2f}"

    def _adjust_control(self, control: OptionControl, direction: int) -> None:
        """
        Use:
        Apply one increment/decrement or toggle action to a control.

        Inputs:
        - control: Row metadata.
        - direction: -1 for minus, +1 for plus.

        Output:
        None.
        """
        current = self._read_value(control)
        if control.kind == "bool":
            next_value = not bool(current)
        elif control.kind == "int":
            raw = int(current) + int(control.step) * int(direction)
            next_value = max(int(control.minimum), min(int(control.maximum), raw))
        else:
            raw_float = float(current) + float(control.step) * float(direction)
            bounded = max(float(control.minimum), min(float(control.maximum), raw_float))
            next_value = round(bounded, 3)

        self._write_value(control, next_value)
        self.status_message = f"Updated {control.label} -> {self._format_value(control, next_value)}"

    def _draw_button(
        self,
        screen: pygame.Surface,
        rect: pygame.Rect,
        text: str,
        text_size: int,
        border_colour: tuple[int, int, int] = HILITE,
        fill_colour: tuple[int, int, int] = (68, 74, 80),
    ) -> None:
        """
        Use:
        Draw one rounded options-screen button.

        Inputs:
        - screen: Destination surface.
        - rect: Button area.
        - text: Label text.
        - text_size: Font size.
        - border_colour: Border colour.
        - fill_colour: Base fill colour.

        Output:
        None.
        """
        pygame.draw.rect(screen, fill_colour, rect, border_radius=10)
        pygame.draw.rect(screen, border_colour, rect, 2, border_radius=10)
        label = self._font(text_size).render(text, True, FG)
        screen.blit(label, label.get_rect(center=rect.center))

    def handle_events(self, events: List[pygame.event.Event]) -> str:
        """
        Use:
        Process options interactions and return next state token.

        Inputs:
        - events: Pygame events for current frame.

        Output:
        Next state token (`OPTIONS` or `MENU`).
        """
        self._hold_repeat.update()
        controls = list(self._controls_for_tab())

        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.back_button.collidepoint(event.pos):
                    self._hold_repeat.stop()
                    return "MENU"
                if self.tab_general.collidepoint(event.pos):
                    self.active_tab = "general"
                    self._hold_repeat.stop()
                    self.status_message = "Viewing GENERAL options."
                    continue
                if self.tab_advanced.collidepoint(event.pos):
                    self.active_tab = "advanced"
                    self.advanced_page_index = 0
                    self._hold_repeat.stop()
                    self.status_message = "Viewing ADVANCED page 1."
                    continue


                for row_index, (_, minus_rect, plus_rect) in enumerate(self.row_controls_cache):
                    if minus_rect.collidepoint(event.pos):
                        self._hold_repeat.begin(
                            lambda row=row_index: self._adjust_control(controls[row], -1)
                        )
                        break
                    if plus_rect.collidepoint(event.pos):
                        self._hold_repeat.begin(
                            lambda row=row_index: self._adjust_control(controls[row], +1)
                        )
                        break

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._hold_repeat.stop()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT and self.active_tab == "advanced":
                self._toggle_advanced_page()

        return "OPTIONS"

    def draw(self, screen: pygame.Surface) -> None:
        """
        Use:
        Render options tabs, controls, and active runtime values.

        Inputs:
        - screen: Target surface.

        Output:
        None.
        """
        screen.fill(BG)

        title = self._font(96).render("OPTIONS", True, FG)
        screen.blit(title, title.get_rect(midtop=(self.screen_rect.centerx, 8)))

        pygame.draw.rect(screen, PANEL_BG, self.left_panel, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, self.left_panel, 2, border_radius=12)
        pygame.draw.rect(screen, PANEL_BG, self.right_panel, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, self.right_panel, 2, border_radius=12)

        self._draw_button(screen, self.back_button, "BACK", 22)
        self._draw_button(
            screen,
            self.tab_general,
            "GENERAL",
            20,
            border_colour=TAB_ACTIVE if self.active_tab == "general" else ACCENT,
            fill_colour=TAB_ACTIVE if self.active_tab == "general" else TAB_INACTIVE,
        )
        self._draw_button(
            screen,
            self.tab_advanced,
            "ADVANCED",
            20,
            border_colour=TAB_ACTIVE if self.active_tab == "advanced" else ACCENT,
            fill_colour=TAB_ACTIVE if self.active_tab == "advanced" else TAB_INACTIVE,
        )

        controls = list(self._controls_for_tab())
        label_font = self._font(22)
        value_font = self._font(30)
        button_font_size = 30

        row_start_y = self.left_panel.y + self._offset_int("row_start_y", 88)
        row_gap = self._offset_int("row_gap", 72)
        minus_x = self.left_panel.x + self._offset_int("minus_x", 26)
        plus_x = self.left_panel.x + self.left_panel.width - self._offset_int("plus_right_pad", 76)
        button_w = self._offset_int("row_button_w", 50)
        button_h = self._offset_int("row_button_h", 42)
        label_x = self.left_panel.x + self._offset_int("row_label_x", 92)
        value_x = self.left_panel.x + self._offset_int("row_value_x", 284)

        self.row_controls_cache = []
        for row_index, control in enumerate(controls):
            y_pos = row_start_y + (row_index * row_gap)
            minus_rect = pygame.Rect(minus_x, y_pos, button_w, button_h)
            plus_rect = pygame.Rect(plus_x, y_pos, button_w, button_h)
            self.row_controls_cache.append((control, minus_rect, plus_rect))

            self._draw_button(screen, minus_rect, "-", button_font_size)
            self._draw_button(screen, plus_rect, "+", button_font_size)

            label = label_font.render(control.label, True, ACCENT)
            screen.blit(label, (label_x, y_pos - 2))

            raw_value = self._read_value(control)
            value_text = self._format_value(control, raw_value)
            value_surface = value_font.render(value_text, True, FG)
            value_rect = value_surface.get_rect(center=(value_x, y_pos + (button_h // 2)))
            screen.blit(value_surface, value_rect)

        info_font = self._font(22)
        details = [
            "GENERAL: run-time simulation + display options.",
            "ADVANCED: training progression thresholds/hyperparams.",
            "Advanced uses two pages; press RIGHT to toggle page.",
            "Hold + or - to accelerate value changes.",
            "All changes write directly to runtime_config.json.",
            f"Current tab: {self.active_tab.upper()}",
        ]
        x_info = self.right_panel.x + 20
        y_info = self.right_panel.y + 24
        for line in details:
            txt = info_font.render(line, True, FG)
            screen.blit(txt, (x_info, y_info))
            y_info += 34

        status = self._font(20).render(self.status_message, True, ACCENT)
        screen.blit(status, (self.left_panel.x + 18, self.left_panel.bottom - 34))

