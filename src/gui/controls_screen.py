"""
Controls screen showing hardcoded control mappings per GUI screen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pygame


BG = (53, 62, 67)
FG = (235, 235, 235)
ACCENT = (180, 180, 180)
HILITE = (255, 215, 0)
PANEL_BG = (46, 53, 58)
PANEL_BORDER = (78, 91, 99)
BUTTON_BG = (70, 82, 95)
BUTTON_BG_ACTIVE = (90, 150, 220)


CONTROL_PAGES: List[Tuple[str, List[str]]] = [
    (
        "GLOBAL",
        [
            "ESC: Return to MENU from any screen.",
            "Mouse Left Click: Activate buttons and controls.",
            "LEFT/RIGHT Arrow (on this screen): switch control pages.",
            "Back button: Return to previous menu context.",
        ],
    ),
    (
        "MENU",
        [
            "TRAIN: Open train setup screen.",
            "EVALUATE: Open evaluation setup screen.",
            "DEMO: Open baseline demo screen.",
            "REPLAYS: Open replay browser.",
            "SETUP: Open map/setup configuration screen.",
            "OPTIONS: Open options and advanced settings.",
            "CONTROLS: Open this controls help screen.",
            "EXIT: Close application.",
        ],
    ),
    (
        "SETUP",
        [
            "SEED box: Regenerate deterministic map seed.",
            "Road Density +/-: Adjust road coverage.",
            "Structure Density +/-: Adjust junction/structure density.",
            "Phase +/-: Change curriculum phase.",
            "Map Level +/-: Change level within current phase.",
            "GEN PREVIEW: Open preview sub-window.",
            "SAVE CONFIG: Save current setup config to logs.",
            "BACK/ESC: Return to MENU.",
        ],
    ),
    (
        "TRAIN",
        [
            "PHASE +/-: Change starting phase.",
            "LEVEL +/-: Change map level.",
            "VISUALISE TRAINING: Toggle rendered/headless execution.",
            "SAVE NETWORK: Save current policy/network state.",
            "LOAD NETWORK: Open replay network loader.",
            "RESET EPISODE: Rebuild environment with current settings.",
            "BEGIN TRAINING: Enter DURING TRAINING runtime view.",
            "BACK/ESC: Return to MENU.",
        ],
    ),
    (
        "DURING TRAINING",
        [
            "PLAY/PAUSE: Pause or resume episode stepping.",
            "LEFT/RIGHT Arrow: Fine speed adjust (-/+ 0.01x).",
            "Speed +/- buttons: Coarse speed adjust (-/+ 1.0x).",
            "SPEED label: Shows current simulation speed multiplier.",
            "SAVE REPLAY: Request episode replay save on completion.",
            "SCREENSHOT: Save image to screenshots folder.",
            "BACK: Leave runtime and return to TRAIN setup.",
        ],
    ),
    (
        "DEMO",
        [
            "PHASE +/-: Choose baseline run phase.",
            "LEVEL +/-: Choose map level for baseline run.",
            "SEED: Regenerate or type deterministic seed.",
            "BEGIN: Start baseline episode run (no learning update).",
            "PLAY/PAUSE and speed controls: Same runtime controls.",
            "BACK/ESC: Return to MENU.",
        ],
    ),
    (
        "EVALUATE",
        [
            "LOAD NETWORK: Select trained network for evaluation.",
            "SEED input: Set deterministic evaluation seed.",
            "PHASE +/- and LEVEL +/-: Set scenario difficulty.",
            "EPISODES +/-: Set episode count for evaluation run.",
            "VISUALISE: Toggle rendered/headless evaluation run.",
            "BEGIN EVALUATION: Run episodes without training updates.",
            "SUMMARY TABLE: Open aggregated results (if available).",
            "BACK/ESC: Return to MENU.",
        ],
    ),
    (
        "REPLAYS",
        [
            "NETWORKS: Open network slot browser (save/load/delete/rename).",
            "EPISODES: Open episode replay slots and playback list.",
            "EXPORT RESULTS: Export selected network metrics as JSON.",
            "RUN (episode slot): Play selected episode replay.",
            "KEEP circle: Protect slot from oldest-first replacement.",
            "RESTART (player view): Restart replay from first step.",
            "BACK/ESC: Return to previous replay context or MENU.",
        ],
    ),
    (
        "OPTIONS",
        [
            "GENERAL/ADVANCED tabs: Switch options category.",
            "RIGHT ARROW page toggle: Cycle advanced option pages.",
            "+/- controls: Change numeric/boolean/device settings.",
            "Device toggle: AUTO/CPU/GPU selection (when available).",
            "PHASE reward scales: Tune per-phase training weighting.",
            "Auto delay and thresholds: Tune curriculum progression.",
            "BACK/ESC: Return to MENU.",
        ],
    ),
]


class ControlsScreen:
    """
    Multi-page controls/help screen with hardcoded controls per GUI screen.
    """

    def __init__(
        self,
        screen_rect: pygame.Rect,
        font_path: Path | None = None,
        ui_offsets: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.screen_rect = screen_rect
        self.font_path = font_path
        self.ui_offsets = ui_offsets or {}
        self.page_index = 0
        self.status_message = "Use LEFT/RIGHT or arrow buttons to switch pages."
        self._build_layout()

    def _font(self, size: int) -> pygame.font.Font:
        """
        Resolve configured font with fallback to system font.
        """
        return pygame.font.Font(self.font_path, size) if self.font_path else pygame.font.SysFont(None, size)

    def _offset_xy(self, key: str, default: tuple[int, int]) -> tuple[int, int]:
        """
        Read one `(x, y)` offset tuple from controls offsets.
        """
        value = self.ui_offsets.get(key, default)
        if isinstance(value, tuple) and len(value) == 2:
            return int(value[0]), int(value[1])
        return default

    def _offset_int(self, key: str, default: int) -> int:
        """
        Read one integer value from controls offsets.
        """
        value = self.ui_offsets.get(key, default)
        if isinstance(value, (int, float)):
            return int(value)
        return int(default)

    def _build_layout(self) -> None:
        """
        Build fixed panel and button geometry for controls page.
        """
        bx, by = self._offset_xy("back_button", (34, 28))
        bw = self._offset_int("back_button_w", 96)
        bh = self._offset_int("back_button_h", 46)
        self.back_button = pygame.Rect(bx, by, bw, bh)

        margin = self._offset_int("panel_margin", 34)
        top = self._offset_int("panel_top", 98)
        bottom_margin = self._offset_int("panel_bottom_margin", 24)
        pw = self.screen_rect.width - 2 * margin
        ph = self.screen_rect.height - top - bottom_margin
        px, py = self._offset_xy("panel", (margin, top))
        self.panel = pygame.Rect(px, py, pw, ph)

        nav_w = self._offset_int("nav_button_w", 54)
        nav_h = self._offset_int("nav_button_h", 44)
        prev_dx, prev_dy = self._offset_xy("prev_button", (0, 0))
        next_dx, next_dy = self._offset_xy("next_button", (0, 0))

        self.prev_button = pygame.Rect(self.panel.left + 18 + prev_dx, self.panel.top + 16 + prev_dy, nav_w, nav_h)
        self.next_button = pygame.Rect(self.panel.right - nav_w - 18 + next_dx, self.panel.top + 16 + next_dy, nav_w, nav_h)

    def _wrap_lines(self, text: str, font: pygame.font.Font, max_width: int) -> List[str]:
        """
        Soft-wrap one line by words to stay inside panel width.
        """
        words = text.split()
        if not words:
            return [""]
        lines: List[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if font.size(candidate)[0] <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def _advance_page(self, step: int) -> None:
        """
        Move page index with wrap-around in either direction.
        """
        page_count = len(CONTROL_PAGES)
        if page_count <= 0:
            self.page_index = 0
            return
        self.page_index = (self.page_index + step) % page_count

    def _draw_button(self, screen: pygame.Surface, rect: pygame.Rect, label: str, *, active: bool = False) -> None:
        """
        Draw one rounded rectangle control button.
        """
        pygame.draw.rect(screen, BUTTON_BG_ACTIVE if active else BUTTON_BG, rect, border_radius=8)
        pygame.draw.rect(screen, HILITE if active else PANEL_BORDER, rect, 2, border_radius=8)
        text = self._font(26).render(label, True, FG)
        screen.blit(text, text.get_rect(center=rect.center))

    def _draw_training_style_button(
        self,
        screen: pygame.Surface,
        rect: pygame.Rect,
        text: str,
        text_size: int,
        border_colour: tuple[int, int, int] = HILITE,
    ) -> None:
        """
        Draw a button using the same gradient style as TRAIN/SETUP/replay back buttons.
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

    def handle_events(self, events: List[pygame.event.Event]) -> str:
        """
        Handle controls interactions and return next state token.
        """
        for event in events:
            if event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
                return "MENU"
            if event.type == pygame.KEYUP and event.key == pygame.K_LEFT:
                self._advance_page(-1)
            if event.type == pygame.KEYUP and event.key == pygame.K_RIGHT:
                self._advance_page(1)
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                pos = event.pos
                if self.back_button.collidepoint(pos):
                    return "MENU"
                if self.prev_button.collidepoint(pos):
                    self._advance_page(-1)
                if self.next_button.collidepoint(pos):
                    self._advance_page(1)
        return "CONTROLS"

    def draw(self, screen: pygame.Surface) -> None:
        """
        Render controls page title, navigation, and current page body.
        """
        screen.fill(BG)

        title_dx, title_dy = self._offset_xy("title", (0, 0))
        header = self._font(56).render("CONTROLS", True, FG)
        screen.blit(header, header.get_rect(center=(self.screen_rect.centerx + title_dx, 56 + title_dy)))

        self._draw_training_style_button(screen, self.back_button, "BACK", 22)

        pygame.draw.rect(screen, PANEL_BG, self.panel, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, self.panel, 2, border_radius=12)

        self._draw_button(screen, self.prev_button, "<")
        self._draw_button(screen, self.next_button, ">")

        page_title, entries = CONTROL_PAGES[self.page_index]

        page_dx, page_dy = self._offset_xy("page_title", (0, 0))
        title_text = self._font(38).render(page_title, True, FG)
        screen.blit(title_text, title_text.get_rect(center=(self.panel.centerx + page_dx, self.panel.top + 38 + page_dy)))

        idx_text = self._font(22).render(
            f"Page {self.page_index + 1}/{len(CONTROL_PAGES)}",
            True,
            ACCENT,
        )
        idx_dx, idx_dy = self._offset_xy("page_index", (0, 0))
        screen.blit(idx_text, idx_text.get_rect(center=(self.panel.centerx + idx_dx, self.panel.top + 72 + idx_dy)))

        body_font = self._font(27)
        line_gap = self._offset_int("line_gap", 14)
        y = self.panel.top + self._offset_int("lines_start_y", 104)
        text_left = self.panel.left + self._offset_int("lines_left_pad", 30)
        max_width = self.panel.width - self._offset_int("lines_right_pad", 54)

        for entry in entries:
            wrapped = self._wrap_lines(entry, body_font, max_width)
            for segment in wrapped:
                rendered = body_font.render(segment, True, FG)
                screen.blit(rendered, (text_left, y))
                y += rendered.get_height() + 4
            y += line_gap

        status_dx, status_dy = self._offset_xy("status", (0, 0))
        status_font = self._font(22)
        status = status_font.render(self.status_message, True, ACCENT)
        screen.blit(
            status,
            status.get_rect(
                center=(self.panel.centerx + status_dx, self.panel.bottom - 18 + status_dy),
            ),
        )
