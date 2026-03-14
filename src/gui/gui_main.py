"""
Main Pygame entry point for the Traffic Control System (TCS) GUI.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Tuple

# Allow direct execution:
#   python src/gui/gui_main.py
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pygame
from src.gui.options_screen import OptionsScreen
from src.gui.setup_screen import SetupScreen
from src.gui.train_screen import TrainScreen
from src.gui.ui_offsets import UI_OFFSETS
from src.utils.run_init import RunContext, init_run, start_train_run


# --------------------------------------------------------------------------- #
# Core constants                                                              #
# --------------------------------------------------------------------------- #

FPS = 60
SCREEN_SIZE = (1200, 700)

# Shared colour palette.
BG = (53, 62, 67)
FG = (235, 235, 235)
ACCENT = (180, 180, 180)
HILITE = (255, 215, 0)

# Larger baseline typography for readability.
UI_TEXT_SIZES: Dict[str, int] = {
    "header": 110,
    "button": 36,
    "button_primary": 44,
    "button_small": 24,
    "screen_title": 40,
    "screen_hint": 26,
}

# Centralised border colours so buttons remain visually distinct.
MENU_BUTTON_COLOURS: Dict[str, tuple[int, int, int]] = {
    "TRAIN": (0, 71, 171),
    "EVALUATE": (128, 0, 128),
    "DEMO": (178, 34, 34),
    "REPLAYS": (85, 130, 20),
    "SETUP": (60, 60, 60),
    "OPTIONS": (0, 51, 102),
    "CONTROLS": (0, 139, 139),
}

# --------------------------------------------------------------------------- #
# Rendering helpers                                                           #
# --------------------------------------------------------------------------- #

def _vertical_gradient(
    size: tuple[int, int],
    top_colour: tuple[int, int, int],
    bottom_colour: tuple[int, int, int],
) -> pygame.Surface:
    """
    Build a vertical gradient surface.

    Args:
        size: Target (width, height) in pixels.
        top_colour: RGB colour at y=0.
        bottom_colour: RGB colour at y=height-1.

    Returns:
        Surface containing a top-to-bottom colour mix.
    """
    width, height = size
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    for y_pos in range(height):
        blend = y_pos / max(1, height - 1)
        red = int(top_colour[0] * (1 - blend) + bottom_colour[0] * blend)
        green = int(top_colour[1] * (1 - blend) + bottom_colour[1] * blend)
        blue = int(top_colour[2] * (1 - blend) + bottom_colour[2] * blend)
        pygame.draw.line(surface, (red, green, blue), (0, y_pos), (width, y_pos))
    return surface


def draw_text_centre(
    surface: pygame.Surface,
    font_path: Path | None,
    text: str,
    centre: tuple[int, int],
    font_size: int,
    colour: tuple[int, int, int],
) -> None:
    """
    Render single-line text centred at the given point.
    """
    font = pygame.font.Font(font_path, font_size) if font_path else pygame.font.SysFont(None, font_size)
    rendered = font.render(text, True, colour)
    rect = rendered.get_rect(center=centre)
    surface.blit(rendered, rect)


# --------------------------------------------------------------------------- #
# Widgets                                                                     #
# --------------------------------------------------------------------------- #

class Button:
    """
    Reusable clickable button with centred label rendering.
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        text: str,
        text_size: int,
        bordercolour: tuple[int, int, int] = HILITE,
        textcolour: tuple[int, int, int] = FG,
        thickness: int = 2,
        font_path: Path | None = None,
    ) -> None:
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.text_size = text_size
        self.textcolour = textcolour
        self.bordercolour = bordercolour
        self.thickness = thickness
        self.font_path = font_path

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Return True for left click release events inside the button rect.
        """
        return (
            event.type == pygame.MOUSEBUTTONUP
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        )

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw gradient fill, rounded border, and centred label.
        """
        gradient = _vertical_gradient((self.rect.width, self.rect.height), (90, 95, 100), (50, 52, 55))

        radius = 10
        mask = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=radius)
        gradient.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        screen.blit(gradient, self.rect.topleft)

        pygame.draw.rect(screen, self.bordercolour, self.rect, self.thickness, border_radius=radius)

        font = pygame.font.Font(self.font_path, self.text_size) if self.font_path else pygame.font.SysFont(None, self.text_size)
        text = font.render(self.text, True, self.textcolour)
        text_rect = text.get_rect(center=self.rect.center)
        shadow = font.render(self.text, True, (0, 0, 0))
        screen.blit(shadow, text_rect.move(1, 1))
        screen.blit(text, text_rect)


class MainMenu:
    """
    Main menu screen.
    """

    def __init__(self, screen_rect: pygame.Rect, font_path: Path | None = None) -> None:
        self.screen_rect = screen_rect
        self.font_path = font_path
        self.bg_colour = BG
        self.line_colour = FG
        self.header_text = "MENU"
        self.header_rect, self.buttons = self._build_buttons()

    def _build_buttons(self) -> tuple[pygame.Rect, dict[str, Button]]:
        """
        Create header placement and all menu buttons from `UI_OFFSETS`.
        """
        offsets = UI_OFFSETS.get("menu", {})
        centre_x = self.screen_rect.centerx
        centre_y = self.screen_rect.centery

        btn_w, btn_h = offsets.get("button_size", (300, 108))
        gap_y = int(offsets.get("button_gap", 16))
        top_y = centre_y - int(btn_h * 1.75)

        header_w, header_h = offsets.get("header_size", (560, 170))
        hx_off, hy_off = offsets.get("header", (0, 0))
        header_rect = pygame.Rect(
            centre_x - header_w // 2 + hx_off,
            top_y - header_h - 12 + hy_off,
            header_w,
            header_h,
        )

        buttons: dict[str, Button] = {}

        tx, ty = offsets.get("train", (0, 0))
        buttons["TRAIN"] = Button(
            centre_x - btn_w // 2 + tx,
            top_y + ty,
            btn_w,
            btn_h,
            "TRAIN",
            int(offsets.get("train_text_size", UI_TEXT_SIZES["button_primary"])),
            thickness=2,
            font_path=self.font_path,
            bordercolour=MENU_BUTTON_COLOURS["TRAIN"],
            textcolour=(230, 240, 255),
        )

        grid_labels = ["EVALUATE", "DEMO", "REPLAYS", "SETUP", "OPTIONS", "CONTROLS"]
        start_y = top_y + btn_h + gap_y
        grid_w = btn_w * 2 + gap_y
        left_x = centre_x - (grid_w // 2)

        for idx, key in enumerate(grid_labels):
            row = idx // 2
            col = idx % 2
            x = left_x + col * (btn_w + gap_y)
            y = start_y + row * (btn_h + gap_y)
            dx, dy = offsets.get(key.lower(), (0, 0))
            buttons[key] = Button(
                x + dx,
                y + dy,
                btn_w,
                btn_h,
                key,
                int(offsets.get("button_text_size", UI_TEXT_SIZES["button"])),
                thickness=2,
                font_path=self.font_path,
                bordercolour=MENU_BUTTON_COLOURS[key],
                textcolour=(240, 240, 240),
            )

        ex, ey = offsets.get("exit", (0, 0))
        ex_w, ex_h = offsets.get("exit_size", (96, 48))
        buttons["EXIT"] = Button(
            self.screen_rect.width - ex_w - 20 + ex,
            30 + ey,
            ex_w,
            ex_h,
            "EXIT",
            int(offsets.get("exit_text_size", UI_TEXT_SIZES["button_small"])),
            thickness=2,
            font_path=self.font_path,
        )
        return header_rect, buttons

    def handle_events(self, events: list[pygame.event.Event]) -> str:
        """
        Translate click events into a next-state token.
        """
        for event in events:
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                for name, button in self.buttons.items():
                    if button.handle_event(event):
                        if name == "EXIT":
                            return "QUIT"
                        if name == "TRAIN":
                            return "TRAIN"
                        return name
        return "MENU"

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw menu header text and menu buttons.
        """
        screen.fill(self.bg_colour)

        font_size = int(UI_OFFSETS.get("menu", {}).get("header_text_size", UI_TEXT_SIZES["header"]))
        draw_text_centre(screen, self.font_path, self.header_text, self.header_rect.center, font_size, FG)

        for button in self.buttons.values():
            button.draw(screen)


# --------------------------------------------------------------------------- #
# Main loop                                                                   #
# --------------------------------------------------------------------------- #

def run_gui() -> None:
    """
    Use:
    Run the TCS GUI state loop.

    Inputs:
    - None.

    Output:
    None. Opens the GUI window and runs until exit.
    """
    run_ctx: RunContext = init_run()
    print(f"[TCS] Startup ready (seed={run_ctx.seed}).")

    pygame.init()

    base_dir = Path(__file__).resolve().parents[2]
    font_path = base_dir / "assets" / "fonts" / "pixel_font-1.ttf"
    if not font_path.exists():
        font_path = None

    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption("TCS")
    clock = pygame.time.Clock()

    menu = MainMenu(screen.get_rect(), font_path=font_path)
    setup_screen = SetupScreen(
        screen.get_rect(),
        font_path=font_path,
        run_ctx=run_ctx,
        ui_offsets=UI_OFFSETS.get("setup", {}),
    )
    train_screen = TrainScreen(
        screen.get_rect(),
        font_path=font_path,
        run_ctx=run_ctx,
        ui_offsets=UI_OFFSETS.get("train", {}),
    )
    options_screen = OptionsScreen(
        screen.get_rect(),
        font_path=font_path,
        run_ctx=run_ctx,
        ui_offsets=UI_OFFSETS.get("options", {}),
    )
    state = "MENU"
    active_title = ""
    back_button = Button(
        40,
        30,
        96,
        48,
        "BACK",
        UI_TEXT_SIZES["button_small"],
        thickness=2,
        font_path=font_path,
    )

    running = True
    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
                if state == "MENU":
                    running = False
                else:
                    state = "MENU"

        if state == "MENU":
            next_state = menu.handle_events(events)
            menu.draw(screen)
            if next_state != "MENU":
                state = next_state
                if state == "TRAIN" and run_ctx.run_id is None:
                    run_ctx = start_train_run(run_ctx)
                    setup_screen.run_ctx = run_ctx
                    train_screen.run_ctx = run_ctx
                    options_screen.run_ctx = run_ctx
                    train_screen.seed = int(run_ctx.seed)
                    train_screen.reset_environment(initial=True)
                    print("[TCS] TRAIN logging initialised.")
                active_title = state
                if state == "QUIT":
                    running = False
        elif state == "SETUP":
            setup_screen.draw(screen)
            next_state = setup_screen.handle_events(events)
            if next_state == "MENU":
                state = "MENU"
        elif state == "OPTIONS":
            options_screen.draw(screen)
            next_state = options_screen.handle_events(events)
            if next_state == "MENU":
                state = "MENU"
        elif state == "TRAIN":
            train_screen.draw(screen)
            next_state = train_screen.handle_events(events)
            if next_state == "MENU":
                state = "MENU"
        else:
            screen.fill(BG)
            back_button.draw(screen)

            draw_text_centre(
                screen,
                font_path,
                active_title,
                (screen.get_width() // 2, screen.get_height() // 2 - 20),
                UI_TEXT_SIZES["screen_title"],
                FG,
            )
            draw_text_centre(
                screen,
                font_path,
                "Press ESC or BACK to return to MENU",
                (screen.get_width() // 2, screen.get_height() // 2 + 28),
                UI_TEXT_SIZES["screen_hint"],
                ACCENT,
            )

            for event in events:
                if back_button.handle_event(event):
                    state = "MENU"

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    run_gui()

