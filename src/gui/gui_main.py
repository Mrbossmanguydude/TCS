"""
Pygame GUI: This is the frontend in its entirety, taking elements from the backend
to run. It consists of many screens and sub-screens (which will be detailed within the same
file as the screens themselves) which allow for a modular, maintainable and easy to test structure.
Pygame is used primarily due to my familiarity with it, as well as for reasons mentioned in
the software requirements section of Analysis within the documentation. 
Once finished, it should align with mockups made as well as interviews with the client.
"""

from pathlib import Path
import sys

# Ensure project root is on sys.path when running as a script (e.g., python src/gui/gui_main.py).
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pygame
from src.utils.run_init import RunContext, init_run, start_train_run

# CONSTANTS -------------------------------------------------------------------
FPS = 60  # Stable frame pacing for smooth input/animation, 60 FPS is typical for most Pygame code.
SCREEN_SIZE = (1200, 700)  # Fixed canvas so layout stays predictable, given in (x, y) form.
BG = (34, 2, 1)  # Dark background for contrast.
FG = (235, 235, 235)  # Primary text/line color for readability.
ACCENT = (180, 180, 180)  # Secondary text for hints/placeholders.

UI_OFFSETS = {
    "menu": {
        "train": (0, 0),
        "evaluate": (0, 0),
        "demo": (0, 0),
        "replays": (0, 0),
        "setup": (0, 0),
        "options": (0, 0),
        "controls": (0, 0),
        "exit": (0, 0),
        "header": (0, 0),
    },
}
HILITE = (255, 215, 0)  # Highlight


###############################################################################
# HELPERS                                                                     #
###############################################################################


def _vertical_gradient(size: tuple, top_color: tuple, bottom_color: tuple) -> pygame.Surface:
    """A simple vertical gradient surface for depth without external assets; for
       even easier implementation. This is a helper procedure.
    """
    w, h = size # Parses the size tuple for the width and height.
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    for y in range(h):
        t = y / max(1, h - 1)
        r = int(top_color[0] * (1 - t) + bottom_color[0] * t)
        g = int(top_color[1] * (1 - t) + bottom_color[1] * t)
        b = int(top_color[2] * (1 - t) + bottom_color[2] * t)
        pygame.draw.line(surf, (r, g, b), (0, y), (w, y)) # Creates lines to create a gradient effect.
    return surf


def draw_text(surface, font_path, text, pos, fontsize, color):
    """Centralised text rendering; uses custom font if one exists in the path given."""
    font = pygame.font.Font(font_path, fontsize) if font_path else pygame.font.SysFont(None, fontsize) # Creates the font obj and uses a system font if custom not available.
    word = font.render(text, True, color) # Word becomes a blit-able Pygame Surface.
    surface.blit(word, pos) # Displays the word to the screen at desired position.

###############################################################################
# CLASSES                                                                     #
###############################################################################


class Button:
    def __init__(
        self,
        x,
        y,
        width,
        height,
        text,
        text_size,
        bordercolor=HILITE,
        textcolor=FG,
        thickness=2,
        font_path=None,
    ):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text  # Label visible to user.
        self.text_size = text_size
        self.textcolor = textcolor
        self.bordercolor = bordercolor
        self.thickness = thickness
        self.font_path = font_path

    def handle_event(self, event):
        """
        Return True on left click release inside the rect.
        Stateless check so the main loop stays simple.
        """
        return (
            event.type == pygame.MOUSEBUTTONUP
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        )

    def draw(self, screen):
        """
        Render a metallic button: gradient fill, soft highlight, rounded border, shadowed text.
        Visual aids help users spot interactable elements quickly, hence the extra GUI elements.
        """
        top = (90, 95, 100)  # Lighter top for sheen.
        bottom = (50, 52, 55)  # Darker base for depth.
        gradient = _vertical_gradient((self.rect.width, self.rect.height), top, bottom)
        radius = 10  # Rounded corners for softer feel.
        mask = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA) # Using a mask to make it pixel perfect.
        pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=radius)
        gradient.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        screen.blit(gradient, self.rect.topleft)

        pygame.draw.rect(screen, self.bordercolor, self.rect, self.thickness, border_radius=radius) # Draws the Button's Rect.

        shadow_offset = 1
        draw_text( # Draws the shadow.
            screen,
            self.font_path,
            self.text,
            (self.rect.x + 12 + shadow_offset, self.rect.y + 10 + shadow_offset),
            self.text_size,
            (0, 0, 0),
        )
        draw_text(screen, self.font_path, self.text, (self.rect.x + 12, self.rect.y + 10), self.text_size, self.textcolor) # Draws the Button's text.
class MainMenu:
    def __init__(self, screen_rect: pygame.Rect, font_path=None):
        self.screen_rect = screen_rect
        self.font_path = font_path
        self.bg_color = BG
        self.line_color = FG
        self.header_text = "MENU"
        self.header_rect, self.buttons = self._build_buttons()

    def _build_buttons(self):
        offsets = UI_OFFSETS.get("menu", {})
        center_x = self.screen_rect.centerx
        center_y = self.screen_rect.centery
        btn_w, btn_h = 220, 80  # Button sizing for prominence.
        gap_y = 12  # Vertical spacing to keep grid breathable.
        top_y = center_y - (btn_h * 3) # Reserve room for header/grid.

        header_w, header_h = 260, 90 # Larger header area for hierarchy.
        hx_off, hy_off = offsets.get("header", (0, 0))
        header_rect = pygame.Rect(
            center_x - header_w // 2 + hx_off,
            top_y - header_h + hy_off,
            header_w,
            header_h,
        )

        buttons = {}
        tx, ty = offsets.get("train", (0, 0))
        buttons["TRAIN"] = Button(center_x - btn_w // 2 + tx, top_y + ty, btn_w, btn_h, "TRAIN", 28, thickness=2, font_path=self.font_path)

        grid_labels = [
            ("EVALUATE", "EVALUATE"),
            ("DEMO", "DEMO"),
            ("REPLAYS", "REPLAYS"),
            ("SETUP", "SETUP"),
            ("OPTIONS", "OPTIONS"),
            ("CONTROLS", "CONTROLS"),
        ]
        start_y = top_y + btn_h + gap_y
        grid_w = btn_w * 2 + gap_y
        left_x = center_x - (grid_w // 2)
        for idx, (key, label) in enumerate(grid_labels):
            row = idx // 2
            col = idx % 2
            x = left_x + col * (btn_w + gap_y)
            y = start_y + row * (btn_h + gap_y)
            dx, dy = offsets.get(key.lower(), (0, 0))
            buttons[key] = Button(x + dx, y + dy, btn_w, btn_h, label, 22, thickness=2, font_path=self.font_path)

        ex, ey = offsets.get("exit", (0, 0))
        buttons["EXIT"] = Button(self.screen_rect.width - 90 + ex, 30 + ey, 70, 35, "EXIT", 18, thickness=2, font_path=self.font_path)
        return header_rect, buttons

    def handle_events(self, events):
        """Route mouse clicks to button names; returns next state string."""
        for event in events:
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                for name, btn in self.buttons.items():
                    if btn.rect.collidepoint(event.pos):
                        if name == "EXIT":
                            return "QUIT"
                        if name == "TRAIN":
                            return "TRAIN"
                        return name
        return "MENU"

    def draw(self, screen: pygame.Surface):
        """Render frame, header, and all menu buttons."""
        screen.fill(self.bg_color)
        margin = 40 # Frame inset to avoid edge crowding, might remove if it doesnt look visually pleasing to the client.
        frame_rect = pygame.Rect(margin, margin, self.screen_rect.width - 2 * margin, self.screen_rect.height - 2 * margin)
        pygame.draw.rect(screen, self.line_color, frame_rect, 2)
        header_color = FG
        draw_text(screen, self.font_path, self.header_text, (self.header_rect.x + 55, self.header_rect.y + 20), 68, header_color)
        pygame.draw.line(
            screen,
            header_color,
            (self.header_rect.x + 10, self.header_rect.bottom - 10),
            (self.header_rect.right - 10, self.header_rect.bottom - 10),
            2,
        )
        for _, btn in self.buttons.items():
            btn.draw(screen) # Draws all buttons.


# MAIN LOOP -------------------------------------------------------------------
def run_gui():
    """
    Main loop with interchangable states.
    """
    # Initialise run context before GUI starts.
    run_ctx: RunContext = init_run()
    print(f"[TCS] Run initialised: run_id={run_ctx.run_id} seed={run_ctx.seed} db={run_ctx.db_path}")

    pygame.init()

    base_dir = Path(__file__).resolve().parents[2] # .../TCS
    font_path = base_dir / "assets" / "fonts" / "pixel_font-1.ttf"
    if not font_path.exists():
        font_path = None # So UI still renders without assets.

    screen = pygame.display.set_mode(SCREEN_SIZE)  # Fixed size keeps layout stable during development, might change to a better resolution as well as with fullscreen capability.
    pygame.display.set_caption("TCS GUI")
    clock = pygame.time.Clock() # Frame pacing/ticks.

    menu = MainMenu(screen.get_rect(), font_path=font_path)  # Centralized menu layout/interaction.
    state = "MENU" # String based states makes it easy to fetch the state, but may change in the future.
    placeholder_title = "" # Carries the last selected state for placeholder labels.
    back_button = Button(40, 30, 70, 35, "BACK", 18, thickness=2, font_path=font_path)

    running = True
    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False # Allow window close.
            if event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
                if state == "MENU":
                    running = False #ESC quits from menu.
                else:
                    state = "MENU" # ESC returns from sub-screens to menu.

        if state == "MENU":
            next_state = menu.handle_events(events) # Delegate click handling to menu.
            menu.draw(screen) # Draw menu layout and buttons.
            if next_state != "MENU":
                state = next_state # Advance to chosen state.
                if state == "TRAIN" and run_ctx.run_id is None:
                    run_ctx = start_train_run(run_ctx)
                    print(f"[TCS] TRAIN run created: run_id={run_ctx.run_id}")
                placeholder_title = state
                if state == "QUIT":
                    running = False # Exit button terminates loop.
        else: 
            # Additionally, for these cases for different states like options, train evaluate; The main processes will be delegated to their respective files. 
            screen.fill(BG)
            back_button.draw(screen)
            draw_text(
                screen,
                font_path,
                f"{placeholder_title} (placeholder)",
                (screen.get_width() // 2 - 140, screen.get_height() // 2 - 20),
                28,
                FG
            )
            draw_text(
                screen,
                font_path,
                "Press ESC or BACK to return to menu",
                (screen.get_width() // 2 - 200, screen.get_height() // 2 + 20),
                20,
                ACCENT
            )
            for event in events:
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if back_button.handle_event(event):
                        state = "MENU" # Back returns to menu.

        pygame.display.flip() # Present the frame.
        clock.tick(FPS) # Cap FPS for consistent timing.

    pygame.quit() # End of program.


if __name__ == "__main__":
    run_gui()

'''
Probable sequence of window implementation:
1) Main Menu
2) Controls (Changed with each new window)
3) PopUp
4) Scenario Setup
5) Gen Preview + Map display
6) Train setup
7) Training
8) Baseline Demo
9) Evaluate
10) Evaluating
11) Usage Settings (General)
12) Usage Settings (Advanced)
13) Replay Browser

Backend to be implemented first, with this skeleton in mind.
'''
