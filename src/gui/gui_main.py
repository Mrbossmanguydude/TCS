import pygame

# INITIALISATION
pygame.init()

FPS = 60
WIDTH, HEIGHT = 1100, 600

# FILE PATHS
FONT = "fonts\\pixel_font-1.ttf"

# COLOURS
GRAY = (61, 68, 72)

# CLASSES
class Button:
    def __init__(self, x, y, width, height, text, text_size, bordercolor=(0, 0, 0), textcolor=(0, 0, 0), thickness=5):
        self.rect = pygame.Rect(x, y, width, height)
        self.rect.topleft = (x, y)
        self.text = text
        self.text_len = len(self.text)
        self.text_size = text_size
        self.textcolor = textcolor
        self.bordercolor = bordercolor
        self.thickness = thickness
        self.clicked_ticks = 0
        self.clicked = False

    def get_clicked(self):
        #Check for a left mouse button click.
        if self.clicked_ticks >= FPS:
            mouse_pos = pygame.mouse.get_pos()
            if self.rect.collidepoint(mouse_pos):
                if pygame.mouse.get_pressed()[0]: #Index 0 specifies left mouse button.
                    self.clicked = True #Button is clicked if a collision with mouse and Rect is detected.
        else:
            self.clicked = False #Otherwise, the button is not clicked.

    def draw(self, screen):
        #Draws out the button and its border.
        draw_highlighted_rect(screen, self.rect, self.bordercolor, self.bordercolor, self.thickness, self.thickness)
        draw_text(screen, FONT, self.text, (self.rect.x + 15, self.rect.y), self.text_size, self.textcolor)

def draw_highlighted_rect(surface : pygame.surface.Surface, rect : pygame.rect.Rect, border_color : tuple, highlight_color : tuple, border_thickness : int, highlight_thickness : int):
    pygame.draw.rect(surface, border_color, rect, border_thickness)
    inner_rect = pygame.Rect(rect.left + border_thickness, rect.top + border_thickness,rect.width - 2 * border_thickness, rect.height - 2 * border_thickness)
    pygame.draw.rect(surface, highlight_color, inner_rect, highlight_thickness)

def draw_text(surface : pygame.surface.Surface, font : pygame.font.Font, text : str, pos : tuple, fontsize : int, color : tuple):
    font = pygame.font.Font(font, fontsize) # Font is reassigned as a pygame.Font obj to be blit.
    word = font.render(text, True, color)
    surface.blit(word, (pos[0], pos[1])) #word blit at right position in given font.

#-----------------------------------------------------------------------------------------------------------------------
# SUBROUTINES
def draw(screen, buttons, state):
    # SCREEN FILL CONSTANTS
    screen.fill((GRAY))

    # BUTTON HANDLING
    for button in buttons[state]:
        button.draw(screen)

#-----------------------------------------------------------------------------------------------------------------------
# HELPER OBJS + VARIABLES

state = "MENU"

buttons = {
    "MENU" : []
} # {state : [back, buttons...]}

screen = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("TCS_testing")
clock = pygame.time.Clock() 

running = True

#-----------------------------------------------------------------------------------------------------------------------
#MAIN LOOP

while running: 
    clock.tick(FPS) 

    draw(screen, buttons, state)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.update()

pygame.quit()