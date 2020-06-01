import pygame
import sys
import randomcolor
from GridWorld.ActionEnums import Dir
from GridWorld.GridWorldStateClass import GridWorldState

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)
RED = (255, 0, 0)
DARK_RED = (89, 6, 16)


class AbstractGridWorldVisualizer():
    def __init__(self, mdp, agent, screen_width=1000, screen_height=1000, cell_size=50, margin=1):
        self.mdp = mdp
        self.agent = agent
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.cell_size = cell_size
        self.margin = margin
        self.agent_size = self.cell_size - self.margin
        # TODO: write docs detailing Parameters and Return ...
        """
        This class uses Pygame to visualize an Abstract GridWorldMDP and an agent in it

        """
    def createAbstractGridWorldMDP(self):
        """
        Creates and returns a Pygame Surface from the Abstract MDP this class is initialized with.
        All cells that belong to the same abstract class are shown in the same color
        :return:
        """
        WIDTH_DIM = self.mdp.get_width()
        HEIGHT_DIM = self.mdp.get_height()
        rand_color = randomcolor.RandomColor()
        #dictionary of abstract state to colors
        abs_to_color = {}

        WINDOW_WIDTH = (self.cell_size + self.margin) * WIDTH_DIM + self.margin
        WINDOW_HEIGTH = (self.cell_size + self.margin) * HEIGHT_DIM + self.margin
        screen = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGTH))
        window = pygame.Rect(0, 0, WINDOW_HEIGTH, WINDOW_WIDTH)
        walls = self.mdp._compute_walls()
        # draw background
        pygame.draw.rect(screen,
                         BLACK,
                         window)
        # draw cells

        for col_idx, column in enumerate(range(1, HEIGHT_DIM + 1, 1)):
            for row_idx, row in enumerate(range(WIDTH_DIM, 0, -1)):
                color = WHITE
                if (column, row) in walls:
                    color = BLACK
                else:
                    ground_state = GridWorldState(column,row)
                    abs_state = self.mdp.get_abstr_from_ground(ground_state)
                    print(ground_state)

                    if ( abs_state in abs_to_color.keys()):
                        color = abs_to_color[abs_state]
                    else:
                        new_color = rand_color.generate()
                        while ( new_color in abs_to_color.values()):
                            new_color = rand_color.generate()
                        abs_to_color[abs_state] = new_color
                    print(new_color[0])
                    color = pygame.Color(new_color[0])
                pygame.draw.rect(screen,
                                 color,
                                 [(self.margin + self.cell_size) * (col_idx) + self.margin,
                                  (self.margin + self.cell_size) * (row_idx) + self.margin,
                                  self.cell_size,
                                  self.cell_size])
        return screen

    def displayAbstractMDP(self):

            """
            Displays a window with this mdp and agent at the current state of the agent
            :return:
            """

            screen = pygame.display.set_mode([self.screen_width, self.screen_height])
            mdp_env = self.createAbstractGridWorldMDP()
            pygame.init()
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: sys.exit()

                screen.blit(mdp_env, (0, 0))
                pygame.display.flip()