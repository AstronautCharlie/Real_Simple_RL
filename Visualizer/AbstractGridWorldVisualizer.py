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
    # Commented out to remove agent
    '''
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
    '''
    def __init__(self, abstr_gridworld_mdp, screen_width=1000, screen_height=1000, cell_size=50, margin=1):
        """
        Initialize a visualizer for the given abstract MDP.
        :param mdp: an abstract MDP on a GridWorldMDP ground MDP
        :param screen_width:
        :param screen_height:
        :param cell_size:
        :param margin:
        """
        self.abstr_mdp = abstr_gridworld_mdp
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.cell_size = cell_size
        self.margin = margin
        self.agent_size = self.cell_size - self.margin

    def createAbstractGridWorldMDP(self):
        """
        Creates and returns a Pygame Surface from the Abstract MDP this class is initialized with.
        All cells that belong to the same abstract class are shown in the same color
        :return:
        """
        WIDTH_DIM = self.abstr_mdp.mdp.get_width()
        HEIGHT_DIM = self.abstr_mdp.mdp.get_height()
        rand_color = randomcolor.RandomColor()
        #dictionary of abstract state to colors
        abs_to_color = {}

        WINDOW_WIDTH = (self.cell_size + self.margin) * WIDTH_DIM + self.margin
        WINDOW_HEIGTH = (self.cell_size + self.margin) * HEIGHT_DIM + self.margin
        screen = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGTH))
        window = pygame.Rect(0, 0, WINDOW_HEIGTH, WINDOW_WIDTH)
        walls = self.abstr_mdp.mdp.compute_walls()
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
                    abs_state = self.abstr_mdp.get_abstr_from_ground(ground_state)
                    print("ground state", ground_state)
                    print("abstract state", abs_state)

                    if (abs_state in abs_to_color):
                        new_color = abs_to_color[abs_state]
                    else:
                        new_color = rand_color.generate()
                        while ( new_color in abs_to_color.values()):
                            new_color = rand_color.generate()
                        abs_to_color[abs_state] = new_color
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

    def createAction(self, action):
        """
        Creates and returns a pygame surface representing the given action
        :param action:
        :return:
        """
        right_arrow = pygame.image.load("Visualizer/viz_resources/right_arrow.png")
        right_arrow = pygame.transform.scale(right_arrow,(self.agent_size,self.agent_size))
        if action==Dir.RIGHT:
            return right_arrow
        up_arrow = right_arrow.copy()
        up_arrow = pygame.transform.rotate(up_arrow, 90)
        if action==Dir.UP:
            return up_arrow
        left_arrow = up_arrow.copy()
        left_arrow = pygame.transform.rotate(left_arrow, 90)
        if action == Dir.LEFT:
            return left_arrow
        down_arrow = left_arrow.copy()
        down_arrow = pygame.transform.rotate(down_arrow, 90)
        return down_arrow

    def placeAction(self,action_surface,ground_state,mdpSurface):
        """
        Returns a PyGame surface with the given action placed on the given fround state in the mdp
        :param action: the action we want to display
        :return:
        """
        action_rect = action_surface.get_rect()
        mdp_env_rect = mdpSurface.get_rect()

        action_rect.left = ((self.margin + self.cell_size) * (ground_state.x - 1)) + self.margin
        # since indexing for state starts on bottom left
        action_rect.top = (mdp_env_rect.height - ((self.margin + self.cell_size) * (ground_state.y))) + self.margin
        mdpSurface.blit(action_surface, action_rect)
        return mdpSurface

    def visualizeLearnedPolicy(self, agent):
        """
        Shows best action learned at each state of the MDP
        :return:
        """
        screen = pygame.display.set_mode([self.screen_width, self.screen_height])
        mdp_env = self.createAbstractGridWorldMDP()
        WIDTH_DIM = self.abstr_mdp.mdp.get_width()
        HEIGHT_DIM = self.abstr_mdp.mdp.get_height()
        walls = self.abstr_mdp.mdp.compute_walls()
        pygame.init()
        complete_viz = False
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
            if (not complete_viz):
                for col_idx, column in enumerate(range(1, HEIGHT_DIM + 1, 1)):
                    for row_idx, row in enumerate(range(WIDTH_DIM, 0, -1)):
                        if(not (column, row) in walls):
                            ground_state = GridWorldState(column, row)
                            abs_state = self.abstr_mdp.get_abstr_from_ground(ground_state)
                            print("abs_state", abs_state)
                            best_action = agent.get_best_action(abs_state)
                            print(best_action)
                            action_img = self.createAction(best_action)
                            mdp_and_action = self.placeAction(action_img, ground_state, mdp_env)
                            screen.blit(mdp_and_action, (0, 0))
                            pygame.display.flip()
            complete_viz = True