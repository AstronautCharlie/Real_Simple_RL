import pygame
import sys
from GridWorld.ActionEnums import Dir

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)
RED = (255, 0, 0)
DARK_RED = (89, 6, 16)
class GridWorldVisualizer():
    def __init__(self,mdp,agent, screen_width=1000, screen_height=1000,cell_size=30,margin=1):
        self.mdp =mdp
        self.agent = agent
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.cell_size = cell_size
        self.margin = margin
        self.agent_size = self.cell_size - self.margin
        #TODO: write docs detailing Parameters and Return ...
        """
        This class uses Pygame to visualize a GridWorldMDP and an agent in it
         
        """


    def createGridWorldMDP(self):
        """
        Creates and returns a Pygame Surface from the MDP this class is initialized with.
        :return:
        """
        WIDTH_DIM = self.mdp.get_width()
        HEIGHT_DIM = self.mdp.get_height()

        WINDOW_WIDTH = (self.cell_size + self.margin) * WIDTH_DIM + self.margin
        WINDOW_HEIGTH = (self.cell_size + self.margin) * HEIGHT_DIM + self.margin
        screen = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGTH))
        window = pygame.Rect(0, 0, WINDOW_HEIGTH, WINDOW_WIDTH)
        walls = self.mdp._compute_walls()
        print(walls)
        # draw background
        pygame.draw.rect(screen,
                         BLACK,
                         window)
        # draw cells

        for col_idx, column in enumerate(range(1, HEIGHT_DIM + 1, 1)):
            for row_idx, row in enumerate(range(WIDTH_DIM, 0, -1)):
                print(row_idx, col_idx)
                print(row, column)
                color = WHITE
                if (column, row) in walls:
                    color = BLACK
                pygame.draw.rect(screen,
                                 color,
                                 [(self.margin + self.cell_size) * (col_idx) + self.margin,
                                  (self.margin + self.cell_size) * (row_idx) + self.margin,
                                  self.cell_size,
                                  self.cell_size])
        return screen

    def createGridWorldAgent(self):
        """
        Creates and returns a Pygame Surface representing the agent this class is initialized with
        :return:
        """
        screen = pygame.Surface((self.agent_size, self.agent_size))
        screen.fill(WHITE)
        pygame.draw.circle(screen, DARK_RED, (round(self.agent_size / 2), round(self.agent_size / 2)),
                           round(self.agent_size / 2))
        return screen

    def placeAgentOnMDP(self,mdpSurface,agentSurface):
        """
        Places the agent on the mdp at it's current state. Takes in a PyGame surface representing an agent and an MDP.
        Does not check if MDP already has other states/actions on it
        :return: Pygame surface with agentSurface placed on to mdpSurface based on the agent's current state
        """
        mdp_env_rect = mdpSurface.get_rect()
        agent_rect = agentSurface.get_rect()
        cur_state = self.agent.get_current_state()
        agent_rect.left = ((self.margin + self.cell_size) * (cur_state.x - 1)) + self.margin
        # since indexing for state starts on bottom left
        agent_rect.top = (mdp_env_rect.height - ((self.margin + self.cell_size) * (cur_state.y))) + self.margin
        mdpSurface.blit(agentSurface, agent_rect)
        return mdpSurface

    def displayGridWorldMDPWithAgent(self):
        """
        Displays a window with this mdp and agent at the current state of the agent
        :return:
        """

        screen = pygame.display.set_mode([self.screen_width, self.screen_height])
        mdp_env = self.createGridWorldMDP()
        agent = self.createGridWorldAgent()
        mdp_and_agent = self.placeAgentOnMDP(mdp_env, agent)
        pygame.init()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()

            screen.blit(mdp_and_agent, (0, 0))
            pygame.display.flip()

    def createAction(self,action):
        """
        Creates and returns a pygame surface representing the given action
        :param action:
        :return:
        """
        right_arrow = pygame.image.load("resources/arrow-icon-arrow-right.jpg")
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

    def placeAction(self,action_surface,state,mdpSurface):
        """
        Returns a PyGame surface with the given action placed on the given state in the mdp
        :param action: the action we want to display
        :return:
        """
        action_rect = action_surface.get_rect()
        mdp_env_rect = mdpSurface.get_rect()
        cur_state = self.agent.get_current_state()
        action_rect.left = ((self.margin + self.cell_size) * (cur_state.x - 1)) + self.margin
        # since indexing for state starts on bottom left
        action_rect.top = (mdp_env_rect.height - ((self.margin + self.cell_size) * (cur_state.y))) + self.margin
        mdpSurface.blit(action_surface, action_rect)
        return mdpSurface

    def displayGridWorldMDPWithAction(self,action):
        """
        Displays a window with this mdp and agent at the current state of the agent
        :return:
        """

        screen = pygame.display.set_mode([self.screen_width, self.screen_height])
        mdp_env = self.createGridWorldMDP()
        action_surface = self.createAction(action)
        mdp_and_action = self.placeAction(action_surface,self.agent.get_current_state,mdp_env)
        pygame.init()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()

            screen.blit(mdp_and_action, (0, 0))
            pygame.display.flip()




    def visualizeExploration(self,steps,alpha_gain=10):
        #TODO add another surface visualizing distribution of policies
        """
        Creates a visualization showing the distribution of various states visited during exploration of the mdp
        carried out by the agent based on the explore method of the mdp class. Updates the agent in this visualization
        with the knowledge gained through the exploration
        :param steps: The number of steps to take during this exploration
        param alpha_gain: initial alpha value of a state how much the alpha chanel increases by when a state is revisited
        :return:
        """
        screen = pygame.display.set_mode([self.screen_width, self.screen_height])
        mdp_env = self.createGridWorldMDP()
        agent = self.createGridWorldAgent()
        agent.set_alpha(alpha_gain)
        # see how often each state is visited



        mdp_and_agent = self.placeAgentOnMDP(mdp_env, agent)
        pygame.init()
        for i in range(steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
            current_state, action, next_state, _ = self.agent.explore()
            action_img = self.createAction(action)
            action_img.set_alpha(15)
            mdp_and_agent = self.placeAgentOnMDP(mdp_env, agent)
            mdp_and_agent_and_action = self.placeAction(action_img,current_state,mdp_and_agent)
            screen.blit(mdp_and_agent_and_action, (0, 0))
            pygame.display.flip()

    def visualizeLearnedPolicy(self):
        """
        Run after a pollicy is learnt, visualized the best policy learned by the agent so far
        :return:
        """








