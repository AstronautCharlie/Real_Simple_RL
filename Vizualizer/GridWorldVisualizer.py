import pygame
import sys


#set the size of the entire screen
# This sets the WIDTH and HEIGHT of each grid location
CELL_WIDTH = 30
CELL_HEIGHT = 30

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)
RED = (255, 0, 0)

# This sets the margin between each cell
MARGIN = 2

AGENT_WIDTH = CELL_WIDTH-MARGIN
"""
returns a pygame surface representing the instance of gridworld MDP passed into the function
"""
"""
Ideas: have this be a class, have methods that take in agents?
"""
def createGridWorldMDP(GridWorldMDP):
    WIDTH_DIM = GridWorldMDP.get_width()
    HEIGHT_DIM = GridWorldMDP.get_height()

    WINDOW_WIDTH= (CELL_WIDTH + MARGIN) * WIDTH_DIM + MARGIN
    WINDOW_HEIGTH =  (CELL_HEIGHT + MARGIN) * HEIGHT_DIM + MARGIN
    screen = pygame.Surface((WINDOW_WIDTH,WINDOW_HEIGTH))
    window = pygame.Rect(0,0,WINDOW_HEIGTH,WINDOW_WIDTH)
    walls = GridWorldMDP._compute_walls()
    print(walls)
    #draw background
    pygame.draw.rect(screen,
                     BLACK,
                     window)
    #draw cells

    for col_idx, column in enumerate(range(1, HEIGHT_DIM + 1, 1)):
        for row_idx,row in enumerate(range(WIDTH_DIM,0,-1)):
            print(row_idx,col_idx)
            print(row,column)
            color = WHITE
            if (column,row) in walls:
                color = BLACK
            pygame.draw.rect(screen,
                             color,
                             [(MARGIN + CELL_WIDTH) * (col_idx) + MARGIN,
                              (MARGIN + CELL_HEIGHT) * (row_idx) + MARGIN,
                              CELL_WIDTH,
                              CELL_HEIGHT])
    return screen
"""
returns a pygame surface representing an agent with it's coordinates according to the MDP and it's current state
"""
def createGridWorldAgent(GridWorldAgent):
    screen = pygame.Surface((AGENT_WIDTH,AGENT_WIDTH))
    screen.fill(WHITE)
    AGENT_COLOR = (255, 0, 0)
    pygame.draw.circle(screen, AGENT_COLOR, (round(AGENT_WIDTH/2),round(AGENT_WIDTH/2)), round(AGENT_WIDTH/2))
    return screen
"""
Takes in a pygame surface representing an agent,pygame surface representing an mdp and a GridWorldAgent.PLaces places the agent on it's 
current state in the mdp. Returns a pygame surface
"""
def add_agent_to_MDP(GridWorldMDPSurface,GridWorldAgentSurface,GridWorldAgent):

    mdp_env_rect = GridWorldMDPSurface.get_rect()
    agent_rect = GridWorldMDPSurface.get_rect()
    cur_state = GridWorldAgent.get_current_state()
    agent_rect.left = ((MARGIN + CELL_WIDTH) * (cur_state.x-1) )+ MARGIN
    #since indexing for state starts on bottom left
    agent_rect.top = (mdp_env_rect.height-((MARGIN + CELL_HEIGHT) * (cur_state.y))) + MARGIN
    GridWorldMDPSurface.blit(GridWorldAgentSurface,agent_rect)
    return GridWorldMDPSurface

def displayGridWorldMDPWithAgent(GridWorldMDP,GridWorldAgent):
    screen = pygame.display.set_mode([1000,1000])
    mdp_env = createGridWorldMDP(GridWorldMDP)
    agent = createGridWorldAgent(GridWorldAgent)
    mdp_and_agent = add_agent_to_MDP(mdp_env,agent,GridWorldAgent)
    pygame.init()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

        screen.blit(mdp_and_agent,(0,0))
        pygame.display.flip()
def visualizeExploration(GridWorldMDP,GridWorldAgent,steps):
    screen = pygame.display.set_mode([1000, 1000])
    mdp_env = createGridWorldMDP(GridWorldMDP)
    agent = createGridWorldAgent(GridWorldAgent)
    #see how often each state is visited
    agent.set_alpha(10)
    mdp_and_agent = add_agent_to_MDP(mdp_env, agent, GridWorldAgent)
    pygame.init()
    for i in range(steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
        current_state, action, next_state, _ = GridWorldAgent.explore()
        mdp_and_agent = add_agent_to_MDP(mdp_env, agent, GridWorldAgent)
        screen.blit(mdp_and_agent, (0, 0))
        pygame.display.flip()













