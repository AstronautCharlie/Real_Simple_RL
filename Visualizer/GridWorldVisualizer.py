#TODO: implement visualization of true rollouts. generation of true rollouts works. Now I just need to visualize it.

import pygame
import sys
import randomcolor
import pandas as pd
import ast
import numpy as np
from GridWorld.ActionEnums import Dir
from GridWorld.GridWorldStateClass import  GridWorldState
from GridWorld.GridWorldMDPClass import GridWorldMDP
from GridWorld.ActionEnums import Dir
from MDP.StateAbstractionClass import StateAbstraction
from MDP.AbstractMDPClass import AbstractMDP

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)
RED = (255, 0, 0)
DARK_RED = (89, 6, 16)

# Dimension variables used in display_corrupt_visualization
WIDTH = 11
HEIGHT = 11
class GridWorldVisualizer():
    def __init__(self, agent=None, screen_width=600, screen_height=600, cell_size=50, margin=1):
        #self.mdp = mdp
        #self.agent = agent
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.cell_size = cell_size
        self.margin = margin
        self.agent_size = self.cell_size - self.margin
        #TODO: write docs detailing Parameters and Return ...
        """
        This class uses Pygame to visualize a GridWorldMDP and an agent in it
        """
    '''
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
                pygame.draw.rect(screen,
                                 color,
                                 [(self.margin + self.cell_size) * (col_idx) + self.margin,
                                  (self.margin + self.cell_size) * (row_idx) + self.margin,
                                  self.cell_size,
                                  self.cell_size])
        return screen
    '''

    def display_surface(self, surface):
        """
        Display the given surface
        :param surface: a pygame surface
        """
        screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.init()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                screen.blit(surface, (0, 0))
                pygame.display.flip()

    def create_gridworld_mdp(self, mdp):
        """
        Create a surface with the given GridWorld. Creates a black background and then draws white squares on it
        such that there is a border between them.
        :param mdp: a GridWorldMDP to be visualized
        :return: screen, a pygame surface
        """
        mdp_width = mdp.get_width()
        mdp_height = mdp.get_height()

        # Fence post calculation
        window_width = self.margin + (self.margin + self.cell_size) * mdp_width
        window_height = self.margin + (self.margin + self.cell_size) * mdp_height

        # Set up surface and background
        screen = pygame.Surface((window_width, window_height))
        window = pygame.Rect(0, 0, window_width, window_height)
        walls = mdp.compute_walls()

        # Draw background
        pygame.draw.rect(screen, BLACK, window)

        # Draw grid
        for col_idx, column in enumerate(range(1, mdp_width + 1, 1)):
            for row_idx, row in enumerate(range(mdp_height, 0, -1)):
                color = WHITE
                if (column, row) in walls:
                    color = BLACK
                cell = pygame.Rect((self.margin + self.cell_size) * col_idx + self.margin,
                                   (self.margin + self.cell_size) * row_idx + self.margin,
                                   self.cell_size,
                                   self.cell_size)
                pygame.draw.rect(screen, color, cell)

        return screen

    '''
    def display_gridworld_mdp(self, mdp):
        """
        Display a generic FourRooms MDP
        :return:
        """
        screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        mdp = self.create_gridworld_mdp(mdp)

        pygame.init()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
            screen.blit(mdp, (0, 0))
            pygame.display.flip()
    '''

    # Create abstract gridworld surface from file

    def create_abstract_gridworld_mdp_from_file(self, abstraction_file, key, err_list=None):
        """
        Create a pygame surface from the given abstract mdp. Color all cells
        sharing an abstract state as the same color.
        :param mdp: abstract gridworld MDP
        :param err_file: optional list of 3-tuples where first entry is ground state, second entry is true
                        abstract state, and third entry is incorrect abstract state
        :return: surface, a pygame surface with this mdp color coded
        """
        # First draw a regular gridworld
        ground_mdp = GridWorldMDP()
        screen = self.create_gridworld_mdp(ground_mdp)
        rand_color = randomcolor.RandomColor()
        
        # Read in abstraction file and create abstract MDP from it 
        abstr_df = pd.read_csv(abstraction_file, names=['abstr_type', 'abstr_eps', 'abstr_dict'])
        abstr_str = abstr_df.loc[(abstr_df['abstr_type'] == str(key[0]))
                                  & (abstr_df['abstr_eps'] == key[1])]['abstr_dict'].values[0]
        abstr_list = ast.literal_eval(abstr_str)
        s_a = {}
        for el in abstr_list:
            ground_state = GridWorldState(el[0][0], el[0][1])
            abstr_state = el[1]
            s_a[ground_state] = abstr_state
        for k, value in s_a.items():
            print(k, value)
        mdp = AbstractMDP(ground_mdp, StateAbstraction(s_a, key[0]))

        # This will hold the abstract state to color mapping
        abs_to_color = {}

        # Unpack the err_list argument if it is given. Create dictionary of ground states to 2-tuples of
        #  (true abstr state, corrupt abstr state)
        if err_list:
            print("Got err_list argument")
            err_dict = {}
            for err in err_list:
                err_dict[err[0]] = (err[1], err[2])
            print(err_dict)

        for col_idx, col in enumerate(range(1, mdp.get_width() + 1, 1)):
            for row_idx, row in enumerate(range(mdp.get_height(), 0, -1)):
                if (col, row) not in mdp.mdp.compute_walls():
                    ground_state = GridWorldState(col, row)
                    abstr_state_class = mdp.get_abstr_from_ground(ground_state)
                    abstr_state = abstr_state_class.data
                    if (abstr_state in abs_to_color):
                        new_color = abs_to_color[abstr_state]
                    else:
                        new_color = rand_color.generate()
                        while (new_color in abs_to_color.values()):
                            new_color = rand_color.generate()
                        abs_to_color[abstr_state] = new_color
                    color = pygame.Color(new_color[0])
                    cell = pygame.Rect((self.margin + self.cell_size) * col_idx + self.margin,
                                       (self.margin + self.cell_size) * row_idx + self.margin,
                                       self.cell_size,
                                       self.cell_size)
                    pygame.draw.rect(screen, color, cell)
                    if err_list and (col, row) in list(err_dict.keys()):
                        print("Split coloring", (col, row))
                        cell = pygame.Rect((self.margin + self.cell_size) * col_idx + self.margin,
                                           (self.margin + self.cell_size) * row_idx + self.margin,
                                           self.cell_size,
                                           self.cell_size / 2)
                        true_abstr = err_dict[(col, row)][0]
                        if (true_abstr in abs_to_color):
                            new_color = abs_to_color[true_abstr]
                        else:
                            new_color = rand_color.generate()
                            while (new_color in abs_to_color.values()):
                                new_color = rand_color.generate()
                            abs_to_color[true_abstr] = new_color
                        color = pygame.Color(new_color[0])
                        pygame.draw.rect(screen, color, cell)
        return screen

    # Create abstract gridwold surface from MDP

    def create_abstract_gridworld_mdp(self, mdp, err_list=None):
        """
        Create a pygame surface from the given abstract mdp. Color all cells
        sharing an abstract state as the same color.
        :param mdp: abstract gridworld MDP
        :param err_file: optional list of 3-tuples where first entry is ground state, second entry is true
                        abstract state, and third entry is incorrect abstract state
        :return: surface, a pygame surface with this mdp color coded
        """
        # First draw a regular gridworld
        screen = self.create_gridworld_mdp(mdp)
        rand_color = randomcolor.RandomColor()

        # This will hold the abstract state to color mapping
        abs_to_color = {}

        # Unpack the err_list argument if it is given. Create dictionary of ground states to 2-tuples of
        #  (true abstr state, corrupt abstr state)
        if err_list:
            print("Got err_list argument")
            err_dict = {}
            for err in err_list:
                err_dict[err[0]] = (err[1], err[2])
            print(err_dict)

        for col_idx, col in enumerate(range(1, mdp.get_width() + 1, 1)):
            for row_idx, row in enumerate(range(mdp.get_height(), 0, -1)):
                if (col, row) not in mdp.mdp.compute_walls():
                    ground_state = GridWorldState(col, row)
                    abstr_state_class = mdp.get_abstr_from_ground(ground_state)
                    abstr_state = abstr_state_class.data
                    if (abstr_state in abs_to_color):
                        new_color = abs_to_color[abstr_state]
                    else:
                        new_color = rand_color.generate()
                        while (new_color in abs_to_color.values()):
                            new_color = rand_color.generate()
                        abs_to_color[abstr_state] = new_color
                    color = pygame.Color(new_color[0])
                    cell = pygame.Rect((self.margin + self.cell_size) * col_idx + self.margin,
                                       (self.margin + self.cell_size) * row_idx + self.margin,
                                       self.cell_size,
                                       self.cell_size)
                    pygame.draw.rect(screen, color, cell)
                    if err_list and (col, row) in list(err_dict.keys()):
                        print("Split coloring", (col, row))
                        cell = pygame.Rect((self.margin + self.cell_size) * col_idx + self.margin,
                                           (self.margin + self.cell_size) * row_idx + self.margin,
                                           self.cell_size,
                                           self.cell_size / 2)
                        true_abstr = err_dict[(col, row)][0]
                        if (true_abstr in abs_to_color):
                            new_color = abs_to_color[true_abstr]
                        else:
                            new_color = rand_color.generate()
                            while (new_color in abs_to_color.values()):
                                new_color = rand_color.generate()
                            abs_to_color[true_abstr] = new_color
                        color = pygame.Color(new_color[0])
                        pygame.draw.rect(screen, color, cell)
        return screen


    def display_abstract_gridworld_mdp(self, mdp):
        """
        Display the abstract MDP with cells color-coded by their abstract state
        :param mdp: an abstract MDP to be displayed
        """
        screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        mdp = self.create_abstract_gridworld_mdp(mdp)
        pygame.init()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                screen.blit(mdp, (0, 0))
                pygame.display.flip()

    def create_corruption_visualization(self, key, corrupt_abstr_file, err_file=None):
        """
        Visualize the corrupt abstraction indicated by the key
        :param key: tuple of (Abstr_type, abstr_epsilon, corr_type, corr_proportion, batch_num)
        :param corrupt_abstr_file: path to the file containing the corrupt abstraction
        :param err_file: optional path to the file containing the list of states that are errors
        :return: a surface showing the corrupt abstract MDP, color-coded, with error states marked by a red circle
        """
        # Open files and isolate row of interest
        corr_names = ['AbstrType', 'AbstrEpsilon', 'CorrType', 'CorrProp', 'Batch', 'S_A']
        corr_df = pd.read_csv(corrupt_abstr_file, names=corr_names)

        # Load the state abstraction file into an abstract MDP
        s_a_list = corr_df.loc[(corr_df['AbstrType'] == str(key[0]))
                                & (corr_df['AbstrEpsilon'] == key[1])
                                & (corr_df['CorrType'] == str(key[2]))
                                & (corr_df['CorrProp'] == key[3])
                                & (corr_df['Batch'] == key[4])]['S_A'].values[0]
        s_a_list = ast.literal_eval(s_a_list)
        s_a = {}
        for i in range(len(s_a_list)):
            ground_state = GridWorldState(s_a_list[i][0][0], s_a_list[i][0][1])
            abstr_state = s_a_list[i][1]
            s_a[ground_state] = abstr_state
        corr_abstr_mdp = AbstractMDP(GridWorldMDP(), StateAbstraction(s_a, key[0]))

        # Load error_file data if that argument is given
        err_arg = None
        if err_file:
            names=['AbstrType', 'AbstrEps', 'Prop', 'Batch', 'ErrorStates']
            error_df = pd.read_csv(err_file, names=names)
            error_list = ast.literal_eval(error_df.loc[(error_df['AbstrType'] == str(key[0]))
                                                       & (error_df['AbstrEps'] == key[1])
                                                       & (error_df['Prop'] == key[3])
                                                       & (error_df['Batch'] == key[4])]['ErrorStates'].values[0])

            error_states = []
            for val in error_list:
                error_states.append(((ast.literal_eval(val[0])),
                                     (ast.literal_eval(val[1])),
                                     (ast.literal_eval(val[2]))))
            err_arg = error_states

        # Create the surface for the corrupt abstract MDP
        corr_surface = self.create_abstract_gridworld_mdp(corr_abstr_mdp, err_list=err_arg)

        return corr_surface

    def draw_errors(self, surface, key, error_file):
        # Parse error list; error_states contains a list of tuples of the coordinates of error states
        error_names = ['AbstrType', 'AbstrEps', 'Prop', 'Batch', 'ErrorStates']
        error_df = pd.read_csv(error_file, names=error_names)

        error_list = ast.literal_eval(error_df.loc[(error_df['AbstrType'] == str(key[0]))
                                                   & (error_df['AbstrEps'] == key[1])
                                                   & (error_df['Prop'] == key[3])
                                                   & (error_df['Batch'] == key[4])]['ErrorStates'].values[0])

        error_states = []
        for val in error_list:
            error_states.append(((ast.literal_eval(val[0])),
                                 (ast.literal_eval(val[1])),
                                 (ast.literal_eval(val[2]))))
        # Mark errors with circles
        for error in error_states:
            col = int((self.margin + self.cell_size) * (error[0][0] - 1) + self.margin + np.floor(self.cell_size / 2))
            row = int((self.margin + self.cell_size) * (HEIGHT - error[0][1]) + self.margin + np.floor(self.cell_size / 2))
            radius = int(np.floor(self.cell_size / 3))
            pygame.draw.circle(surface, RED, (col, row), radius)

        return surface

    '''
    def display_corrupt_visualization(self, key, corrupt_abstr_file, error_file, display_errors=True):
        """
        Create the surface for the corrupt visualization and display it
        """
        screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        mdp = self.create_corruption_visualization(key, corrupt_abstr_file, error_file)
        if display_errors:
            mdp = self.create_errors(mdp, key, error_file)
        pygame.init()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                screen.blit(mdp, (0, 0))
                pygame.display.flip()
    '''
    def parse_action_string(self, action_string):
        """
        Return the direction enum associated with the string. Necessary because when we read in data from files,
        all the action enums are strings
        :param action_string:
        :return: action_enum
        """
        if action_string == 'Dir.RIGHT':
            return Dir.RIGHT
        elif action_string == 'Dir.LEFT':
            return Dir.LEFT
        elif action_string == 'Dir.UP':
            return Dir.UP
        elif action_string == 'Dir.DOWN':
            return Dir.DOWN
        else:
            raise ValueError("Cannot parse action " + action_string)

    def generate_true_abstract_rollout(self, key, policy_file, abstraction_file, agent_num):
        """
        Generate a sequence of (state, action) tuples ending in either a terminal state or in a loop. Assumes
        policy_file/abstraction_file are for true abstract MDP
        :param policy_file: file containing the learned policies
        :param abstraction_file: file containing the abstraction mappings
        :param key: (Abstr_type, abstr_epsilon, corr_type, corr_prop, batch_num) tuple
        :param agent_num: number of the agent in the ensemble, stored in policy_file
        :return: rollout, a list of (State, Action) tuples
        """
        # Create MDP
        mdp = GridWorldMDP()

        # Load policy into dictionary, identifying proper abstractions/agents based on key and agent_num
        policy_names = ['Key', 'AgentNum', 'PolicyDict']
        policy_df = pd.read_csv(policy_file, names=policy_names)
        policy_df['AbstrType'], policy_df['AbstrEps'] = policy_df['Key'].str.split(', ').str
        policy_df['AbstrType'] = policy_df['AbstrType'].map(lambda x: x.strip('(<: 1234>'))
        policy_df = policy_df.loc[policy_df['AbstrType'] != 'ground']
        policy_df['AbstrEps'] = policy_df['AbstrEps'].map(lambda x: x.strip('(<: >)'))
        #print(policy_df.dtypes)
        #print(policy_df.to_string())
        policy_string = ast.literal_eval(policy_df.loc[(policy_df['AbstrType'] == str(key[0]))
                                                       & (policy_df['AbstrEps'] == str(key[1]))
                                                       & (policy_df['AgentNum'] == agent_num)]['PolicyDict'].values[0])
        policy_dict = {}
        for policy_key in policy_string.keys():
            policy_dict[int(policy_key)] = self.parse_action_string(policy_string[policy_key])

        # Load abstraction mapping into dictionary, identifying proper abstractions/agents based on key and agent_num
        abstr_names = ['AbstrType', 'AbstrEps', 'AbstrDict']
        abstr_df = pd.read_csv(abstraction_file, names=abstr_names)
        abstr_string = abstr_df.loc[(abstr_df['AbstrType'] == str(key[0]))
                                    & (abstr_df['AbstrEps'] == key[1])]['AbstrDict'].values[0]
        abstr_string = ast.literal_eval(abstr_string)
        abstr_dict = {}
        for entry in abstr_string:
            abstr_dict[entry[0]] = entry[1]

        # Generate rollout by repeatedly getting the ground state from the MDP, converting it to an abstract state
        #  via abstr_dict, select an action based on policy_dict, and applying the action to the MDP.
        # If we ever hit a state we have seen before, break
        state = mdp.init_state
        visited_states = []
        rollout = [(state.x, state.y)]
        i = 0
        while not (state.is_terminal()) and (state not in visited_states):
            i += 1
            abstr_state = abstr_dict[(state.x, state.y)]
            action = policy_dict[abstr_state]
            next_state = mdp.transition(state, action)
            rollout.append((next_state.x, next_state.y))
            visited_states.append(state)
            state = next_state
        return rollout

    def generate_corrupt_abstract_rollout(self, key, policy_file, abstraction_file, agent_num):
        """
        Generate a sequence of (state, action) tuples ending in either a terminal state or in a loop. Assumes
        policy_file/abstraction_file are for corrupted abstract MDPs
        :param policy_file: file containing the learned policies
        :param abstraction_file: file containing the abstraction mappings
        :param key: (Abstr_type, abstr_epsilon, corr_type, corr_prop, batch_num) tuple
        :param agent_num: number of the agent in the ensemble, stored in policy_file
        :return: rollout, a list of (State, Action) tuples
        """
        # Create MDP
        mdp = GridWorldMDP()

        # Load policy into dictionary, identifying proper abstractions/agents based on key and agent_num
        policy_names = ['Key', 'AgentNum', 'PolicyDict']
        policy_df = pd.read_csv(policy_file, names=policy_names)
        policy_df['AbstrType'], policy_df['AbstrEps'], policy_df['CorrType'], policy_df['CorrProp'], policy_df['BatchNum'] = policy_df['Key'].str.split(', ').str
        policy_df['AbstrType'] = policy_df['AbstrType'].map(lambda x: x.strip('(<: 1234>'))
        policy_df['BatchNum'] = policy_df['BatchNum'].map(lambda x: x.strip(')'))
        policy_df['CorrType'] = policy_df['CorrType'].map(lambda x: x.strip('<>: 1234'))
        policy_string = ast.literal_eval(policy_df.loc[(policy_df['AbstrType'] == str(key[0]))
                                                       & (policy_df['AbstrEps'] == str(key[1]))
                                                       & (policy_df['CorrType'] == str(key[2]))
                                                       & (policy_df['CorrProp'] == str(key[3]))
                                                       & (policy_df['BatchNum'] == str(key[4]))
                                                       & (policy_df['AgentNum'] == agent_num)]['PolicyDict'].values[0])
        policy_dict = {}
        for policy_key in policy_string.keys():
            policy_dict[int(policy_key)] = self.parse_action_string(policy_string[policy_key])

        # Load abstraction mapping into dictionary, identifying proper abstractions/agents based on key and agent_num
        abstr_names = ['AbstrType', 'AbstrEps', 'CorrType', 'CorrProp', 'BatchNum', 'AbstrDict']
        abstr_df = pd.read_csv(abstraction_file, names=abstr_names)
        abstr_string = abstr_df.loc[(abstr_df['AbstrType'] == str(key[0]))
                                    & (abstr_df['AbstrEps'] == key[1])
                                    & (abstr_df['CorrType'] == str(key[2]))
                                    & (abstr_df['CorrProp'] == key[3])
                                    & (abstr_df['BatchNum'] == key[4])]['AbstrDict'].values[0]
        abstr_string = ast.literal_eval(abstr_string)
        abstr_dict = {}
        for entry in abstr_string:
            abstr_dict[entry[0]] = entry[1]

        # Generate rollout by repeatedly getting the ground state from the MDP, converting it to an abstract state
        #  via abstr_dict, select an action based on policy_dict, and applying the action to the MDP.
        # If we ever hit a state we have seen before, break
        state = mdp.init_state
        visited_states = []
        rollout = [(state.x, state.y)]
        i = 0
        while not (state.is_terminal()) and (state not in visited_states):
            i += 1
            abstr_state = abstr_dict[(state.x, state.y)]
            action = policy_dict[abstr_state]
            next_state = mdp.transition(state, action)
            rollout.append((next_state.x, next_state.y))
            visited_states.append(state)
            state = next_state
        return rollout

    def create_step_rectangle(self, state, next_state):
        """
        Create a pygame rectangle going from the middle of the cell representing State to the middle of the cell
        representing NextState
        :param state: State
        :param next_state: State
        :return: pygame Rect
        """

        left = min(state[0], next_state[0]) - 1
        top = min(HEIGHT - state[1], HEIGHT - next_state[1])
        width = abs(state[0] - next_state[0]) + 1
        height = abs(state[1] - next_state[1]) + 1

        return pygame.Rect(self.cell_size * left + (self.cell_size / 3),
                           self.cell_size * top + (self.cell_size / 3),
                           width * (self.cell_size / 3),
                           height * (self.cell_size / 3))

    def draw_gridworld_rollout(self, surface, rollout, color, agent_num):
        """
        Draw the rollout on the given surface
        :param surface: a pygame Surface
        :param rollout: a list of [state, action, state, action...]
        :return: surface with the rollout visualized
        """
        i = 0
        while i < len(rollout) - 1:
            state = rollout[i]
            next_state = rollout[i + 1]
            state_pos = (self.cell_size * (state[0] - 1) + (self.cell_size / 2) + int(3 * agent_num / 2),
                         self.cell_size * (HEIGHT - state[1]) + (self.cell_size / 2) + int(3 * agent_num / 2))
            next_state_pos = (self.cell_size * (next_state[0] - 1) + (self.cell_size / 2) + int(3 * agent_num / 2),
                         self.cell_size * (HEIGHT - next_state[1]) + (self.cell_size / 2) + int(3 * agent_num/ 2))
            pygame.draw.line(surface, color, state_pos, next_state_pos, 2)
            i += 1
        return surface

    def draw_true_ensemble_rollouts(self, surface, key, policy_file, abstraction_file, num_agents):
        """
        Generate rollouts for each agent in the ensemble on the MDP matching the 'key' argument and display them
        on the given surface. Assumes data is for a true (uncorrupted) ensemble.
        :param surface: an abstract gridworld surface
        :param key: usual key tuple
        :param policy_file: path to policy dictionary (State -> Action)
        :param abstraction_file: path to abstraction dictionary (ground state -> abstract state)
        :param num_agents: number of agents in each ensemble
        """
        random_color = randomcolor.RandomColor()
        colors_used = []
        print("Generating rollouts for key", key)
        for i in range(num_agents):
            rollout = self.generate_true_abstract_rollout(key, policy_file, abstraction_file, i)
            print(rollout)
            color = random_color.generate()
            while color in colors_used:
                color = random_color.generate()
            color = pygame.Color(color[0])
            surface = self.draw_gridworld_rollout(surface, rollout, color, i)
        return surface

    def draw_corrupt_ensemble_rollouts(self, surface, key, policy_file, abstraction_file, num_agents):
        """
        Generate rollouts for each agent in the ensemble on the MDP matching the 'key' argument and display them
        on the given surface. Assumes data is for corrupt abstractions.
        :param surface: an abstract gridworld surface
        :param key: usual key tuple
        :param policy_file: path to policy dictionary (State -> Action)
        :param abstraction_file: path to abstraction dictionary (ground state -> abstract state)
        :param num_agents: number of agents in each ensemble
        """
        random_color = randomcolor.RandomColor()
        colors_used = []
        print("Generating rollouts for key", key)
        for i in range(num_agents):
            rollout = self.generate_corrupt_abstract_rollout(key, policy_file, abstraction_file, i)
            print(rollout)
            color = random_color.generate()
            while color in colors_used:
                color = random_color.generate()
            color = pygame.Color(color[0])
            surface = self.draw_gridworld_rollout(surface, rollout, color, i)
        return surface

    '''
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
        right_arrow = pygame.image.load("Visualizer/viz_resources/arrow-icon-arrow-right.jpg")
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

        action_rect.left = ((self.margin + self.cell_size) * (state.x - 1)) + self.margin
        # since indexing for state starts on bottom left
        action_rect.top = (mdp_env_rect.height - ((self.margin + self.cell_size) * (state.y))) + self.margin
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
            mdp_and_agent = self.placeAgentOnMDP(mdp_env, agent)
            current_state, action, next_state, _ = self.agent.explore()
            action_img = self.createAction(action)
            action_img.set_alpha(15)
            mdp_and_agent_and_action = self.placeAction(action_img,current_state,mdp_and_agent)
            screen.blit(mdp_and_agent_and_action, (0, 0))
            pygame.display.flip()

    def visualizeLearnedTrajectory(self, steps, alpha_gain=10):
        """
        Run after a policy is learnt, visualize the the trajectory learned (starting at initial state) and taking
        the given number of steps
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
            mdp_and_agent = self.placeAgentOnMDP(mdp_env, agent)
            current_state, action, next_state,  = self.agent.apply_best_action()
            action_img = self.createAction(action)
            action_img.set_alpha(15)
            mdp_and_agent_and_action = self.placeAction(action_img,current_state,mdp_and_agent)
            screen.blit(mdp_and_agent_and_action, (0, 0))
            pygame.display.flip()

    def visualizeLearnedPolicy(self):
        """
        Shows best action learned at each state of the MDP
        :return:
        """
        screen = pygame.display.set_mode([self.screen_width, self.screen_height])
        mdp_env = self.createGridWorldMDP()
        WIDTH_DIM = self.mdp.get_width()
        HEIGHT_DIM = self.mdp.get_height()
        walls = self.mdp.compute_walls()
        pygame.init()
        complete_viz = False
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
            if (not complete_viz):
                for col_idx, column in enumerate(range(1, HEIGHT_DIM + 1, 1)):
                    for row_idx, row in enumerate(range(WIDTH_DIM, 0, -1)):
                        if(not (column, row) in walls):
                            state = GridWorldState(column,row)

                            best_action = self.agent.get_best_action(state)
                            action_img = self.createAction(best_action)
                            mdp_and_action = self.placeAction(action_img, state, mdp_env)
                            screen.blit(mdp_and_action, (0, 0))
                            pygame.display.flip()
            complete_viz = True
    '''












