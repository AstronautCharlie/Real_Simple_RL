#TODO: implement visualization of true rollouts. generation of true rollouts works. Now I just need to visualize it.

import pygame
import sys
import randomcolor
import ast
import pandas as pd
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

    def create_gridworld_mdp(self, mdp, background=BLACK):
        """
        Create a surface with the given GridWorld. Creates a black background and then draws white squares on it
        such that there is a border between them.
        :param mdp: a GridWorldMDP to be visualized
        :return: screen, a pygame surface
        """
        mdp_width = mdp.get_width()
        mdp_height = mdp.get_height()
        if background is None:
            background = BLACK
        # Fence post calculation
        window_width = self.margin + (self.margin + self.cell_size) * mdp_width
        window_height = self.margin + (self.margin + self.cell_size) * mdp_height

        # Set up surface and background
        screen = pygame.Surface((window_width, window_height))
        window = pygame.Rect(0, 0, window_width, window_height)
        walls = mdp.compute_walls()

        # Draw background
        pygame.draw.rect(screen, background, window)

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
        #for k, value in s_a.items():
            #print(k, value)
        mdp = AbstractMDP(ground_mdp, StateAbstraction(s_a, key[0]))

        # This will hold the abstract state to color mapping
        abs_to_color = {}

        # Unpack the err_list argument if it is given. Create dictionary of ground states to 2-tuples of
        #  (true abstr state, corrupt abstr state)
        if err_list:
            #print("Got err_list argument")
            err_dict = {}
            for err in err_list:
                err_dict[err[0]] = (err[1], err[2])
            #print(err_dict)

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
                        #print("Split coloring", (col, row))
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
            #print("Got err_list argument")
            err_dict = {}
            for err in err_list:
                err_dict[err[0]] = (err[1], err[2])
            #print(err_dict)

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
                        #print("Split coloring", (col, row))
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

    def create_corruption_visualization(self, key, corrupt_abstr_file, error_file=None):
        """
        Visualize the corrupt abstraction indicated by the key
        :param key: tuple of (Abstr_type, abstr_epsilon, corr_type, corr_proportion, batch_num)
        :param corrupt_abstr_file: path to the file containing the corrupt abstraction
        :param error_file: optional path to the file containing the list of states that are errors
        :return: a surface showing the corrupt abstract MDP, color-coded, with error states marked by a red circle
        """
        # Open files and isolate row of interest
        '''
        corr_names = ['AbstrType', 'AbstrEpsilon', 'CorrType', 'CorrProp', 'Batch', 'S_A']
        corr_df = pd.read_csv(corrupt_abstr_file, names=corr_names)

        # Load the state abstraction file into an abstract MDP
        print(corr_df.to_string())
        s_a_list = corr_df.loc[(corr_df['AbstrType'] == str(key[0]))
                                & (corr_df['AbstrEpsilon'] == key[1])
                                & (corr_df['CorrType'] == str(key[2]))
                                & (corr_df['CorrProp'] == key[3])
                                & (corr_df['Batch'] == key[4])]['S_A'].values[0]
        s_a_list = ast.literal_eval(s_a_list)
        '''
        s_a_list = self.parse_file_for_dict(key, corrupt_abstr_file)
        s_a = {}
        for i in range(len(s_a_list)):
            ground_state = GridWorldState(s_a_list[i][0][0], s_a_list[i][0][1])
            abstr_state = s_a_list[i][1]
            s_a[ground_state] = abstr_state

        corr_abstr_mdp = AbstractMDP(GridWorldMDP(), StateAbstraction(s_a, key[0]))

        # Load error_file data if that argument is given
        err_arg = None
        if error_file:
            '''
            names=['AbstrType', 'AbstrEps', 'Prop', 'Batch', 'ErrorStates']
            error_df = pd.read_csv(error_file, names=names)
            error_list = ast.literal_eval(error_df.loc[(error_df['AbstrType'] == str(key[0]))
                                                       & (error_df['AbstrEps'] == key[1])
                                                       & (error_df['Prop'] == key[3])
                                                       & (error_df['Batch'] == key[4])]['ErrorStates'].values[0])

            '''
            error_list = self.parse_file_for_dict(key, error_file)
            error_states = []
            for val in error_list:
                error_states.append(((ast.literal_eval(val[0])),
                                     (ast.literal_eval(val[1])),
                                     (ast.literal_eval(val[2]))))
            err_arg = error_states

        # Create the surface for the corrupt abstract MDP
        corr_surface = self.create_abstract_gridworld_mdp(corr_abstr_mdp, err_list=err_arg)
        if error_file:
            corr_surface = self.draw_errors(corr_surface, key, error_file)

        return corr_surface

    def draw_errors(self, surface, key, error_file):
        # Parse error list; error_states contains a list of tuples of the coordinates of error states
        '''
        error_names = ['AbstrType', 'AbstrEps', 'Prop', 'Batch', 'ErrorStates']
        error_df = pd.read_csv(error_file, names=error_names)

        error_list = ast.literal_eval(error_df.loc[(error_df['AbstrType'] == str(key[0]))
                                                   & (error_df['AbstrEps'] == key[1])
                                                   & (error_df['Prop'] == key[3])
                                                   & (error_df['Batch'] == key[4])]['ErrorStates'].values[0])
        '''
        error_list = self.parse_file_for_dict(key, error_file)
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

    def parse_file_for_dict(self, key, file, agent_num=None):
        """
        Parse the given file and return the value associated with the key and batch num. Will work on any file where
        key is a string all together and batch num is its own column.
        :param key: a 2-tuple for a true abstraction, a 5-tuple for a corrupt abstraction, or the string "ground"
        :param file: a csv file mapping keys and batch numbers to saved values of some kind
        :param agent_num: optional integer indicating a particular agent. If provided, we'll look at a particular
                            agent within the given key
        :return: the value from the file matching the key and batch num
        """
        if agent_num is not None:
            names = ['key', 'agent_num', 'dict']
        else:
            names = ['key', 'dict']
        df = pd.read_csv(file, names=names)
        # In this case we're in the ground state and the key is just 'ground'
        if key == "ground":
            df = df.loc[df['key'] == key]
            if agent_num is not None:
                df = df.loc[df['agent_num'] == agent_num]
            value = ast.literal_eval(df['dict'].values[0])
        # In this case we're in a true state abstraction, key is (Abstr_type, abstr_eps)
        elif len(key) == 2:
            # Split key into abstr_type and abstr_eps
            #print(df['key'])
            df['abstr_type'], df['abstr_eps'] = df['key'].str.split(', ').str
            # Fill in abstr_eps field (for ground)
            df['abstr_eps'] = df['abstr_eps'].fillna('0.0')
            df['abstr_type'] = df['abstr_type'].map(lambda x: x.strip('(<: 1234>'))
            df['abstr_eps'] = df['abstr_eps'].map(lambda x: x.strip('(<: >)'))
            df = df.loc[(df['abstr_type'] == str(key[0]))
                        & (df['abstr_eps'] == str(key[1]))]
            if agent_num is not None:
                df = df.loc[df['agent_num'] == agent_num]
            value = ast.literal_eval(df['dict'].values[0])
        # In this case we're in a corrupt abstraction, key is (Abstr_type, abstr_eps, corr_type, corr_prop, mdp_num)
        elif len(key) == 5:
            df['abstr_type'], df['abstr_eps'], df['corr_type'], df['corr_prop'], df['batch_num'] = df['key'].str.split(', ').str
            df['abstr_type'] = df['abstr_type'].map(lambda x: x.strip('(<: 1234>'))
            df['corr_type'] = df['corr_type'].map(lambda x: x.strip('<>: 1234'))
            df['batch_num'] = df['batch_num'].map(lambda x: x.strip(')'))
            df = df.loc[(df['abstr_type'] == str(key[0]))
                        & (df['abstr_eps'].astype(float) == key[1])
                        & (df['corr_type'] == str(key[2]))
                        & (df['corr_prop'].astype(float) == key[3])
                        & (df['batch_num'].astype(int) == key[4])]
            if agent_num is not None:
                df = df.loc[df['agent_num'] == agent_num]
            value = ast.literal_eval(df['dict'].values[0])
        else:
            raise ValueError('Key provided is not of valid type (either "ground", 2-tuple, or 5-tuple)')

        return value

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
        Generate a sequence of states ending in either a terminal state or in a loop. If ends in terminal state,
        final element is None. Otherwise final element is the last action taken. Assumes
        policy_file/abstraction_file are for corrupted abstract MDPs. Final element
        :param policy_file: file containing the learned policies
        :param abstraction_file: file containing the abstraction mappings
        :param key: (Abstr_type, abstr_epsilon, corr_type, corr_prop, batch_num) tuple
        :param agent_num: number of the agent in the ensemble, stored in policy_file
        :return: rollout, a list of states with the final action or None as last element
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
            policy_dict[ast.literal_eval(policy_key)] = self.parse_action_string(policy_string[policy_key])

        # Load abstraction mapping into dictionary, identifying proper abstractions/agents based on key and agent_num
        '''
        abstr_names = ['AbstrType', 'AbstrEps', 'CorrType', 'CorrProp', 'BatchNum', 'AbstrDict']
        abstr_df = pd.read_csv(abstraction_file, names=abstr_names)
        abstr_string = abstr_df.loc[(abstr_df['AbstrType'] == str(key[0]))
                                    & (abstr_df['AbstrEps'] == key[1])
                                    & (abstr_df['CorrType'] == str(key[2]))
                                    & (abstr_df['CorrProp'] == key[3])
                                    & (abstr_df['BatchNum'] == key[4])]['AbstrDict'].values[0]
        '''
        abstr_string = self.parse_file_for_dict(key, abstraction_file)
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
            #abstr_state = abstr_dict[(state.x, state.y)]
            #print(state.x, state.y)
            #print(policy_dict)
            #print(abstr_state)
            #action = policy_dict[abstr_state]
            action = policy_dict[(state.x, state.y)]
            next_state = mdp.transition(state, action)
            rollout.append((next_state.x, next_state.y))
            visited_states.append(state)
            state = next_state
        rollout.append(action)
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
        # Draw a single line indicating the path the agent takes by following the greedy policy
        while i < len(rollout) - 2:
            state = rollout[i]
            next_state = rollout[i + 1]
            state_pos = (self.cell_size * (state[0] - 1) + (self.cell_size / 2) + int(3 * agent_num / 2),
                         self.cell_size * (HEIGHT - state[1]) + (self.cell_size / 2) + int(3 * agent_num / 2))
            next_state_pos = (self.cell_size * (next_state[0] - 1) + (self.cell_size / 2) + int(3 * agent_num / 2),
                         self.cell_size * (HEIGHT - next_state[1]) + (self.cell_size / 2) + int(3 * agent_num/ 2))
            pygame.draw.line(surface, color, state_pos, next_state_pos, 2)
            i += 1
        # If we're not in the goal state, draw a small black square in the direction of the last action taken which
        #  resulted in a loop
        if rollout[-1] != (11, 11):
            final_cell = rollout[-2]
            action = rollout[-1]
            shrink_factor = 0.7
            if action == Dir.UP:
                margin = (0,1*shrink_factor)
            elif action == Dir.DOWN:
                margin = (0,-1*shrink_factor)
            elif action == Dir.RIGHT:
                margin = (1*shrink_factor,0)
            elif action == Dir.LEFT:
                margin = (-1*shrink_factor,0)
            mid_loc = (self.cell_size * (final_cell[0] - 1) + (self.cell_size / 2) + int(3 * agent_num / 2),
                       self.cell_size * (HEIGHT - final_cell[1]) + (self.cell_size / 2) + int(3 * agent_num / 2))
            end_loc = (self.cell_size * (final_cell[0] - 1) + (self.cell_size / 2) + int(3 * agent_num / 2) + self.cell_size * margin[0],
                       self.cell_size * (HEIGHT - final_cell[1]) + (self.cell_size / 2) + int(3 * agent_num / 2) + self.cell_size * margin[1])
            pygame.draw.line(surface, BLACK, mid_loc, end_loc, 3)
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
        #for i in range(num_agents):
        for agent_num in num_agents:
            #rollout = self.generate_corrupt_abstract_rollout(key, policy_file, abstraction_file, i)
            rollout = self.generate_corrupt_abstract_rollout(key, policy_file, abstraction_file, agent_num)
            print(rollout)
            color = random_color.generate()
            while color in colors_used:
                color = random_color.generate()
            color = pygame.Color(color[0])
            #surface = self.draw_gridworld_rollout(surface, rollout, color, i)
            surface = self.draw_gridworld_rollout(surface, rollout, color, agent_num)
        return surface

    def draw_state_value_gradient(self, key, agent_num, state_value_file, background=None, save=False):
        """
        Draw FourRooms with a gradient from dark to light, where high-value states are light and low-value states
        are dark
        :return: pygame surface
        """
        # Draw basic grid
        mdp = GridWorldMDP()
        surface = self.create_gridworld_mdp(mdp, background=background)

        # Read in dictionary from file and parse dictionary
        value_dict = self.parse_file_for_dict(key, state_value_file, agent_num=agent_num)
        #for key, value in value_dict.items():
        #    print(key, value)

        # Draw rectangles that are dark if the state is low value and bright if the state is high value. This is
        #  indexed against the maximum state value observed, such that the highest-value state observed will be white.
        value_max = max(value_dict.values())
        for square in value_dict.keys():
            value_color = (np.floor(255 * value_dict[square] / value_max),
                           np.floor(255 * value_dict[square] / value_max),
                           np.floor(255 * value_dict[square] / value_max))
            #print('square', square, 'value', value_dict[square], 'color', value_color)
            cell = pygame.Rect((self.margin + self.cell_size) * (square[0] - 1) + self.margin,
                               (self.margin + self.cell_size) * (11 - square[1]) + self.margin,
                               self.cell_size,
                               self.cell_size)
            pygame.draw.rect(surface, value_color, cell)
        if save:
            abstr_string = self.get_abstr_name(key[0])
            file_name = 'state_value_gradient_' + abstr_string + '_' + str(key[3]) + '_'\
                        + str(key[4]) + '_' + agent_num + '.png'
            pygame.image.save(surface, file_name)

        return surface

    def draw_misaggregations(self, surface, key, error_file, abstraction_file, save=False):
        """
        Visualize state aggregation errors on the given surface by drawing circles over the error states
        and squares over the other ground states that share an abstract state with the error states. Each circle/square
        will be colored such that one color represents one abstract state.
        :param surface: a pygame Surface
        :param key: a 5-tuple representing a key for a corrupt abstraction
        :param error_file: file denoting the errors
        :param abstraction_file: a file denoting the state abstractions for the given key
        :return: the surface with the errors drawn on it
        """
        # Read in the error file and abstraction file and parse the contents of error_list
        error_list = self.parse_file_for_dict(key, error_file)
        abstr_list = self.parse_file_for_dict(key, abstraction_file)
        # Error_dict is a list of tuples where first element is ground tuple, second is true state integer, third is
        #  corr state integer
        error_list = [(ast.literal_eval(error[0]),
                       ast.literal_eval(error[1]),
                       ast.literal_eval(error[2])) for error in error_list]

        # Create abstr_lookup_dict, which will map error states (or tuples of error states) to random colors, and
        #  abstr_apply_dict, which will map colors to lists of ground states which are correctly mapped to the
        #  abstract state represented by the color
        abstr_lookup_dict = {}
        abstr_apply_dict = {}

        # Aggregate error groups
        agged_states = []
        for i in range(len(error_list)):
            error_state = error_list[i][0]
            true_state_int = error_list[i][1]
            corr_state_int = error_list[i][2]
            if corr_state_int in agged_states:
                continue
            temp = [error_state]
            for j in range(i, len(error_list)):
                if corr_state_int == error_list[j][2] and error_state != error_list[j][0]:
                    temp.append(error_list[j][0])
            #print('Assigning', true_state_int, 'to', temp)
            abstr_lookup_dict[corr_state_int] = tuple(temp)
            for state in temp:
                agged_states.append(corr_state_int)
        # Abstr_lookup_dict now maps abstract states to error_groups

        error_states = [error_tuple[0] for error_tuple in error_list]
        # Get mapping of abstract states to ground states correctly mapped to them for all abstract states with
        #  errors in them
        for abstr_state in abstr_lookup_dict.keys():
            temp = []
            for pairing in abstr_list:
                if pairing[0] not in error_states and pairing[1] == abstr_state:
                    temp.append(pairing[0])
            abstr_apply_dict[abstr_state] = tuple(temp)

        # Map each abstract state with an error group to a random color and store in color_map_dict
        color_map_dict = {}
        rc = randomcolor.RandomColor()
        colors_used = []
        for abstr_state in abstr_lookup_dict.keys():
            color = rc.generate()
            while color in colors_used:
                color = rc.generate()
            colors_used.append(color)
            color = pygame.Color(color[0])
            color_map_dict[abstr_state] = color

        # Finally, for each abstract state with an error group, draw circles on the states in the error group and
        #  squares on the true states
        for abstr_state in abstr_lookup_dict.keys():
            color = color_map_dict[abstr_state]
            # Draw circles on error group
            for err_state in abstr_lookup_dict[abstr_state]:
                col = int((self.margin + self.cell_size) * (err_state[0] - 1) + self.margin
                          + np.floor(self.cell_size / 2))
                row = int((self.margin + self.cell_size) * (HEIGHT - err_state[1]) + self.margin
                          + np.floor(self.cell_size / 2))
                radius = int(np.floor(self.cell_size / 3))
                pygame.draw.circle(surface, color, (col, row), radius)
            # Draw squares on correct ground states
            for true_state in abstr_apply_dict[abstr_state]:
                left = int((self.margin + self.cell_size) * (true_state[0] - 1) + self.margin + np.floor(self.cell_size / 3))
                top = int((self.margin + self.cell_size) * (HEIGHT - true_state[1]) + self.margin + np.floor(self.cell_size / 3))
                cell = pygame.Rect(left, top, int(np.floor(self.cell_size / 3)), int(np.floor(self.cell_size / 3)))
                pygame.draw.rect(surface, color, cell)


        return surface

    def get_abstr_name(self, abstr):
        abstr_string = str(abstr)
        abstr_string = abstr_string[abstr_string.find('.') + 1:]
        abstr_string = abstr_string[:abstr_string.find('_')]
        abstr_string = abstr_string.lower()
        return abstr_string









