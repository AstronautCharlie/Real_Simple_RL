"""
Modification of StateAbstraction to handle continuous environments. Abstract states
are represented as a tuple of buckets, where each element corresponds to the 1d cell
in that dimension
"""
import numpy as np


class DiscretizationAbstraction:
    def __init__(self, env, finest_mesh=1000, starting_mesh=10):
        """
        :param env: the OpenAIGym environment
        :param finest_mesh: resolution of the finest mesh (applied along each dimension)
        """
        print('start of making discretization abstraction')
        self.env = env
        self.space = env.observation_space
        self.finest_mesh = finest_mesh
        self.cell_to_abstract_cell = []  # maps cell number (in finest mesh) to abstract cell

        self.n = self.space.shape
        self.ranges = self.space.high - self.space.low
        # Create bucket size for each dimension - this is the cell size of the finest resolution
        self.bucket_sizes = self.ranges / finest_mesh
        self.starting_group_size = finest_mesh // starting_mesh

        print('About to start making grid')
        # self.grid holds the cutoff values in observation space for each cell (real numbers)
        # self.cell_to_abstract_cell maps finest mesh cell number to abstract cell number
        self.grid = []
        for i in range(self.n[0]):  # TODO this assumes all environments have shape tuple of form "(n,)"
            row = []
            dimension_dict = {}
            start_point = self.space.low[i]  # lower bound of this dimension
            bucket_size = self.bucket_sizes[i]  # bucket size of this dimension
            for j in range(finest_mesh):
                cell_threshold = j * bucket_size + start_point
                dimension_dict[j] = (j // self.starting_group_size) * self.starting_group_size
                row.append(cell_threshold)
            self.grid.append(row)
            self.cell_to_abstract_cell.append(dimension_dict)

        print('About to start making group_dict')
        # Create dictionary mapping each abstract state to its constituent abstract states
        self.group_dict = self.create_group_dict()

    def get_abstr_from_ground(self, cell):
        """
        Get the abstract state corresponding to the given cell. Abstract state is a tuple
        of size n (where n is the dimensionality of the observation space)
        :param cell: observation from the environments state space
        :return: n-tuple indicating abstract space
        """
        abstr_state = []
        for i in range(len(cell)):
            abstr_cell = self.cell_to_abstract_cell[i][cell[i]]
            abstr_state.append(abstr_cell)
        return tuple(abstr_state)

    def divide_cell(self, dimension, abstr_cell):
        """
        Split the given cell in the given dimension in half by updating self.dimension_dict
        :param dimension: int, the dimension in the state space of the cell to be divided
        :param abstr_cell: int, the cell to divide
        :return: None
        """
        cell_group = []

        # Get all cells in finest mesh matching given abstract cell
        for key, value in self.cell_to_abstract_cell[dimension].items():
            if value == abstr_cell:
                cell_group.append(key)
        cell_group.sort()
        next_largest_cell = abstr_cell + len(cell_group)
        if len(cell_group) == 0:
            raise ValueError('divide_cell called on non-existent cell; dimension: {}, cell: {}'.format(dimension,
                                                                                                       abstr_cell))

        # Split cell group in half
        new_abstr_cell = (abstr_cell + next_largest_cell) // 2
        split_index = len(cell_group) // 2
        for cell in cell_group[split_index:]:
            self.cell_to_abstract_cell[dimension][cell] = new_abstr_cell

    def divide_abstr_state(self, abstr_tuple):
        # TODO Handle reset of abstract state occupancy count
        """
        Split all cells in the given abstract state
        :param abstr_tuple: tuple representing the abstract state
        :return: None
        """
        for i in range(len(abstr_tuple)):
            self.divide_cell(i, abstr_tuple[i])
        self.group_dict = self.create_group_dict()

    def get_cell_from_observation(self, observation):
        """
        Convert an observation from the observation space to a cell in the finest mesh
        :param observation: sample from the environment's observation space
        :return: tuple of cell in finest mesh
        """
        cell = []
        for i in range(len(observation)):
            dim = observation[i]
            lower_bound = self.space.low[i]
            bucket_size = self.bucket_sizes[i]
            bucket = int((dim - lower_bound) // bucket_size)
            cell.append(bucket)
        return tuple(cell)

    def create_group_dict(self):
        """
        Create dictionary mapping each abstract state to a list of all constituent finest mesh hypercells
        :return: dict (abstract-to-finest mesh mapping)
        """
        # Get all abstract states by getting all possible combinations of abstract cells
        abstr_cell_values = []
        for i in range(self.n[0]):
            abstr_cell_values.append(list(set(self.cell_to_abstract_cell[i].values())))
        abstr_states = np.array(np.meshgrid(*abstr_cell_values)).T.reshape(-1, self.n[0])

        # Get all ground states by getting all possible combinations of finest mesh cells
        ground_cells = np.array([i for i in range(self.finest_mesh)] * self.n[0]).reshape(self.n[0], self.finest_mesh)
        # ground_cells is an n-d array where n is number of dimensions and entries are finest mesh buckets
        ground_cells = np.array(np.meshgrid(*ground_cells)).T.reshape(-1, self.n[0])

        print('About to start the awful shit')
        group_dict = {}
        i = 0
        for abstr_state in abstr_states:
            group_dict[tuple(abstr_state)] = []
        #print(group_dict)
        for ground_cell in ground_cells:
            abstr_cell = self.get_abstr_from_ground(ground_cell)
            #print('ground: {} abstr: {}'.format(ground_cell, abstr_cell))
            group_dict[abstr_cell].append(tuple(ground_cell))
            #for i in range(len(ground_cell)):
            #    if self.cell_to_abstract_cell[i][ground_cell[i]] != abstr_state[i]:
        #group_dict[tuple(abstr_state)] = group

        return group_dict
