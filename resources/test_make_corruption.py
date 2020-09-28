"""
This file tests the make_corruption method in the AbstractionCorrupters file
"""

from GridWorld.GridWorldMDPClass import GridWorldMDP
from MDP.StateAbstractionClass import StateAbstraction
from MDP.AbstractMDPClass import AbstractMDP
from MDP.ValueIterationClass import ValueIteration
from resources.AbstractionTypes import Abstr_type
from resources.AbstractionCorrupters import make_corruption
from resources.AbstractionMakers import make_abstr

import numpy as np

# Number of states to corrupt
STATE_NUM = 20

# Create abstract MDP
mdp = GridWorldMDP()
vi = ValueIteration(mdp)
vi.run_value_iteration()
q_table = vi.get_q_table()
state_abstr = make_abstr(q_table, Abstr_type.PI_STAR)
abstr_mdp = AbstractMDP(mdp, state_abstr)

# Randomly select our list of states and print them out
states_to_corrupt = np.random.choice(mdp.get_all_possible_states(), size=STATE_NUM, replace=False)
for state in states_to_corrupt:
    print(state)

# Create a corrupt MDP
corr_mdp = make_corruption(abstr_mdp, states_to_corrupt)

for state in states_to_corrupt:
    print(abstr_mdp.get_abstr_from_ground(state), corr_mdp.get_abstr_from_ground(state))