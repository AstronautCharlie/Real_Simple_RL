"""
This file defines methods to introduce noise into a state abstraction
"""

from MDP.StateAbstractionClass import StateAbstraction
from resources.CorruptionTypes import Corr_type
import copy
import numpy as np

def uniform_random(s_a, proportion=1.0):
    """
    Scramble a state abstraction by reassigning ground states to abstract states with uniform probability. Note that
    this enforces that a ground state cannot be randomly assigned to its correct ground state. 'Proportion'
    parameter indicates what portion of the ground states are to be reassigned. This does not create any new abstract
    states.
    :param s_a: the state abstraction to be scrambled
    :param proportion: the proportion of states to be reassigned
    :return: c_s_a: the corrupted state abstraction
    """
    if proportion > 1.0:
        raise ValueError("Cannot have proporton greater than 1")

    # Get the original dictionary mapping ground states to abstract states and the lists of ground and abstract states
    orig_dict = s_a.get_abstr_dict()
    ground_states = list(orig_dict.keys())
    abstr_states = list(orig_dict.values())

    # Create a deep copy of the original dictionary. This will become the corrupted state abstraction
    corrupt_dict = copy.deepcopy(orig_dict)

    # Randomly a proportion of ground states. These will be randomly reassigned to abstract states
    corrupt_states = np.random.choice(ground_states, size=int(np.floor(proportion * len(ground_states))), replace=False)
    for state in corrupt_states:
        while corrupt_dict[state] == orig_dict[state]:
            corrupt_dict[state] = np.random.choice(abstr_states)

    c_s_a = StateAbstraction(corrupt_dict, abstr_type=s_a.abstr_type, epsilon=s_a.epsilon)
    return c_s_a

def make_corruption(s_a, type, proportion):
    if type == Corr_type.UNI_RAND:
        return uniform_random(s_a, proportion=proportion)