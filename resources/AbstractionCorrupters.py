"""
This file defines methods to introduce noise into a state abstraction
"""

from MDP.StateAbstractionClass import StateAbstraction
from resources.CorruptionTypes import Corr_type
import copy
import numpy as np

def uniform_random(s_a, count=0):#proportion=1.0):
    """
    Scramble a state abstraction by reassigning ground states to abstract states with uniform probability. Note that
    this enforces that a ground state cannot be randomly assigned to its correct ground state. 'Proportion'
    parameter indicates what portion of the ground states are to be reassigned. This does not create any new abstract
    states.
    :param s_a: the state abstraction to be scrambled
    :param proportion: the proportion of states to be reassigned
    :return: c_s_a: the corrupted state abstraction
    """
    #if proportion > 1.0:
    #    raise ValueError("Cannot have proporton greater than 1")

    # Get the original dictionary mapping ground states to abstract states and the lists of ground and abstract states
    orig_dict = s_a.get_abstr_dict()
    ground_states = list(orig_dict.keys())
    abstr_states = list(orig_dict.values())

    # Create a deep copy of the original dictionary. This will become the corrupted state abstraction
    corrupt_dict = copy.deepcopy(orig_dict)

    # Randomly a proportion of ground states. These will be randomly reassigned to abstract states
    #corrupt_states = np.random.choice(ground_states, size=int(np.floor(proportion * len(ground_states))), replace=False)
    corrupt_states = np.random.choice(ground_states, size=count, replace=False)
    print('corrupt states are', corrupt_states)
    for state in corrupt_states:
        while corrupt_dict[state] == orig_dict[state]:
            corrupt_dict[state] = np.random.choice(abstr_states)

    c_s_a = StateAbstraction(corrupt_dict, abstr_type=s_a.abstr_type, epsilon=s_a.epsilon)
    return c_s_a

'''
def make_corruption(s_a, type, proportion):
    if type == Corr_type.UNI_RAND:
        return uniform_random(s_a, proportion=proportion)
'''

def make_corruption(abstr_mdp, states_to_corrupt=None, corr_type=Corr_type.UNI_RAND, reassignment_dict=None):
    """
    Corrupt the given state abstraction. If states to corrupt and type are not null, randomly reassign the given states
    to incorrect abstract states. If reassignment dict is not null, explicitly reassign key states to the same
    abstract state as value states
    :param abstr_mdp: (AbstractMDP) the mdp to be corrupted
    :param states_to_corrupt: (list of States) the ground states to be reassigned
    :param corr_type: method of reassigning the states
    :param reassignment_dict: dictionary mapping error states to corrupted states
    :return: c_s_a, a corrupted state abstraction with the states in states_to_corrupt randomly reassigned
    """
    orig_dict = abstr_mdp.get_state_abstr().get_abstr_dict()
    corrupt_dict = copy.deepcopy(orig_dict)
    abstr_states = list(orig_dict.values())

    # In this case, map keys in reassignment dict to the same abstract state as the value
    if reassignment_dict is not None:
        for error_state, corrupt_state in reassignment_dict.items():
            #try:
            new_abstr_state = orig_dict[corrupt_state]
            corrupt_dict[error_state] = new_abstr_state
            #except:
            #    print('Failed with', corrupt_state, error_state)
            #    quit()
    # In this case, randomly reassign the given states
    elif corr_type == Corr_type.UNI_RAND:
        for state in states_to_corrupt:
            while corrupt_dict[state] == orig_dict[state]:
                corrupt_dict[state] = np.random.choice(abstr_states)
    else:
        raise ValueError(str(corr_type) + " is not a supported abstraction type")

    c_s_a = StateAbstraction(corrupt_dict,
                             abstr_type=abstr_mdp.get_state_abstr().abstr_type,
                             epsilon=abstr_mdp.get_state_abstr().epsilon)
    return c_s_a
