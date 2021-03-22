"""
This file defines methods to introduce noise into a state abstraction
"""

from MDP.StateAbstractionClass import StateAbstraction
from MDP.ValueIterationClass import ValueIteration
from MDP.AbstractMDPClass import AbstractMDP
from resources.CorruptionTypes import Corr_type
from resources.AbstractionMakers import make_abstr
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

def apply_noise_from_distribution(ground_mdp,
                                  abstr_type,
                                  approximation_epsilon=0.0,
                                  distribution=None,
                                  distribution_parameters=None,
                                  per_state_distribution=None,
                                  per_state_parameters=None,
                                  seed=None):
    """
    Run value iteration on ground MDP to get true abstraction of given type. Then apply noise by sampling from given
    distribution and add the sampled value to the Q-values. Then create approximate abstraction by grouping together
    based on given epsilon
    :param ground_mdp: the ground mdp with no abstractions
    :param abstr_type: what type of abstraction is desired
    :param distribution: a scipy distribution
    :param distribution_parameters: a dictionary of parameters passed to the distribution when sampling
    :param approximation_epsilon: the epsilon used in making approximate abstractions
    :param per_state_distribution: dictionary mapping states to distributions
    :param per_state_parameters: dictionary mapping states to parameters used for their per-state distributions
    """
    # Get Q-table
    vi = ValueIteration(ground_mdp)
    vi.run_value_iteration()
    q_table = vi.get_q_table()

    # Apply noise sampled from distribution to Q-table
    for (state, action), value in q_table.items():
        #print(state, action, value)
        # If there is a specific per-state distribution, apply that
        if per_state_distribution:
            if state in per_state_distribution.keys():
                dist = per_state_distribution[state]
                args = per_state_parameters[state]
                noise = dist.rvs(**args)
                q_table[(state, action)] += noise
        # Otherwise apply mdp-wide distribution
        else:
            noise = distribution.rvs(**distribution_parameters)
            print(noise)
            q_table[(state, action)] += noise
        #print('New value:', q_table[(state, action)],'\n')

    # Make new epsilon-approximate abstraction
    new_s_a = make_abstr(q_table,
                         abstr_type,
                         epsilon=approximation_epsilon,
                         combine_zeroes=True,
                         threshold=0.0,
                         seed=seed)

    # Create abstract MDP with this corrupted s_a
    corr_mdp = AbstractMDP(ground_mdp, new_s_a)

    return corr_mdp