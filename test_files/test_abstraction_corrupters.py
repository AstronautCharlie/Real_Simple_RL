"""
This file tests the abstraction corruption functions
"""

from MDP.AbstractMDPClass import AbstractMDP
from GridWorld.GridWorldMDPClass import GridWorldMDP
from MDP.ValueIterationClass import ValueIteration
from resources.AbstractionMakers import *
from resources.AbstractionCorrupters import *

def test_fourrooms(abstr_type, noise=0.0):
    """
    Test the corruption of a Q* abstraction in fourrooms
    :param abstr_type: the type of abstraction to be tested
    :param noise: the proportion of states to be scrambled
    :return:
    """
    # Make a grid world MDP and create an abstraction of the given type from it
    mdp = GridWorldMDP()
    vi = ValueIteration(mdp)
    vi.run_value_iteration()
    q_table = vi.get_q_table()
    true_abstr = make_abstr(q_table, abstr_type=abstr_type)

    # Corrupt the true results
    corrupt_results = uniform_random(true_abstr, proportion=0.1)

    true_dict = true_abstr.get_abstr_dict()
    corrupt_dict = corrupt_results.get_abstr_dict()

    for key in true_dict.keys():
        if true_dict[key] != corrupt_dict[key]:
            print(key, true_dict[key], corrupt_dict[key])


if __name__ == '__main__':
    test_fourrooms(Abstr_type.A_STAR)