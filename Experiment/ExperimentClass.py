'''
This class creates and runs a test of q-learning on the given MDP for the given abstraction
types and epsilon values, and compares the value of each trajectory to the value of the
optimal ground-state trajectory from that point
'''
from MDP.MDPClass import MDP
from MDP.StateAbstractionClass import StateAbstraction
from resources.AbstractionMakers import make_abstr
from resources.AbstractionTypes import Abstr_type

class Experiment():
    def __init__(self, mdp, abstr_epsilon_list):
        '''
        Create an experiment based on the given MDP and the given set of abstraction types and corresponding
        epsilons
        :param mdp: MDP
        :param abstr_epsilon_list: list of tuples, where first element is abstraction type and
        second is the epsilon
        '''
        self.ground_mdp = mdp
        for val in abstr_epsilon_list:
            if val[0] not in Abstr_type or val[1] < 0 or val[1] > 1:
                raise ValueError('Abstraction Epsilon List is invalid', abstr_epsilon_list)
        self.abstr_epsilon_list = abstr_epsilon_list

        # Create abstract MDPs for element of abstr_epsilon_list:
        

