"""
Simplification of StateAbstractionClass compatible with OpenAI Gym env

Major changes:
- group_dict moved from Agent to StateAbstraction
- Assumes all ground states have abstract states, even if just singletons

"""

from collections import defaultdict
from gym.spaces.discrete import Discrete


# Helper methods
def reverse_abstr_dict(mapping):
    """
    Given a mapping of ground states to abstract states, create a dictionary
    mapping abstr state to list of ground states
    :param mapping: dictionary(ground state: abstr state)
    :return: dictionary(abstr state : List(ground states))
    """
    abstr_to_ground = defaultdict(lambda: [])
    for ground, abstr in mapping.items():
        abstr_to_ground[abstr].append(ground)
    return abstr_to_ground


class StateAbstraction:
    def __init__(self, abstr_dict=None):
        """
        :param abstr_dict: dict (ground states:abstract states)
        """
        if abstr_dict:
            self.abstr_dict = abstr_dict
        else:
            self.abstr_dict = {}

        # Create group dict, which maps abstract state to list of ground
        #  states
        self.group_dict = reverse_abstr_dict(self.abstr_dict)

    def __str__(self):
        result = 'abstr -> ground'
        for abstr in self.group_dict.keys():
            result += str(abstr) + '->' + str(self.group_dict[abstr]) + '\n'
        return result

    def get_abstr_from_ground(self, ground):
        """
        Get the abstract state corresponding to the given ground state
        :param ground: ground state
        :return: abstr state
        """
        try:
            if self.abstr_dict[ground] is None:
                raise ValueError('Searched ground state {} and abstr state'
                                 'was None [StateAbstraction.get_abstr_from'
                                 'ground]'.format(ground))
            return self.abstr_dict[ground]
        except KeyError:
            raise ValueError('Searched ground state {} with no corresponding '
                             'abstract state [StateAbstraction.get_abstr_from'
                             '_ground]'.format(ground))

    def get_ground_from_abstr(self, abstr_state):
        """
        Get a list of ground states associated with the given abstract state
        :param abstr_state: abstract state of interest
        :return: List(ground states)
        """
        ground_states = self.group_dict[abstr_state]
        if len(ground_states) == 0:
            raise ValueError('Searched abstract state {} with no '
                             'corresponding ground states [StateAbstraction'
                             '.get_ground_from_abstr]'.format(abstr_state))
        return ground_states

    def make_trivial_abstraction(self, env):
        """
        Create an abstraction where each state is mapped to its own abstract
        state.
        :param env: environment supporting OpenAI Gym interface
        """
        if isinstance(env.observation_space, Discrete):
            for i in range(env.observation_space.n):
                self.abstr_dict[i] = i
                self.group_dict = reverse_abstr_dict(self.abstr_dict)
        else:
            raise ValueError('StateAbstraction.make_trivial_abstraction'
                             ' is only supported for Discrete'
                             ' environments')

    def is_singleton(self, state, state_type='ground'):
        """
        Return boolean indicating whether state is a singleton state
        state_type can be 'ground' or 'abstract'
        :param state
        :param state_type: str ('ground' or 'abstract')
        :return: boolean
        """
        # Check if state is not in abstract mapping - if so, then it is a singleton
        if state_type == 'ground' and state not in self.abstr_dict.keys():
            return True
        if state_type == 'abstract' and state not in self.group_dict.keys():
            return True

        if state_type not in ['ground', 'abstract']:
            raise ValueError('TrackingAgent.is_singleton called with '
                             'state_type = {}; needs to be "ground"'
                             ' or "abstract"')
        if state_type == 'abstract':
            temp = state
        else:
            temp = self.abstr_dict[state]
        return len(self.group_dict[temp]) == 1
