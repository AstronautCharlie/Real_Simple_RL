'''
This class represents a state abstraction from set of ground
states to a set of abstract states
'''

from MDP.StateClass import State


class StateAbstraction():
    def __init__(self, abstr_dict=None, abstr_type=None, epsilon=1e-12):
        """
        :param abstr_dict: dictionary(ground states:abstract states)
        :param abstr_type: the type of abstraction
        :param epsilon: the difference threshold between states aggregated together
        The abstr_dict is a mapping of the ground states to abstract
        states. If no abstr_dict is provided, assume the trivial
        abstraction of mapping each state to itself
        """
        self.abstr_dict = abstr_dict
        self.abstr_type = abstr_type
        #if abstr_dict is not None and abstr_type is None:
        #    raise ValueError('Cannot create non-trivial state abstraction without abstraction type argument. '
        #                     'Please provide abstr_type=(type of abstraction)')
        self.epsilon = epsilon

    def get_abstr_dict(self):
        return self.abstr_dict

    def get_abstr_from_ground(self, state):
        """
        Get the abstract state corresponding to the given state. If the given state does not occur in abstr_dict,
        return the state itself
        :param state:
        :return: abstr_state, the abstract state corresponding to the given ground state
        """
        if self.abstr_dict is not None and state in self.abstr_dict.keys():
            abstr_state = State(data=(self.abstr_dict[state]), is_terminal=state.is_terminal())
            return abstr_state

        else:
            print('No Abstract state corresponding to', state, 'Returning state.')
            print('Abstr dict is')
            for key, value in self.abstr_dict.items():
                print(key, value)
            return state

    def get_all_abstr_states(self):
        if self.abstr_dict is None:
            print('You shouldn\'t be here')
            return None
        abstr_states = []
        for state in self.abstr_dict.keys():
            abstr_states.append(self.get_abstr_from_ground(state))
        return abstr_states

    def get_ground_from_abstr(self, abstr_state):
        """
        Returns a list of all ground states associated with the given abstr_state
        """
        result = []
        for ground in self.abstr_dict.keys():
            try:
                if self.abstr_dict[ground] == abstr_state.data:
                    result.append(ground)
            except:
                print(self.abstr_dict[ground], type(self.abstr_dict[ground]),
                      abstr_state, type(abstr_state))
                print('failed in get_ground_from_abstr')
                quit()
        return result

    def __str__(self):
        abstr_states_temp = list(self.abstr_dict.values())

        abstr_states = []
        for state in abstr_states_temp:
            if state not in abstr_states:
                abstr_states.append(state)

        result = ""

        abstr_states.sort()
        for state in abstr_states:
            for key in self.abstr_dict.keys():
                if self.abstr_dict[key] == state:
                    result += 'ground -> abstr: ' + str(key) + ' -> ' + str(self.abstr_dict[key])
                    result += '\n'
        return result

    # TODO Make compatible with gym environments
    def make_trivial_abstraction(self, mdp, abstr_type=None):
        """
        Make a state abstraction for an MDP where each ground state maps to its own abstract state
        """
        if abstr_type:
            self.abstr_type = abstr_type
        self.abstr_dict = {}
        for state in mdp.get_all_possible_states():
            i = 0
            self.abstr_dict[state] = State(i, is_terminal = state.is_terminal)
            i += 1


