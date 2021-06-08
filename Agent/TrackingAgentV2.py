"""
Improved and simplified TrackingAgent
"""

from MDP.StateAbstractionV2 import StateAbstraction
from collections import defaultdict
from gym.spaces.discrete import Discrete
import numpy as np


class TrackingAgent:
    def __init__(self, env, **kwargs):
        # Basic parameters
        self.env = env
        self.params = kwargs
        self.current_state = env.reset()
        self._q_table = defaultdict(lambda x: 0)
        # Used for annealing learning parameters
        self._step_counter = 0

        # If no abstraction provided, make trivial abstraction
        if 's_a' not in self.params.keys() or self.params['s_a'] is None:
            self.params['s_a'] = StateAbstraction()
            self.params['s_a'].make_trivial_abstraction(env)

        # Fill in any missing default arguments
        default_vals = {'alpha': 0.1, 'epsilon': 0.1,
                        'decay_exploration': True}
        for key, value in default_vals.items():
            if key not in self.params.keys():
                self.params[key] = value

        # Trackers
        self.state_occupancy_record = defaultdict(lambda: 0)
        self.reachable_state_dict = defaultdict(lambda: set())

    # ---------------------
    # Exploration functions
    # ---------------------
    def epsilon_greedy(self, state):
        """
        Return best action for the given state with probability 1-self.epsilon,
        otherwise return random action
        :param state:
        :return: the selected action
        """
        if np.random.uniform() < self.params['epsilon']:
            action = self.env.action_space.sample()
        else:
            action = self.get_best_action(state)
        return action

    def explore(self):
        """
        Epsilon-greedy exploration with recording of state occupancies
        :return: (reward:float, done:Bool)
        """
        # Select and apply action
        action = self.epsilon_greedy(self.current_state)
        next_state, reward, done, _ = self.env.step(action)
        # Apply Q-value update
        self.update_q_value(self.current_state, action, next_state, reward)

        # Update current state and trackers
        self.state_occupancy_record[next_state] += 1
        self.reachable_state_dict[self.current_state].add(next_state)
        self.current_state = next_state
        self._step_counter += 1

        # Decay learning rate and exploration rate if q-value is not 0
        if self.get_q_value(self.current_state, action) != 0:
            self._update_learning_parameters()

        return reward, done

    # ---------------------
    # Abstraction functions
    # ---------------------

    def detach_state(self, state, reset_q_value=False):
        """
        Detach the given state from its abstract state and assign it
        to a new (singleton) abstract state
        :param state: the state to detach
        :param reset_q_value: string or bool indicating how to assign a
                Q-value to next state; True means set to 0, False means
                leave as is, and 'neighbor' means to take the average
                of neighboring squares
        :return:
        """
        if reset_q_value not in [True, False, 'neighbor', 'rollout']:
            raise ValueError('TrackingAgent.detach_state called with '
                             'invalid reset_q_value = {}; must be '
                             'True, False, "neighbor", or "rollout"'.format(reset_q_value))


        # If already singleton, do nothing
        if self.is_singleton(state, state_type='ground'):
            return

        # New abstract state is 1 more than maximum current abstract state
        old_abstr_state = self.params['s_a'].abstr_dict[state]
        new_abstr_state = max(list(self.params['s_a'].abstr_dict.keys())) + 1

        # Reassign state in StateAbstraction group_dict
        self.params['s_a'].group_dict[old_abstr_state].remove(state)
        self.params['s_a'].group_dict[new_abstr_state] = [state]

        # Assign ground state to new abstract state
        self.params['s_a'].abstr_dict[state] = new_abstr_state

        # Handle resetting Q-value
        if not isinstance(self.env.action_space, Discrete):
            raise ValueError('Environment does not have Discrete action '
                             'space [TrackingAgent.detach_state]')
        # TODO Implement all of these
        if reset_q_value == 'neighbor':
            raise NotImplementedError('TrackingAgent.detach_state not '
                                      'supported q-value reset parameter '
                                      '"neighbor"')
        elif reset_q_value == 'rollout':
            raise NotImplementedError('TrackingAgent.detach_state not '
                                      'supported q-value reset parameter '
                                      '"rollout"')
        elif reset_q_value:
            if not isinstance(self.env.action_space, Discrete):
                raise NotImplementedError('TrackingAgent.detach_state '
                                          'not supported for environments'
                                          ' with continuous action spaces')
            for i in range(self.env.action_space.n):
                self.set_q_value(state, i, 0)

    # -------
    # Utility
    # -------
    def get_best_action_value_pair(self, state):
        """
        Get the action with highest q-value for the given state and it's
        q-value
        :param state: the state for which we're looking up the Q-value
        :return: (action, action q-value)
        """
        max_val = float("-inf")
        best_action = None

        actions = [i for i in range(self.env.action_space.n)]
        np.random.shuffle(actions)
        for action in actions:
            q_value = self.get_q_value(state, action)
            if q_value > max_val:
                max_val = q_value
                best_action = action
        return best_action, max_val

    def get_best_action(self, state):
        action, _ = self.get_best_action_value_pair(state)
        return action

    def get_best_action_value(self, state):
        _, value = self.get_best_action_value_pair(state)
        return value

    def get_q_value(self, state, action):
        return self._q_table[(state, action)]

    def set_q_value(self, state, action, new_value):
        self._q_table[(state, action)] = new_value

    def update_q_value(self, state, action, next_state, reward):
        """
        Update Q-value with the new (s,a,s',r) tuple
        """
        old_q_value = self.get_q_value(state, action)
        td_error = reward + self.params['gamma'] * self.get_best_action_value(next_state) - old_q_value
        new_q_value = old_q_value + self.params['alpha'] * td_error
        self.set_q_value(state, action, new_q_value)

    def _update_learning_parameters(self):
        """
        Decay exploration and learning rate parameters
        """
        if self.params['decay_exploration']:
            self.params['alpha'] = self.params['init_alpha'] / (1.0 + self._step_counter / 2000000)
            self.params['epsilon'] = self.params['init_epsilon'] / (1.0 + self._step_counter / 2000000)

    def is_singleton(self, state, state_type='ground'):
        """
        Return boolean indicating whether state is a singleton state
        state_type can be 'ground' or 'abstract'
        :param state
        :param state_type: str ('ground' or 'abstract')
        :return: boolean
        """
        if state_type not in ['ground', 'abstract']:
            raise ValueError('TrackingAgent.is_singleton called with '
                             'state_type = {}; needs to be "ground"'
                             ' or "abstract"')
        if state_type == 'abstract':
            temp = state
        else:
            temp = self.params['s_a'].abstr_dict[state]
        return len(self.params['s_a'].group_dict[temp]) == 0
