"""
Simplified and improved version of Agent to be compatible with OpenAI Gym
interface
"""
from collections import defaultdict
import numpy as np


class Agent:
    def __init__(self,
                 env,
                 **kwargs):
        self.env = env
        self.current_state = env.reset()
        self._q_table = defaultdict(lambda: 0.0)
        self._step_counter = 0
        default_vals = {'alpha': 0.1, 'epsilon': 0.1, 'decay_exploration': True}
        for key, value in default_vals:
            if key not in kwargs.items():
                kwargs[key] = value
        self.params = kwargs

    # ---------------------
    # Exploration functions
    # ---------------------
    def epsilon_greedy_action(self, state):
        """
        Select best action with probability (1-epsilon), random action otherwise
        :param state: the state for which we're selecting the action
        :return: the action selected
        """
        if np.random.uniform() < self.params['epsilon']:
            action = self.env.action_space.sample()
        else:
            action = self.get_best_action(state)
        return action

    def _update_learning_parameters(self):
        """
        Decay exploration and learning rate parameters
        """
        if self.params['decay_exploration']:
            self.params['alpha'] = self.params['init_alpha'] / (1.0 + self._step_counter / 2000000)
            self.params['epsilon'] = self.params['init_epsilon'] / (1.0 + self._step_counter / 2000000)

    def explore(self):
        """
        Apply epsilon-greedy exploration and update Q-table with resulting data
        :return: (reward, done)
        """


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

    def get_q_value(self, state, action):
        return self._q_table[(state, action)]
