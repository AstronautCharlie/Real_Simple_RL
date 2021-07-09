"""
Improved and simplified TrackingAgent

Required parameters:
- alpha (learning rate, default 0.1)
- epsilon-greedy factor (default 0.1)
- decay_exploration (default True)
- gamma (default 1)
"""

from MDP.StateAbstractionV2 import StateAbstraction, reverse_abstr_dict
from MDP.DiscretizationAbstraction import DiscretizationAbstraction
from collections import defaultdict
from gym.spaces.discrete import Discrete
import numpy as np
import time


class TrackingAgent:
    def __init__(self, env, **kwargs):
        # Basic parameters
        self.env = env
        self.params = kwargs
        self._q_table = defaultdict(lambda: 0)
        # Used for annealing learning parameters
        self._step_counter = 0

        # Make abstraction if applicable (i.e. no abstraction provided or abstraction type is discretization
        if ('s_a' not in self.params.keys() or self.params['s_a'] is None) \
                and ('abstraction_type' in self.params.keys() and self.params['abstraction_type'] == 'temporal'):
            self.params['s_a'] = StateAbstraction()
            self.params['s_a'].make_trivial_abstraction(env)
        print('About to make discretization abstraction')
        if 'abstraction_type' in self.params.keys() and self.params['abstraction_type'] == 'discretization':
            self.make_discretization_abstraction()
        print('Done making discretization abstraction')

        # Starting state
        starting_state = env.reset()
        if self.params['abstraction_type'] == 'temporal':
            self.current_state = starting_state
        elif self.params['abstraction_type'] == 'discretization':
            starting_state = self.params['s_a'].get_cell_from_observation(starting_state)
            self.current_state = starting_state

        # Fill in any missing default arguments
        default_vals = {'alpha': 0.1, 'epsilon': 0.1,
                        'decay_exploration': True, 'gamma': 1}
        for key, value in default_vals.items():
            if key not in self.params.keys():
                print('adding value {} {}'.format(key, value))
                self.params[key] = value
        self.params['init_alpha'] = self.params['alpha']
        self.params['init_epsilon'] = self.params['epsilon']

        # Trackers
        #if isinstance(self.env.observation_space, Discrete):  # Only track ground states if state space is discrete
        self.state_occupancy_record = defaultdict(lambda: 0)  # Ground states
        #else:
            #self.state_occupancy_record = None
        self.abstr_state_occupancy_record = defaultdict(lambda: 0)  # Abstract states
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
        #print('observation: {}'.format(next_state))
        if self.params['abstraction_type'] == 'discretization':
            next_state = self.params['s_a'].get_cell_from_observation(next_state)
        #print('current state: {} next_state: {}'.format(self.current_state, next_state))
        # Update all states in group dict
        abstr_state = self.params['s_a'].get_abstr_from_ground(self.current_state)
        for ground_state in self.params['s_a'].group_dict[abstr_state]:
            self.update_q_value(ground_state, action, next_state, reward)
            #print('state: {} next_state: {} abstract_state: {} q-value: {}'.format(ground_state, next_state, abstr_state,
            #                                                                       self.get_q_value(ground_state, action)))
        # Get abstract state from current state
        abstr_state = self.params['s_a'].get_abstr_from_ground(self.current_state)

        # Update current state and trackers
        #if self.state_occupancy_record:
        self.state_occupancy_record[self.current_state] += 1
        self.reachable_state_dict[self.current_state].add(next_state)
        self.abstr_state_occupancy_record[abstr_state] += 1
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
            for i in range(self.env.action_space.n):
                neighbor_vals = [self.get_q_value(neighbor, i) for neighbor in self.reachable_state_dict[state]]
                self.set_q_value(state, i, np.mean(neighbor_vals))

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

    def detach_most_visited_ground_state(self):
        """
        Detach the most visited ground state
        :return: the detached state
        """
        state_to_detach = self.get_most_visited_ground_state()
        if state_to_detach is None:
            return []
        self.detach_state(state_to_detach, reset_q_value=self.params['reset_q_value'])
        return [state_to_detach]

    def refine_abstraction(self):
        """
        Refine the agents abstraction by calling the appropriate refinement method
        :return: list of detached states
        """
        supported_types = ['temporal', 'discretization']
        if 'refinement_type' not in self.params.keys():
            raise ValueError('"refinement_type" parameter not set')
        if self.params['refinement_type'] not in supported_types:
            raise NotImplementedError('Refinement type {} is not supported'.format(self.params['refinement_type']))

        if self.params['refinement_type'] == 'temporal':
            detached_states = self.detach_most_visited_ground_state()
            return detached_states

        if self.params['refinement_type'] == 'discretization':
            most_visited_state = self.get_most_visited_abstr_state()
            self.abstr_state_occupancy_record[most_visited_state] = 0
            print('Dividing state {}'.format(most_visited_state))
            self.params['s_a'].divide_abstr_state(most_visited_state)
            return most_visited_state

    def make_abstraction(self):
        """
        Create an abstraction depending on the type of abstraction as defined by keyword. Set created abstraction
        to self.params['s_a']
        """
        supported_types = ['temporal', 'discretization']
        if 'abstraction_type' not in self.params.keys():
            raise ValueError('"abstraction_type" parameter not set')
        if self.params['abstraction_type'] not in supported_types:
            raise NotImplementedError('Abstraction type {} is not supported'.format(self.params['abstraction_type']))

        if self.params['abstraction_type'] == 'temporal':
            # TODO make more generic
            self.make_temporal_abstraction(1)

        if self.params['abstraction_type'] == 'discretization':
            self.make_discretization_abstraction()

    def make_temporal_abstraction(self, n=1):
        """
        Create a neighbor-based abstraction and assign it as the agent's
        state abstraction
        :param n: the neighbor factor
        :return: None
        """
        if not isinstance(self.env.observation_space, Discrete):
            raise NotImplementedError('TrackingAgent.make_temporal_abstraction only supported '
                                      'for environments with Discrete observation spaces. '
                                      'Current type is {}'.format(type(self.env.observation_space)))

        # TODO verify this line for getting all states is valid
        states = list(self.state_occupancy_record.keys())
        np.random.shuffle(states)

        new_abstr_dict = {}
        abstr_state_counter = 1

        # Iterate through all states, get neighbors, and group them together
        #  (unless they've already been assigned)
        for state in states:
            if state in new_abstr_dict.keys():
                continue

            # Add seed state to abstract state
            new_abstr_dict[state] = abstr_state_counter

            # Add n-neighbors to abstract state
            state_queue = [[state]]
            for i in range(n):
                temp = state_queue.pop(0)
                reachable_states = []
                for s in temp:
                    reachable_states.extend(self.reachable_state_dict[s])
                to_push = []
                for r_state in reachable_states:
                    if r_state in new_abstr_dict.keys():
                        continue
                    new_abstr_dict[r_state] = abstr_state_counter
                    to_push.append(r_state)
                state_queue.append(to_push)
            abstr_state_counter += 1

        # Assign it to this agent's state abstraction
        self.params['s_a'] = StateAbstraction(new_abstr_dict)

        # Update group dict
        self.params['s_a'].group_dict = reverse_abstr_dict(self.params['s_a'].abstr_dict)

    def make_discretization_abstraction(self):
        """
        Make a discretization of the state space by uniformly bucketing each dimension of the state space by the
        agent's 'discretization' parameter. Set the discretization to the agent's state abstraction.
        """
        finest_mesh = 1000
        starting_mesh = 10

        if 'starting_mesh' in self.params.keys():
            starting_mesh = self.params['starting_mesh']
        if 'finest_mesh' in self.params.keys():
            finest_mesh = self.params['finest_mesh']

        self.params['s_a'] = DiscretizationAbstraction(self.env, finest_mesh, starting_mesh)

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
        return self.params['s_a'].is_singleton(state, state_type)

    def get_most_visited_ground_state(self, exclude_singleton=True):
        """
        Return the ground state that has been visited the most
        :param exclude_singleton: If true, return the most visited state
            mapped with at least one other ground state
        :return: state
        """
        visit_counts = list(self.state_occupancy_record.values())
        visit_counts.sort(reverse=True)

        for max_count in visit_counts:
            for state, count in self.state_occupancy_record.items():
                if max_count == count:
                    if exclude_singleton and self.is_singleton(state):
                        continue
                    return state

    def get_most_visited_abstr_state(self):
        """
        Return the abstract state that has been visited the most
        :return: state
        """
        visit_counts = list(self.abstr_state_occupancy_record.values())
        visit_counts.sort(reverse=True)

        for max_count in visit_counts:
            for state, count in self.abstr_state_occupancy_record.items():
                if max_count == count:
                    return state
