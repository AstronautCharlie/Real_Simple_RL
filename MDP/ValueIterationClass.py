from collections import defaultdict
import numpy as np

class ValueIteration():
    def __init__(self,mdp, gamma, delta):
        self.mdp = mdp
        self.gamma = gamma
        #stop when the learned q-values for all state action pairs are within delta of each other
        self.delta = delta
        self._q_table = defaultdict(lambda: 0.0)

    def get_best_action_value_pair(self, state):
        '''
        Get the action with the best q-value and the q-value associated
        with that action

        Parameters:
            state:State

        Returns:
            best_action:Enum
            max_val:float
        '''
        # Initialize best action to be a random choice (in case no )
        max_val = float("-inf")
        best_action = None

        # Iterate through actions and find action with highest q-value
        # in q-table. Shuffle actions so that if best actions have
        # the same value, a random one is chosen
        shuffled_actions = self.mdp.actions.copy()
        np.random.shuffle(shuffled_actions)
        for action in shuffled_actions:
            q_value = self.get_q_value(state, action)
            if q_value > max_val:
                max_val = q_value
                best_action = action
        return best_action, max_val

    def get_best_action(self, state):
        '''
        Return the action with the max q-value for the given state

        Parameters:
            state:State

        Returns:
            best_action:Enum
        '''
        best_action, _ = self.get_best_action_value_pair(state)
        return best_action

    def get_best_action_value(self, state):
        '''
        Return the q-value of the action with the max q-value for
        the given state

        Parameters:
            state:State

        Returns:
            reward:float
        '''
        _, reward = self.get_best_action_value_pair(state)
        return reward

    def get_action_values(self, state):
        '''
        Get all the action-value pairs for the given state and
        return them as a list of tuples

        Parameters:
            state:State

        Returns:
            action_value_list:list
        '''
        action_value_list = []
        for action in self.mdp.actions:
            pair = tuple([action, self.get_q_value(state, action)])
            action_value_list.append(pair)
        return action_value_list

    def get_q_value(self, state, action):
        '''
        Query the q-table for the value of the given state-action
        pair

        Parameters:
            state:State
            action:Enum

        returns:
            q-value:float
        '''
        return self._q_table[(state, action)]

    def get_mdp(self):
        return self.mdp

    def get_q_table(self):
        return self._q_table

    def update_q_table(self, new_q_table):
        """
        Update the entire q_table to a new one
        :return:
        """
        self._q_table = new_q_table

    def __str__(self):
        '''
        Print out the Q-table
        :return:
        '''
        '''
        result = 'state, value:'
        q_table = self.get_q_table()
        for key in q_table.keys():
            result += str(key[0]) + ',' + str(key[1]) + ' ' + str(q_table[key])
            result += '\n'
        return result
        '''
        result = 'state, value:\n'
        value_func = self.get_q_table()
        for key in value_func.keys():
            result += str(key) + ' ' + str(value_func[key]) + '\n'
        return result

    def doValueIteration(self,steps):
        all_states = self.mdp.get_all_possible_states()
        stop = False
        for i in range(steps):
            #q table at i + 1'th iteration
            if(stop == False):
                print(i)
                Q_i = defaultdict(lambda: 0.0)
                stop = True
                for state in all_states:
                    for action in self.mdp.actions:
                        next_states = self.mdp.next_possible_states(state,action)
                        q_value = 0
                        for next_state in next_states.keys():
                            next_state_cur_value = self.get_best_action_value(next_state)
                            transition_prob = next_states[next_state]
                            reward = self.mdp.reward(state,action,next_state)
                            q_value += transition_prob*(reward + self.gamma*next_state_cur_value)
                            print(q_value)
                        Q_i[(state, action)] = q_value
                        stop = stop and (abs(q_value - self.get_q_value(state,action)) < self.delta )
                self.update_q_table(Q_i)



