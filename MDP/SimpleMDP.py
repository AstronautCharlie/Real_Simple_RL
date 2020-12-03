"""
This class represents a simple 4-state MDP which is used as a proof-of-concept for the detachment algorithm

Schematic of MDP:

s2          (right)  e
(up/down)
s1
(down/up)
s3          (right)  e

Starting state is s1, from which the agent can either go left or right. If left, it enters
state s2. If right, it enters s3. From s2/s3, it can either
"""

from MDP.MDPClass import MDP
from MDP.StateClass import State

# This enumerates the states
s1 = State(data='s1')
s2 = State(data='s2')
s3 = State(data='s3')
e = State(data='e', is_terminal=True)

# These are the actions
actions = ('up', 'down', 'right')

simple_states = [s1, s2, s3, e]
simple_actions = actions

GAMMA = 0.9
GOOD_REWARD = 1
BAD_REWARD = -0.5

class SimpleMDP(MDP):
    def __init__(self, actions=actions):
        super().__init__(actions=actions,
                         init_state=s1,
                         gamma=0.9)

    # --------------
    # Main functions
    # --------------
    def transition(self, state, action):
        next_state = state

        # If in s1, up = s2, down = s3, right = s1
        if state == s1:
            if action == 'up':
                next_state = s2
            elif action == 'down':
                next_state = s3
            elif action == 'right':
                next_state = s1
            else:
                raise ValueError('Invalid action ' + action)

        # If in s2, up = s2, down = s1, right = e
        if state == s2:
            if action == 'down':
                next_state = s1
            elif action == 'right':
                next_state = e
            elif action == 'up':
                next_state = s2
            else:
                raise ValueError('Invalid action ' + action)

        # if in s3, up = s1, down = s3, right = e
        if state == s3:
            if action == 'up':
                next_state = s1
            elif action == 'right':
                next_state = e
            elif action == 'down':
                next_state = s3
            else:
                raise ValueError('Invalid action ' + action)

        return next_state

    def reward(self, state, action, next_state):
        if state == s2 and action == 'right' and next_state == e:
            return GOOD_REWARD
        elif state == s3 and action == 'right' and next_state == e:
            return BAD_REWARD
        else:
            return 0

    def act(self, action):
        state = self.current_state
        next_state = self.transition(state, action)
        reward = self.reward(state, action, next_state)

        self.set_current_state(next_state)

        if next_state.is_terminal():
            self.reset_to_init()

        return next_state, reward

    # -----------------
    # Utility functions
    # -----------------
    def reset_to_init(self):
        self.set_current_state(self.get_init_state())

    def copy(self):
        return SimpleMDP()

    # --------------------
    # VI related functions
    # --------------------
    def get_all_possible_states(self):
        return [s1, s2, s3, e]

    def get_next_possible_states(self, state, action):
        # e
        if state == e:
            return {e: 1}

        # s1
        if state == s1:
            if action == 'up':
                return {s2: 1}
            elif action == 'down':
                return {s3: 1}
            elif action == 'right':
                return {s1: 1}
            else:
                raise ValueError('Invalid state/action' + str(state) + str(action))

        # s2
        if state == s2:
            if action == 'up':
                return {s2: 1}
            elif action == 'down':
                return {s1: 1}
            elif action == 'right':
                return {e: 1}
            else:
                raise ValueError('Invalid state/action' + str(state) + str(action))

        # s3
        if state == s3:
            if action == 'up':
                return {s1: 1}
            elif action == 'down':
                return {s3: 1}
            elif action == 'right':
                return {e: 1}
            else:
                raise ValueError('Invalid state/action' + str(state) + str(action))
