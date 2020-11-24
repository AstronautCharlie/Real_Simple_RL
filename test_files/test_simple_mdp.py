"""
This file tests the SimpleMDP and makes sure the transitions behave as expected and that
value iteration works as expected.
"""

from MDP.SimpleMDP import SimpleMDP
from MDP.StateClass import State
from MDP.ValueIterationClass import ValueIteration
import unittest

# This enumerates the states
s1 = State(data='s1')
s2 = State(data='s2')
s3 = State(data='s3')
e = State(data='e', is_terminal=True)

# These are the actions
actions = ('up', 'down', 'right')

class test_mdp_actions(unittest.TestCase):
    def test_start(self):
        mdp = SimpleMDP()
        self.assertEqual(mdp.get_current_state(), s1)

    def test_actions(self):
        mdp = SimpleMDP()

        # s1, up -> s2
        next_state, reward = mdp.act('up')
        self.assertEqual(next_state, s2)
        self.assertEqual(reward, 0)

        # s2, down -> s1
        next_state, reward = mdp.act('down')
        self.assertEqual(next_state, s1)
        self.assertEqual(reward, 0)

        # s1, down -> s3
        next_state, reward = mdp.act('down')
        self.assertEqual(next_state, s3)
        self.assertEqual(reward, 0)

        # s3, up -> s1
        next_state, reward = mdp.act('up')
        self.assertEqual(next_state, s1)
        self.assertEqual(reward, 0)

        # s1, up -> s2, right -> e
        _, _ = mdp.act('up')
        next_state, reward = mdp.act('right')
        self.assertEqual(next_state, e)
        self.assertEqual(reward, 1)
        self.assertEqual(mdp.get_current_state(), s1)

        # s1, down -> s3, right -> e
        _, _ = mdp.act('down')
        next_state, reward = mdp.act('right')
        self.assertEqual(next_state, e)
        self.assertEqual(reward, -0.5)
        self.assertEqual(mdp.get_current_state(), s1)

def test_mdp_vi():
    mdp = SimpleMDP()
    vi = ValueIteration(mdp)
    vi.run_value_iteration()
    q_table = vi.get_q_table()
    for key, value in q_table.items():
        print(key[0], key[1], value)


if __name__ == '__main__':
    test_mdp_vi()
    unittest.main()

