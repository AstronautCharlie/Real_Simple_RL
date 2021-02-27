"""
Test that all the functionality in AbstractionAgent that is not in the Agent class works as expected
"""

from Agent.AbstractionAgent import AbstractionAgent
from GridWorld.GridWorldMDPClass import GridWorldMDP
from GridWorld.GridWorldStateClass import GridWorldState
from MDP.StateAbstractionClass import StateAbstraction
from Visualizer.GridWorldVisualizer import GridWorldVisualizer
from util import *
from GridWorld.TwoRoomsMDP import TwoRoomsMDP

import pandas as pd
import ast
import copy
import csv

# Test that detaching a given state sets its Q-table to 0
def test_detach_state(agent):
    # Test that detach_state both removes the state from the abstraction dictionary and resets the Q-table to 0
    #  We select this state to remove since we are guaranteed to always interact with it
    state_to_remove = GridWorldState(1, 1)
    print('State and abstr state prior to detach:', state_to_remove, agent.s_a.abstr_dict[state_to_remove])
    print('Other states in this abstract state: ', end = '')
    for temp_state in agent.mdp.get_all_possible_states():
        if agent.s_a.abstr_dict[temp_state] == agent.s_a.abstr_dict[state_to_remove]:
            print(temp_state, end = ' ')
    print()
    for i in range(5000):
        agent.explore()
    print()
    print('Q-value of state after exploring: (should be non-zero)')
    for i in range(len(agent.mdp.actions)):
        print(agent.mdp.actions[i], agent.get_q_value(state_to_remove, agent.mdp.actions[i]))
    print()
    agent.detach_state(state_to_remove, reset_q_value=True)

    print('State and abstr state after detach:', state_to_remove, agent.s_a.abstr_dict[state_to_remove])
    print('Q-value of state after detaching: (should be zero)')
    for i in range(len(agent.mdp.actions)):
        print(agent.mdp.actions[i], agent.get_q_value(state_to_remove, agent.mdp.actions[i]))
    print()
    for i in range(5000):
        agent.explore()
    print('Q-value of state after exploring again: (should be non-zero)')
    for i in range(len(agent.mdp.actions)):
        print(agent.mdp.actions[i], agent.get_q_value(state_to_remove, agent.mdp.actions[i]))
    print('\n'*3)
    print('Full Q-table:')
    for key, value in agent.get_q_table().items():
        print(key[0], key[1], value)

    # Check that the ground -> abstr and abstr -> ground mappings correspond
    for key in agent.group_dict.keys():
        for state in agent.all_possible_states:
            if agent.s_a.abstr_dict[state] == key and state not in agent.group_dict[key]:
                print('FUCK', key, state)
    print('Success!')

# Test that generating a rollout works as expected
def test_generate_rollout(agent):
    # Generate a roll-out from an untrained agent. Should be random and short
    rollout = agent.generate_rollout()
    print('Roll-out for untrained model')
    for state in rollout:
        print(state, end=', ')
    print('\n'*2)

# Compare training a model with incorrect abstraction for 10,000 steps to training for 5,000, detaching,
#  and training for 5,000 more
def test_rollout_adjustment(key):
    """
    Train the agent on a state abstraction with fatal errors. Then generate a roll-out, detach the first state that's
    part of a cycle, and restart learning.
    """
    # Load a poorly-performing abstraction
    names = ['AbstrType', 'AbstrEps', 'CorrType', 'CorrProp', 'Batch', 'Dict']
    df = pd.read_csv('../abstr_exp/corrupted/corrupted_abstractions.csv', names=names)
    abstr_string = df.loc[(df['AbstrType'] == str(key[0]))
                        & (df['AbstrEps'] == key[1])
                        & (df['CorrType'] == str(key[2]))
                        & (df['CorrProp'] == key[3])
                        & (df['Batch'] == key[4])]['Dict'].values[0]
    abstr_list = ast.literal_eval(abstr_string)
    abstr_dict = {}
    for el in abstr_list:
        is_term = el[0][0] == 11 and el[0][1] == 11
        state = GridWorldState(el[0][0], el[0][1], is_terminal=is_term)
        abstr_dict[state] = el[1]

    # Create an agent with this abstraction
    s_a = StateAbstraction(abstr_dict, abstr_type=Abstr_type.PI_STAR)
    mdp = GridWorldMDP()
    agent = AbstractionAgent(mdp, s_a=s_a)

    # This is useful for later
    agent2 = copy.deepcopy(agent)

    # Generate a roll-out from a trained agent after 10000 steps
    for i in range(5000):
        agent.explore()
    rollout = agent.generate_rollout()
    print('Roll-out for model with no adjustment, 5,000 steps')
    for state in rollout:
        print(state, end=', ')
    for i in range(5000):
        agent.explore()
    rollout = agent.generate_rollout()
    print('Roll-out for model with no adjustment, 10,000 steps')
    for state in rollout:
        print(state, end=', ')
    print('\n')

    # Train an agent for 5000 steps, detach the first state in the cycle, and train for another 5000 steps
    #  The hope is that this will get further than the 10000 step one
    for i in range(5000):
        agent2.explore()
    rollout = agent2.generate_rollout()
    print('Roll-out for model pre-adjustment, 5,000 steps')
    for state in rollout:
        print(state, end=', ')
    print()
    print('Detaching state', rollout[-1])
    agent2.detach_state(rollout[-1])
    for i in range(5000):
        agent2.explore()
    rollout = agent2.generate_rollout()
    print('Roll-out for model post-adjustment, 10,000 steps')
    for state in rollout:
        print(state, end=', ')

# Check that string representation of agent's state abstraction is correct
def test_get_abstraction_as_string(agent):
    print(agent.get_abstraction_as_string())

# Test if iteratively detaching until a terminal roll-out produces non-trivial abstraction
def iterate_detachment(mdp_key, batch_size=5000):
    """
    Load an incorrect abstraction. Train the model, generate a roll-out, detach the first cycle state. Repeat until
    the roll-out achieves a terminal state. Save the adjusted abstraction and learned policy. Visualize the original
    incorrect abstraction with roll-outs from original agents and the adjusted abstraction with a roll-out from the
    new agent
    :param key: key for incorrect (poorly performing) abstraction
    :param batch_size: Number of steps to train between state detachments
    """
    # Load a poorly-performing abstraction
    names = ['AbstrType', 'AbstrEps', 'CorrType', 'CorrProp', 'Batch', 'Dict']
    df = pd.read_csv('../abstr_exp/corrupted/corrupted_abstractions.csv', names=names)
    abstr_string = df.loc[(df['AbstrType'] == str(mdp_key[0]))
                        & (df['AbstrEps'] == mdp_key[1])
                        & (df['CorrType'] == str(mdp_key[2]))
                        & (df['CorrProp'] == mdp_key[3])
                        & (df['Batch'] == mdp_key[4])]['Dict'].values[0]
    abstr_list = ast.literal_eval(abstr_string)
    abstr_dict = {}
    for el in abstr_list:
        is_term = el[0][0] == 11 and el[0][1] == 11
        state = GridWorldState(el[0][0], el[0][1], is_terminal=is_term)
        abstr_dict[state] = el[1]

    # Create an agent with this abstraction
    s_a = StateAbstraction(abstr_dict, abstr_type=Abstr_type.PI_STAR)
    mdp = GridWorldMDP()
    agent = AbstractionAgent(mdp, s_a=s_a)

    # Generate a roll-out from untrained model (should be random and short)
    rollout = agent.generate_rollout()
    print('Roll-out from untrained model')
    for state in rollout:
        print(state, end=', ')
    print()

    # Until roll-out leads to terminal state, explore and detach last state of roll-out. Record each of the detached
    #  states so they can be visualized later
    detached_states = []
    step_counter = 0
    while not rollout[-1].is_terminal():
        for i in range(batch_size):
            agent.explore()
        step_counter += batch_size
        rollout = agent.generate_rollout()
        print('Roll-out after', step_counter, 'steps')
        for state in rollout:
            print(state, end=', ')
        print()
        print('State Q-value pre-detach:')
        for action in agent.mdp.actions:
            print(rollout[-1], action, agent.get_q_value(rollout[-1], action))
        detach_flag = agent.detach_state(rollout[-1])
        if detach_flag == 0:
            print('Detaching state', rollout[-1])
            detached_states.append(rollout[-1])
        elif detach_flag == 1:
            print(rollout[-1], 'already a singleton state. No change.')
        print('State Q-value post-detach:')
        for action in agent.mdp.actions:
            print(rollout[-1], action, agent.get_q_value(rollout[-1], action))
        print()
    for key, value in agent.get_q_table():
        print(key, value)

    # Save resulting adapted state abstraction and learned policy
    s_a_file = open('../abstr_exp/adapted/adapted_abstraction.csv', 'w', newline='')
    s_a_writer = csv.writer(s_a_file)
    print(mdp_key)
    s_a_writer.writerow((mdp_key[0], mdp_key[1], mdp_key[2], mdp_key[3], mdp_key[4], agent.get_abstraction_as_string()))
    s_a_file.close()

    policy_file = open('../abstr_exp/adapted/learned_policy.csv', 'w', newline='')
    policy_writer = csv.writer(policy_file)
    policy_writer.writerow((mdp_key[0], mdp_key[1], mdp_key[2], mdp_key[3], mdp_key[4],
                            agent.get_learned_policy_as_string()))
    policy_file.close()

    # Visualize the adapted state abstraction and learned policy, along with the original for comparison
    viz = GridWorldVisualizer()
    surface = viz.create_corruption_visualization(mdp_key,
                                                  '../abstr_exp/adapted/adapted_abstraction.csv',
                                                  error_file='../abstr_exp/corrupted/error_states.csv')
    # Draw small white circles over the states that were detached
    for state in detached_states:
        print(state, end=', ')
    #for d_state in
    viz.display_surface(surface)

def test_get_ground_states_from_abstact_state():
    mdp = GridWorldMDP()
    abstr_mdp = mdp.make_abstr_mdp(Abstr_type.PI_STAR)
    agent = AbstractionAgent(mdp, s_a=abstr_mdp.state_abstr)

    for value in agent.s_a.abstr_dict.values():
        print(value, end= ' ')
        ground_states = agent.get_ground_states_from_abstract_state(value)
        for state in ground_states:
            print(state, end = ' ')
        print()

def test_check_for_optimal_action_and_value(states, num_steps):
    """
    Create a list of actions generated by following the greedy policy, starting at the given state
    """
    mdp = GridWorldMDP()
    abstr_mdp = mdp.make_abstr_mdp(Abstr_type.Q_STAR)
    agent = AbstractionAgent(mdp, s_a=abstr_mdp.state_abstr)
    for i in range(100000):
        if i % 1000 == 0:
            print('On step', i)
        agent.explore()

    # print(agent.get_learned_policy_as_string())
    policy = agent.get_learned_policy()
    #for key, value in agent.get_learned_policy_as_string().items():
    #    print(key, value, agent.get_q_value(key[0], key[1]))
    for s in agent.mdp.get_all_possible_states():
        #for a in agent.mdp.actions:
        print(s, agent.get_best_action_value(s))

    for state in states:
        mdp_state = GridWorldState(state[0], state[1])
        action, value = agent.check_for_optimal_action_value_next_state(mdp_state, verbose=True)
        print()

def test_check_abstract_state_consistency():
    mdp = GridWorldMDP()
    abstr_mdp = mdp.make_abstr_mdp(Abstr_type.A_STAR)
    agent = AbstractionAgent(mdp, s_a = abstr_mdp.state_abstr)
    for i in range(100000):
        if i % 1000 == 0:
            print('On step', i)
        agent.explore()
    # Get all abstract states
    abstr_states = []
    for value in agent.s_a.abstr_dict.values():
        abstr_states.append(value)
    abstr_states = agent.get_abstract_states()
    for abstr_state in abstr_states:
        agent.check_abstract_state_consistency(abstr_state, verbose=True)

def test_detach_inconsistent_states(abstr_type):
    mdp = GridWorldMDP()
    abstr_mdp = mdp.make_abstr_mdp(abstr_type)
    agent = AbstractionAgent(mdp, s_a=abstr_mdp.state_abstr)
    for i in range(1000000):
        if i % 1000 == 0:
            print('On step', i)
        agent.explore()
    error_states = agent.detach_inconsistent_states(verbose=True)

def test_update(agent):
    print('Abstr dict is', end = ' ')
    for key, value in agent.s_a.abstr_dict.items():
        print(key, value, end = ' ')
    print()
    print('Group dict is', end = ' ')
    for key, value in agent.group_dict.items():
        print(key, end = ' ')
        for val in value:
            print(val, end = ' ')
        print()
    for i in range(20):
        agent.explore()


if __name__ == '__main__':
    #mdp = GridWorldMDP()
    #abstr_mdp = mdp.make_abstr_mdp(Abstr_type.PI_STAR)
    mdp = TwoRoomsMDP(lower_height=2, lower_width=2, upper_width=2, upper_height=2, hallway_states=[2])
    abstr_mdp = mdp.make_abstr_mdp(mdp)
    agent = AbstractionAgent(mdp, s_a=abstr_mdp.state_abstr)
    test_update(agent)

    # Basic functions
    #test_detach_state(agent)
    #test_generate_rollout(agent)
    #test_get_abstraction_as_string(agent)
    #test_get_ground_states_from_abstact_state()

    # Roll-out functions
    #key = (Abstr_type.Q_STAR, 0.0, Corr_type.UNI_RAND, 0.05, 2)
    #test_rollout_adjustment(key)
    #iterate_detachment(key, batch_size=50000)

    # Roll-out from a particular state
    #states = [(1, 1), (2, 2), (3, 6), (6, 3), (8, 8), (10, 10)]
    #test_check_for_optimal_action_and_value(states, 2)

    #test_check_abstract_state_consistency()

    #test_detach_inconsistent_states(Abstr_type.Q_STAR)