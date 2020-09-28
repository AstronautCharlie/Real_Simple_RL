"""
Test that all the functionality in AbstractionAgent that is not in the Agent class works as expected
"""

from Agent.AbstractionAgent import AbstractionAgent
from Agent.AgentClass import Agent
from GridWorld.GridWorldMDPClass import GridWorldMDP
from GridWorld.GridWorldStateClass import GridWorldState
from resources.AbstractionTypes import Abstr_type
from resources.AbstractionCorrupters import Corr_type
from MDP.StateAbstractionClass import StateAbstraction
from Visualizer.GridWorldVisualizer import GridWorldVisualizer

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
    for i in range(5000):
        agent.explore()
    print()
    print('Q-value of state after exploring: (should be non-zero)')
    for i in range(len(agent.mdp.actions)):
        print(agent.mdp.actions[i], agent.get_q_value(state_to_remove, agent.mdp.actions[i]))
    print()
    agent.detach_state(state_to_remove)

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
def iterate_detachment(key, batch_size=5000):
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
    for key, value in agent.s_a.abstr_dict.items():
        print(key, value)

    # Generate a roll-out from untrained model (should be random and short)
    rollout = agent.generate_rollout()
    print('Roll-out from untrained model')
    for state in rollout:
        print(state, end=', ')
    print()

    # Until rollout leads to terminal state, explore and detach last state of rollout
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
        detach_flag = agent.detach_state(rollout[-1])
        if detach_flag == 0:
            print('Detaching state', rollout[-1])
        elif detach_flag == 1:
            print(rollout[-1], 'already a singleton state. No change.')
        print()

    # Save resulting adapted state abstraction and learned policy
    s_a_file = open('abstr_exp/adapted/adapted_abstraction.csv', 'a')
    s_a_writer = csv.writer(s_a_file)
    s_a_writer.writerow((key[0], key[1], key[2], key[3], key[4], agent.get_abstraction_as_string()))

    policy_file = open('abstr_exp/adapted/learned_policy.csv', 'a')
    policy_writer = csv.writer(policy_file)
    policy_writer.writerow((key[0], key[1], key[2], key[3], key[4], agent.get_learned_policy_as_string()))

    # Visualize the adapted state abstraction and learned policy, along with the original for comparison
    viz =

if __name__ == '__main__':
    mdp = GridWorldMDP()
    abstr_mdp = mdp.make_abstr_mdp(Abstr_type.PI_STAR)
    agent = AbstractionAgent(mdp, s_a=abstr_mdp.state_abstr)

    # Basic functions
    #test_detach_state(agent)
    #test_generate_rollout(agent)
    #test_get_abstraction_as_string(agent)

    # Roll-out functions
    key = (Abstr_type.PI_STAR, 0.0, Corr_type.UNI_RAND, 0.05, 2)
    #test_rollout_adjustment(key)
    #iterate_detachment(key, batch_size=10000)