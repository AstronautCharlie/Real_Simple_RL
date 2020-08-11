"""
Test that the Agent's make_abstraction function works
"""

from GridWorld.GridWorldMDPClass import GridWorldMDP
from GridWorld.GridWorldStateClass import GridWorldState
from Agent.AgentClass import Agent
from resources.AbstractionTypes import Abstr_type

import numpy as np

EP_COUNT = 35
PARAM_CUT = 1
TRIAL_COUNT = 50
ABSTR_TYPE = Abstr_type.Q_STAR
THRESHOLD=0.05
IGNORE_ZEROES = True

def test_agent_abstraction():

    # Make agent and MDP
    mdp = GridWorldMDP()
    agent = Agent(mdp)

    # Run for some number of steps
    while agent._episode_counter < EP_COUNT:
        agent.explore()

    # Make a new abstraction based on the learned q-table
    num_abstr_states, num_reduced_ground_states = agent.make_abstraction(Abstr_type.PI_STAR, epsilon=THRESHOLD, ignore_zeroes=IGNORE_ZEROES)
    agent._epsilon = agent._epsilon / PARAM_CUT

    # Count the number of abstract states so we can see how much this is changing
    key_count = 0
    ground_key_count = 0
    for key in agent._q_table.keys():
        key_count += 1
        #if isinstance(key[0], TaxiState):
        if isinstance(key[0], GridWorldState):
            ground_key_count += 1
    #print(key_count - ground_key_count)

    # Print the state abstraction so we can see what's up
    #for key, value in agent.mdp.state_abstr.abstr_dict.items():
    #    print(key, value)

    #for key, value in agent._q_table.items():
    #    print(key[0], key[1], value)

    # Run for an episode to see what the reward is
    curr_state = agent.get_current_state()
    cumu_reward = 0
    discount = 1
    while not curr_state.is_terminal():
        state, action, next_state, reward = agent.explore()
        cumu_reward += reward * discount
        discount *= agent.mdp.gamma
        curr_state = next_state

    #print("Test", cumu_reward, end=' ')


    # Control agent, no abstraction performed
    mdp_control = GridWorldMDP()
    agent_control = Agent(mdp_control)

    # Train the agent for the same number of episodes
    curr_state = agent_control.mdp.get_current_state()
    while agent_control._episode_counter < EP_COUNT:
        agent_control.explore()

    # Run control agent for an episode to see what the reward is
    curr_state = agent_control.get_current_state()
    control_cumu_reward = 0
    discount = 1
    while not curr_state.is_terminal():
        state, action, next_state, reward = agent_control.explore()
        control_cumu_reward += reward * discount
        discount *= agent_control.mdp.gamma
        curr_state = next_state
    #print("Control", cumu_reward)
    #print("Delta", cumu_reward - control_cumu_reward)
    return cumu_reward, control_cumu_reward, key_count - ground_key_count, num_reduced_ground_states - num_abstr_states



if __name__ == '__main__':
    test_rewards = []
    control_rewards = []
    reduced_state_counter = []
    #for i in range(50):
    while len(test_rewards) < TRIAL_COUNT:
        print(len(test_rewards))
        test_reward, control_reward, num_abstr_states, reduced_states = test_agent_abstraction()
        if num_abstr_states > 0:
            test_rewards.append(test_reward)
            control_rewards.append(control_reward)
            reduced_state_counter.append(reduced_states)
    #print(len(deltas))
    print("++++++\nRESULTS\n++++++")
    test_avg = sum(test_rewards) / len(test_rewards)
    test_std = np.std(test_rewards)
    control_avg = sum(control_rewards) / len(control_rewards)
    control_std = np.std(control_rewards)

    print("avg, stdev of test rewards:", test_avg, test_std)
    print("avg, stdev of control rewards:", control_avg, control_std)
    print("delta b/w averages:", test_avg - control_avg)
    print("avg num states reduced", sum(reduced_state_counter) / len(reduced_state_counter))