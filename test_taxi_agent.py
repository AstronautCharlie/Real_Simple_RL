'''
This file tests the agent in the taxi environment
'''
from resources.AbstractionMakers import make_abstr
from resources.AbstractionTypes import Abstr_type
from GridWorld.TaxiMDPClass import TaxiMDP
from Agent.AgentClass import Agent

def print_action_values(val_list):
	'''
	print the given list of action-value pairs
	'''
	for pair in val_list:
		(action, value) = pair
		print("action : value =", action, ":", value)

def test_explore():
    # slip probability is 0 for testing purposes
    mdp = TaxiMDP(slip_prob = 0.0)
    agent = Agent(mdp)
    print(mdp)

    states_explored = [agent.get_current_state()]
    for i in range(100000):
        agent.explore()
        curr_state = agent.get_current_state()
        if curr_state not in states_explored:
            states_explored.append(curr_state)

    for state in states_explored:
        print("values learned at state", state)
        print_action_values(agent.get_action_values(state))

def test_abstractions():
    mdp = TaxiMDP()
    agent = Agent(mdp)
    print(mdp)

    states_explored = [agent.get_current_state()]
    for i in range(100000):
        agent.explore()
        curr_state = agent.get_current_state()
        if curr_state not in states_explored:
            states_explored.append(curr_state)

    for state in states_explored:
        print("values learned at state", state)
        print_action_values(agent.get_action_values(state))

    state_abstr = make_abstr(agent.get_q_table(), Abstr_type.A_STAR, epsilon = 1)
    print(state_abstr)

if __name__ == '__main__':
    #test_explore()
    test_abstractions()