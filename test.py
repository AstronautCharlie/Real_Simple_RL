# Imports 
from GridWorld.GridWorldMDPClass import GridWorldMDP
from GridWorld.ActionEnums import Dir
from GridWorld.GridWorldStateClass import GridWorldState
from Agent.AgentClass import Agent 
import random 
import numpy as np 


def go_right(state):
	return Dir.RIGHT

def go_left(state):
	return Dir.LEFT 

def go_up(state):
	return Dir.UP

def go_down(state):
	return Dir.DOWN 

def go_up_right(state):
	if random.random() < 0.5:
		return Dir.UP
	else: 
		return Dir.RIGHT

def print_action_values(val_list):
	'''
	print the given list of action-value pairs 
	'''
	for pair in val_list: 
		(action, value) = pair 
		print("action : value =", action, ":", value)

def apply_trajectory(agent, action_list):
	'''
	Apply the given trajectory to the given agent. Then print out all the 
	state-action values for all states passed through in the trajectory
	'''
	states_accessed = [] 
	print('Applying trajectory', action_list)
	for action in action_list:
		states_accessed.append(agent.get_current_state())
		agent.apply_action(action)

	print("Completed trajectory. Values learned are below")
	for state in states_accessed:
		print("Values at state =", state)
		print_action_values(agent.get_action_values(state))
		print()




def main():

	grid_mdp = GridWorldMDP(height=9, width=9, slip_prob=0.0, gamma=0.95, build_walls=True)

	agent = Agent(grid_mdp)
	agent.set_current_state(GridWorldState(1,1))

	print(grid_mdp.goal_location)

	# Testing if epsilon-greedy policy works properly 
	trajectory = [] 
	for i in range(10000):
		#print("At step", i)
		#print("parameters are", agent._alpha, agent.mdp.gamma)
		current_state, action, next_state, _ = agent.explore()
		#print("At", str(current_state), "took action", action, "got to", str(next_state))
		#print("Values learned for", str(current_state), "is")
		#print_action_values(agent.get_action_values(current_state))
		trajectory.append(current_state)
		#print()

	#print("Went through the following states:")
	#for state in trajectory:
	#	print(str(state))
	already_printed = [] 
	for state in trajectory:
		if state not in already_printed:
			#print("values learned at state", state)
			#print_action_values(agent.get_action_values(state))
			already_printed.append(state)
	#print(grid_mdp.walls)

	agent.reset_to_init()

	for i in range(25):
		current_state, action, next_state = agent.apply_best_action()
		print('At', str(current_state), 'taking action', str(action), 'now at', str(next_state))

	# Testing a few trajectories to make sure the q-table updates
	# properly 
	'''
	test_trajectory = [Dir.UP, Dir.RIGHT, Dir.UP, Dir.RIGHT]
	for i in range(5):
		apply_trajectory(agent, test_trajectory)
		agent.set_current_state(GridWorldState(9,9))

	test_trajectory = [Dir.RIGHT, Dir.RIGHT, Dir.UP, Dir.UP]
	apply_trajectory(agent, test_trajectory)
	agent.set_current_state(GridWorldState(9,9))

	test_trajectory = [Dir.UP, Dir.UP, Dir.RIGHT, Dir.RIGHT]
	apply_trajectory(agent, test_trajectory)
	'''
	
	# Testing motion, reward at goal state, and reset to 
	# initial state at terminal state 
	'''
	agent = Agent(grid_mdp, go_up_right)
	for i in range(30):
		agent.act()
	print(grid_mdp.walls)
	'''

	# Testing getter for best action/value given state 
	'''
	agent = Agent(grid_mdp, go_right, alpha=0.5)
	current_state = agent.get_current_state() 
	test_action = Dir.UP

	# Set q_value for init_state, Dir.UP = 1.0
	agent._set_q_value(current_state, test_action, 1.0)

	# Should give Dir.UP, 1.0 
	print("should give (Dir.UP, 1.0)", agent.get_best_action_value_pair(current_state))

	# Go right by one 
	agent.act()
	print("Currently at", agent.get_current_state())
	# Should give random action with value = 0 
	print("Should give (random_action, 0.0)", agent.get_best_action_value_pair(agent.get_current_state()))
	# Update q-values of this state
	agent._set_q_value(agent.get_current_state(), Dir.UP, -1.0)
	agent._set_q_value(agent.get_current_state(), Dir.DOWN, -1.0)
	agent._set_q_value(agent.get_current_state(), Dir.LEFT, -1.0)
	agent._set_q_value(agent.get_current_state(), Dir.RIGHT, 0.1)
	# Should give Dir.RIGHT, 0.1
	print("Should give (Dir.RIGHT, 0.1)", agent.get_best_action_value_pair(agent.get_current_state()))

	print()
	# Checking that all values were updated properly
	for action in agent.mdp.actions:
		print("action:q-value = ", action, ":", agent.get_q_value(agent.get_current_state(), action))
	'''

	# Testing single instance of the act, update flow
	# Start agent at (10,11), go one right, get reward, 
	# check that update happened 
	'''
	agent = Agent(grid_mdp, go_right)
	init_state = GridWorldState(10,11)
	agent.set_current_state(init_state)
	agent.act()
	action_values = agent.get_action_values(init_state)
	for pair in action_values:
		print(pair)
	print() 

	# Reset agent to (9,11), act twice, then check 
	# value of each of the two states that the agent
	# passed through 
	init_state = GridWorldState(9,11)
	agent.set_current_state(init_state)
	# Go right twice 
	agent.act()
	agent.act() 
	# Check that state-action values for (9,11) and 
	# (10,11) got updated 
	print("Action-value pairs for (9,11)")
	action_values = agent.get_action_values(init_state)
	for pair in action_values:
		print(pair)
	print()

	print("Action-value pairs for (10,11)")
	action_values = agent.get_action_values(GridWorldState(10,11))
	for pair in action_values:
		print(pair)
	'''

if __name__ == '__main__':
	main()