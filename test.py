# Imports 
from GridWorld.GridWorldMDPClass import GridWorldMDP
from GridWorld.ActionEnums import Dir
from GridWorld.GridWorldStateClass import GridWorldState
from GridWorld.AbstractGridWorldMDPClass import AbstractGridWorldMDP
from MDP.StateAbstractionClass import StateAbstraction 
from Agent.AgentClass import Agent 
from resources.AbstractionMakers import make_abstr
from resources.AbstractionTypes import Abstr_type
import random 
import numpy as np 

# -----------------------------
# Functions defined for testing
# -----------------------------

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


# ---------
# Main test 
# ---------

def main():

	# Testing what a Q* abstraction looks like in 
	# four rooms

	# Make MDP and train an agent in it
	grid_mdp = GridWorldMDP(height=9, width=9, slip_prob=0.0, gamma=0.99)
	agent = Agent(grid_mdp)
	
	# Train the agent for 10000 steps 
	trajectory = [] 
	for i in range(100000):
		if i % 1000 == 0:
			print("epsilon, alpha:", agent._epsilon, agent._alpha)
		current_state, action, next_state, _ = agent.explore()
		trajectory.append(current_state)

	already_printed = [] 
	for state in trajectory:
		if state not in already_printed:
			already_printed.append(state)

	# Print the action values learned at each state 
	for state in already_printed:
		print("values learned at state", state)
		print_action_values(agent.get_action_values(state))
		print()

	# Make an abstraction from the agent's q-table 
	state_abstr = make_abstr(agent.get_q_table(), Abstr_type.Q_STAR, epsilon=0.05)
	print(state_abstr) 

	# Testing that Pi* abstraction works 
	'''
	# Create toy q_table to build abstraction from 
	q_table = {(GridWorldState(1,1), Dir.UP): 0.9,
				(GridWorldState(1,1), Dir.DOWN): 0.8,
				(GridWorldState(1,1), Dir.LEFT): 0.7,
				(GridWorldState(1,1), Dir.RIGHT): 0.6,

				# Same optimal action and action value as (1,1)
				(GridWorldState(1,2), Dir.UP): 0.9,
				(GridWorldState(1,2), Dir.DOWN): 0.0,
				(GridWorldState(1,2), Dir.LEFT): 0.2,
				(GridWorldState(1,2), Dir.RIGHT): 0.5,

				# val(UP) = 0.9 but val(DOWN) = 0.91
				(GridWorldState(2,2), Dir.UP): 0.9,
				(GridWorldState(2,2), Dir.DOWN): 0.91,
				(GridWorldState(2,2), Dir.LEFT): 0.8,
				(GridWorldState(2,2), Dir.RIGHT): 0.9,

				# val(UP) = 0.89, max val
				(GridWorldState(2,1), Dir.UP): 0.9,
				(GridWorldState(2,1), Dir.DOWN): 0.9,
				(GridWorldState(2,1), Dir.LEFT): 0.90000000001,
				(GridWorldState(2,1), Dir.RIGHT): 0.7,

				# val(UP) = 0.93, max val 
				(GridWorldState(3,1), Dir.UP): 1000,
				(GridWorldState(3,1), Dir.DOWN): 0.89,
				(GridWorldState(3,1), Dir.LEFT): 0.89,
				(GridWorldState(3,1), Dir.RIGHT): 0.89}
	
	state_abstr = make_abstr(q_table, Abstr_type.PI_STAR)
	print("(1,1), (1,2), and (3,1) should all get mapped together")
	print(state_abstr)
	'''


	# Testing that A* abstraction works
	'''
	# Create toy q_table to build abstraction from 
				# Optimal action/val is UP/0.9
	q_table = {(GridWorldState(1,1), Dir.UP): 0.9,
				(GridWorldState(1,1), Dir.DOWN): 0.8,
				(GridWorldState(1,1), Dir.LEFT): 0.7,
				(GridWorldState(1,1), Dir.RIGHT): 0.6,

				# Same optimal action and action value as (1,1)
				(GridWorldState(1,2), Dir.UP): 0.9,
				(GridWorldState(1,2), Dir.DOWN): 0.0,
				(GridWorldState(1,2), Dir.LEFT): 0.2,
				(GridWorldState(1,2), Dir.RIGHT): 0.5,

				# val(UP) = 0.9 but val(DOWN) = 0.91
				(GridWorldState(2,2), Dir.UP): 0.9,
				(GridWorldState(2,2), Dir.DOWN): 0.91,
				(GridWorldState(2,2), Dir.LEFT): 0.8,
				(GridWorldState(2,2), Dir.RIGHT): 0.9,

				# val(UP) = 0.89, max val
				(GridWorldState(2,1), Dir.UP): 0.89,
				(GridWorldState(2,1), Dir.DOWN): 0.88,
				(GridWorldState(2,1), Dir.LEFT): 0.8,
				(GridWorldState(2,1), Dir.RIGHT): 0.7,

				# val(UP) = 0.93, max val 
				(GridWorldState(3,1), Dir.UP): 0.93,
				(GridWorldState(3,1), Dir.DOWN): 0.89,
				(GridWorldState(3,1), Dir.LEFT): 0.89,
				(GridWorldState(3,1), Dir.RIGHT): 0.89}
	state_abstr = make_abstr(q_table, Abstr_type.A_STAR)
	print("Epsilon = 0. (1,1) and (1,2) should be mapped together")
	print(state_abstr)

	state_abstr = make_abstr(q_table, Abstr_type.A_STAR, epsilon=0.015)
	print("Epsilon = 0.015. (1,1), (1,2), and (2,1) should all be mapped together")
	print(state_abstr)

	state_abstr = make_abstr(q_table, Abstr_type.A_STAR, epsilon=0.031)
	print("Epsilon = 0.031. (1,1), (1,2), (2,1), (3,1) should all be mapped together")
	print(state_abstr)
	'''


	# Testing that Q* abstraction function works
	'''
	# Create toy q_table to build the abstraction from
	q_table = {(GridWorldState(1,1), Dir.UP): 1.0,
				(GridWorldState(1,1), Dir.DOWN): 2.5,
				(GridWorldState(1,1), Dir.LEFT): 2.3,
				(GridWorldState(1,1), Dir.RIGHT): 5.0,

				(GridWorldState(2,1), Dir.UP): 1.0,
				(GridWorldState(2,1), Dir.DOWN): 2.5,
				(GridWorldState(2,1), Dir.LEFT): 2.3,
				(GridWorldState(2,1), Dir.RIGHT): 5.05,

				(GridWorldState(2,2), Dir.UP): 1.1,
				(GridWorldState(2,2), Dir.DOWN): 2.4,
				(GridWorldState(2,2), Dir.LEFT): 2.3,
				(GridWorldState(2,2), Dir.RIGHT): 4.8,

				(GridWorldState(1,2), Dir.UP): 1.3,
				(GridWorldState(1,2), Dir.DOWN): 2.0,
				(GridWorldState(1,2), Dir.LEFT): 2.0,
				(GridWorldState(1,2), Dir.RIGHT): 4.8
				}
	state_abstr = make_abstr(q_table, Abstr_type.Q_STAR)
	print("Epsilon = 0. No shapes should be mapped together.")
	print(str(state_abstr))

	state_abstr = make_abstr(q_table, Abstr_type.Q_STAR, epsilon=0.3)
	print("Epsilon = 0.3. (1,1), (2,1), (2,2) should all be mapped together")
	print(str(state_abstr))

	state_abstr = make_abstr(q_table, Abstr_type.Q_STAR, epsilon=0.1)
	print("Epsilon = 0.1. (1,1), (2,1) should be mapped together. (2,2) should not.")
	print(str(state_abstr))

	state_abstr = make_abstr(q_table, Abstr_type.Q_STAR, epsilon=0.5)
	print("Epsilon = 0.5. (1,1), (2,1), (1,2), (2,2) should all be mapped together")
	print(str(state_abstr))
	'''


	# Testing Q-learning in abstract Four Rooms 
	'''
	# Map all the states in the bottom-right room to the same abstract state 
	abstr_dict = {} 
	for i in range(6,12):
		for j in range(1,6):
			abstr_dict[GridWorldState(i,j)] = 'oneroom'

	state_abstr = StateAbstraction(abstr_dict)

	abstr_mdp = AbstractGridWorldMDP(height=11, 
										width=11,
										slip_prob=0.0,
										gamma=0.95,
										build_walls=True,
										state_abstr=state_abstr)
	agent = Agent(abstr_mdp)

	trajectory = [] 
	for i in range(100000):
		#print("At step", i)
		#print("parameters are", agent._alpha, agent.mdp.gamma)
		current_state, action, next_state, _ = agent.explore()
		#print("At", str(current_state), "took action", action, "got to", str(next_state))
		#print("Values learned for", str(current_state), "is")
		#print_action_values(agent.get_action_values(current_state))
		trajectory.append(current_state)
		#print()

	already_printed = [] 
	for state in trajectory:
		if state not in already_printed:
			print("values learned at state", state)
			print_action_values(agent.get_action_values(state))
			already_printed.append(state)

	agent.reset_to_init()
	for i in range(25):
		current_state, action, next_state = agent.apply_best_action()
		print('At', str(current_state), 'taking action', str(action), 'now at', str(next_state))
	'''


	# Testing Q-learning in toy abstract MDP
	'''
	# Simple abstraction in a grid where all states above the start-to-goal
	# diagonal are grouped together and all states below that diagonal
	# are grouped together 
	toy_abstr = StateAbstraction({GridWorldState(2,1): 'up', 
									GridWorldState(3,1): 'up',
									GridWorldState(3,2): 'up',
									GridWorldState(4,1): 'up',
									GridWorldState(4,2): 'up',
									GridWorldState(4,3): 'up',
									GridWorldState(5,1): 'up',
									GridWorldState(5,2): 'up',
									GridWorldState(5,3): 'up',
									GridWorldState(5,4): 'up',
									GridWorldState(1,2): 'right',
									GridWorldState(1,3): 'right',
									GridWorldState(1,4): 'right',
									GridWorldState(1,5): 'right',
									GridWorldState(2,3): 'right',
									GridWorldState(2,4): 'right',
									GridWorldState(2,5): 'right',
									GridWorldState(3,4): 'right',
									GridWorldState(3,5): 'right',
									GridWorldState(4,5): 'right'})
	#print("states covered by abstraction are", toy_abstr.abstr_dict.keys())
	

	abstr_mdp = AbstractGridWorldMDP(height=5, 
							width=5, 
							slip_prob=0.0, 
							gamma=0.95, 
							build_walls=False,
							state_abstr=toy_abstr)

	#print(abstr_mdp.state_abstr.get_abstr_from_ground(GridWorldState(1,1)))
	agent = Agent(abstr_mdp)
	
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

	already_printed = [] 
	for state in trajectory:
		if state not in already_printed:
			print("values learned at state", state)
			print_action_values(agent.get_action_values(state))
			already_printed.append(state)
	'''	

	# Testing both epsilon-greedy and application of best learned 
	# policy in ground MDP 
	'''
	grid_mdp = GridWorldMDP(height=9, width=9, slip_prob=0.0, gamma=0.95, build_walls=True)

	agent = Agent(grid_mdp)
	#agent.set_current_state(GridWorldState(1,1))

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
			print("values learned at state", state)
			print_action_values(agent.get_action_values(state))
			already_printed.append(state)
	#print(grid_mdp.walls)

	agent.reset_to_init()

	for i in range(25):
		current_state, action, next_state = agent.apply_best_action()
		print('At', str(current_state), 'taking action', str(action), 'now at', str(next_state))
	'''

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