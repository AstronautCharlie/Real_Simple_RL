# Imports 
from GridWorld.GridWorldMDPClass import GridWorldMDP
from GridWorld.ActionEnums import Dir
from Agent.AgentClass import Agent 
import random 


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

def main(): 

	grid_mdp = GridWorldMDP(slip_prob=0.0)
	
	# Testing motion, reward at goal state, and reset to 
	# initial state at terminal state 
	'''
	agent = Agent(grid_mdp, go_up_right)
	for i in range(30):
		agent.act()
	print(grid_mdp.walls)
	'''

	# Testing getter for best action/value given state 
	agent = Agent(grid_mdp, None, alpha=0.5)
	current_state = agent.get_current_state() 
	test_action = Dir.UP

	# Set q_value for init_state, Dir.UP = 1.0
	key = tuple([current_state, test_action])
	#print(type(key)) 
	agent.q_table[key] = 1.0

	# Should give Dir.UP, 1.0 
	print(agent.get_best_action_value_pair(current_state))

if __name__ == '__main__':
	main()