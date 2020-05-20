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
	agent = Agent(grid_mdp, go_up_right)
	for i in range(30):
		agent.act()
	print(grid_mdp.walls)

if __name__ == '__main__':
	main()