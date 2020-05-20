# Imports 
from GridWorldMDPClass import GridWorldMDP
from Functions import * 
from ActionEnums import Dir 
from AgentClass import Agent 

def go_right(state):
	return Dir.RIGHT

def main(): 
	grid_mdp = GridWorldMDP()
	agent = Agent(grid_mdp, go_right)
	for i in range(5):
		agent.act()

if __name__ == '__main__':
	main()