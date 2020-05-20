# Imports 
from GridWorldMDPClass import GridWorldMDP
from Functions import * 
from ActionEnums import Dir 
from AgentClass import Agent 

def main(): 
	grid_mdp = GridWorldMDP()
	init = grid_mdp.get_init_state()
	action = Dir.RIGHT
	state, reward = grid_mdp.act(init, action)
	print(state, reward)

if __name__ == '__main__':
	main()