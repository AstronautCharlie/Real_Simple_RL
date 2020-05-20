# Imports 
from GridWorld.GridWorldMDPClass import GridWorldMDP
from GridWorld.ActionEnums import Dir


def main(): 
	grid_mdp = GridWorldMDP()
	init = grid_mdp.get_init_state()
	action = Dir.RIGHT
	state, reward = grid_mdp.act(init, action)
	print(state, reward)

if __name__ == '__main__':
	main()