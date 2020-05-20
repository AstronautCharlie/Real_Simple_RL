# Imports 
from GridWorld.GridWorldMDPClass import GridWorldMDP
from GridWorld.ActionEnums import Dir


def go_right(state):
	return Dir.RIGHT

def main(): 
	grid_mdp = GridWorldMDP()
	agent = Agent(grid_mdp, go_right)
	for i in range(5):
		agent.act()

if __name__ == '__main__':
	main()