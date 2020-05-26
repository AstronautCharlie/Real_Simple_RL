from Visualizer.GridWorldVisualizer import GridWorldVisualizer
from GridWorld.GridWorldMDPClass import GridWorldMDP
from Agent.AgentClass import Agent
from GridWorld.GridWorldStateClass import GridWorldState
from GridWorld.ActionEnums import Dir

grid_mdp = GridWorldMDP(height=11, width=11, slip_prob=0.0, gamma=0.95, build_walls=True)
agent = Agent(grid_mdp)
agent.set_current_state(GridWorldState(1,1))

g_viz = GridWorldVisualizer(grid_mdp,agent)
#g_viz.displayGridWorldMDPWithAction(Dir.LEFT)
g_viz.visualizeExploration(10000)




