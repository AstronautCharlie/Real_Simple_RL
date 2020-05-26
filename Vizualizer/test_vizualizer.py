from  Vizualizer.GridWorldVisualizer import displayGridWorldMDPWithAgent, visualizeExploration
from GridWorld.GridWorldMDPClass import GridWorldMDP
from Agent.AgentClass import Agent
from GridWorld.GridWorldStateClass import GridWorldState

grid_mdp = GridWorldMDP(height=11, width=11, slip_prob=0.0, gamma=0.95, build_walls=True)
agent = Agent(grid_mdp)
agent.set_current_state(GridWorldState(1,1))
#displayGridWorldMDPWithAgent(grid_mdp,agent)

visualizeExploration(grid_mdp,agent,10000)


