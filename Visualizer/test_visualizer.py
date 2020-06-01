from Visualizer.GridWorldVisualizer import GridWorldVisualizer
from GridWorld.GridWorldMDPClass import GridWorldMDP
from Agent.AgentClass import Agent
from GridWorld.GridWorldStateClass import GridWorldState
from GridWorld.ActionEnums import Dir

grid_mdp = GridWorldMDP(height=11, width=11, slip_prob=0.0, gamma=0.95, build_walls=True)
agent = Agent(grid_mdp)
grid_mdp.set_current_state(GridWorldState(1,1))

g_viz = GridWorldVisualizer(grid_mdp,agent)
#g_viz.displayGridWorldMDPWithAgent()
# for i in range(100000):
#     if i % 1000 == 0:
#         print("epsilon, alpha:", agent._epsilon, agent._alpha)
#     current_state, action, next_state,_  = agent.explore()
#     if i > 90000:
#         if current_state==(GridWorldState(1,1)):
#             print(action)





g_viz.visualizeExploration(100000)
g_viz.visualizeLearnedPolicy(200)




