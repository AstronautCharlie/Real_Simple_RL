from Visualizer.AbstractGridWorldVisualizer import AbstractGridWorldVisualizer
from GridWorld.GridWorldMDPClass import GridWorldMDP
from GridWorld.AbstractGridWorldMDPClass import AbstractGridWorldMDP
from Agent.AgentClass import Agent
from resources.AbstractionMakers import make_abstr
from resources.AbstractionTypes import Abstr_type
from GridWorld.GridWorldStateClass import GridWorldState

grid_mdp = GridWorldMDP(height=11, width=11, slip_prob=0.0, gamma=0.95, build_walls=True)
agent = Agent(grid_mdp)
for i in range(100000):
    if i % 1000 == 0:
        print("epsilon, alpha:", agent._epsilon, agent._alpha)
    current_state, action, next_state, _ = agent.explore()
state_abstr = make_abstr(agent.get_q_table(), Abstr_type.Q_STAR, epsilon=0.05)
print(state_abstr.abstr_dict[GridWorldState(1,1)])
# abstr_grid_mdp = AbstractGridWorldMDP(state_abstr=state_abstr)
# print(abstr_grid_mdp.get_abstr_from_ground(GridWorldState(1,1)))
#

# abs_g_viz = AbstractGridWorldVisualizer(abstr_grid_mdp,agent)
# abs_g_viz.displayAbstractMDP()