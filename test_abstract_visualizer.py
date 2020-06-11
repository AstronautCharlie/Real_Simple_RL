from Visualizer.AbstractGridWorldVisualizer import AbstractGridWorldVisualizer
from GridWorld.GridWorldMDPClass import GridWorldMDP
from GridWorld.AbstractGridWorldMDPClass import AbstractGridWorldMDP
from Agent.AgentClass import Agent
from resources.AbstractionMakers import make_abstr
from resources.AbstractionTypes import Abstr_type
from GridWorld.GridWorldStateClass import GridWorldState
from MDP.ValueIterationClass import ValueIteration

# grid_mdp = GridWorldMDP(height=11, width=11, slip_prob=0.0, gamma=0.95, build_walls=True)
# # agent = Agent(grid_mdp)
# # for i in range(1000000):
# #     if i % 1000 == 0:
# #         print("epsilon, alpha:", agent._epsilon, agent._alpha)
# #     current_state, action, next_state, _ = agent.explore()
# # state_abstr = make_abstr(agent.get_q_table(), Abstr_type.Q_STAR, epsilon=0.05)
# # abstr_grid_mdp = AbstractGridWorldMDP(state_abstr=state_abstr)
# # abs_agent = Agent(abstr_grid_mdp)
# # abs_g_viz = AbstractGridWorldVisualizer(abstr_grid_mdp,abs_agent)
# # #abs_g_viz.displayAbstractMDP()
# # for i in range(100000):
# #     if i % 1000 == 0:
# #         print("epsilon, alpha:", abs_agent._epsilon, abs_agent._alpha)
# #     current_state, action, next_state,_  = abs_agent.explore()
# #
# # abs_g_viz.visualizeLearnedPolicy()


#Q-STAR - USING VI
# mdp = GridWorldMDP(slip_prob=0, gamma=0.99)
# vi = ValueIteration(mdp)
# vi.doValueIteration()
# q_table = vi.get_q_table()
# q_star_abstr = make_abstr(q_table, Abstr_type.Q_STAR, epsilon=0.01)
# abstr_grid_mdp = AbstractGridWorldMDP(state_abstr=q_star_abstr)
# abs_agent = Agent(abstr_grid_mdp)
# abs_g_viz = AbstractGridWorldVisualizer(abstr_grid_mdp,abs_agent)
# #abs_g_viz.displayAbstractMDP()
# for i in range(100000):
#     if i % 1000 == 0:
#         print("epsilon, alpha:", abs_agent._epsilon, abs_agent._alpha)
#     current_state, action, next_state,_  = abs_agent.explore()
#
# abs_g_viz.visualizeLearnedPolicy()

#A-STAR - USING VI
# mdp = GridWorldMDP(slip_prob=0, gamma=0.99)
# vi = ValueIteration(mdp)
# vi.doValueIteration()
# q_table = vi.get_q_table()
# a_star_abstr = make_abstr(q_table, Abstr_type.A_STAR, epsilon=0.01)
# abstr_grid_mdp = AbstractGridWorldMDP(state_abstr=a_star_abstr)
# abs_agent = Agent(abstr_grid_mdp)
# abs_g_viz = AbstractGridWorldVisualizer(abstr_grid_mdp,abs_agent)
# # abs_g_viz.displayAbstractMDP()
# for i in range(100000):
#     if i % 1000 == 0:
#         print("epsilon, alpha:", abs_agent._epsilon, abs_agent._alpha)
#     current_state, action, next_state,_  = abs_agent.explore()
#
# abs_g_viz.visualizeLearnedPolicy()



#PI-STAR - USING VI
mdp = GridWorldMDP(slip_prob=0, gamma=0.99)
vi = ValueIteration(mdp)
vi.doValueIteration()
q_table = vi.get_q_table()
pi_star_abstr = make_abstr(q_table, Abstr_type.PI_STAR, epsilon=0.01)
abstr_grid_mdp = AbstractGridWorldMDP(state_abstr=pi_star_abstr)
abs_agent = Agent(abstr_grid_mdp)
abs_g_viz = AbstractGridWorldVisualizer(abstr_grid_mdp,abs_agent)
# abs_g_viz.displayAbstractMDP()
for i in range(100000):
    if i % 1000 == 0:
        print("epsilon, alpha:", abs_agent._epsilon, abs_agent._alpha)
    current_state, action, next_state,_  = abs_agent.explore()

abs_g_viz.visualizeLearnedPolicy()
