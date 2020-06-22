'''
This tests the experiment class
'''

from Experiment.ExperimentClass import Experiment
from GridWorld.GridWorldMDPClass import GridWorldMDP
from Agent.AgentClass import Agent
from resources.AbstractionTypes import Abstr_type

def test_gridworld():
    mdp = GridWorldMDP()
    #abstr_epsilon_list = [(Abstr_type.Q_STAR, 0.0)]
    exp = Experiment(mdp, num_agents=2)#, abstr_epsilon_list)
    print(exp)
    for agent in exp.agents['ground']:
        print(agent.mdp)
        
    for i in range(10):
        exp.agents['ground'][0].explore()
    for agent in exp.agents['ground']:
        print(agent.mdp)
    #print(exp.agents['ground'])
    #actual_reward, optimal_reward = exp.run_trajectory(exp.agents['ground'][0])
    #print(actual_reward, optimal_reward)

if __name__ == '__main__':
    test_gridworld()