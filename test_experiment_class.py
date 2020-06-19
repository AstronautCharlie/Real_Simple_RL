'''
This tests the experiment class
'''

from Experiment.ExperimentClass import Experiment
from GridWorld.GridWorldMDPClass import GridWorldMDP
from Agent.AgentClass import Agent
from resources.AbstractionTypes import Abstr_type

def test_gridworld():
    mdp = GridWorldMDP()
    abstr_epsilon_list = [(Abstr_type.Q_STAR, 0.0)]
    exp = Experiment(mdp, abstr_epsilon_list)
    print(exp)

if __name__ == '__main__':
    test_gridworld()