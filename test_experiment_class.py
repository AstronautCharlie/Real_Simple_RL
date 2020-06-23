'''
This tests the experiment class
'''

from Experiment.ExperimentClass import Experiment
from GridWorld.GridWorldMDPClass import GridWorldMDP
from GridWorld.TaxiMDPClass import TaxiMDP
from Agent.AgentClass import Agent
from resources.AbstractionTypes import Abstr_type

def test_gridworld():
    #mdp = GridWorldMDP()
    mdp = TaxiMDP()


    abstr_epsilon_list = [(Abstr_type.Q_STAR, 0.0), (Abstr_type.A_STAR, 0.0), (Abstr_type.PI_STAR, 0.0)]
    exp = Experiment(mdp, num_agents=10, abstr_epsilon_list=abstr_epsilon_list)

    # Testing that one agent in an ensemble acting on its MDP won't affect another agent
    '''
    print(exp)
    for agent in exp.agents['ground']:
        print(agent.mdp)
    print()
    for i in range(100):
        exp.agents['ground'][0].explore()
    for agent in exp.agents['ground']:
        print(agent.mdp)
    '''


    # Testing run_trajectory
    '''
    for i in range(20):
        actual, optimal = exp.run_trajectory(exp.agents['ground'][0])
        print(actual, optimal)

    print('\n\n\n')

    for i in range(20):
        actual, optimal = exp.run_trajectory(exp.agents[(Abstr_type.PI_STAR, 0.0)][0])
        print(actual, optimal)
    '''

    # Testing run_ensemble
    #print(exp.run_ensemble(exp.agents[(Abstr_type.Q_STAR, 0.0)]))

    # Testing writing to file
    data, steps = exp.run_all_ensembles(num_episodes=100)

    # Testing plotting results
    exp.visualize_results(data, 'results/exp_graph.png')

    exp.visualize_results(steps, 'results/step_counts.png')

if __name__ == '__main__':
    test_gridworld()