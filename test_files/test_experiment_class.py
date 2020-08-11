'''
This tests the experiment class
'''

from Experiment.ExperimentClass import Experiment
from GridWorld.GridWorldMDPClass import GridWorldMDP
from GridWorld.TaxiMDPClass import TaxiMDP
from Agent.AgentClass import Agent
from resources.AbstractionTypes import Abstr_type

def test_gridworld():
    mdp = GridWorldMDP()
    mdp_str = 'rooms'
    #mdp = TaxiMDP()
    #mdp_str = 'taxi'

    eps = 0.0

    abstr_epsilon_list = [(Abstr_type.Q_STAR, eps), (Abstr_type.A_STAR, eps), (Abstr_type.PI_STAR, eps)]
    #abstr_epsilon_list = [(Abstr_type.A_STAR, eps), (Abstr_type.PI_STAR, eps)]
    exp = Experiment(mdp, num_agents=20, abstr_epsilon_list=abstr_epsilon_list)

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
    data, steps = exp.run_all_ensembles(num_episodes=500)

    # Testing plotting results
    exp.visualize_results(data, 'results/exp_graph_' + mdp_str + '_' + str(eps) + '.png')

    exp.visualize_results(steps, 'results/step_counts_' + mdp_str + '_' + str(eps) + '.png')

if __name__ == '__main__':
    test_gridworld()