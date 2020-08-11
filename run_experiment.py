"""
Run this to conduct an experiment
"""

from Experiment.ExperimentClass import Experiment
from GridWorld.GridWorldMDPClass import GridWorldMDP
from GridWorld.TaxiMDPClass import TaxiMDP
from Agent.AgentClass import Agent
from resources.AbstractionTypes import Abstr_type
from resources.AbstractionCorrupters import *

NUM_EPISODES = 50
NUM_CORR_MDPS = 10
NUM_AGENTS = 20
EPS = 0.0
#ABSTR_EPSILON_LIST = [(Abstr_type.Q_STAR, EPS), (Abstr_type.A_STAR, EPS), (Abstr_type.PI_STAR, EPS)]
ABSTR_EPSILON_LIST = [(Abstr_type.Q_STAR, EPS)]
#CORRUPTION_LIST = [(Corr_type.UNI_RAND, 0.0), (Corr_type.UNI_RAND, 0.01), (Corr_type.UNI_RAND, 0.05), (Corr_type.UNI_RAND, 0.1), (Corr_type.UNI_RAND, 0.2), (Corr_type.UNI_RAND, 0.3)]
CORRUPTION_LIST = [(Corr_type.UNI_RAND, 0.0), (Corr_type.UNI_RAND, 0.01)]
MDP = GridWorldMDP()
MDP_STR = 'rooms'
# MDP = TaxiMDP()
# MDP_STR = 'taxi'

def run_experiment():

    exp = Experiment(MDP,
                     num_agents=NUM_AGENTS,
                     abstr_epsilon_list=ABSTR_EPSILON_LIST,
                     corruption_list=CORRUPTION_LIST,
                     num_corrupted_mdps=NUM_CORR_MDPS,
                     num_episodes=NUM_EPISODES)

    # Run experiment. This will write results to files
    # Commented out for testing visualization
    #data, steps, corr_data, corr_steps = exp.run_all_ensembles(include_corruption=True)

    # Plot results
    #exp.visualize_results(data, 'results/exp_graph_' + MDP_STR + '_' + str(EPS) + '.png')

    #exp.visualize_results(steps, 'results/step_counts_' + MDP_STR + '_' + str(EPS) + '.png')

    #exp.visualize_corrupt_mdps(corr_data, 'results/corrupted_exp_graph_' + MDP_STR + '_' + str(EPS) + '.png')

    #exp.visualize_corrupt_mdps(corr_steps, 'results/corrupted_step_counts_' + MDP_STR + '_' + str(EPS) + '.png')

    exp.visualize_corrupt_mdps('exp_output/corrupted/exp_output.csv', 'results/corrupted_exp_graph_' + MDP_STR + '_' + str(EPS) + '.png')

if __name__ == '__main__':
    run_experiment()