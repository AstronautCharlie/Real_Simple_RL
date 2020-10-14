"""
Run this to conduct an experiment
"""
import os
import time

from Experiment.ExperimentClass import Experiment
from GridWorld.GridWorldMDPClass import GridWorldMDP
from GridWorld.TaxiMDPClass import TaxiMDP
from Agent.AgentClass import Agent
from resources.AbstractionTypes import Abstr_type
from resources.AbstractionCorrupters import *

NUM_EPISODES = 1000
NUM_CORR_MDPS = 5
NUM_AGENTS = 5
EPS = 0.0
ABSTR_EPSILON_LIST = [(Abstr_type.A_STAR, EPS), (Abstr_type.PI_STAR, EPS), (Abstr_type.Q_STAR, EPS)]
CORRUPTION_LIST = [(Corr_type.UNI_RAND, 0.05), (Corr_type.UNI_RAND, 0.1)]
MDP = GridWorldMDP()
MDP_STR = 'rooms'

def run_experiment():
    start_time = time.time()

    exp = Experiment(MDP,
                     num_agents=NUM_AGENTS,
                     abstr_epsilon_list=ABSTR_EPSILON_LIST,
                     corruption_list=CORRUPTION_LIST,
                     num_corrupted_mdps=NUM_CORR_MDPS,
                     num_episodes=NUM_EPISODES,
                     results_dir='exp_output/hot',
                     agent_type='abstraction',
                     agent_exploration_epsilon=0.5,
                     decay_exploration=False,
                     exploring_starts=False,
                     step_limit=5000)

    # Run experiment. This will write results to files
    # Commented out for testing visualization
    data, steps, corr_data, corr_steps = exp.run_all_ensembles(include_corruption=True)

    # Plot performance for true results
    exp.visualize_results(data, outfilename='true_aggregated_results.png')

    # Plot performance for corrupt results
    if len(CORRUPTION_LIST) > 0:
        exp.visualize_corrupt_results(corr_data, outfilename='corrupt_aggregated_results.png')

    # Write a summary of the parameters to this file
    param_file = open(os.path.join(exp.results_dir, 'param_summary.txt'), 'w')
    abs_sum = 'Abstraction types:'.ljust(30) + str(ABSTR_EPSILON_LIST) + '\n'
    corr_sum = 'Corruptions applied:'.ljust(30) + str(CORRUPTION_LIST) + '\n'
    mdp_num_sum = '# corrupt MDPs:'.ljust(30) + str(NUM_CORR_MDPS) + '\n'
    ep_num_sum = '# episodes trained:'.ljust(30) + str(NUM_EPISODES) + '\n'
    exp_num_sum = 'Starting exploration epsilon:'.ljust(30) + str(exp.agent_exploration_epsilon) + '\n'
    decay_sum = 'Decay exploration?:'.ljust(30) + str(exp.decay_exploration) + '\n'
    exp_sum = 'Exploring starts?:'.ljust(30) + str(exp.exploring_starts) + '\n'
    step_limit = 'Step limit:'.ljust(30) + str(exp.step_limit) + '\n'
    runtime = 'Runtime:'.ljust(30) + str(round(time.time() - start_time))

    param_file.write(abs_sum + corr_sum + mdp_num_sum + ep_num_sum + exp_num_sum + decay_sum + exp_sum + step_limit
                     + runtime)

    # Plot step counts for true results
    #exp.visualize_results(steps, outfilename='true_step_counts.png')

    # Plot step counts for corrupt results
    #exp.visualize_corrupt_results(corr_steps, outfilename='corrupt_step_counts.png')

if __name__ == '__main__':
    run_experiment()