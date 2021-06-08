"""
Run this to conduct an experiment
# TODO Get online abstraction agents working


# TODO If action from optimal one-step rollout results in cycle, make sure that state is separated from
# TODO   other states where action from optimal one-step rollout is same. E.g. if right results in a cycle for s, we
# TODO   want to map s to its own state, not with other states where right is optimal and non-cycle

"""
import os
import time
import matplotlib.pyplot as plt

from Experiment.ExperimentClass import Experiment
from GridWorld.GridWorldMDPClass import GridWorldMDP
from GridWorld.GridWorldStateClass import GridWorldState
from GridWorld.TaxiMDPClass import TaxiMDP
from GridWorld.LargeTaxiMDPClass import LargeTaxiMDP
from GridWorld.TwoRoomsMDP import TwoRoomsMDP
from Agent.AgentClass import Agent
from resources.AbstractionTypes import Abstr_type
from resources.AbstractionCorrupters import *
from util import *
from Visualizer.QValueVisualizer import QValueVisualizer
import scipy.stats

# MDP details
MDP = GridWorldMDP()
MDP = TaxiMDP(same_goal=True)
#MDP = LargeTaxiMDP(same_goal=True, gamma=0.9)
mdp_sum = 'Taxi MDP'
'''
MDP = TwoRoomsMDP(upper_height=3,
                  upper_width=3,
                  lower_height=3,
                  lower_width=3,
                  hallway_states=[3], goal_location=[(1,5)])

MDP = TwoRoomsMDP(lower_width=1,
                  lower_height=1,
                  hallway_states=[1],
                  upper_height=0,
                  upper_width=0,
                  goal_location=[(1,5)],
                  hallway_height=5,
                  init_state=(1,1))
mdp_sum = 'TwoRooms MDP'
'''


# Experiment parameters
NUM_EPISODES = 300
NUM_CORR_MDPS = 1
NUM_AGENTS = 10
EPS = 0.0
EXPLORATION_EPSILON = 0.1
AGENT_TYPE = 'tracking'
STEP_LIMIT = float("inf")
DECAY_EXPLORATION = False
EXPLORING_STARTS = False
VERBOSE = True

# Detach-related
DETACH_INTERVAL = None #1000
DETACH_POINTS = [i for i in range(21, NUM_EPISODES)] #[99]
PREVENT_CYCLES = True
RESET_Q_VALUE = 'neighbor'
VARIANCE_THRESHOLD = None
AGENT_DETACH_METHOD = 'abstr'
DETACH_REASSIGNMENT = 'group'
STATES_PER_DETACH = 1

# State-tracking
STATES_TO_TRACK = None#[s for s in MDP.get_all_possible_states()]

# Online abstraction
INCLUDE_ONLINE_ABSTR = True
ONLINE_TRAINING_EPS = 20
ONLINE_EPSILON = 0.05

# Noisy error distributions
ABSTR_ERROR_DISTRIBUTION = None#stats.norm
ABSTR_ERROR_PARAMS = None#{'loc': 0, 'scale': 0.1}
PER_STATE_ERROR_DISTRIBUTIONS = None
PER_STATE_ERROR_PARAMS = None
NOISY_ABSTR_TYPES = [Abstr_type.Q_STAR, Abstr_type.A_STAR, Abstr_type.PI_STAR]
NOISY_ABSTR_EPSILON = 0.05

# Per-run
ABSTR_EPSILON_LIST = []#[(Abstr_type.Q_STAR, 0.0)]#, (Abstr_type.A_STAR, 0.0), (Abstr_type.PI_STAR, 0.0)]
CORRUPTION_LIST = None
ERROR_DICTS = None #[{GridWorldState(1,2): GridWorldState(1,4)}]
GROUND_ONLY = True
SKIP_TRUE = False
DETACH_ONLY = False
NEIGHBOR_FACTOR = 1

# Create notes
NOTES = str(NUM_EPISODES) + ' episodes, ' + str(ONLINE_TRAINING_EPS) + ' training episodes, '\
        + str(min(DETACH_POINTS)) + ' to ' + str(max(DETACH_POINTS)) + ', ' + str(NUM_AGENTS) + ' agents, ' \
        + ', neighbor factor = ' + str(NEIGHBOR_FACTOR)
#+ str(MDP.width) + 'x' + str(MDP.height) + ', neighbor factor = ' + str(NEIGHBOR_FACTOR)



def run_experiment():
    start_time = time.time()
    print('Total number of states:', len(MDP.get_all_possible_states()))



    exp = Experiment(MDP,
                     num_agents=NUM_AGENTS,
                     abstr_epsilon_list=ABSTR_EPSILON_LIST,
                     corruption_list=CORRUPTION_LIST,
                     error_dicts=ERROR_DICTS,
                     num_corrupted_mdps=NUM_CORR_MDPS,
                     num_episodes=NUM_EPISODES,
                     results_dir='exp_output/hot',
                     agent_type=AGENT_TYPE,
                     agent_exploration_epsilon=EXPLORATION_EPSILON,
                     decay_exploration=DECAY_EXPLORATION,
                     exploring_starts=EXPLORING_STARTS,
                     step_limit=STEP_LIMIT,
                     detach_interval=DETACH_INTERVAL,
                     prevent_cycles=PREVENT_CYCLES,
                     reset_q_value=RESET_Q_VALUE,
                     agent_detach=AGENT_DETACH_METHOD,
                     detach_reassignment=DETACH_REASSIGNMENT,
                     detach_points=DETACH_POINTS,
                     states_to_track=STATES_TO_TRACK,
                     detach_only=DETACH_ONLY,
                     track_error_states=True,
                     include_online_abstraction=INCLUDE_ONLINE_ABSTR,
                     online_abstraction_training_episodes=ONLINE_TRAINING_EPS,
                     online_abstraction_epsilon=ONLINE_EPSILON,
                     abstr_error_distribution=ABSTR_ERROR_DISTRIBUTION,
                     abstr_error_parameters=ABSTR_ERROR_PARAMS,
                     noisy_abstr_types=NOISY_ABSTR_TYPES,
                     per_state_abstr_error_distribution=PER_STATE_ERROR_DISTRIBUTIONS,
                     per_state_abstr_error_parameters=PER_STATE_ERROR_PARAMS,
                     noisy_abstr_epsilon=NOISY_ABSTR_EPSILON,
                     ground_only=GROUND_ONLY,
                     skip_true=SKIP_TRUE,
                     neighbor_factor=NEIGHBOR_FACTOR,
                     states_per_detach=STATES_PER_DETACH)

    # Run experiment. This will write results to files
    # Commented out for testing visualization
    include_corruption = not exp.detach_only

    data, steps, corr_data, corr_steps, corr_detach_data, corr_detach_steps = exp.run_all_ensembles(include_corruption=include_corruption,
                                                                                                    skip_ground=True,
                                                                                                    verbose=VERBOSE)
    '''
    if (DETACH_INTERVAL is not None or DETACH_POINTS is not None) and include_corruption:
        print('fail1')
        data, steps, corr_data, corr_steps, corr_detach_data, corr_detach_steps = exp.run_all_ensembles(include_corruption=include_corruption,
                                                                                                        skip_ground=True,
                                                                                                        verbose=VERBOSE)
    elif include_corruption:
        print('fail2')
        data, steps, corr_data, corr_steps = exp.run_all_ensembles(include_corruption=include_corruption,
                                                                   skip_ground=True,
                                                                   verbose=VERBOSE)
    elif (DETACH_INTERVAL is not None or DETACH_POINTS is not None):
        print('fail3')
        data, steps, corr_detach_data, corr_detach_steps = exp.run_all_ensembles(include_corruption=include_corruption,
                                                                                 skip_ground=True,
                                                                                 verbose=VERBOSE)
    else:
        print('fail4')
        data, steps = exp.run_all_ensembles(include_corruption=include_corruption,
                                            skip_ground=True,
                                            verbose=VERBOSE)
    '''

    # Record volatilities
    if exp.agent_type == 'tracking':
        exp.record_volatilities('volatilities.csv')

    # Visualize step counts
    exp_res = open(steps, "r")
    plt.style.use('seaborn-whitegrid')
    ax = plt.subplot(111)

    for mdp in exp_res:
        # splitting on double quotes
        mdp = mdp.split("\"")

        # if ground, first list item will have the word "ground"
        if ("ground" in mdp[0]):
            # and will contain everything we need as a comma seperated string
            mdp = mdp[0].split(",")
        else:
            # if not, the name of the abstraction will be the second list item
            # and everything else we need will be in the 3rd list item
            # which needs to be cleaned of empty strings
            mdp = [mdp[1]] + [m for m in mdp[2].split(",") if m != ""]

        episodes = [i for i in range(1, len(mdp))]
        ax.plot(episodes, [float(i) for i in mdp[1:]], label="%s" % (mdp[0],))

    plt.xlabel('Episode Number')
    plt.ylabel('Average Cumulative Steps Taken')
    plt.suptitle('Cumulative Steps Taken in True Abstractions and Ground MDP')
    leg = ax.legend(bbox_to_anchor=(0.6, 0.25), loc='upper left', ncol=1, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.savefig(os.path.join(exp.results_dir, 'true_step_counts.png'))
    plt.clf()

    # Plot performance for true results
    exp.visualize_results(data, outfilename='true_aggregated_results.png')

    # Plot performance for corrupt results
    if (CORRUPTION_LIST is not None or ERROR_DICTS is not None) and not DETACH_ONLY:
        exp.visualize_corrupt_results(corr_data, outfilename='corrupt_aggregated_results.png',
                                      title='Q-Learning Performance in Corrupt MDPs')

        # Visualize step counts
        exp_res = open(corr_steps, "r")
        plt.style.use('seaborn-whitegrid')
        ax = plt.subplot(111)

        for mdp in exp_res:
            # splitting on double quotes
            mdp = mdp.split("\"")

            # if ground, first list item will have the word "ground"
            if ("ground" in mdp[0]):
                # and will contain everything we need as a comma seperated string
                mdp = mdp[0].split(",")
            else:
                # if not, the name of the abstraction will be the second list item
                # and everything else we need will be in the 3rd list item
                # which needs to be cleaned of empty strings
                mdp = [mdp[1]] + [m for m in mdp[2].split(",") if m != ""]

            episodes = [i for i in range(1, len(mdp))]
            ax.plot(episodes, [float(i) for i in mdp[1:]], label="%s" % (mdp[0],))

        plt.xlabel('Episode Number')
        plt.ylabel('Average Cumulative Steps Taken')
        plt.suptitle('Cumulative Steps Taken in Corrupted MDPs')
        leg = ax.legend(bbox_to_anchor=(0.6, 0.25), loc='upper left', ncol=1, mode="expand", shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.5)
        print('Saving corrupted step counts to', os.path.join(exp.results_dir, 'corrupted_step_counts.png'))
        plt.savefig(os.path.join(exp.results_dir, 'corrupted_step_counts.png'))
        plt.clf()

    # Plot performance for corrupt w/ detach results
    if (CORRUPTION_LIST is not None or ERROR_DICTS is not None) and (DETACH_INTERVAL is not None or DETACH_POINTS is not None):
        exp.visualize_corrupt_results(corr_detach_data,
                                      outfilename='corrupt_detach_aggregated_results.png',
                                      individual_mdp_dir='corrupted_w_detach',
                                      title='Q-Learning Performance in Corrupt MDPs with Detachment Algorithm')

    # Write a summary of the parameters to this file
    param_file = open(os.path.join(exp.results_dir, 'param_summary.txt'), 'w')
    abs_sum = 'Abstraction types:'.ljust(30) + str(ABSTR_EPSILON_LIST) + '\n'
    corr_sum = 'Corruptions applied:'.ljust(30) + str(CORRUPTION_LIST) + '\n'
    error_sum = ''
    if exp.error_dicts is not None:
        error_sum += 'Error dictionary:'.ljust(30)
        for dic in exp.error_dicts:
            error_sum += '{'
            for key, value in dic.items():
                error_sum += str(key) + ': ' + str(value) + ', '
            # Cut off last comma
            error_sum = error_sum[:-2]
            error_sum += '}\n'
    mdp_num_sum = '# corrupt MDPs:'.ljust(30) + str(NUM_CORR_MDPS) + '\n'
    ep_num_sum = '# episodes trained:'.ljust(30) + str(NUM_EPISODES) + '\n'
    exp_num_sum = 'Starting exploration epsilon:'.ljust(30) + str(exp.agent_exploration_epsilon) + '\n'
    decay_sum = 'Decay exploration?:'.ljust(30) + str(exp.decay_exploration) + '\n'
    exp_sum = 'Exploring starts?:'.ljust(30) + str(exp.exploring_starts) + '\n'
    detach_sum = 'Detach interval:'.ljust(30) + str(exp.detach_interval) + '\n'\
                 + 'Reset Q-value:'.ljust(30) + str(exp.reset_q_value) + '\n'\
                 + 'Prevent cycles?:'.ljust(30) + str(exp.prevent_cycle) + '\n'\
                 + 'Variance threshold:'.ljust(30) + str(exp.variance_threshold) + '\n'
    step_limit = 'Step limit:'.ljust(30) + str(exp.step_limit) + '\n'
    runtime = 'Runtime:'.ljust(30) + str(round(time.time() - start_time)) + '\n'
    detach_sum += 'Detachment method:'.ljust(30) + AGENT_DETACH_METHOD + '\n'\
                 + 'Reassignment method:'.ljust(30) + DETACH_REASSIGNMENT + '\n'

    param_file.write(mdp_sum
                     + abs_sum
                     + corr_sum
                     + mdp_num_sum
                     + error_sum
                     + detach_sum
                     + ep_num_sum
                     + exp_num_sum
                     + decay_sum
                     + exp_sum
                     + detach_sum
                     + step_limit
                     + runtime
                     + NOTES + '\n')
    if isinstance(MDP, TwoRoomsMDP):
        param_file.write(MDP.get_params())

    # Write a categorization of detached states
    if VERBOSE and (DETACH_POINTS or DETACH_INTERVAL):
        for key in exp.corr_detach_agents.keys():
            for num in range(exp.num_agents):
                print('Categorizing detach for key', key)
                corrupted_abstr_file = os.path.join(exp.results_dir, 'corrupted/corrupted_abstractions.csv')
                error_file = os.path.join(exp.results_dir, 'corrupted/error_states.csv')
                detached_state_file = os.path.join(exp.results_dir, 'corrupted_w_detach/detached_states.csv')
                categorize_detached_states(key, num, corrupted_abstr_file, error_file, detached_state_file)

    # run Q-value visualizer
    if exp.states_to_track:
        v = QValueVisualizer(exp, exp.states_to_track)
        v.graph_q_values(aggregate=False)


if __name__ == '__main__':
    run_experiment()