"""
Run this to conduct an experiment
"""

from Experiment.ExperimentClass import Experiment
from GridWorld.GridWorldMDPClass import GridWorldMDP
from GridWorld.TaxiMDPClass import TaxiMDP
from Agent.AgentClass import Agent
from resources.AbstractionTypes import Abstr_type
from resources.AbstractionCorrupters import *

NUM_EPISODES = 100
NUM_CORR_MDPS = 10
NUM_AGENTS = 10
EPS = 0.0
ABSTR_EPSILON_LIST = [(Abstr_type.A_STAR, EPS), (Abstr_type.PI_STAR, EPS), (Abstr_type.Q_STAR, EPS)]
#ABSTR_EPSILON_LIST = [(Abstr_type.PI_STAR, EPS)]
#CORRUPTION_LIST = [(Corr_type.UNI_RAND, 0.0), (Corr_type.UNI_RAND, 0.01), (Corr_type.UNI_RAND, 0.05), (Corr_type.UNI_RAND, 0.1), (Corr_type.UNI_RAND, 0.2), (Corr_type.UNI_RAND, 0.3)]
CORRUPTION_LIST = [(Corr_type.UNI_RAND, 0.1), (Corr_type.UNI_RAND, 0.25)]
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
    data, steps, corr_data, corr_steps = exp.run_all_ensembles(include_corruption=True)

    # Plot results
    #exp.visualize_results(data, 'results/exp_graph_' + MDP_STR + '_' + str(EPS) + '.png')

    #exp.visualize_results(steps, 'results/step_counts_' + MDP_STR + '_' + str(EPS) + '.png')
    #print(corr_data)

    exp.visualize_corrupt_results(corr_data, 'results/corrupted_exp_graph_' + MDP_STR + '_' + str(EPS) + '.png')

    for i in range(NUM_CORR_MDPS):
        diff_dict = exp.get_corrupt_policy_differences((Abstr_type.PI_STAR, 0.0, Corr_type.UNI_RAND, 0.1, i))
        print(i)
        for key, value in diff_dict.items():
            print(key, value)

    #exp.visualize_corrupt_mdps(corr_steps, 'results/corrupted_step_counts_' + MDP_STR + '_' + str(EPS) + '.png')

    #corr_mdp = ((Abstr_type.PI_STAR, 0.0, Corr_type.UNI_RAND, 0.01, 1))
    '''
    print()
    # PI*, 0.01
    for i in range(NUM_CORR_MDPS):
        print("On batch_num", i)
        corr_mdp = ((Abstr_type.PI_STAR, 0.0, Corr_type.UNI_RAND, 0.01, i))
        # 'errors' is a list of tuples of the form (ground_state, true_abstr_state, corr_abstr_state)
        errors = exp.get_corruption_errors(corr_mdp)

        for error in errors:
            ground_state = error[0]
            print("ground state is", ground_state)
            print("Optimal action based on VI is", exp.vi.get_all_optimal_actions(ground_state))
            print("Actions learned by agents on corrupt MDPs are",
                  exp.get_optimal_actions_corrupt_mdp(ground_state, corr_mdp))
        print()
    # A*, 0.01
    for i in range(NUM_CORR_MDPS):
        print("On batch_num", i)
        corr_mdp = ((Abstr_type.A_STAR, 0.0, Corr_type.UNI_RAND, 0.01, i))

        errors = exp.get_corruption_errors(corr_mdp)

        for error in errors:
            ground_state = error[0]
            print("ground state is", ground_state)
            print("Optimal action based on VI is", exp.get_optimal_action_from_vi(ground_state))
            print("Actions learned by agents on corrupt MDPs are",
                  exp.get_optimal_actions_corrupt_mdp(ground_state, corr_mdp))
        print()
    # PI*, 0.1
    for i in range(NUM_CORR_MDPS):
        print("On batch_num", i)
        corr_mdp = ((Abstr_type.PI_STAR, 0.0, Corr_type.UNI_RAND, 0.1, i))

        errors = exp.get_corruption_errors(corr_mdp)

        for error in errors:
            ground_state = error[0]
            print("ground state is", ground_state)
            print("Optimal action based on VI is", exp.get_optimal_action_from_vi(ground_state))
            print("Actions learned by agents on corrupt MDPs are",
                  exp.get_optimal_actions_corrupt_mdp(ground_state, corr_mdp))
        print()
    # A*, 0.1
    for i in range(NUM_CORR_MDPS):
        print("On batch_num", i)
        corr_mdp = ((Abstr_type.A_STAR, 0.0, Corr_type.UNI_RAND, 0.1, i))

        errors = exp.get_corruption_errors(corr_mdp)

        for error in errors:
            ground_state = error[0]
            print("ground state is", ground_state)
            print("Optimal action based on VI is", exp.get_optimal_action_from_vi(ground_state))
            print("Actions learned by agents on corrupt MDPs are",
                  exp.get_optimal_actions_corrupt_mdp(ground_state, corr_mdp))
        print()
    '''
    #exp.visualize_corrupt_results('exp_output/corrupted/exp_output.csv', 'results/corrupted_exp_graph_' + MDP_STR + '_' + str(EPS) + '.png')

    #exp.visualize_corrupt_abstraction(corr_mdp)

if __name__ == '__main__':
    run_experiment()