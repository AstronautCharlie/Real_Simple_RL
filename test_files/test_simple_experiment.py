"""
This file tests the SimpleExperiment class
"""
from MDP.SimpleMDP import SimpleMDP, simple_states
from Experiment.SimpleExperiment import SimpleExperiment
from MDP.StateClass import State

# Define state abstraction
[s1, s2, s3, e] = simple_states
#state_abstrs = [{s1: State(1),
#                 s2: State(2),
#                 s3: State(2),
#                 e: State(3)}]
state_abstrs = [{s1: 1,
                 s2: 2,
                 s3: 2,
                 e: 3}]

# Parameters
NUM_AGENTS = 5
NUM_CORR_MDPS = 1
NUM_EPISODES = 50
DETACH_INTERVAL = 10
RESET_Q_VALUE = True
LEARNING_RATE = 0.1
EXPLORATION_EPSILON = 0.2


if __name__ == '__main__':
    mdp = SimpleMDP()
    exp = SimpleExperiment(mdp,
                           abstr_dicts=state_abstrs,
                           num_agents=NUM_AGENTS,
                           num_corrupted_mdps=NUM_CORR_MDPS,
                           num_episodes=NUM_EPISODES,
                           detach_interval=DETACH_INTERVAL,
                           reset_q_value=RESET_Q_VALUE,
                           agent_learning_rate=LEARNING_RATE,
                           agent_exploration_epsilon=EXPLORATION_EPSILON
                           )

    # Test that agents were created correctly
    '''
    agent = exp.agents['ground'][0]
    print(agent)
    for i in range(10):
        print('Trajectory', i)
        print('Starting state', agent.get_current_state())
        a, b, c = exp.run_trajectory(agent)
        print(a, b, c)
        q_table = agent.get_q_table()
        for key, value in q_table.items():
            print(key[0], key[1], value)
        print(agent._alpha)

        print()
    '''

    # Test that run_ensemble works correctly
    '''
    for i in range(10):
        print('Trajectory', i)
        r, s = exp.run_ensemble(exp.agents['ground'])
        print(r, s)
        print()
    '''

    # Test run_all_ensembles
    data, steps, corr_data, corr_steps, corr_detach_data, corr_detach_steps = exp.run_all_ensembles()

    # Plot performance for true results
    exp.visualize_results(data, outfilename='true_aggregated_results.png')

    # Plot performance for corrupt results
    exp.visualize_corrupt_results(corr_data, outfilename='corrupt_aggregated_results.png')

    # Plot performance for corrupt w/ detach results
    exp.visualize_corrupt_results(corr_detach_data,
                                  outfilename='corrupt_detach_aggregated_results.png',
                                  individual_mdp_dir='corrupted_w_detach')
