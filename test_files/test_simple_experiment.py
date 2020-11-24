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
NUM_EPISODES = 20
DETACH_INTERVAL = 10
RESET_Q_VALUE = True

if __name__ == '__main__':
    mdp = SimpleMDP()
    exp = SimpleExperiment(mdp,
                           abstr_dicts=state_abstrs,
                           num_agents=1,
                           num_corrupted_mdps=1,
                           num_episodes=NUM_EPISODES,
                           detach_interval=DETACH_INTERVAL,
                           reset_q_value=RESET_Q_VALUE
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
    exp.run_all_ensembles()

