"""
Define parameters and run Experiment

TrackingAgent required parameters:
- alpha (learning rate, default 0.1)
- epsilon-greedy factor (default 0.1)
- decay_exploration (default True)
- gamma (default 1)

AgentWrapper parameters (all optional):
- 'make_abstraction': List of ints (when the abstractions happen)
- 'do_detach': list of ints (where detach happens)
- 'reset_q_value': True, False, 'neighbor', or 'rollout'

Experiment parameters
- num_agents: number of agents in each ensemble
- num_episodes: number of episodes each ensemble is run for
- agent_params: dictionary mapping ensemble names to AgentWrapper parameter sets

"""

import gym
import gym_classics
from Experiment.ExperimentV2 import Experiment
import copy

NUM_EPISODES = 250
NUM_AGENTS = 1

if __name__ == '__main__':
    # Define environment and parameters
    env_name = 'MountainCar-v0'
    env = gym.make(env_name).env
    # Baseline parameters
    base_params = {'alpha': 0.1,
                   'epsilon': 0.1,
                   'decay_exploration': True,
                   'gamma': 0.99,
                   'abstraction_type': 'discretization',
                   'refinement_type': 'discretization',
                   'starting_mesh': 10,
                   'finest_mesh': 1000}
    # Ground agent parameters
    ground_agent_params = copy.copy(base_params)
    # Abstract agent parameters
    abstr_agent_params = copy.copy(base_params)
    #abstr_agent_params['make_abstraction'] = [15]
    #abstr_agent_params['do_detach'] = [i for i in range(16, 150)]
    abstr_agent_params['reset_q_value'] = 'neighbor'
    abstr_agent_params['refine_abstraction'] = [i for i in range(NUM_EPISODES)]

    # Experiment parameters
    experiment_params = {'num_agents': NUM_AGENTS,
                         'num_episodes': NUM_EPISODES,
                         'agent_params': {'baseline': ground_agent_params,
                                          'abstr': abstr_agent_params},
                         'output_dir': 'experiment_output-v2/' + env_name}

    # Create experiment
    exp = Experiment(env, **experiment_params)

    # Run experiment
    exp.run_all_ensembles()
