"""
New and improved Experiment class. Assumes the MDP adheres to the Env
interface laid out by OpenAI Gym

Experiment needs to take in:
- num_agents: number of agents in each ensemble
- num_episodes: number of episodes each ensemble is run for
- agent_params: dictionary mapping ensemble names to AgentWrapper parameter sets

Needs to do:
- Copying of environments (need 1 per agent)
- Run all the agents (AgentWrapper will handle the detachment, anything that has to happen during the loop)
- Collect and save results
"""
from Agent.AgentWrapper import AgentWrapper
import pandas as pd
import os


class Experiment:
    def __init__(self, env, **kwargs):
        self.env = env
        self.params = kwargs
        required_args = ['agent_params', 'num_agents', 'num_episodes']
        for req in required_args:
            if req not in self.params.keys():
                raise ValueError('Missing required argument: {}'.format(req))
        if 'output_dir' not in self.params.keys():
            self.params['output_dir'] = './exp_output'

    def run_ensemble(self, params):
        """
        Run an ensemble of AgentWrappers with the given parameters; number
        of agents dictated by self.params['num_agents']
        :param params: dictionary of AgentWrapper parameters
        :return: step counts (List of Lists), rewards (List of Lists),
            online abstractions made (List of dicts; empty list if that
            parameter isn't set), detached states (List of Lists; empty
            list if that parameter isn't set)
        """
        ensemble_steps = []
        ensemble_rewards = []
        ensemble_abstractions = []
        ensemble_detached_states = []
        for i in range(self.params['num_agents']):
            print('On agent number {}'.format(i))
            print('about to make agent wrapper')
            agent = AgentWrapper(self.env, **params)
            steps, rewards, abstraction, detached_states = agent.run_all_episodes(self.params['num_episodes'])
            ensemble_steps.append(steps)
            ensemble_rewards.append(rewards)
            ensemble_abstractions.append(abstraction)
            ensemble_detached_states.append(detached_states)
        return ensemble_steps, ensemble_rewards, ensemble_abstractions, ensemble_detached_states

    def run_all_ensembles(self):
        """
        Run an ensemble for every set of agent_params. Return a dictionary mapping
        the ensemble name to a dictionary, which itself maps 'steps', 'rewards', 'abstractions', and
        'detachments' to Lists of Lists containing those values per agent, per episode

        Save files to experiment output path and return values
        :return: dictionary {ensemble_name: {field name: List of Lists}}
        """
        final_results = {}

        for ensemble_name, param_set in self.params['agent_params'].items():
            # Run ensemble
            print('Running ensemble {} with params {}'.format(ensemble_name, param_set))
            steps, rewards, abstractions, detachments = self.run_ensemble(param_set)
            ensemble_results = {'steps': steps,
                                'rewards': rewards,
                                'abstractions': abstractions,
                                'detachments': detachments}

            # Save results
            for key, value in ensemble_results.items():
                df = pd.DataFrame(value)
                if not df.empty:
                    path = os.path.join(self.params['output_dir'], ensemble_name)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    df.to_csv(os.path.join(path, key + '.csv'))

            final_results[ensemble_name] = ensemble_results
