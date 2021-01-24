"""
This file has most of the same functionality as ExperimentClass, but runs it specifically on a SimpleMDP
"""

from MDP.ValueIterationClass import ValueIteration
from MDP.AbstractMDPClass import AbstractMDP
from MDP.SimpleMDP import SimpleMDP
from Agent.AbstractionAgent import AbstractionAgent
from resources.AbstractionCorrupters import *
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os


class SimpleExperiment:
    def __init__(self,
                 mdp,
                 abstr_dicts=None,
                 num_corrupted_mdps=1,
                 num_agents=10,
                 num_episodes=100,
                 results_dir='exp_results/simple',
                 agent_exploration_epsilon=0.1,
                 agent_learning_rate=0.1,
                 detach_interval=None,
                 prevent_cycles=False,
                 variance_threshold=False,
                 reset_q_value=False):

        self.ground_mdp = mdp
        self.abstr_dicts = abstr_dicts
        self.num_agents = num_agents
        self.num_corrupted_mdps = num_corrupted_mdps
        self.num_episodes = num_episodes
        self.results_dir = results_dir
        self.num_episodes = num_episodes
        self.agent_exploration_epsilon = agent_exploration_epsilon
        self.agent_learning_rate = agent_learning_rate
        self.detach_interval = detach_interval
        self.prevent_cycles = prevent_cycles
        self.variance_threshold = variance_threshold
        self.reset_q_value = reset_q_value

        # Run VI and get q-table. Used for graphing results
        vi = ValueIteration(mdp)
        vi.run_value_iteration()
        q_table = vi.get_q_table()
        self.vi_table = q_table
        self.vi = vi

        # This will hold all the agents. Key is ('explicit errors', abstraction dict number, mdp number),
        #  value is the MDP itself
        self.agents = {}

        # Create the corrupt MDPs from the provided abstraction dictionaries
        self.corrupt_mdp_dict = {}
        if self.abstr_dicts is not None:
            if not os.path.exists(os.path.join(self.results_dir, 'corrupted')):
                os.makedirs(os.path.join(self.results_dir, 'corrupted'))
            for i in range(len(self.abstr_dicts)):
                abstr_dict = self.abstr_dicts[i]
                for j in range(self.num_corrupted_mdps):
                    # Make a state abstraction that corresponds to given abstraction dictionary
                    s_a = StateAbstraction(abstr_dict=abstr_dict, epsilon=0)
                    abstr_mdp = AbstractMDP(mdp, s_a)
                    self.corrupt_mdp_dict[('explicit errors', i, j)] = abstr_mdp

        # Create the agents on the ground MDP
        ground_agents = []
        for i in range(self.num_agents):
            temp_mdp = SimpleMDP()
            agent = AbstractionAgent(temp_mdp,
                                     epsilon=agent_exploration_epsilon,
                                     alpha=agent_learning_rate,
                                     decay_exploration=False)
            ground_agents.append(agent)
        self.agents['ground'] = ground_agents

        # Create agents on the corrupt MDPs
        self.corr_agents = {}
        for key in self.corrupt_mdp_dict.keys():
            corr_ensemble = []
            for i in range(self.num_agents):
                # This makes an AbstractionAgent from the state abstraction corresponding to the abstract MDP
                temp_mdp = copy.deepcopy(SimpleMDP())
                corr_mdp = copy.deepcopy(self.corrupt_mdp_dict[key].copy())
                s_a = copy.deepcopy(corr_mdp.state_abstr)
                agent = AbstractionAgent(temp_mdp,
                                         s_a,
                                         epsilon=agent_exploration_epsilon,
                                         alpha=agent_learning_rate,
                                         decay_exploration=False)
                corr_ensemble.append(agent)
            self.corr_agents[key] = corr_ensemble

        # If detach interval is set, create another set of agents that will run detachment algorithm
        self.corr_detach_agents = {}
        for key in self.corrupt_mdp_dict.keys():
            corr_ensemble = []
            for i in range(self.num_agents):
                print('making detach agent', i)
                temp_mdp = copy.deepcopy(SimpleMDP())
                corr_mdp = copy.deepcopy(self.corrupt_mdp_dict[key].copy())
                s_a = copy.deepcopy(corr_mdp.state_abstr)
                agent = AbstractionAgent(temp_mdp,
                                         s_a,
                                         epsilon=agent_exploration_epsilon,
                                         alpha=agent_learning_rate,
                                         decay_exploration=False)
                corr_ensemble.append(agent)
            self.corr_detach_agents[key] = corr_ensemble

    # ----------------------------
    # Functions for running agents
    # ----------------------------

    def run_trajectory(self, agent):
        """
        Run an agent on the MDP until it reaches a terminal state. Record sum of discounted rewards had along the way.
        :return: sum of rewards, optimal value of starting state, step count
        """
        step_count = 0
        # Bypass abstraction function to get current ground state
        starting_state = agent.mdp.current_state
        optimal_value = self.get_optimal_state_value(starting_state)
        current_state = starting_state

        # This will track the rewards accumulated along the trajectory
        sum_rewards = 0
        discount = 1

        # Explore until a terminal state is reached
        while not current_state.is_terminal():
            current_state, action, next_state, reward = agent.explore()
            #print(current_state, action, next_state, reward)
            sum_rewards += discount * reward
            current_state = next_state
            discount *= agent.mdp.gamma
            step_count += 1
        #print()
        agent.mdp.reset_to_init()

        return sum_rewards, optimal_value, step_count

    def run_ensemble(self, ensemble):
        """
        Run each agent in an ensemble for a single trajectory and track the fraction of the optimal value
        each agent captured
        :return:
        """
        reward_fractions = []
        step_counts = []
        for agent in ensemble:
            actual_rewards, optimal_rewards, step_count = self.run_trajectory(agent)
            reward_fractions.append(actual_rewards / optimal_rewards)
            step_counts.append(step_count)
        return reward_fractions, step_counts

    def detach_ensemble(self, ensemble):
        """
        For each agent, check the consistency of abstract states and detach inconsistent ground states
        """
        detached_states = {}
        print('Number of agents to detach', len(ensemble))
        for i in range(len(ensemble)):
            agent = ensemble[i]
            print('Detaching for agent number', i)
            error_states = agent.detach_inconsistent_states(prevent_cycles=self.prevent_cycles,
                                                            reset_q_value=self.reset_q_value)
            if error_states is not None:
                detached_states[i] = error_states
        return detached_states

    def run_all_ensembles(self, include_corruption=False):
        """
        Central method for running an experiment.

        Run all ensembles in the experiment and write the average fraction of optimal rewards
        achieved by each ensemble at each episode to a file.

        If include_corruption, do the same for corrupt MDPs.

        Also write learned policies to file
        """

        #
        # This chunk runs the ground level ensemble
        #
        # Create folders if they don't already exist and the files to hold the results
        true_dir = os.path.join(self.results_dir, 'true')
        if not os.path.exists(true_dir):
            os.makedirs(true_dir)
        reward_file = open(os.path.join(true_dir, 'rewards.csv'), 'w', newline='')
        step_file = open(os.path.join(true_dir, 'step_counts.csv'), 'w', newline='')
        policy_file = open(os.path.join(true_dir, 'learned_policies.csv'), 'w', newline='')
        value_file = open(os.path.join(true_dir, 'state_values.csv'), 'w', newline='')
        reward_writer = csv.writer(reward_file)
        step_writer = csv.writer(step_file)
        policy_writer = csv.writer(policy_file)
        value_writer = csv.writer(value_file)

        # Run the ensemble in the ground environment
        avg_reward_fractions = ['ground']
        avg_step_counts = ['ground']
        for episode in range(self.num_episodes):
            print('On episode', episode)
            reward_fractions, step_counts = self.run_ensemble(self.agents['ground'])
            avg_reward_fraction = sum(reward_fractions) / len(reward_fractions)
            avg_reward_fractions.append(avg_reward_fraction)
            avg_step_count = sum(step_counts) / len(step_counts)
            # Hacky step to make step-counts cumulative
            if len(avg_step_counts) > 1:
                avg_step_counts.append(avg_step_count + avg_step_counts[-1])
            else:
                avg_step_counts.append(avg_step_count)
            # Write q-values
            for i in range(len(self.agents['ground'])):
                value_string = '{'
                q_table = self.agents['ground'][i].get_q_table()
                for key, value in q_table.items():
                    value_string += '(' + str(key[0]) + ',' + str(key[1]) + '): ' + str(value) + ', '
                value_string += '}'
                value_writer.writerow(('ground', i, episode, value_string))

        # Write results
        reward_writer.writerow(avg_reward_fractions)
        step_writer.writerow(avg_step_counts)
        # Policy and value data would go here if needed. Copy from Experiment
        for i in range(len(self.agents['ground'])):
            policy_writer.writerow(('ground', i, self.agents['ground'][i].get_learned_policy_as_string()))

        #
        # This chunk runs the abstracted MDPs w/ no detachment
        #
        if self.abstr_dicts is not None:
            # Create files/folders
            corrupt_dir = os.path.join(self.results_dir, 'corrupted')
            if not os.path.exists(corrupt_dir):
                os.makedirs(corrupt_dir)
            reward_file = open(os.path.join(corrupt_dir, 'rewards.csv'), 'w', newline='')
            step_file = open(os.path.join(corrupt_dir, 'step_counts.csv'), 'w', newline='')
            policy_file = open(os.path.join(corrupt_dir, 'learned_policies.csv'), 'w', newline='')
            value_file = open(os.path.join(corrupt_dir, 'q_values.csv'), 'w', newline='')
            reward_writer = csv.writer(reward_file)
            step_writer = csv.writer(step_file)
            policy_writer = csv.writer(policy_file)
            value_writer = csv.writer(value_file)

            # Run ensembles on abstracted MDP
            for ensemble_key in self.corr_agents.keys():
                print(ensemble_key)
                avg_reward_fractions = [ensemble_key]
                avg_step_counts = [ensemble_key]
                for episode in range(self.num_episodes):
                    # Run ensemble
                    print('On episode', episode)
                    reward_fractions, step_counts = self.run_ensemble(self.corr_agents[ensemble_key])
                    avg_reward_fraction = sum(reward_fractions) / len(reward_fractions)
                    avg_reward_fractions.append(avg_reward_fraction)
                    avg_step_count = sum(step_counts) / len(step_counts)
                    if len(avg_step_counts) > 1:
                        avg_step_counts.append(avg_step_count + avg_step_counts[-1])
                    else:
                        avg_step_counts.append(avg_step_count)
                    for i in range(len(self.corr_agents[ensemble_key])):
                        value_string = '{'
                        q_table = self.corr_agents[ensemble_key][i].get_q_table()
                        for key, value in q_table.items():
                            value_string += '(' + str(key[0]) + ',' + str(key[1]) + '): ' + str(value) + ', '
                        value_string += '}'
                        value_writer.writerow((ensemble_key, i, episode, value_string))
                # Write results
                reward_writer.writerow(avg_reward_fractions)
                step_writer.writerow(avg_step_counts)
                for i in range(len(self.corr_agents[ensemble_key])):
                    policy_writer.writerow((ensemble_key, i, self.corr_agents[ensemble_key][i].get_learned_policy_as_string()))
        #
        # This chunk runs abstract MDPs w/ detachment
        #
        if self.detach_interval is not None:
            # Create files/folders
            detach_dir = os.path.join(self.results_dir, 'corrupted_w_detach')
            if not os.path.exists(detach_dir):
                os.makedirs(detach_dir)
            reward_file = open(os.path.join(detach_dir, 'rewards.csv'), 'w', newline='')
            step_file = open(os.path.join(detach_dir, 'step_counts.csv'), 'w', newline='')
            policy_file = open(os.path.join(detach_dir, 'learned_policies.csv'), 'w', newline='')
            #value_file = open(os.path.join(detach_dir, 'learned_values.csv'), 'w', newline='')
            detach_file = open(os.path.join(detach_dir, 'detached_states.csv'), 'w', newline='')
            finalSAfile = open(os.path.join(detach_dir, 'final_s_a.csv'), 'w', newline='')
            q_value_file = open(os.path.join(detach_dir, 'q_values.csv'), 'w', newline='')
            reward_writer = csv.writer(reward_file)
            step_writer = csv.writer(step_file)
            policy_writer = csv.writer(policy_file)
            #value_writer = csv.writer(value_file)
            detach_writer = csv.writer(detach_file)
            finalSA_writer = csv.writer(finalSAfile)
            q_value_writer = csv.writer(q_value_file)
            # Run ensemble w/ detachment
            for ensemble_key in self.corr_detach_agents.keys():
                print(ensemble_key, 'detaching states')
                avg_reward_fractions = [ensemble_key]
                avg_step_counts = [ensemble_key]
                detached_state_record = {}
                for agent_num in range(self.num_agents):
                    detached_state_record[agent_num] = []
                for episode in range(self.num_episodes):
                    print('On episode', episode)
                    reward_fractions, step_counts = self.run_ensemble(self.corr_detach_agents[ensemble_key])
                    avg_reward_fraction = sum(reward_fractions) / len(reward_fractions)
                    avg_reward_fractions.append(avg_reward_fraction)
                    avg_step_count = sum(step_counts) / len(step_counts)
                    if len(avg_step_counts) > 1:
                        avg_step_counts.append(avg_step_count + avg_step_counts[-1])
                    else:
                        avg_step_counts.append(avg_step_count)
                    # Run detachment at given interval
                    if episode > 0 and episode % self.detach_interval == 0:
                        print('In experiment, about to detach states')
                        detach_dict = self.detach_ensemble(self.corr_detach_agents[ensemble_key])
                        for key, value in detach_dict.items():
                            detached_states = []
                            for state in value:
                                detached_states.append((state.data, episode))
                            detached_state_record[key] += detached_states
                        print('Detach state record is', detached_state_record)
                        # If on last episode, write all detached states to a file
                    if episode == self.num_episodes - 1:
                        print('Writing detached states to file')
                        for i in range(len(self.corr_detach_agents[ensemble_key])):
                            detach_writer.writerow((ensemble_key, i, detached_state_record[i]))
                    # Write q-values
                    for i in range(len(self.corr_detach_agents[ensemble_key])):
                        value_string = '{'
                        q_table = self.corr_detach_agents[ensemble_key][i].get_q_table()
                        for key, value in q_table.items():
                            value_string += '(' + str(key[0]) + ',' + str(key[1]) + '): ' + str(value) + ', '
                        value_string += '}'
                        q_value_writer.writerow((ensemble_key, i, episode, value_string))
                # Write the results
                reward_writer.writerow(avg_reward_fractions)
                step_writer.writerow(avg_step_counts)
                # Write final state abstractions
                for i in range(len(self.corr_detach_agents[ensemble_key])):
                    finalSA_writer.writerow((ensemble_key, i, self.corr_detach_agents[ensemble_key][i].get_abstraction_as_string()))
                # Write the final learned policy
                for i in range(len(self.corr_detach_agents[ensemble_key])):
                    policy_writer.writerow((ensemble_key, i, self.corr_detach_agents[ensemble_key][i].get_learned_policy_as_string()))

        if self.detach_interval is not None:
            return os.path.join(self.results_dir, 'true/rewards.csv'), \
                   os.path.join(self.results_dir, 'true/step_counts.csv'), \
                   os.path.join(self.results_dir, 'corrupted/rewards.csv'), \
                   os.path.join(self.results_dir, 'corrupted/step_counts.csv'), \
                   os.path.join(self.results_dir, 'corrupted_w_detach/rewards.csv'), \
                   os.path.join(self.results_dir, 'corrupted_w_detach/step_counts.csv')
        else:
            return os.path.join(self.results_dir, 'true/rewards.csv'), \
                   os.path.join(self.results_dir, 'true/step_counts.csv')

    # ------------------------
    # Visualizations functions
    # ------------------------
    def visualize_results(self, infilepath, outdirpath=None, outfilename=None):
        """
        Graph the results in the given file and save the results to the given outfile

        Copied from Experiment Class
        """
        if outdirpath is None:
            outdirpath = self.results_dir
        if not os.path.exists(outdirpath):
            os.makedirs(outdirpath)

        exp_res = open(infilepath, 'r')
        plt.style.use('seaborn-whitegrid')
        ax = plt.subplot()

        for mdp in exp_res:
            mdp = mdp.split('\"')
            if ('ground' in mdp[0]):
                mdp = mdp[0].split(',')
            else:
                mdp = [mdp[1]] + [m for m in mdp[2].split(',') if m != ""]

            episodes = [i for i in range(1, len(mdp))]
            plt.plot(episodes, [float(i) for i in mdp[1:]], label='%s' % (mdp[0],))

        plt.xlabel('Episode Number')
        plt.ylabel('Proportion of Value of Optimal Policy Captured')
        plt.ylim(-0.5,1)
        plt.suptitle('Performance of Q-Learning in the Ground Environment')
        #leg = plt.legend(loc='best', ncol=2, mode='expand', shadow=True, fancybox=True)
        #leg.get_frame().set_alpha(0.5)

        if outfilename is None:
            outfilename = 'true_results.png'
        plt.savefig(os.path.join(outdirpath, outfilename))
        plt.clf()

    def visualize_corrupt_results(self,
                                  infilepath,
                                  outdirpath=None,
                                  outfilename=False,
                                  individual_mdp_dir=None,
                                  graph_between=False,
                                  title=None):
        """
        Graph the results of the corrupted MDPs
        """
        if outdirpath is None:
            outdirpath = self.results_dir
        if not os.path.exists(outdirpath):
            os.makedirs(outdirpath)
        plt.style.use('seaborn-whitegrid')
        plt.ylim(-0.5, 1)

        # Read in data as dataframe, get 'key' which consists of the abstraction type, the abstraction epsilon,
        #  the corruption value, and the number within that batch
        # infile looks like: key | ep 1 | ep 2 | ep3...
        names = ['key'] + [i for i in range(self.num_episodes)]
        df = pd.read_csv(infilepath, names=names)
        # Turn string representing key into a list of values
        df['key'] = df['key'].str.replace('(', '').str.replace(')','').str.replace(', ',',').str.split(',')

        # This extracts the batch number from the key and parses the key into usable columns
        def remove_batch_num(row):
            return tuple(row['key'][:-1])
        def convert_key_to_tuple(row):
            return tuple(row['key'])
        # key is the unique identifier for a single ensemble on one corrupt MDP
        df['key'] = df.apply(convert_key_to_tuple, axis=1)
        # batch is the combination of abstract MDP and corruption data; will match to as many rows as we have
        #  exp.num_corr_mdps
        df['batch'] = df.apply(remove_batch_num, axis=1)
        df[['key', 'mdp', 'batch_num']] = pd.DataFrame(df.key.tolist(), index=df.index)

        # This section calculates the averages and standard deviations of fractions of optimal value captured
        #  across different abstraction types, graphs the results across episodes, and saves the figure
        avg_df = df.groupby('batch').mean()
        std_df = df.groupby('batch').std()

        episodes = [i for i in range(1, self.num_episodes + 1)]
        for i in range(avg_df.shape[0]):
            upper = avg_df.iloc[i] + std_df.iloc[i]
            lower = avg_df.iloc[i] - std_df.iloc[i]
            plt.plot(episodes, list(avg_df.iloc[i]), label="%s" % ([avg_df.index[i][0]]))#, avg_df.index[i][3]]))
            if graph_between:
                plt.fill_between(episodes, upper, lower, alpha=0.2)
        plt.suptitle(title)
        plt.xlabel('Episode Number')
        plt.ylabel('Proportion of Value of Optimal Policy Captured')
        if outfilename is None:
            outfilename = 'corrupt_results.png'
        plt.savefig(os.path.join(outdirpath, outfilename))
        plt.clf()

        # This section graphs the average performance of each ensemble on each corrupt MDP separately and saves
        #  the results
        # Get all abstr_type/abstr_epsilon/corr_type/corr_prop combinations
        batch_list = list(df['batch'].drop_duplicates())

        # Iterate through each of these values and subset the full dataframe for those rows matching the given
        #  parameters
        for batch in batch_list:
            temp_df = df.loc[df['batch'] == batch]

            # This is used for the filename later
            a_t = str(temp_df['key'].values[0])
            start = a_t.find('.') + 1
            end = a_t.find(':')
            a_t = a_t[start:end].lower()

            # Strip away everything except the data itself
            temp_df = temp_df.drop(columns=['key'])#, 'batch', 'abstr_type', 'abstr_epsilon', 'corr_type', 'corr_prop'])
            temp_df = temp_df.set_index('batch_num')

            # Iterate through all batches and graph them
            for index, row in temp_df.iterrows():
                #print('episodes\n', episodes)
                #print('row\n', row, row.shape)
                plt.plot(episodes, row[:-2], label="%s" % (index,))
            #plt.legend(loc='best', fancybox=True)
            plt.title("%s" % (batch,))

            # This creates a nice file name for the graph
            file_name = str(a_t)# + '_' +  str(c_p)
            file_name = file_name.replace('.', '')
            if individual_mdp_dir is None:
                individual_mdp_dir = 'corrupted'
            file_name = individual_mdp_dir + '/{}{}'.format(file_name, '.png')
            plt.savefig(os.path.join(outdirpath, file_name))
            plt.clf()

    # -----------------
    # Getters & setters
    # -----------------

    def get_optimal_state_value(self, state):
        '''
        Get the value of the given state under the optimal policy, as dictated by the VI table
        :return: float: value of the state under optimal policy
        '''
        optimal_state_value = float("-inf")
        for key in self.vi_table.keys():
            if key[0] == state:
                if self.vi_table[key] > optimal_state_value:
                    optimal_state_value = self.vi_table[key]
        return optimal_state_value
