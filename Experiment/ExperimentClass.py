#TODO: Right now we're not recalculating value of optimal policy at the start of each episode
"""
This class creates and runs a test of q-learning on the given MDP for the given abstraction
types and epsilon values, and compares the value of each trajectory to the value of the
optimal ground-state trajectory from that point

NOTE: this class currently stores true agents and corrupted agents in different dictionaries.
"""
from MDP.MDPClass import MDP
from MDP.StateAbstractionClass import StateAbstraction
from MDP.ValueIterationClass import ValueIteration
from MDP.AbstractMDPClass import AbstractMDP
from Agent.AgentClass import Agent
from Agent.AbstractionAgent import AbstractionAgent
from resources.AbstractionMakers import make_abstr
from resources.AbstractionTypes import Abstr_type
from resources.AbstractionCorrupters import *
from Visualizer.AbstractGridWorldVisualizer import AbstractGridWorldVisualizer as vis
from collections import defaultdict
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy

class Experiment():
    def __init__(self,
                 mdp,
                 abstr_epsilon_list=(),
                 corruption_list=(),
                 num_corrupted_mdps=10,
                 num_agents=10,
                 num_episodes=100,
                 results_dir='exp_results',
                 agent_type='abstraction',
                 agent_exploration_epsilon=0.1,
                 decay_exploration=True,
                 exploring_starts=False,
                 step_limit=10000):
        """
        Create an experiment, which will hold the ground MDP, the abstract MDPs (parameters dictated by abstr_epsilon_list),
        and an ensemble of (num_agents) q-learning agents on each MDP.
        :param mdp: MDP
        :param abstr_epsilon_list: list of tuples, where first element is abstraction type and second is the epsilon
        :param corruption_list: list of tuples, where first element is the corruption type and second is the proportion
        :param num_corrupted_mdps: number of corrupted MDPs to create for each entry in corruption_list
        :param num_agents: the number of agents making up an ensemble on a given MDP
        :param num_episodes: the number of episodes each ensemble will be run on each MDP
        :param results_dir: directory where output of experiment will be stored
        :param agent_type: either 'standard' if we're doing Q-learning in an abstract MDP or 'abstraction' if we're
                            using an abstraction agent that generalizes
        :param agent_exploration_epsilon: starting exploration rate of agent (probability of taking random action)
        :param decay_exploration: if true, decay exploration rate
        :param exploring_starts: if true, restart episode at random state instead of initial state
        :param step_limit: cap on the number of steps in an episode. If limit is hit, reset agent
        """
        # Check that agent_type is valid
        if agent_type not in ['standard', 'abstraction']:
            raise ValueError('"agent_type" variable must be "standard" or "abstraction". Is currently '+str(agent_type))

        self.ground_mdp = mdp
        for val in abstr_epsilon_list:
            if val[0] not in Abstr_type or val[1] < 0 or val[1] > 1:
                raise ValueError('Abstraction Epsilon List is invalid', abstr_epsilon_list)
        self.abstr_epsilon_list = abstr_epsilon_list
        self.corruption_list = corruption_list
        self.num_agents = num_agents
        self.num_corrupted_mdps = num_corrupted_mdps
        self.results_dir = results_dir
        self.num_episodes = num_episodes
        self.agent_type = agent_type
        self.decay_exploration = decay_exploration
        self.exploring_starts = exploring_starts
        self.step_limit = step_limit
        self.agent_exploration_epsilon = agent_exploration_epsilon

        # Agent ensembles will be stored in a dict where key is the (abstr_type, epsilon) tuple ('ground' in the case
        # of the ground MDP) and values are lists of agents. In the case of corrupted MDPs, the key will be
        # (abstr_type, epsilon, corruption_type, proportion)
        self.agents = {}

        # Run Value Iteration to get q-table for abstractions and to hold value of optimal policies
        vi = ValueIteration(mdp)
        vi.run_value_iteration()
        q_table = vi.get_q_table()
        self.vi_table = q_table
        self.vi = vi

        # Create abstract MDPs from each element of abstr_epsilon_list. val[0] is abstraction type, val[1] is epsilon
        self.abstr_mdp_dict = {}
        file_string = 'true/abstractions.csv'
        with open(os.path.join(self.results_dir, file_string), 'w', newline='') as abstr_file:
            abstr_writer = csv.writer(abstr_file)
            for val in abstr_epsilon_list:
                state_abstr = make_abstr(q_table, val[0], val[1])
                self.abstr_mdp_dict[(val[0], val[1])] = AbstractMDP(mdp, state_abstr)
                abstr_writer.writerow((val[0], val[1], AbstractMDP(mdp, state_abstr).abstr_to_string()))

        # Create (self.num_corrupted_mdps) corrupted versions of MDPs (if applicable) from each element of
        #  corruption_list
        # This is stored in a dictionary, mapping tuples of (abstractMDP type, abstraction_epsilon, corruption_type,
        #  proportion, number) to a corrupted abstract MDP
        # This is a messy way of storing things, but it doesn't really matter because the self.agents dictionary
        #  is what we use to run the experiment
        # This also writes the corrupted state abstractions to a file
        # These are generated in such a way that all corrupt MDPs with the same batch number will have errors
        #  in the same ground states
        self.corrupt_mdp_dict = {}
        if len(corruption_list) > 0:
            corr_abstr_file = open(os.path.join(self.results_dir, 'corrupted/corrupted_abstractions.csv'), 'w', newline='')
            err_state_file = open(os.path.join(self.results_dir, 'corrupted/error_states.csv'), 'w', newline='')
            abstr_writer = csv.writer(corr_abstr_file)
            err_writer = csv.writer(err_state_file)
            for val in corruption_list:
                # Unpack the values in corruption list; first is corruption type and second is proportion
                corr_type = val[0]
                prop = val[1]
                # We generate 1 corrupt MDP per abstraction type at a time. This way, we can ensure that all
                # corrupt MDPs with the same key and batch number have the same error states. Also write to file
                # so they can be visualized later
                for i in range(self.num_corrupted_mdps):
                    states_to_corrupt = np.random.choice(self.ground_mdp.get_all_possible_states(),
                                                         size=int(np.floor(len(self.ground_mdp.get_all_possible_states()) * prop)),
                                                         replace=False)
                    for key in self.abstr_mdp_dict.keys():
                        # Create a corrupt state abstraction for this batch number and list of states
                        abstr_mdp = self.abstr_mdp_dict[key]
                        c_s_a = make_corruption(abstr_mdp, states_to_corrupt, type=corr_type)
                        # Get the correct and incorrect abstract states for the corrupted ground states
                        #  and write all these to a file
                        error_line = []
                        for state in states_to_corrupt:
                            true_state = abstr_mdp.get_abstr_from_ground(state)
                            corr_state = c_s_a.get_abstr_from_ground(state)
                            error_line.append((str(state), str(true_state), str(corr_state)))
                        temp_key = (abstr_mdp.abstr_type, abstr_mdp.abstr_epsilon, corr_type, prop, i)
                        # Record the error states
                        err_writer.writerow((temp_key, error_line))
                        # Make an abstract MDP with this corrupted state abstraction
                        corrupt_abstr_mdp = AbstractMDP(self.ground_mdp, c_s_a)
                        # Record the dictionary describing the state abstraction
                        abstr_writer.writerow((temp_key, corrupt_abstr_mdp.abstr_to_string()))
                        self.corrupt_mdp_dict[(abstr_mdp.abstr_type,
                                               abstr_mdp.abstr_epsilon,
                                               corr_type,
                                               prop,
                                               i)] = corrupt_abstr_mdp

        # Create agents on ground mdp
        ground_agents = []
        for i in range(self.num_agents):
            temp_mdp = self.ground_mdp.copy()
            # If agent_type == 'standard', use regular q-learning. If 'abstraction', use abstraction agent
            if self.agent_type == 'standard':
                agent = Agent(temp_mdp, epsilon=agent_exploration_epsilon, decay_exploration=decay_exploration)
            else:
                agent = AbstractionAgent(temp_mdp, epsilon=agent_exploration_epsilon, decay_exploration=decay_exploration)
            ground_agents.append(agent)
        self.agents['ground'] = ground_agents

        # Create agents on abstract MDPs
        for abstract_mdp in self.abstr_mdp_dict.values():
            abstract_mdp_ensemble = []
            for i in range(self.num_agents):
                # Create agents according to agent_type parameter
                if self.agent_type == 'standard':
                    temp_mdp = abstract_mdp.copy()
                    agent = Agent(temp_mdp, epsilon=agent_exploration_epsilon, decay_exploration=decay_exploration)
                    abstract_mdp_ensemble.append(agent)
                else:
                    temp_mdp = self.ground_mdp.copy()
                    s_a = copy.deepcopy(abstract_mdp.state_abstr)
                    agent = AbstractionAgent(temp_mdp, s_a, epsilon=agent_exploration_epsilon, decay_exploration=decay_exploration)
                    abstract_mdp_ensemble.append(agent)
            self.agents[(abstract_mdp.abstr_type, abstract_mdp.abstr_epsilon)] = abstract_mdp_ensemble

        # Create agents on corrupted abstract MDPs. Remember that we have self.num_corrupted_mdps ensembles for each
        #  combination of abstractMDP type and entry in self.corruption_list.
        # self.corr_agents is now a dictionary mapping (abstr_type, epsilon, corr_type, proportion, batch_num) to
        #  a list of agents
        self.corr_agents = {}
        for corr_key in self.corrupt_mdp_dict.keys():
            corr_ensemble = []
            for i in range(self.num_agents):
                # Create agents according to agent_type parameter
                if self.agent_type == 'standard':
                    temp_mdp = self.corrupt_mdp_dict[corr_key].copy()
                    agent = Agent(temp_mdp, epsilon=agent_exploration_epsilon, decay_exploration=decay_exploration)
                    corr_ensemble.append(agent)
                else:
                    temp_mdp = self.ground_mdp.copy()
                    corr_mdp = self.corrupt_mdp_dict[corr_key].copy()
                    s_a = corr_mdp.state_abstr
                    agent = AbstractionAgent(temp_mdp, s_a, epsilon=agent_exploration_epsilon, decay_exploration=decay_exploration)
                    corr_ensemble.append(agent)

            self.corr_agents[corr_key] = corr_ensemble

    def __str__(self):
        result = 'key: num agents\n'
        for key in self.agents.keys():
            result += str(key) + ': ' + str(len(self.agents[key])) + '\n'
        return result

    def run_trajectory(self, agent, step_limit):
        """
        Run an agent on its MDP until it reaches a terminal state. Record the discounted rewards achieved along the way
        and the starting state
        :param agent: Q-learning agent
        :param step_limit: limit the number of steps in the trajectory to force termination
        :return: sum_rewards: sum of discounted rewards gotten over the course of the trajectory
        :return: optimal_value: value of the starting state under the optimal policy
        :return: step_count: the number of steps taken in this trjaectory
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
        while not current_state.is_terminal(): #and step_count < step_limit:
            if step_limit is not None and step_count > step_limit:
                break
            _, action, next_state, reward = agent.explore()
            sum_rewards += discount * reward
            current_state = next_state
            discount *= agent.mdp.gamma
            step_count += 1
        # Reset agent's MDP to initial state (or random state if self.exploring_starts)
        if self.exploring_starts:
            random_state = np.random.choice(agent.mdp.get_all_possible_states())
            while random_state.is_terminal():
                random_state = np.random.choice(agent.mdp.get_all_possible_states())
            agent.mdp.set_current_state(random_state)
        else:
            agent.mdp.reset_to_init()
        # Return the sum of discounted rewards from the trajectory and value of optimal policy
        return sum_rewards, optimal_value, step_count

    def run_ensemble(self, ensemble):
        """
        Run each agent in an ensemble for a single trajectory and track the fraction of the optimal value the
        agent captured
        :param ensemble: a list of agents
        :return: a list of the fraction of the optimal value each agent captured
        """
        reward_fractions = []
        step_counts = []
        for agent in ensemble:
            actual_rewards, optimal_rewards, step_count = self.run_trajectory(agent, step_limit=self.step_limit)
            reward_fractions.append(actual_rewards / optimal_rewards)
            step_counts.append(step_count)
        return reward_fractions, step_counts

    def run_all_ensembles(self, include_corruption=False):
        """
        This is the central method for running an experiment.

        Run all ensembles in the experiment and write the average fraction of optimal
        reward achieved by each ensemble at each episode to a file.

        If include_corruption, do the same for the corrupt MDPs in the experiment.

        Also write the learned policies to a file

        TODO fix these param/return comments to reflect addition of corrupt info
        :param include_corruption: boolean indicating whether or not to include corrupt MDPs
        :return: 2 file paths. First is a path to a file with the average fraction of the optimal reward achieved
                    by each ensemble at each episode. Second is the same for the corrupt MDPs
        """
        # This chunk runs the ensembles on the ground MDP and all the correct abstract MDPs
        # csvfile contains the average rewards obtained in every episode
        # stepfile contains the average number of steps per episode
        step_file = open(os.path.join(self.results_dir, "true/step_counts.csv"), 'w', newline='')
        csv_file = open(os.path.join(self.results_dir, "true/exp_output.csv"), 'w', newline='')
        policy_file = open(os.path.join(self.results_dir, "true/learned_policies.csv"), 'w', newline='')
        value_file = open(os.path.join(self.results_dir, 'true/learned_state_values.csv'), 'w', newline='')
        writer = csv.writer(csv_file)
        step_writer = csv.writer(step_file)
        policy_writer = csv.writer(policy_file)
        value_writer = csv.writer(value_file)
        # The for loop below iterates through all ensembles in self.agents. Each key in the self.agents dict
        #  represents a single ensemble of agents on a single MDP
        for ensemble_key in self.agents.keys():
            print(ensemble_key)
            # The two lines below are done so that each list of data is preceded by the key for that
            #  ensemble.
            #  This is done so we know which data belongs to which ensemble
            avg_reward_fractions = [ensemble_key]
            avg_step_counts = [ensemble_key]
            for episode in range(self.num_episodes):
                if episode % 10 == 0:
                    print("On episode", episode)
                # Reward_fractions and step_counts are lists of the fraction of the optimal policy captured
                #  and the number of steps taken (respectively) in a single trajectory by each agent in the
                #  ensemble
                reward_fractions, step_counts = self.run_ensemble(self.agents[ensemble_key])
                # We average these across the whole ensemble
                avg_reward_fraction = sum(reward_fractions) / len(reward_fractions)
                avg_reward_fractions.append(avg_reward_fraction)
                avg_step_count = sum(step_counts) / len(step_counts)
                # Hacky step to make step-counts cumulative
                if len(avg_step_counts) > 1:
                    avg_step_counts.append(avg_step_count + avg_step_counts[-1])
                else:
                    avg_step_counts.append(avg_step_count)
            # Write average rewards per episode
            writer.writerow(avg_reward_fractions)
            # Write average step count per episode
            step_writer.writerow(avg_step_counts)
            # Write the policy learned by each agent
            for i in range(len(self.agents[ensemble_key])):
                policy_writer.writerow((ensemble_key, i, self.agents[ensemble_key][i].get_learned_policy_as_string()))
            for i in range(len(self.agents[ensemble_key])):
                value_writer.writerow((ensemble_key, i, self.agents[ensemble_key][i].get_learned_state_values()))

        # This chunk runs the ensembles on all the corrupted MDPs
        if include_corruption:
            stepfile = open(os.path.join(self.results_dir, "corrupted/step_counts.csv"), 'w', newline='')
            csvfile = open(os.path.join(self.results_dir, "corrupted/exp_output.csv"), 'w', newline='')
            policyfile = open(os.path.join(self.results_dir, "corrupted/learned_policies.csv"), 'w', newline='')
            valuefile = open(os.path.join(self.results_dir, 'corrupted/learned_state_values.csv'), 'w', newline='')
            reward_writer = csv.writer(csvfile)
            step_writer = csv.writer(stepfile)
            policy_writer = csv.writer(policyfile)
            value_writer = csv.writer(valuefile)
            for ensemble_key in self.corr_agents.keys():
                print(ensemble_key)
                avg_reward_fractions = [ensemble_key]
                avg_step_counts = [ensemble_key]
                for episode in range(self.num_episodes):
                    if episode % 10 == 0:
                        print("On episode", episode)
                    reward_fractions, step_counts = self.run_ensemble(self.corr_agents[ensemble_key])
                    avg_reward_fraction = sum(reward_fractions) / len(reward_fractions)
                    avg_reward_fractions.append(avg_reward_fraction)
                    avg_step_count = sum(step_counts) / len(step_counts)
                    if len(avg_step_counts) > 1:
                        avg_step_counts.append(avg_step_count + avg_step_counts[-1])
                    else:
                        avg_step_counts.append(avg_step_count)
                # This is the fraction of the optimal policy captured, on average, per agent per episode
                reward_writer.writerow(avg_reward_fractions)
                # This is the average number of step counts per agent per episode
                step_writer.writerow(avg_step_counts)
                # Write learned policy
                for i in range(len(self.corr_agents[ensemble_key])):
                    policy_writer.writerow((ensemble_key, i, self.corr_agents[ensemble_key][i].get_learned_policy_as_string()))
                # Write learned state values
                for i in range(len(self.corr_agents[ensemble_key])):
                    value_writer.writerow((ensemble_key, i, self.corr_agents[ensemble_key][i].get_learned_state_values()))
        if include_corruption:
            return os.path.join(self.results_dir, "true/exp_output.csv"), \
                   os.path.join(self.results_dir, "true/step_counts.csv"), \
                   os.path.join(self.results_dir, "corrupted/exp_output.csv"), \
                   os.path.join(self.results_dir, "corrupted/step_counts.csv")
        else:
            return os.path.join(self.results_dir, "exp_output.csv"), os.path.join(self.results_dir, "step_counts.csv")

    def visualize_results(self, infilepath, outdirpath=None, outfilename=None):
        """
        :param infilepath: the name of the file from which to read the results of the experiment
        :param outdirpath: optional argument to folder where to save the figure generated
        :param outfilename: optional argument for name of saved file
        """
        if outdirpath is None:
            outdirpath = self.results_dir

        exp_res = open(infilepath, "r")
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
            plt.plot(episodes, [float(i) for i in mdp[1:]], label="%s" % (mdp[0],))

        leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.5)

        if outfilename is None:
            outfilename = 'true_results.png'
        plt.savefig(os.path.join(outdirpath, outfilename))
        plt.clf()

    def visualize_corrupt_results(self, infilepath, outdirpath=None, outfilename=False, graph_between=False):
        """
        Graph the results from the corrupted MDPs
        :param infilepath: name of the file with the data to be graphed
        :param outdirpath: optional argument to folder where results will be saved
        """
        if outdirpath is None:
            outdirpath = self.results_dir
        plt.style.use('seaborn-whitegrid')

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
        df[['abstr_type', 'abstr_epsilon', 'corr_type', 'corr_prop', 'batch_num']] = pd.DataFrame(df.key.tolist(),
                                                                                                  index=df.index)
        # This section calculates the averages and standard deviations of fractions of optimal value captured
        #  across different abstraction types, graphs the results across episodes, and saves the figure
        avg_df = df.groupby('batch').mean()
        std_df = df.groupby('batch').std()

        episodes = [i for i in range(1, self.num_episodes + 1)]
        for i in range(avg_df.shape[0]):
            upper = avg_df.iloc[i] + std_df.iloc[i]
            lower = avg_df.iloc[i] - std_df.iloc[i]
            plt.plot(episodes, list(avg_df.iloc[i]), label="%s" % ([avg_df.index[i][0], avg_df.index[i][3]]))
            if graph_between:
                plt.fill_between(episodes, upper, lower, alpha=0.2)
        leg = plt.legend(loc='upper left', fancybox=True)
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
            a_t = str(temp_df['abstr_type'].values[0])
            start = a_t.find('.') + 1
            end = a_t.find(':')
            a_t = a_t[start:end].lower()
            c_p = temp_df['corr_prop'].values[0]

            # Strip away everything except the data itself
            temp_df = temp_df.drop(columns=['key', 'batch', 'abstr_type', 'abstr_epsilon', 'corr_type', 'corr_prop'])
            temp_df = temp_df.set_index('batch_num')

            # Iterate through all batches and graph them
            for index, row in temp_df.iterrows():
                plt.plot(episodes, row, label="%s" % (index,))
            plt.legend(loc='best', fancybox=True)
            plt.title("%s" % (batch,))

            # This creates a nice file name for the graph
            file_name = str(a_t) + '_' +  str(c_p)
            print(file_name)
            file_name = file_name.replace('.','')
            file_name = 'corrupted/{}{}'.format(file_name, '.png')
            print(file_name)
            plt.savefig(os.path.join(outdirpath, file_name))
            plt.clf()

    def visualize_corrupt_abstraction(self, key):
        """
        TODO: Complete this
        Visualize the abstraction with the given key from self.corrupt_mdp_dict. Also prints the states where the
        corrupt and true MDPs differ
        :param key: tuple consisting of abstraction type, abstraction epsilon, corruption type, corruption proportion,
                    and batch number
        """
        print(key)
        corr_vis = vis(self.corrupt_mdp_dict[key])
        corr_vis.displayAbstractMDP()

    def get_corrupt_policy_differences(self, key):
        """
        Get the differences between the modal policy learned by the ensemble on the corrupt MDP with the given key
        and the optimal policy in the ground MDP (as dictated by VI)
        :param key: tuple of (Abstr_type, abstr_epsilon, Corr_type, proportion, batch_num)
        :return: dictionary mapping ground states to tuples of the form (optimal action from VI, modal action learned,
                    how many agents learned that action)
        """
        # This will hold final result
        policy_diff_dict = {}

        # Get a list of all ground states in the MDP
        states = self.ground_mdp.get_all_possible_states()

        # Iterate through all ground states in the MDP
        for state in states:
            # Get list of optimal actions as dictated by VI
            true_optimal_actions = self.vi.get_all_optimal_actions(state)

            # Get dictionary of optimal actions -> number of agents who learned that action as optimal for the given
            #   state
            corr_optimal_actions = self.get_optimal_actions_corrupt_mdp(state, key)

            # Get modal action(s) from corr_optimal_actions
            modal_actions = []
            action_count = 0
            for action in corr_optimal_actions.keys():
                if corr_optimal_actions[action] > action_count:
                    action_count = corr_optimal_actions[action]
            for action in corr_optimal_actions.keys():
                if corr_optimal_actions[action] == action_count:
                    modal_actions.append(action)

            # Check if any of the modal actions are not optimal under VI
            for modal_action in modal_actions:
                if modal_action not in true_optimal_actions:
                    policy_diff_dict[state] = (true_optimal_actions, modal_action, corr_optimal_actions[modal_action])

        return policy_diff_dict

    # -----------------
    # Getters & setters
    # -----------------
    def set_num_episodes(self, new_num):
        """
        Set the number of episodes for the experiment to a new value
        :param new_num: new number of episodes
        """
        if new_num <= 0:
            raise ValueError("Cannot have number of episodes less than 1. Invalid argument is " + str(new_num))
        self.num_episodes = new_num

    def get_abstr_mdp(self, key):
        """
        Query self.abstr_mdps for the given key
        :param key: Tuple of (Abstr_type, abstr_epsilon)
        """
        return self.abstr_mdp_dict[key]

    def get_corrupt_mdp(self, key):
        """
        Query self.corrupt_mdp_dict for the given key.
        :param key: Tuple of (Abstr_type, abstr_epsilon, corr_type, proportion, batch_num)
        """
        return self.corrupt_mdp_dict[key]

    def get_corruption_errors(self, key):
        """
        Return the ground states where a corrupt abstract mdp differs from the true mdp
        :param key: tuple of (Abstr_type, abstr_epsilon, corr_type, proportion, batch_num)
        :return: list of tuples representing errors of the form (ground state, true abstr state, corrupt abstr state)
        """
        corr_mdp = self.get_corrupt_mdp(key)
        true_mdp = self.get_abstr_mdp((key[0], key[1]))
        corr_abstr_dict = corr_mdp.state_abstr.get_abstr_dict()
        true_abstr_dict = true_mdp.state_abstr.get_abstr_dict()
        errors = []
        for key in true_abstr_dict.keys():
            if true_abstr_dict[key] != corr_abstr_dict[key]:
                errors.append((key, true_abstr_dict[key], corr_abstr_dict[key]))
        return errors

    def get_optimal_action_from_vi(self, state):
        """
        Query self.q_table for action values associated with state and return optimal action
        :param state: a state in the ground MDP for which the optimal action will be returned
        :return: optimal action for that state
        """
        opt_val = float("-inf")
        opt_action = None
        for key in self.vi_table.keys():
            if key[0] == state:
                if self.vi_table[key] > opt_val:
                    opt_val = self.vi_table[key]
                    opt_action = key[1]
        return opt_action

    def get_optimal_actions_corrupt_mdp(self, state, key):
        """
        Get the optimal actions from every agent in the ensemble on the corrupt MDP corresponding to the given key
        :param state: ground state
        :param key: tuple of (Abstr_type, abstr_epsilon, corruption_type, proportion, batch_num) corresponding to a
                    corrupted mdp
        :return: dictionary of actions to numbers where keys are optimal actions and values are the number of agents
                    in the ensemble for whom that is the optimal action
        """
        # 'ensemble' is a list of agents
        ensemble = self.corr_agents[key]

        # Iterate through ensemble, query each agent for the best action in the given state, and record the value in a
        #  dictionary
        opt_action_dict = defaultdict(lambda: 0)
        for agent in ensemble:
            best_action = agent.get_best_action(state)
            opt_action_dict[best_action] += 1

        return opt_action_dict

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

