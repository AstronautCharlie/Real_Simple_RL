'''
This class creates and runs a test of q-learning on the given MDP for the given abstraction
types and epsilon values, and compares the value of each trajectory to the value of the
optimal ground-state trajectory from that point
'''
from MDP.MDPClass import MDP
from MDP.StateAbstractionClass import StateAbstraction
from MDP.ValueIterationClass import ValueIteration
from MDP.AbstractMDPClass import AbstractMDP
from Agent.AgentClass import Agent
from resources.AbstractionMakers import make_abstr
from resources.AbstractionTypes import Abstr_type
import csv
import os
import matplotlib.pyplot as plt
import numpy as np

out_path = 'results'

class Experiment():
    def __init__(self, mdp, abstr_epsilon_list=[], num_agents=10):
        '''
        Create an experiment, which will hold the ground MDP, the abstract MDPs (parameters dictated by abstr_epsilon_list),
        and an ensemble of (num_agents) q-learning agents on each MDP.
        :param mdp: MDP
        :param abstr_epsilon_list: list of tuples, where first element is abstraction type and
        second is the epsilon
        '''
        self.ground_mdp = mdp
        for val in abstr_epsilon_list:
            if val[0] not in Abstr_type or val[1] < 0 or val[1] > 1:
                raise ValueError('Abstraction Epsilon List is invalid', abstr_epsilon_list)
        self.abstr_epsilon_list = abstr_epsilon_list
        self.num_agents = num_agents
        # Agent ensembles will be stored in a dict where key is the (abstr_type, epsilon) tuple ('ground' in the case
        # of the ground MDP) and values are lists of agents
        self.agents = {}

        # Run Value Iteration to get q-table for abstractions and to hold value of optimal policies
        vi = ValueIteration(mdp)
        vi.run_value_iteration()
        q_table = vi.get_q_table()
        self.vi_table = q_table

        # Create abstract MDPs for element of abstr_epsilon_list:
        self.abstract_mdps = []
        for val in abstr_epsilon_list:
            state_abstr = make_abstr(q_table, val[0], val[1])
            self.abstract_mdps.append(AbstractMDP(mdp, state_abstr))

        # Create agents on ground mdp
        ground_agents = []
        for i in range(self.num_agents):
            mdp = self.ground_mdp.copy()
            agent = Agent(mdp)
            ground_agents.append(agent)
        self.agents['ground'] = ground_agents

        # Create agents on abstract MDPs
        for abstract_mdp in self.abstract_mdps:
            abstract_mdp_ensemble = []
            for i in range(self.num_agents):
                mdp = abstract_mdp.copy()
                agent = Agent(mdp)
                abstract_mdp_ensemble.append(agent)
            self.agents[(abstract_mdp.abstr_type, abstract_mdp.abstr_epsilon)] = abstract_mdp_ensemble

    def __str__(self):
        result = 'key: num agents\n'
        for key in self.agents.keys():
            result += str(key) + ': ' + str(len(self.agents[key])) + '\n'
        return result

    def run_trajectory(self, agent, step_limit=10000):
        '''
        Run an agent on its MDP until it reaches a terminal state. Record the discounted rewards achieved along the way
        and the starting state
        :param agent: Q-learning agent
        :param step_limit: limit the number of steps in the trajectory to force termination
        :return: reward: sum of discounted rewards gotten over the course of the trajectory
        :return: optimal_reward: value of the state under the optimal policy
        '''
        step_count = 0
        # Bypass abstraction function to get current ground state
        starting_state = agent.mdp.current_state
        optimal_value = self.get_optimal_state_value(starting_state)
        #print("starting at: ", starting_state)
        current_state = starting_state

        # This will track the rewards accumulated along the trajectory
        sum_rewards = 0
        discount = 1

        # Explore until a terminal state is reached
        while not current_state.is_terminal() and step_count < step_limit:
            _, action, next_state, reward = agent.explore()
            sum_rewards += discount * reward
            current_state = next_state
            discount *= agent.mdp.gamma
            step_count += 1
        # Reset agent's MDP to initial state
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
        for agent in ensemble:
            actual_rewards, optimal_rewards,_ = self.run_trajectory(agent)
            reward_fractions.append(actual_rewards / optimal_rewards)
        return reward_fractions

    def run_all_ensembles(self, num_episodes):
        """
        Run all ensembles in the experiment for the given number of episodes and write the average fraction of optimal
        reward achieved by each ensemble at each episode to a file
        :param num_episodes: the number of episodes for which we run the experiment
        :return: a path to a file with the average fraction of the optimal reward achieved by each ensemble at each episode
        """
        outpath = os.path.join(out_path, "exp_results.csv")
        with open(outpath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for ensemble_key in self.agents.keys():
                print(ensemble_key)
                avg_reward_fractions = [ensemble_key]
                for episode in range(num_episodes):
                    print("On episode", episode)
                    reward_fractions = self.run_ensemble(self.agents[ensemble_key])
                    avg_reward_fraction = sum(reward_fractions) / len(reward_fractions)
                    avg_reward_fractions.append(avg_reward_fraction)
                    print(reward_fractions)
                    print(avg_reward_fraction)
                writer.writerow(avg_reward_fractions)
        return outpath

    def plot_results(self, outpath):
        """
        Plot the experiment results in the given outpath
        :param outpath: path to a csv file of experiment results, one row per ensemble
        """
        with open(outpath, 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            for row in plots:
                label = row[0]
                data = row[1:]
                plt.plot(data, label=label)
                print(data)
        #print(plt.yticks())
        plt.show()



    # -------
    # Utility
    # -------
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


