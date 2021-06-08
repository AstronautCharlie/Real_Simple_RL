"""
This class creates and runs a test of q-learning on the given MDP for the given abstraction
types and epsilon values, and compares the value of each trajectory to the value of the
optimal ground-state trajectory from that point

NOTE: this class currently stores true agents and corrupted agents in different dictionaries.

#TODO: consider ignoring zeroes when making online abstraction
#TODO: see what the effects of different thresholds are
"""
from MDP.MDPClass import MDP
from MDP.StateAbstractionClass import StateAbstraction
from MDP.ValueIterationClass import ValueIteration
from MDP.AbstractMDPClass import AbstractMDP
from Agent.AgentClass import Agent
from Agent.AbstractionAgent import AbstractionAgent
from Agent.TrackingAgent import TrackingAgent
from GridWorld.GridWorldStateClass import GridWorldState
from resources.AbstractionMakers import make_abstr
from resources.AbstractionTypes import Abstr_type
from resources.AbstractionCorrupters import *
from Visualizer.AbstractGridWorldVisualizer import AbstractGridWorldVisualizer as vis
from collections import defaultdict
from util import *
from scipy.stats import norm

import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import ast
import shutil

"""
self.agents <- dictionary, ensemble_key -> list of agents [ground and true abstractions]
self.corr_agents <- dictionary, ensemble_key -> list of agents [corrupt abstractions no detach]
self.corr_detach_agents <- dictionary, ensemble_key -> list of agents [corrupt abstractions w/ detach]
"""


class Experiment():
    def __init__(self,
                 mdp,
                 abstr_epsilon_list=(),
                 corruption_list=(),
                 error_dicts=None,
                 num_corrupted_mdps=10,
                 num_agents=10,
                 num_episodes=200,
                 results_dir='exp_results',
                 agent_type='abstraction',
                 agent_exploration_epsilon=0.1,
                 agent_learning_rate=0.1,
                 decay_exploration=True,
                 exploring_starts=False,
                 step_limit=10000,
                 detach_interval=None,
                 prevent_cycles=False,
                 variance_threshold=None,
                 reset_q_value=False,
                 agent_detach='abstr',
                 detach_reassignment='group',
                 detach_points=None,
                 states_to_track=[],
                 record_trajectories=False,
                 seed=None,
                 detach_only=False,
                 track_error_states=False,
                 include_online_abstraction=False,
                 online_abstraction_training_episodes=10,
                 online_abstraction_epsilon=0.05,
                 noisy_abstr_types=[],
                 abstr_error_distribution=None,
                 abstr_error_parameters=None,
                 per_state_abstr_error_distribution=None,
                 per_state_abstr_error_parameters=None,
                 noisy_abstr_epsilon=0.0,
                 ground_only=False,
                 skip_true=False,
                 neighbor_factor=1,
                 states_per_detach=1):
        """
        Create an experiment, which will hold the ground MDP, the abstract MDPs (parameters dictated by abstr_epsilon_list),
        and an ensemble of (num_agents) q-learning agents on each MDP.
        :param mdp: MDP
        :param abstr_epsilon_list: list of tuples, where first element is abstraction type and second is the epsilon
        :param corruption_list: list of tuples, where first element is the corruption type and second is the proportion
        :param error_dictionary: dictionary mapping error states to corrupted states they will share an abstract state
                                with
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
        :param include_online_abstraction: boolean indicating whether to include a set of agents that make online
                abstractions
        """
        # Check that agent_type is valid
        if agent_type not in ['standard', 'abstraction', 'tracking']:
            raise ValueError('"agent_type" variable must be "standard", "abstraction", or "tracking". Is currently '+str(agent_type))

        self.ground_mdp = mdp
        if abstr_epsilon_list is not None:
            for val in abstr_epsilon_list:
                if val[0] not in Abstr_type or val[1] < 0 or val[1] > 1:
                    raise ValueError('Abstraction Epsilon List is invalid', abstr_epsilon_list)
        self.abstr_epsilon_list = abstr_epsilon_list
        self.corruption_list = corruption_list
        self.error_dicts = error_dicts
        self.num_agents = num_agents
        self.num_corrupted_mdps = num_corrupted_mdps
        self.results_dir = results_dir
        self.num_episodes = num_episodes
        self.agent_type = agent_type
        self.decay_exploration = decay_exploration
        self.exploring_starts = exploring_starts
        self.step_limit = step_limit
        self.agent_exploration_epsilon = agent_exploration_epsilon
        self.agent_learning_rate = agent_learning_rate
        self.detach_interval = detach_interval
        self.prevent_cycle = prevent_cycles
        self.variance_threshold = variance_threshold
        self.reset_q_value = reset_q_value
        self.agent_detach = agent_detach
        self.detach_reassignment = detach_reassignment
        self.detach_points = detach_points
        self.states_to_track = states_to_track
        self.record_trajectories = record_trajectories
        self.seed = seed
        self.detach_only = detach_only
        self.include_online_abstraction = include_online_abstraction
        self.online_abstraction_training_episodes = online_abstraction_training_episodes
        self.online_abstraction_epsilon = online_abstraction_epsilon
        self.abstr_error_distribution = abstr_error_distribution
        self.abstr_error_parameters = abstr_error_parameters
        self.noisy_abstr_types = noisy_abstr_types
        self.per_state_abstr_error_distribution = per_state_abstr_error_distribution
        self.per_state_abstr_error_parameters = per_state_abstr_error_parameters
        self.noisy_abstr_epsilon = noisy_abstr_epsilon
        self.include_noisy_abstractions = abstr_error_distribution or per_state_abstr_error_distribution
        self.ground_only = ground_only
        self.skip_true = skip_true
        self.neighbor_factor = neighbor_factor
        self.states_per_detach = 1

        # Create results dir if it doesn't exist
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        # Clear out contents of results directory
        for filename in os.listdir(self.results_dir):
            full_file = os.path.join(self.results_dir, filename)
            if os.path.isdir(full_file):
                shutil.rmtree(full_file, ignore_errors=True)
            else:
                os.remove(full_file)

        # Make sure directory has all the necessary folders (true, corrupted, corrupted_w_detach)
        for folder_to_make in ['true', 'corrupted', 'corrupted_w_detach']:
            if not os.path.exists(os.path.join(self.results_dir, folder_to_make)):
                os.makedirs(os.path.join(self.results_dir, folder_to_make))


        # Agent ensembles will be stored in a dict where key is the (abstr_type, epsilon) tuple ('ground' in the case
        # of the ground MDP) and values are lists of agents. In the case of corrupted MDPs, the key will be
        # (abstr_type, epsilon, corruption_type, proportion)
        self.agents = {}

        all_ground_states = mdp.get_all_possible_states()

        # Run Value Iteration to get q-table for abstractions and to hold value of optimal policies
        vi = ValueIteration(mdp)
        vi.run_value_iteration()
        q_table = vi.get_q_table()
        self.vi_table = q_table
        self.vi = vi

        print('Finished vi')
        # Create abstract MDPs from each element of abstr_epsilon_list. val[0] is abstraction type, val[1] is epsilon
        self.abstr_mdp_dict = {}
        file_string = 'true/abstractions.csv'
        if not os.path.exists(os.path.join(self.results_dir, 'true')):
            os.makedirs(os.path.join(self.results_dir, 'true'))
        with open(os.path.join(self.results_dir, file_string), 'w', newline='') as abstr_file:
            abstr_writer = csv.writer(abstr_file)
            for val in abstr_epsilon_list:
                print('Making abstraction', val)
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
        if self.corruption_list is not None:
            if not os.path.exists(os.path.join(self.results_dir, 'corrupted')):
                os.makedirs(os.path.join(self.results_dir, 'corrupted'))
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
                                                         #size=int(np.floor(len(self.ground_mdp.get_all_possible_states()) * prop)),
                                                         size=int(prop),
                                                         replace=False)
                    for state in states_to_corrupt:
                        while (state.x, state.y) in mdp.goal_location or len(np.unique(states_to_corrupt)) < prop:
                            state = np.random.choice(self.ground_mdp.get_all_possible_states())
                        # Record in states_to_track if track_error_states is true
                        if track_error_states:
                            self.states_to_track.append(state)
                    for key in self.abstr_mdp_dict.keys():
                        # Create a corrupt state abstraction for this batch number and list of states
                        abstr_mdp = self.abstr_mdp_dict[key]
                        c_s_a = make_corruption(abstr_mdp, states_to_corrupt, corr_type=corr_type)
                        # Get the correct and incorrect abstract states for the corrupted ground states
                        #  and write all these to a file
                        error_line = []
                        for state in states_to_corrupt:
                            true_state = abstr_mdp.get_abstr_from_ground(state)
                            corr_state = c_s_a.get_abstr_from_ground(state)
                            error_line.append(((state.x, state.y), true_state.data, corr_state.data))
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
        elif self.error_dicts is not None:
            # Record in states_to_track if applicable
            #for error_state in self.error_dicts.keys():
            #    self.states_to_track.append(error_state)

            corr_abstr_file = open(os.path.join(self.results_dir, 'corrupted/corrupted_abstractions.csv'), 'w', newline='')
            err_state_file = open(os.path.join(self.results_dir, 'corrupted/error_states.csv'), 'w', newline='')
            abstr_writer = csv.writer(corr_abstr_file)
            err_writer = csv.writer(err_state_file)

            # We generate 1 corrupt MDP per abstraction type at a time. This way, we can ensure that all
            # corrupt MDPs with the same key and batch number have the same error states. Also write to file
            # so they can be visualized later
            for j in range(len(self.error_dicts)):
                error_dict = self.error_dicts[j]
                for i in range(self.num_corrupted_mdps):
                    for key in self.abstr_mdp_dict.keys():
                        # Create a corrupt state abstraction for this batch number and list of states
                        abstr_mdp = self.abstr_mdp_dict[key]
                        c_s_a = make_corruption(abstr_mdp, reassignment_dict=error_dict)
                        # Get the correct and incorrect abstract states for the corrupted ground states
                        #  and write all these to a file
                        error_line = []
                        for state in error_dict.keys():
                            true_state = abstr_mdp.get_abstr_from_ground(state)
                            corr_state = c_s_a.get_abstr_from_ground(state)
                            error_line.append(((state.x, state.y), true_state.data, corr_state.data))
                        temp_key = (abstr_mdp.abstr_type, abstr_mdp.abstr_epsilon, 'explicit errors', j, i)
                        # Record the error states
                        err_writer.writerow((temp_key, error_line))
                        # Make an abstract MDP with this corrupted state abstraction
                        corrupt_abstr_mdp = AbstractMDP(self.ground_mdp, c_s_a)
                        # Record the dictionary describing the state abstraction
                        abstr_writer.writerow((temp_key, corrupt_abstr_mdp.abstr_to_string()))
                        self.corrupt_mdp_dict[(abstr_mdp.abstr_type,
                                               abstr_mdp.abstr_epsilon,
                                               'explicit errors',
                                               j,
                                               i)] = corrupt_abstr_mdp

        # Create random noise abstractions if that parameter is set
        if self.abstr_error_distribution or self.per_state_abstr_error_distribution:
            self.noisy_abstr_dict = {}
            if not os.path.isdir(os.path.join(self.results_dir, 'noisy')):
                os.mkdir(os.path.join(self.results_dir, 'noisy'))
            file_string = 'noisy/abstractions.csv'
            noisy_abstr_file = open(os.path.join(self.results_dir, file_string), 'w', newline='')
            noisy_abstr_writer = csv.writer(noisy_abstr_file)
            for abstr_type in self.noisy_abstr_types:
                for i in range(self.num_corrupted_mdps):
                    #TODO Check that this is returning what you think it is. This is where the problem is
                    noisy_mdp = apply_noise_from_distribution(self.ground_mdp,
                                                              abstr_type,
                                                              approximation_epsilon=self.noisy_abstr_epsilon,
                                                               distribution=self.abstr_error_distribution,
                                                               distribution_parameters=self.abstr_error_parameters,
                                                               per_state_distribution=self.per_state_abstr_error_distribution,
                                                               per_state_parameters=self.per_state_abstr_error_parameters
                                                               )
                    print(type(mdp), type(noisy_mdp))
                    self.noisy_abstr_dict[(abstr_type, i)] = noisy_mdp
                    noisy_abstr_writer.writerow((abstr_type, i, noisy_mdp.abstr_to_string()))

        print('Making agents')
        # Create agents on ground mdp
        ground_agents = []
        for i in range(self.num_agents):
            if self.seed:
                temp_seed = self.seed + i
            else:
                temp_seed = None
            temp_mdp = self.ground_mdp.copy()
            # If agent_type == 'standard', use regular q-learning. If 'abstraction', use abstraction agent
            if self.agent_type == 'standard':
                agent = Agent(temp_mdp,
                              epsilon=agent_exploration_epsilon,
                              alpha=self.agent_learning_rate,
                              decay_exploration=decay_exploration,
                              seed=temp_seed)

            # Because tracking agent doesn't support agent w/o abstraction, so make it abstraction agent
            elif self.agent_type == 'abstraction':
                agent = AbstractionAgent(temp_mdp,
                                         epsilon=agent_exploration_epsilon,
                                         alpha=self.agent_learning_rate,
                                         decay_exploration=decay_exploration,
                                         seed=temp_seed)
            else:
                print('tracking agent', i)
                agent = TrackingAgent(temp_mdp,
                                      epsilon=agent_exploration_epsilon,
                                      alpha=self.agent_learning_rate,
                                      decay_exploration=decay_exploration,
                                      seed=temp_seed,
                                      ground_states=all_ground_states)
            ground_agents.append(agent)
        self.agents['ground'] = ground_agents

        # Create MDPs for online abstraction construction if applicable
        self.online_abstr_agents = []
        if self.include_online_abstraction:
            self.online_abstr_agents = {}
            file_string = 'online/abstractions.csv'
            if not os.path.exists(os.path.join(self.results_dir, 'online')):
                os.makedirs(os.path.join(self.results_dir, 'online'))
            online_abstr_file = open(os.path.join(self.results_dir, file_string), 'w', newline='')
            online_abstr_writer = csv.writer(online_abstr_file)
            #for abstr_type, eps in self.abstr_epsilon_list:
            for j in range(self.num_corrupted_mdps):
                ensemble = []
                for i in range(self.num_agents):
                    mdp = copy.deepcopy(self.ground_mdp)
                    if self.agent_type == 'standard':
                        agent = Agent(mdp,
                                      epsilon=agent_exploration_epsilon,
                                      alpha=self.agent_learning_rate,
                                      decay_exploration=decay_exploration)
                    elif self.agent_type == 'abstraction':
                        agent = AbstractionAgent(mdp, None,
                                                 epsilon=agent_exploration_epsilon,
                                                 alpha=self.agent_learning_rate,
                                                 decay_exploration=decay_exploration)
                    elif self.agent_type == 'tracking':
                        agent = TrackingAgent(mdp, s_a=None, epsilon=agent_exploration_epsilon,
                                              alpha=self.agent_learning_rate,
                                              decay_exploration=decay_exploration,
                                              ground_states=all_ground_states)
                    else:
                        raise ValueError("Agent Type '" + str(self.agent_type) + "' is not supported")
                    ensemble.append(agent)
                self.online_abstr_agents[(j, 'online')] = ensemble

        # Create agents on abstract MDPs
        for abstract_mdp in self.abstr_mdp_dict.values():
            abstract_mdp_ensemble = []
            for i in range(self.num_agents):
                if self.seed:
                    temp_seed = self.seed + i
                else:
                    temp_seed = None
                # Create agents according to agent_type parameter
                if self.agent_type == 'standard':
                    temp_mdp = abstract_mdp.copy()
                    agent = Agent(temp_mdp,
                                  epsilon=agent_exploration_epsilon,
                                  alpha=self.agent_learning_rate,
                                  decay_exploration=decay_exploration,
                                  seed=temp_seed)
                    abstract_mdp_ensemble.append(agent)
                elif self.agent_type == 'abstraction':
                    temp_mdp = self.ground_mdp.copy()
                    s_a = copy.deepcopy(abstract_mdp.state_abstr)
                    agent = AbstractionAgent(temp_mdp,
                                             s_a,
                                             epsilon=agent_exploration_epsilon,
                                             alpha=self.agent_learning_rate,
                                             decay_exploration=decay_exploration,
                                             seed=temp_seed)
                    abstract_mdp_ensemble.append(agent)
                elif self.agent_type == 'tracking':
                    temp_mdp = self.ground_mdp.copy()
                    s_a = copy.deepcopy(abstract_mdp.state_abstr)
                    agent = TrackingAgent(temp_mdp,
                                          s_a,
                                          epsilon=agent_exploration_epsilon,
                                          alpha=self.agent_learning_rate,
                                          decay_exploration=decay_exploration)
                    abstract_mdp_ensemble.append(agent)
            self.agents[(abstract_mdp.abstr_type, abstract_mdp.abstr_epsilon)] = abstract_mdp_ensemble

        # Create agents on noisily-abstracted MDPs if that parameter is set
        if self.abstr_error_distribution:
            self.noisy_abstr_agents = {}
            self.noisy_abstr_detach_agents = {}
            for (abstr_type, num) in self.noisy_abstr_dict.keys():
                ensemble = []
                ensemble2 = []
                for i in range(self.num_agents):
                    #mdp = copy.deepcopy(self.noisy_abstr_dict[(abstr_type, num)])
                    mdp = copy.deepcopy(self.ground_mdp)
                    noisy_s_a = self.noisy_abstr_dict[(abstr_type, num)].state_abstr
                    if self.agent_type == 'tracking':
                        agent = TrackingAgent(mdp,
                                              s_a=noisy_s_a,
                                              epsilon=agent_exploration_epsilon,
                                              alpha=self.agent_learning_rate,
                                              decay_exploration=decay_exploration)
                    else:
                        raise ValueError("Agent type " + str(self.agent_type) + " not supported with noisy abstraction")
                    ensemble.append(agent)
                    t_agent = copy.deepcopy(agent)
                    ensemble2.append(t_agent)
                self.noisy_abstr_agents[(abstr_type, num)] = ensemble
                self.noisy_abstr_detach_agents[(abstr_type, num)] = ensemble2

        # Create agents on corrupted abstract MDPs. Remember that we have self.num_corrupted_mdps ensembles for each
        #  combination of abstractMDP type and entry in self.corruption_list.
        # self.corr_agents is now a dictionary mapping (abstr_type, epsilon, corr_type, proportion, batch_num) to
        #  a list of agents
        self.corr_agents = {}
        for corr_key in self.corrupt_mdp_dict.keys():
            corr_ensemble = []
            for i in range(self.num_agents):
                if self.seed:
                    temp_seed = self.seed + i
                else:
                    temp_seed = None
                    # Create agents according to agent_type parameter
                if self.agent_type == 'standard':
                    temp_mdp = self.corrupt_mdp_dict[corr_key].copy()
                    agent = Agent(temp_mdp,
                                  epsilon=agent_exploration_epsilon,
                                  alpha=self.agent_learning_rate,
                                  decay_exploration=decay_exploration,
                                  seed=temp_seed)
                    corr_ensemble.append(agent)
                else:
                    temp_mdp = copy.deepcopy(self.ground_mdp.copy())
                    corr_mdp = copy.deepcopy(self.corrupt_mdp_dict[corr_key])
                    s_a = copy.deepcopy(corr_mdp.state_abstr)
                    if self.agent_type == 'abstraction':
                        agent = AbstractionAgent(temp_mdp,
                                                 s_a,
                                                 epsilon=agent_exploration_epsilon,
                                                 alpha=self.agent_learning_rate,
                                                 decay_exploration=decay_exploration,
                                                 seed=temp_seed)
                    elif self.agent_type == 'tracking':
                        agent = TrackingAgent(temp_mdp,
                                                 s_a,
                                                 epsilon=agent_exploration_epsilon,
                                                 alpha=self.agent_learning_rate,
                                                 decay_exploration=decay_exploration)
                    else:
                        raise ValueError("Agent type \'" + str(self.agent_type) + "\' is not supported")
                    corr_ensemble.append(agent)
            self.corr_agents[corr_key] = corr_ensemble

        # If detach_state_interval is set, create another set of agents which will detach inconsistent abstract states
        #   every (detach_state_interval) episodes
        self.corr_detach_agents = {}
        for corr_key in self.corrupt_mdp_dict.keys():
            corr_ensemble = []
            for i in range(self.num_agents):
                if self.seed:
                    temp_seed = self.seed + i
                else:
                    temp_seed = None
                    # Create agents according to agent_type parameter
                if self.agent_type == 'standard':
                    temp_mdp = self.corrupt_mdp_dict[corr_key].copy()
                    agent = Agent(temp_mdp,
                                  epsilon=agent_exploration_epsilon,
                                  alpha=self.agent_learning_rate,
                                  decay_exploration=decay_exploration,
                                  seed=temp_seed)
                    corr_ensemble.append(agent)
                else:
                    temp_mdp = copy.deepcopy(self.ground_mdp.copy())
                    corr_mdp = copy.deepcopy(self.corrupt_mdp_dict[corr_key].copy())
                    s_a = copy.deepcopy(corr_mdp.state_abstr)
                    if self.agent_type == 'abstraction':
                        agent = AbstractionAgent(temp_mdp,
                                                 s_a,
                                                 epsilon=agent_exploration_epsilon,
                                                 alpha=self.agent_learning_rate,
                                                 decay_exploration=decay_exploration,
                                                 consistency_check=self.agent_detach,
                                                 detach_reassignment=self.detach_reassignment,
                                                 seed=temp_seed)
                    elif self.agent_type == 'tracking':
                        agent = TrackingAgent(temp_mdp,
                                                 s_a,
                                                 epsilon=agent_exploration_epsilon,
                                                 alpha=self.agent_learning_rate,
                                                 decay_exploration=decay_exploration,
                                                 consistency_check=self.agent_detach,
                                                 detach_reassignment=self.detach_reassignment)
                    else:
                        raise ValueError("Agent type \'" + str(self.agent_type) + "\' is not supported")
                    corr_ensemble.append(agent)
            self.corr_detach_agents[corr_key] = corr_ensemble
        print('Done making agents')

    def __str__(self):
        result = 'key: num agents\n'
        for key in self.agents.keys():
            result += str(key) + ': ' + str(len(self.agents[key])) + '\n'
        return result

    # ----------------------------
    # Functions for running agents
    # ----------------------------

    def run_trajectory(self, agent, step_limit, verbose=False):
        """
        Run an agent on its MDP until it reaches a terminal state. Record the discounted rewards achieved along the way
        and the starting state
        :param agent: Q-learning agent
        :param step_limit: limit the number of steps in the trajectory to force termination
        :return: sum_rewards: sum of discounted rewards gotten over the course of the trajectory
        :return: optimal_value: value of the starting state under the optimal policy
        :return: step_count: the number of steps taken in this trjaectory
        :return: state_occupancy: {state -> {action -> number of times that state-action pair was taken}}
        """
        step_count = 0
        # Bypass abstraction function to get current ground state
        starting_state = agent.mdp.current_state
        optimal_value = self.get_optimal_state_value(starting_state)
        current_state = starting_state

        # This will track the rewards accumulated along the trajectory
        sum_rewards = 0
        discount = 1

        # This holds the counts of each state-action pair taken
        occupancy_dict = {}
        for state in agent.mdp.get_all_possible_states():
            occupancy_dict[state] = {}
            for action in agent.mdp.actions:
                occupancy_dict[state][action] = 0

        # Explore until a terminal state is reached
        while not current_state.is_terminal():
            if step_limit is not None and step_count > step_limit:
                break
            blah, action, next_state, reward = agent.explore()
            #print(blah, action, next_state, reward)
            try:
                occupancy_dict[blah][action] += 1
            except:
                print(blah, action, next_state, reward)
                for key, value in occupancy_dict.items():
                    print(key, value)
                quit()
            sum_rewards += discount * reward
            current_state = next_state
            discount *= agent.mdp.gamma
            step_count += 1
        if verbose:
            print('Finished episode with step count', step_count)
        # Reset agent's MDP to initial state
        # If exploring_starts, then randomly select state-action pair to be applied first
        if self.exploring_starts:
            random_state = np.random.choice(agent.mdp.get_all_possible_states())
            while random_state.is_terminal():
                random_state = np.random.choice(agent.mdp.get_all_possible_states())
            agent.mdp.set_current_state(random_state)
            random_action = np.random.choice(agent.mdp.actions)
            agent.apply_action_w_update(random_action)
        else:
            agent.mdp.reset_to_init()
        # Return the sum of discounted rewards from the trajectory and value of optimal policy
        return sum_rewards, optimal_value, step_count, occupancy_dict

    def run_ensemble(self, ensemble, verbose=False):
        """
        Run each agent in an ensemble for a single trajectory and track the fraction of the optimal value the
        agent captured
        :param ensemble: a list of agents
        :return: a list of the fraction of the optimal value each agent captured
        """
        reward_fractions = []
        step_counts = []
        occupancy_dict = {}
        for state in ensemble[0].mdp.get_all_possible_states():
            occupancy_dict[state] = {}
            for action in ensemble[0].mdp.actions:
                occupancy_dict[state][action] = 0
        for agent in ensemble:
            actual_rewards, optimal_rewards, step_count, agent_occupancy_dict = self.run_trajectory(agent, step_limit=self.step_limit, verbose=verbose)
            try:
                reward_fractions.append(actual_rewards / optimal_rewards)
            except:
                reward_fractions.append(actual_rewards / (0.99**18))
            step_counts.append(step_count)
            for state, action_dict in occupancy_dict.items():
                for action, count in action_dict.items():
                    occupancy_dict[state][action] += agent_occupancy_dict[state][action]
        return reward_fractions, step_counts, occupancy_dict

    def detach_ensemble(self, ensemble, verbose=False):
        """
        For each agent, check the consistency of abstract states and detach inconsistent ground states
        """
        detached_states = {}
        for i in range(len(ensemble)):
            agent = ensemble[i]
            print('On agent', i)
            print('Current abstraction is', end = ' ')
            s_a = agent.s_a.abstr_dict
            for key, value in s_a.items():
                print(key, value, end='   ')
            error_states = agent.detach_inconsistent_states(variance_threshold=self.variance_threshold,
                                                            prevent_cycles=self.prevent_cycle,
                                                            reset_q_value=self.reset_q_value,
                                                            verbose=False)
            print('Error states are', end = ' ')
            for state in error_states:
                print(state, end = ' ')
            print()
            if error_states is not None:
                detached_states[i] = error_states
        return detached_states

    def run_all_ensembles(self,
                          include_corruption=False,
                          skip_ground=False,
                          verbose=False):
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
        print('About to run all ensembles')
        # This chunk runs the ensembles on the ground MDP and all the correct abstract MDPs
        # csvfile contains the average rewards obtained in every episode
        # stepfile contains the average number of steps per episode
        step_file = open(os.path.join(self.results_dir, "true/step_counts.csv"), 'w', newline='')
        csv_file = open(os.path.join(self.results_dir, "true/exp_output.csv"), 'w', newline='')
        policy_file = open(os.path.join(self.results_dir, "true/learned_policies.csv"), 'w', newline='')
        value_file = open(os.path.join(self.results_dir, 'true/learned_state_values.csv'), 'w', newline='')
        state_occupancy_file = open(os.path.join(self.results_dir, 'true/state_occupancy.csv'), 'w', newline='')
        per_agent_reward_file = open(os.path.join(self.results_dir, 'true/per_agent_rewards.csv'), 'w', newline='')
        per_agent_step_file = open(os.path.join(self.results_dir, 'true/per_agent_steps.csv'), 'w', newline='')
        writer = csv.writer(csv_file)
        step_writer = csv.writer(step_file)
        policy_writer = csv.writer(policy_file)
        value_writer = csv.writer(value_file)
        state_occupancy_writer = csv.writer(state_occupancy_file)
        per_agent_writer = csv.writer(per_agent_reward_file)
        per_agent_step_writer = csv.writer(per_agent_step_file)
        if self.states_to_track:
            q_value_file = open(os.path.join(self.results_dir, 'true/q_values.csv'), 'w', newline='')
            #q_value_df = pd.DataFrame(columns=['ensemble_key', 'agent_num', 'episode', 'state', 'action', 'q_value'])
            q_value_lists = [['ensemble_key', 'agent_num', 'episode', 'state', 'action', 'q_value']]

        # The for loop below iterates through all ensembles in self.agents. Each key in the self.agents dict
        #  represents a single ensemble of agents on a single MDP
        if not self.skip_true:
            for ensemble_key in self.agents.keys():
                print('Running true agents:', ensemble_key)
                if skip_ground and ensemble_key[0] == "ground":
                    print('SKIPPING GROUND')
                    continue
                if self.ground_only and ensemble_key != "ground":
                    print('SKIPPING', ensemble_key[0])
                    continue
                # The two lines below are done so that each list of data is preceded by the key for that
                #  ensemble.
                #  This is done so we know which data belongs to which ensemble
                avg_reward_fractions = [ensemble_key]
                avg_step_counts = [ensemble_key]
                reward_fractions_dict = {}
                steps_dict = {}
                for i in range(len(self.agents[ensemble_key])):
                    reward_fractions_dict[i] = []
                for i in range(len(self.agents[ensemble_key])):
                    steps_dict[i] = []
                    #steps_dict[i] = 0
                for episode in range(self.num_episodes):
                    reward_fractions, step_counts, occupancy_dict = self.run_ensemble(self.agents[ensemble_key], verbose=verbose)
                    # Track individual rewards
                    for i in range(len(reward_fractions)):
                        reward_fractions_dict[i].append(reward_fractions[i])
                    # Track individual steps
                    for i in range(len(step_counts)):
                        steps_dict[i].append(step_counts[i])
                        #steps_dict[i] += step_counts[i]
                    # We average these across the whole ensemble
                    avg_reward_fraction = sum(reward_fractions) / len(reward_fractions)
                    avg_reward_fractions.append(avg_reward_fraction)
                    avg_step_count = sum(step_counts) / len(step_counts)
                    # Hacky step to make step-counts cumulative
                    if len(avg_step_counts) > 1:
                        avg_step_counts.append(avg_step_count + avg_step_counts[-1])
                    else:
                        avg_step_counts.append(avg_step_count)
                    # Write state-occupancy counts - no longer needed

                    if episode == self.num_episodes - 1:
                        for i in range(len(self.agents[ensemble_key])):
                            agent = self.agents[ensemble_key][i]
                            #print('not here', ensemble_key)
                            counts = list(agent.state_occupancy_record.values())
                            counts.sort(reverse=True)
                            for val in counts:
                                for state, count in agent.state_occupancy_record.items():
                                    if count == val:
                                        abstr_state = agent.get_abstr_from_ground(state)
                                        #print(state, count, end=' ')
                                        t = ''
                                        for s in agent.get_ground_states_from_abstract_state(abstr_state):
                                            t += str(s)
                                        #print()
                                        action_counts = ''
                                        for action in self.ground_mdp.actions:
                                            try:
                                                action_counts += str(action) + ': ' + str(agent.state_action_pair_counts[state][action]) + ', '
                                            except:
                                                print('failed in run_ensembles with state, action', state, action)
                                                quit()
                                        state_occupancy_writer.writerow((ensemble_key, i, state, count, action_counts, t))

                    # Write q-values
                    if self.states_to_track:
                        for i in range(len(self.agents[ensemble_key])):
                            q_table = self.agents[ensemble_key][i].get_q_table()
                            for key, value in q_table.items():
                                if key[0] in self.states_to_track:
                                    q_value_lists.append([ensemble_key, i, episode, key[0], key[1], value])
                                    #temp_dict = {'ensemble_key': [ensemble_key], 'agent_num': [i], 'episode': [episode],
                                    #             'state': [key[0]], 'action': [key[1]], 'q_value': [value]}
                                    #temp_df = pd.DataFrame.from_dict(temp_dict)
                                    #q_value_df = pd.concat([q_value_df, temp_df], ignore_index=True)
                # Write per-agent rewards
                for agent_num, rewards in reward_fractions_dict.items():
                    per_agent_writer.writerow((ensemble_key, agent_num, rewards))
                # Write per-agent step counts
                for agent_num, steps in steps_dict.items():
                    per_agent_step_writer.writerow((ensemble_key, agent_num, steps))
                # Write average rewards per episode
                writer.writerow(avg_reward_fractions)
                # Write average step count per episode
                step_writer.writerow(avg_step_counts)
                # Write the policy learned by each agent
                for i in range(len(self.agents[ensemble_key])):
                    policy_writer.writerow((ensemble_key, i, self.agents[ensemble_key][i].get_learned_policy_as_string()))
                for i in range(len(self.agents[ensemble_key])):
                    value_writer.writerow((ensemble_key, i, self.agents[ensemble_key][i].get_learned_state_values()))
        else:
            print('Skipping true abstractions and ground MDP')
        # Track Q-values if that option is set
        if self.states_to_track:
            q_value_df = pd.DataFrame(q_value_lists[1:], columns=q_value_lists[0])
            #print(q_value_lists)
            #print(q_value_df)
            q_value_df.to_csv(q_value_file, index=False)

        # This chunk runs the ensembles on all the corrupted MDPs
        if include_corruption:
            # Create files
            print('including corruption')
            if not os.path.exists(os.path.join(self.results_dir, 'corrupted')):
                os.makedirs(os.path.join(self.results_dir, 'corrupted'))
            stepfile = open(os.path.join(self.results_dir, "corrupted/step_counts.csv"), 'w', newline='')
            csvfile = open(os.path.join(self.results_dir, "corrupted/exp_output.csv"), 'w', newline='')
            policyfile = open(os.path.join(self.results_dir, "corrupted/learned_policies.csv"), 'w', newline='')
            valuefile = open(os.path.join(self.results_dir, 'corrupted/learned_state_values.csv'), 'w', newline='')
            reward_writer = csv.writer(csvfile)
            step_writer = csv.writer(stepfile)
            policy_writer = csv.writer(policyfile)
            value_writer = csv.writer(valuefile)
            if self.states_to_track:
                q_value_file = open(os.path.join(self.results_dir, 'corrupted/q_values.csv'), 'w', newline='')
                q_value_lists = [['ensemble_key', 'agent_num', 'episode', 'state', 'action', 'q_value']]

            for ensemble_key in self.corr_agents.keys():
                print(ensemble_key)
                avg_reward_fractions = [ensemble_key]
                avg_step_counts = [ensemble_key]
                for episode in range(self.num_episodes):
                    if episode % 1 == 0:
                        print("On episode", episode)
                        reward_fractions, step_counts, _ = self.run_ensemble(self.corr_agents[ensemble_key], verbose=True)
                    else:
                        reward_fractions, step_counts, _ = self.run_ensemble(self.corr_agents[ensemble_key])
                    avg_reward_fraction = sum(reward_fractions) / len(reward_fractions)
                    avg_reward_fractions.append(avg_reward_fraction)
                    avg_step_count = sum(step_counts) / len(step_counts)
                    if len(avg_step_counts) > 1:
                        avg_step_counts.append(avg_step_count + avg_step_counts[-1])
                    else:
                        avg_step_counts.append(avg_step_count)
                    # Write q-values
                    if self.states_to_track:
                        for i in range(len(self.corr_agents[ensemble_key])):
                            q_table = self.corr_agents[ensemble_key][i].get_q_table()
                            for key, value in q_table.items():
                                if key[0] in self.states_to_track:
                                    q_value_lists.append([ensemble_key, i, episode, key[0], key[1], value])
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
            if self.states_to_track:
                q_value_df = pd.DataFrame(q_value_lists[1:], columns=q_value_lists[0])
                q_value_df.to_csv(q_value_file, index=False)

        print('Detach agents')
        # This chunk runs the detach agents, if that parameter is set
        if self.detach_interval is not None or self.detach_points is not None:
            if not os.path.exists(os.path.join(self.results_dir, 'corrupted_w_detach')):
                os.makedirs(os.path.join(self.results_dir, 'corrupted_w_detach'))
            stepfile = open(os.path.join(self.results_dir, "corrupted_w_detach/step_counts.csv"), 'w', newline='')
            csvfile = open(os.path.join(self.results_dir, "corrupted_w_detach/exp_output.csv"), 'w', newline='')
            policyfile = open(os.path.join(self.results_dir, "corrupted_w_detach/learned_policies.csv"), 'w', newline='')
            valuefile = open(os.path.join(self.results_dir, 'corrupted_w_detach/learned_state_values.csv'), 'w', newline='')
            detachfile = open(os.path.join(self.results_dir, 'corrupted_w_detach/detached_states.csv'), 'w', newline='')
            finalSAfile = open(os.path.join(self.results_dir, 'corrupted_w_detach/final_s_a.csv'), 'w', newline='')
            reward_writer = csv.writer(csvfile)
            step_writer = csv.writer(stepfile)
            policy_writer = csv.writer(policyfile)
            value_writer = csv.writer(valuefile)
            detach_writer = csv.writer(detachfile)
            finalSA_writer = csv.writer(finalSAfile)
            if self.states_to_track:
                q_value_file = open(os.path.join(self.results_dir, 'corrupted/q_values.csv'), 'w', newline='')
                q_value_lists = [['ensemble_key', 'agent_num', 'episode', 'state', 'action', 'q_value']]

            for ensemble_key in self.corr_detach_agents.keys():
                print(ensemble_key, 'detaching states')
                avg_reward_fractions = [ensemble_key]
                avg_step_counts = [ensemble_key]
                detached_state_record = {}
                for agent_num in range(self.num_agents):
                    detached_state_record[agent_num] = []
                total_steps = 0
                for episode in range(self.num_episodes):
                    if episode % 1 == 0:
                        print("On episode", episode, end = ' ')
                        reward_fractions, step_counts, _ = self.run_ensemble(self.corr_detach_agents[ensemble_key], verbose=True)
                    else:
                        reward_fractions, step_counts, _ = self.run_ensemble(self.corr_detach_agents[ensemble_key])
                    avg_reward_fraction = sum(reward_fractions) / len(reward_fractions)
                    avg_reward_fractions.append(avg_reward_fraction)
                    avg_step_count = sum(step_counts) / len(step_counts)
                    if len(avg_step_counts) > 1:
                        avg_step_counts.append(avg_step_count + avg_step_counts[-1])
                    else:
                        avg_step_counts.append(avg_step_count)
                    for counts in step_counts:
                        total_steps += counts
                    print('Running total of steps is', total_steps)

                    # See if episode number is one meriting a detach
                    do_detach = False
                    if self.detach_interval and episode % self.detach_interval == 0:
                        do_detach = True
                    elif self.detach_points and episode in self.detach_points:
                        do_detach = True
                    if episode > 0 and do_detach:
                        detach_dict = self.detach_ensemble(self.corr_detach_agents[ensemble_key], verbose=verbose)
                        for key, value in detach_dict.items():
                            detached_states = []
                            for state in value:
                                detached_states.append(((state.x, state.y), episode))
                            """
                            for state in value:
                                print(state)
                                if isinstance(detached_states, list):
                                    print("if, detached_states are", detached_states)
                                    for state in detached_states:
                                        print('state is', state)
                                        detached_states.append(((state.x, state.y), episode))
                                else:
                                #try:
                                    print("else")
                                    detached_states.append(((state.x, state.y), episode))
                                #except:
                            """
                            #print('Detached states to be written to file is', detached_states)
                            detached_state_record[key] += detached_states
                    # If on the last episode, write all the detached states to a file
                    if episode == self.num_episodes - 1:
                        for i in range(len(self.corr_detach_agents[ensemble_key])):
                            detach_writer.writerow((ensemble_key, i, detached_state_record[i]))
                    # Write q-values
                    if self.states_to_track:
                        for i in range(len(self.corr_agents[ensemble_key])):
                            q_table = self.corr_agents[ensemble_key][i].get_q_table()
                            for key, value in q_table.items():
                                if key[0] in self.states_to_track:
                                    q_value_lists.append([ensemble_key, i, episode, key[0], key[1], value])
                # This is the fraction of the optimal policy captured, on average, per agent per episode
                reward_writer.writerow(avg_reward_fractions)
                # This is the average number of step counts per agent per episode
                step_writer.writerow(avg_step_counts)
                # Write learned policy
                for i in range(len(self.corr_detach_agents[ensemble_key])):
                    policy_writer.writerow(
                        (ensemble_key, i, self.corr_detach_agents[ensemble_key][i].get_learned_policy_as_string()))
                # Write learned state values
                for i in range(len(self.corr_detach_agents[ensemble_key])):
                    value_writer.writerow(
                        (ensemble_key, i, self.corr_detach_agents[ensemble_key][i].get_learned_state_values()))
                # Write final state abstractions
                for i in range(len(self.corr_detach_agents[ensemble_key])):
                    finalSA_writer.writerow(
                        (ensemble_key, i, self.corr_detach_agents[ensemble_key][i].get_abstraction_as_string())
                    )
                # Write summary of original abstraction and summary of each agent abstraction
                #  Get number of abstract states and constituent counts
            if self.states_to_track:
                q_value_df = pd.DataFrame(q_value_lists[1:], columns=q_value_lists[0])
                q_value_df.to_csv(q_value_file, index=False)

        print('Running online agents...')
        # Run online agents
        if self.online_abstr_agents:
            if not os.path.exists(os.path.join(self.results_dir, 'online')):
                os.makedirs(os.path.join(self.results_dir, 'online'))
            stepfile = open(os.path.join(self.results_dir, "online/step_counts.csv"), 'w', newline='')
            csvfile = open(os.path.join(self.results_dir, "online/exp_output.csv"), 'w', newline='')
            policyfile = open(os.path.join(self.results_dir, "online/learned_policies.csv"), 'w', newline='')
            valuefile = open(os.path.join(self.results_dir, 'online/learned_state_values.csv'), 'w', newline='')
            detachfile = open(os.path.join(self.results_dir, 'online/detached_states.csv'), 'w', newline='')
            onlineabstrfile = open(os.path.join(self.results_dir, 'online/starting_s_a.csv'), 'w', newline='')
            finalSAfile = open(os.path.join(self.results_dir, 'online/final_s_a.csv'), 'w', newline='')
            state_occ_file = open(os.path.join(self.results_dir, 'online/state_occupancy.csv'), 'w', newline='')
            per_agent_reward_file = open(os.path.join(self.results_dir, 'online/per_agent_rewards.csv'), 'w', newline='')
            per_agent_step_file = open(os.path.join(self.results_dir, 'online/per_agent_steps.csv'), 'w', newline='')
            reward_writer = csv.writer(csvfile)
            step_writer = csv.writer(stepfile)
            policy_writer = csv.writer(policyfile)
            value_writer = csv.writer(valuefile)
            detach_writer = csv.writer(detachfile)
            finalSA_writer = csv.writer(finalSAfile)
            startingSA_writer = csv.writer(onlineabstrfile)
            state_occ_writer = csv.writer(state_occ_file)
            per_agent_writer = csv.writer(per_agent_reward_file)
            per_agent_step_writer = csv.writer(per_agent_step_file)

            for ensemble_key in self.online_abstr_agents.keys():
                print('online abstraction: ' + str(ensemble_key))
                abstr_type = ensemble_key[0]
                avg_reward_fractions = [ensemble_key]
                avg_step_counts = [ensemble_key]
                detached_state_record = {}
                for agent_num in range(self.num_agents):
                    detached_state_record[agent_num] = []
                per_agent_reward_dict = {}
                per_agent_step_dict = {}
                for i in range(len(self.online_abstr_agents[ensemble_key])):
                    per_agent_reward_dict[i] = []
                for i in range(len(self.online_abstr_agents[ensemble_key])):
                    per_agent_step_dict[i] = []
                for episode in range(self.num_episodes):
                    if episode % 1 == 0:
                        print('On episode', episode)
                        reward_fractions, step_counts, _ = self.run_ensemble(self.online_abstr_agents[ensemble_key],
                                                                             verbose=True)
                    else:
                        reward_fractions, step_counts, _ = self.run_ensemble(self.online_abstr_agents[ensemble_key])
                    for i in range(len(reward_fractions)):
                        per_agent_reward_dict[i].append(reward_fractions[i])
                    for i in range(len(step_counts)):
                        per_agent_step_dict[i].append(step_counts[i])
                        #per_agent_step_dict[i] += step_counts[i]
                    avg_reward_fraction = sum(reward_fractions) / len(reward_fractions)
                    avg_reward_fractions.append(avg_reward_fraction)
                    avg_step_count = sum(step_counts) / len(step_counts)
                    if len(avg_step_counts) > 1:
                        avg_step_counts.append(avg_step_count + avg_step_counts[-1])
                    else:
                        avg_step_counts.append(avg_step_count)

                    # Check if we should do detach
                    do_detach = False
                    if self.detach_interval and episode % self.detach_interval == 0:
                        do_detach = True
                    elif self.detach_points and episode in self.detach_points:
                        do_detach = True
                    if episode > 0 and do_detach:
                        #for agent in self.online_abstr_agents[ensemble_key]:
                        '''
                        for i in range(len(self.online_abstr_agents[ensemble_key])):
                            detached_states = []
                            agent = self.online_abstr_agents[ensemble_key][i]
                            #for i in range(self.detaches_per_episode):
                            for i in range(self.states_per_detach):
                                most_visited_state = agent.get_most_visited_grouped_state()
                                if most_visited_state:
                                    agent.detach_state(most_visited_state, reset_q_value=self.reset_q_value)
                                    try:
                                        detached_states.append(((most_visited_state.x, most_visited_state.y), episode))
                                    except:
                                        detached_states.append((most_visited_state.data, episode))
                                    detached_state_record[i] += detached_states
                        '''
                        # Also do most volatile detach
                        detach_dict = self.detach_ensemble(self.online_abstr_agents[ensemble_key], verbose=verbose)
                        print('detach dict is')
                        for key, value in detach_dict.items():
                            print(key, end=' ')
                            for val in value:
                                print(val, end = ' ')
                            print()
                            detached_states = []
                            for state in value:
                                try:
                                    detached_states.append(((state.x, state.y), episode))
                                except:
                                    detached_states.append(((state._taxi_loc, state._passenger_loc, state._goal_loc), episode))
                            print('Detached states to be written to file is', detached_states)
                            detached_state_record[key] += detached_states

                    # If on the last episode, write all the detached states to a file
                    if episode == self.num_episodes - 1:
                        for i in range(len(self.online_abstr_agents[ensemble_key])):
                            detach_writer.writerow((ensemble_key, i, detached_state_record[i]))
                        #for agent in self.online_abstr_agents[ensemble_key]:
                        # Write state occupancy counts to file - no longer needed

                        for i in range(len(self.online_abstr_agents[ensemble_key])):
                            agent = self.online_abstr_agents[ensemble_key][i]
                            print(ensemble_key)
                            counts = list(agent.state_occupancy_record.values())
                            counts.sort(reverse=True)
                            for val in counts:
                                for state, count in agent.state_occupancy_record.items():
                                    if count == val:
                                        #print(state, count)
                                        s = ''
                                        abstr_state = agent.get_abstr_from_ground(state)
                                        for other_state in agent.get_ground_states_from_abstract_state(abstr_state):
                                            if other_state != state:
                                                s += str(other_state)
                                        action_counts = ''
                                        for action in self.ground_mdp.actions:
                                            action_counts += str(action) + ': ' + str(agent.state_action_pair_counts[state][action]) + ', '
                                        state_occ_writer.writerow((ensemble_key, i, state, count, action_counts, s))

                    # Write q-values
                    if self.states_to_track:
                        for i in range(len(self.online_abstr_agents[ensemble_key])):
                            q_table = self.online_abstr_agents[ensemble_key][i].get_q_table()
                            for key, value in q_table.items():
                                if key[0] in self.states_to_track:
                                    q_value_lists.append([ensemble_key, i, episode, key[0], key[1], value])
                    # Check if we are supposed to make an online abstraction:
                    if episode == self.online_abstraction_training_episodes:
                        for i in range(len(self.online_abstr_agents[ensemble_key])):
                            abstr_type = ensemble_key[0]
                            temp_agent = self.online_abstr_agents[ensemble_key][i]
                            #temp_agent.s_a = make_abstr(temp_agent._q_table,
                            #                            abstr_type,
                            #                            epsilon=self.online_abstraction_epsilon,
                            #                            combine_zeroes=False,
                            #                            seed=None)
                            #temp_agent.make_online_abstraction(abstr_type,
                            #                                   epsilon=self.online_abstraction_epsilon)
                            temp_agent.make_temporal_abstraction(n=self.neighbor_factor)
                            startingSA_writer.writerow(
                                (ensemble_key, i, episode, temp_agent.get_abstraction_as_string())
                            )

                # Per-agent rewards
                for i in range(self.num_agents):
                    per_agent_writer.writerow((ensemble_key, i, per_agent_reward_dict[i]))
                # Per-agent step counts
                for i in range(self.num_agents):
                    per_agent_step_writer.writerow((ensemble_key, i, per_agent_step_dict[i]))
                # Fraction of optimal policy value
                reward_writer.writerow(avg_reward_fractions)
                # Average number of step counts per episode
                step_writer.writerow(avg_step_counts)
                # Learned policy
                for i in range(len(self.online_abstr_agents[ensemble_key])):
                    policy_writer.writerow(
                        (ensemble_key, i, self.online_abstr_agents[ensemble_key][i].get_learned_policy_as_string())
                    )
                # Learned state values
                for i in range(len(self.online_abstr_agents[ensemble_key])):
                    value_writer.writerow(
                        (ensemble_key, i, self.online_abstr_agents[ensemble_key][i].get_learned_state_values())
                    )
                # Final state abstractions
                for i in range(len(self.online_abstr_agents[ensemble_key])):
                    finalSA_writer.writerow(
                        (ensemble_key, i, self.online_abstr_agents[ensemble_key][i].get_abstraction_as_string())
                    )

            if self.states_to_track:
                q_value_df = pd.DataFrame(q_value_lists[1:], columns=q_value_lists[0])
                q_value_df.to_csv(q_value_file, index=False)
            csvfile.close()
            csvfile = open(os.path.join(self.results_dir, "online/exp_output.csv"), 'r', newline='')
            # TODO this
            # Graph online abstraction results
            outdirpath = self.results_dir
            plt.style.use('seaborn-whitegrid')
            ax = plt.subplot(111)

            names = ['key'] + [i for i in range(self.num_episodes)]
            df = pd.read_csv(csvfile, names=names)
            df['key'] = df['key'].str.replace('(', '').str.replace(')', '').str.replace(', ', ',').str.split(',')

            def convert_key_to_tuple(row):
                return tuple(row['key'])

            df['key'] = df.apply(convert_key_to_tuple, axis=1)
            df[['abstr_type', 'batch_num']] = pd.DataFrame(df.key.tolist(),
                                                           index=df.index)

            episodes = [i for i in range(1, self.num_episodes + 1)]
            for abstr_type in df['abstr_type'].drop_duplicates():
                temp_df = df.loc[df['abstr_type'] == abstr_type]
                for num in temp_df['batch_num'].drop_duplicates():
                    s_temp_df = temp_df.loc[temp_df['batch_num'] == num]
                    a_t = str(s_temp_df['abstr_type'].values[0])
                    start = a_t.find('.') + 1
                    end = a_t.find(':')
                    a_t = a_t[start:end].lower()

                    if 'a_star' in a_t:
                        abstr_string = 'A* Abstraction'
                    elif 'pi_star' in a_t:
                        abstr_string = 'Pi* Abstraction'
                    elif 'q_star' in a_t:
                        abstr_string = 'Q* Abstraction'
                    else:
                        abstr_string = 'Ground MDP'

                    # Iterate through all batches and graph them
                    for index, row in s_temp_df.iterrows():
                        batch_num = row['batch_num']
                        row = row.drop(columns=['key', 'abstr_type', 'batch_num'])
                        #print("batch num is", batch_num)
                        values = []
                        #print(episodes)
                        #print(row)
                        for ep in episodes:
                            values.append(row[ep - 1])
                        # plt.plot(episodes, row, label=str(batch_num))
                        #print(values)
                        plt.plot(episodes, values, label=str(batch_num))
                    plt.legend(bbox_to_anchor=(0.75, 0.2), loc='upper left', fancybox=True)
                    plt.xlabel('Episode Number')
                    plt.ylim(0,1)
                    plt.ylabel('Average Proportion of Optimal Policy Captured By Ensemble')
                    plt.title('Neighbor-based Abstraction with n=' + str(self.neighbor_factor))
                    #plt.title(abstr_string + ', Noisy Abstraction')

                # This creates a nice file name for the graph
                file_name = 'neighbor_based_results'#str('a_t')
                print(file_name)
                file_name = file_name.replace('.', '')
                file_name = 'online' + '/{}{}'.format(file_name, '.png')
                print(file_name)
                plt.savefig(os.path.join(outdirpath, file_name))
                plt.clf()

        # Run noisy abstraction agents
        if self.include_noisy_abstractions:
            if not os.path.exists(os.path.join(self.results_dir, 'noisy')):
                os.makedirs(os.path.join(self.results_dir, 'noisy'))
            if not os.path.exists(os.path.join(self.results_dir, 'noisy_w_detach')):
                os.makedirs(os.path.join(self.results_dir, 'noisy_w_detach'))
            stepfile = open(os.path.join(self.results_dir, "noisy/step_counts.csv"), 'w', newline='')
            stepfile_detach = open(os.path.join(self.results_dir, "noisy_w_detach/step_counts.csv"), 'w', newline='')
            csvfile = open(os.path.join(self.results_dir, "noisy/exp_output.csv"), 'w', newline='')
            csvfile_detach = open(os.path.join(self.results_dir, "noisy_w_detach/exp_output.csv"), 'w', newline='')
            policyfile = open(os.path.join(self.results_dir, "noisy/learned_policies.csv"), 'w', newline='')
            policyfile_detach = open(os.path.join(self.results_dir, "noisy_w_detach/learned_policies.csv"), 'w', newline='')
            valuefile = open(os.path.join(self.results_dir, 'noisy/learned_state_values.csv'), 'w', newline='')
            valuefile_detach = open(os.path.join(self.results_dir, 'noisy_w_detach/learned_state_values.csv'), 'w', newline='')
            detachfile = open(os.path.join(self.results_dir, 'noisy_w_detach/detached_states.csv'), 'w', newline='')
            reward_writer = csv.writer(csvfile)
            reward_detach_writer = csv.writer(csvfile_detach)
            step_writer = csv.writer(stepfile)
            step_detach_writer = csv.writer(stepfile_detach)
            policy_writer = csv.writer(policyfile)
            policy_detach_writer = csv.writer(policyfile_detach)
            value_writer = csv.writer(valuefile)
            value_detach_writer = csv.writer(valuefile_detach)
            detach_writer = csv.writer(detachfile)

            if self.states_to_track:
                q_value_file = open(os.path.join(self.results_dir, 'noisy/q_values.csv'), 'w', newline='')
                q_value_detach_file = open(os.path.join(self.results_dir, 'noisy/q_values.csv'), 'w', newline='')
                #q_value_df = pd.DataFrame(columns=['ensemble_key', 'agent_num', 'episode', 'state', 'action', 'q_value'])
                q_value_lists = [['ensemble_key', 'agent_num', 'episode', 'state', 'action', 'q_value']]
                q_value_detach_lists = [['ensemble_key', 'agent_num', 'episode', 'state', 'action', 'q_value']]


            for ensemble_key in self.noisy_abstr_agents.keys():
                print('Noisy abstraction', ensemble_key)
                detached_state_record = {}
                for agent_num in range(self.num_agents):
                    detached_state_record[agent_num] = []
                avg_reward_fractions = [ensemble_key]
                avg_reward_fractions_d = [ensemble_key]
                avg_step_counts = [ensemble_key]
                avg_step_counts_d = [ensemble_key]
                for episode in range(self.num_episodes):
                    if episode % 1 == 0:
                        print("On episode", episode)
                        reward_fractions, step_counts, _ = self.run_ensemble(self.noisy_abstr_agents[ensemble_key], verbose=True)
                        reward_fractions_d, step_counts_d, _ = self.run_ensemble(self.noisy_abstr_detach_agents[ensemble_key], verbose=True)
                    else:
                        reward_fractions, step_counts, _ = self.run_ensemble(self.noisy_abstr_agents[ensemble_key])
                        reward_fractions_d, step_counts_d, _ = self.run_ensemble(self.noisy_abstr_detach_agents[ensemble_key])
                    avg_reward_fraction = sum(reward_fractions) / len(reward_fractions)
                    avg_reward_fraction_d = sum(reward_fractions_d) / len(reward_fractions_d)

                    avg_reward_fractions.append(avg_reward_fraction)
                    avg_reward_fractions_d.append(avg_reward_fraction_d)

                    avg_step_count = sum(step_counts) / len(step_counts)
                    avg_step_count_d = sum(step_counts_d) / len(step_counts_d)

                    if len(avg_step_counts) > 1:
                        avg_step_counts.append(avg_step_count + avg_step_counts[-1])
                    else:
                        avg_step_counts.append(avg_step_count)
                    if len(avg_step_counts_d) > 1:
                        avg_step_counts_d.append(avg_step_count_d + avg_step_counts_d[-1])

                    # Write q-values
                    if self.states_to_track:
                        for i in range(len(self.noisy_abstr_agents[ensemble_key])):
                            q_table = self.noisy_abstr_agents[ensemble_key][i].get_q_table()
                            q_table_d = self.noisy_abstr_detach_agents[ensemble_key][i].get_q_table()
                            for key, value in q_table.items():
                                if key[0] in self.states_to_track:
                                    q_value_lists.append([ensemble_key, i, episode, key[0], key[1], value])
                                    #temp_dict = {'ensemble_key': [ensemble_key], 'agent_num': [i], 'episode': [episode],
                                    #             'state': [key[0]], 'action': [key[1]], 'q_value': [value]}
                                    #temp_df = pd.DataFrame.from_dict(temp_dict)
                                    #q_value_df = pd.concat([q_value_df, temp_df], ignore_index=True)
                            for key, value in q_table_d.items():
                                if key[0] in self.states_to_track:
                                    q_value_detach_lists.append([ensemble_key, i, episode, key[0], key[1], value])
                    # Check if we should do detach
                    do_detach = False
                    if self.detach_interval and episode % self.detach_interval == 0:
                        do_detach = True
                    elif self.detach_points and episode in self.detach_points:
                        do_detach = True
                    if episode > 0 and do_detach:
                        detach_dict = self.detach_ensemble(self.noisy_abstr_detach_agents[ensemble_key], verbose=verbose)
                        for key, value in detach_dict.items():
                            detached_states = []
                            for state in value:
                                detached_states.append(((state.x, state.y), episode))
                            print('Detached states to be written to file is', detached_states)
                            detached_state_record[key] += detached_states
                    # If on the last episode, write all the detached states to a file
                    if episode == self.num_episodes - 1:
                        for i in range(len(self.noisy_abstr_detach_agents[ensemble_key])):
                            detach_writer.writerow((ensemble_key, i, detached_state_record[i]))
                # This is the fraction of the optimal policy captured, on average, per agent per episode
                reward_writer.writerow(avg_reward_fractions)
                reward_detach_writer.writerow(avg_reward_fractions_d)
                # This is the average number of step counts per agent per episode
                step_writer.writerow(avg_step_counts)
                step_detach_writer.writerow(avg_step_counts_d)
                # Write learned policy
                for i in range(len(self.noisy_abstr_agents[ensemble_key])):
                    policy_writer.writerow((ensemble_key, i, self.noisy_abstr_agents[ensemble_key][i].get_learned_policy_as_string()))
                    policy_detach_writer.writerow((ensemble_key, i, self.noisy_abstr_detach_agents[ensemble_key][i].get_learned_policy_as_string()))
                # Write learned state values
                for i in range(len(self.noisy_abstr_agents[ensemble_key])):
                    value_writer.writerow((ensemble_key, i, self.noisy_abstr_agents[ensemble_key][i].get_learned_state_values()))
                    value_detach_writer.writerow((ensemble_key, i, self.noisy_abstr_detach_agents[ensemble_key][i].get_learned_state_values()))
            if self.states_to_track:
                q_value_df = pd.DataFrame(q_value_lists[1:], columns=q_value_lists[0])
                #print(q_value_df)
                q_value_df.to_csv(q_value_file, index=False)

                q_value_df = pd.DataFrame(q_value_detach_lists[1:], columns=q_value_detach_lists[0])
                q_value_df.to_csv(q_value_detach_file, index=False)

            csvfile.close()
            csvfile = open(os.path.join(self.results_dir, "noisy/exp_output.csv"), 'r', newline='')

            # Graph noisy results
            outdirpath = self.results_dir
            plt.style.use('seaborn-whitegrid')
            ax = plt.subplot(111)

            names = ['key'] + [i for i in range(self.num_episodes)]
            df = pd.read_csv(csvfile, names=names)
            df['key'] = df['key'].str.replace('(', '').str.replace(')', '').str.replace(', ', ',').str.split(',')
            def convert_key_to_tuple(row):
                return tuple(row['key'])
            df['key'] = df.apply(convert_key_to_tuple, axis=1)
            df[['abstr_type', 'batch_num']] = pd.DataFrame(df.key.tolist(),
                                                           index=df.index)

            episodes = [i for i in range(1, self.num_episodes + 1)]
            for abstr_type in df['abstr_type'].drop_duplicates():
                temp_df = df.loc[df['abstr_type'] == abstr_type]
                for num in temp_df['batch_num'].drop_duplicates():
                    s_temp_df = temp_df.loc[temp_df['batch_num'] == num]
                    a_t = str(s_temp_df['abstr_type'].values[0])
                    start = a_t.find('.') + 1
                    end = a_t.find(':')
                    a_t = a_t[start:end].lower()

                    if 'a_star' in a_t:
                        abstr_string = 'A* Abstraction'
                    elif 'pi_star' in a_t:
                        abstr_string = 'Pi* Abstraction'
                    elif 'q_star' in a_t:
                        abstr_string = 'Q* Abstraction'
                    else:
                        abstr_string = 'Ground MDP'

                    # Iterate through all batches and graph them
                    for index, row in s_temp_df.iterrows():
                        batch_num = row['batch_num']
                        row = row.drop(columns=['key', 'abstr_type', 'batch_num'])
                        #print("batch num is", batch_num)
                        values = []
                        #print(episodes)
                        #print(row)
                        for ep in episodes:
                            values.append(row[ep-1])
                        #plt.plot(episodes, row, label=str(batch_num))
                        #print(values)
                        plt.plot(episodes, values, label=str(batch_num))
                    plt.legend(bbox_to_anchor=(0.75, 0.2), loc='upper left', fancybox=True)
                    plt.xlabel('Episode Number')
                    plt.ylabel('Average Proportion of Optimal Policy Captured By Ensemble')
                    plt.title(abstr_string + ', Noisy Abstraction')

                # This creates a nice file name for the graph
                file_name = str(a_t)
                print(file_name)
                file_name = file_name.replace('.', '')
                file_name = 'noisy' + '/{}{}'.format(file_name, '.png')
                print(file_name)
                plt.savefig(os.path.join(outdirpath, file_name))
                plt.clf()

            # Graph noisy_w_detach results
            outdirpath = self.results_dir
            plt.style.use('seaborn-whitegrid')
            ax = plt.subplot(111)

            csvfile_detach.close()
            csvfile_detach = open(os.path.join(self.results_dir, "noisy_w_detach/exp_output.csv"), 'r', newline='')


            names = ['key'] + [i for i in range(self.num_episodes)]
            df = pd.read_csv(csvfile_detach, names=names)
            df['key'] = df['key'].str.replace('(', '').str.replace(')', '').str.replace(', ', ',').str.split(',')
            def convert_key_to_tuple(row):
                return tuple(row['key'])
            df['key'] = df.apply(convert_key_to_tuple, axis=1)
            df[['abstr_type', 'batch_num']] = pd.DataFrame(df.key.tolist(),
                                                           index=df.index)

            episodes = [i for i in range(1, self.num_episodes + 1)]
            for abstr_type in df['abstr_type'].drop_duplicates():
                temp_df = df.loc[df['abstr_type'] == abstr_type]
                for num in temp_df['batch_num'].drop_duplicates():
                    print('detach graph', abstr_type, num)
                    s_temp_df = temp_df.loc[temp_df['batch_num'] == num]
                    a_t = str(s_temp_df['abstr_type'].values[0])
                    start = a_t.find('.') + 1
                    end = a_t.find(':')
                    a_t = a_t[start:end].lower()

                    if 'a_star' in a_t:
                        abstr_string = 'A* Abstraction'
                    elif 'pi_star' in a_t:
                        abstr_string = 'Pi* Abstraction'
                    elif 'q_star' in a_t:
                        abstr_string = 'Q* Abstraction'
                    else:
                        abstr_string = 'Ground MDP'

                    # Iterate through all batches and graph them
                    for index, row in s_temp_df.iterrows():
                        batch_num = row['batch_num']
                        row = row.drop(columns=['key', 'abstr_type', 'batch_num'])
                        #print("batch num is", batch_num)
                        values = []
                        #print(episodes)
                        #print(row)
                        for ep in episodes:
                            values.append(row[ep-1])
                        #plt.plot(episodes, row, label=str(batch_num))
                        #print(values)
                        plt.plot(episodes, values, label=str(batch_num))
                    plt.legend(bbox_to_anchor=(0.75, 0.2), loc='upper left', fancybox=True)
                    plt.xlabel('Episode Number')
                    plt.ylabel('Average Proportion of Optimal Policy Captured By Ensemble')
                    plt.title(abstr_string + ', Noisy Abstraction')

                # This creates a nice file name for the graph
                file_name = str(a_t)
                print(file_name)
                file_name = file_name.replace('.', '')
                file_name = 'noisy_w_detach' + '/{}{}'.format(file_name, '.png')
                print(file_name)
                plt.savefig(os.path.join(outdirpath, file_name))
                plt.clf()




        # Return files
        #if include_corruption and (self.detach_points or self.detach_interval):
            #if self.detach_interval is None:
            #    return os.path.join(self.results_dir, "true/exp_output.csv"), \
            #            os.path.join(self.results_dir, "true/step_counts.csv"), \
            #            os.path.join(self.results_dir, "corrupted/exp_output.csv"), \
            #            os.path.join(self.results_dir, "corrupted/step_counts.csv")
            #else:
        print('returning1')
        return os.path.join(self.results_dir, "true/exp_output.csv"), \
                os.path.join(self.results_dir, "true/step_counts.csv"), \
                os.path.join(self.results_dir, "corrupted/exp_output.csv"), \
                os.path.join(self.results_dir, "corrupted/step_counts.csv"), \
                os.path.join(self.results_dir, 'corrupted_w_detach/exp_output.csv'), \
                os.path.join(self.results_dir, 'corrupted_w_detach/step_counts.csv')
        '''
        elif include_corruption:
            print('returning2')
            return os.path.join(self.results_dir, "true/exp_output.csv"), \
                    os.path.join(self.results_dir, "true/step_counts.csv"), \
                    os.path.join(self.results_dir, "corrupted/exp_output.csv"), \
                    os.path.join(self.results_dir, "corrupted/step_counts.csv")
        elif self.detach_interval or self.detach_reassignment:
            print('returning3')
            return os.path.join(self.results_dir, "true/exp_output.csv"), \
                    os.path.join(self.results_dir, "true/step_counts.csv"), \
                    os.path.join(self.results_dir, 'corrupted_w_detach/exp_output.csv'), \
                    os.path.join(self.results_dir, 'corrupted_w_detach/step_counts.csv')
        else:
            print('returning4')
            return os.path.join(self.results_dir, "exp_output.csv"), os.path.join(self.results_dir, "step_counts.csv")
        '''

    def record_volatilities(self, outfile_name=None):
        """
        Calculate volatilities based on the current q-value update records. Only works when running tracking agent.
        Outfile is an optional argument naming the file
        """
        if self.agent_type != 'tracking':
            raise ValueError("Cannot run record_volatilities when agent_type is not 'tracking'")

        if outfile_name is None:
            outfile = 'volatilities.csv'
        outfile = os.path.join(self.results_dir, outfile_name)

        # Calculate volatilities and write to df for...
        # ...true abstract agents
        vol_df = pd.DataFrame()
        for agent_key in self.agents.keys():
            if agent_key != "ground":
                for agent in self.agents[agent_key]:
                    i = 0
                    vol_table = agent.calculate_normalized_volatility()
                    # vol_table: {abstr_state -> {action -> normalized std dev scaled by sqrt population}}
                    temp_dict = {'ensemble_key': [],
                                 'agent_num': [],
                                 'abstr_state': [],
                                 'action': [],
                                 'normalized_vol': []}
                    for abstr_state, action_to_vol in vol_table.items():
                        for action, vol in action_to_vol.items():
                            temp_dict['ensemble_key'].append(agent_key)
                            temp_dict['agent_num'].append(i)
                            temp_dict['abstr_state'].append(abstr_state)
                            temp_dict['action'].append(action)
                            temp_dict['normalized_vol'].append(vol)
                    i += 1
                    temp_df = pd.DataFrame.from_dict(temp_dict)
                    vol_df = vol_df.append(temp_df)
                    #vol_df.append(pd.DataFrame.from_dict(temp_dict))

        # ...corrupt agents (no detach)
        if (self.corruption_list or self.error_dicts) and not self.detach_only:
            for agent_key in self.corr_agents.keys():
                for agent in self.corr_agents[agent_key]:
                    i = 0
                    vol_table = agent.calculate_normalized_volatility()
                    # vol_table: {abstr_state -> {action -> normalized std dev scaled by sqrt population}}
                    temp_dict = {'ensemble_key': [],
                                 'agent_num': [],
                                 'abstr_state': [],
                                 'action': [],
                                 'normalized_vol': []}
                    for abstr_state, action_to_vol in vol_table.items():
                        for action, vol in action_to_vol.items():
                            temp_dict['ensemble_key'].append(agent_key)
                            temp_dict['agent_num'].append(i)
                            temp_dict['abstr_state'].append(abstr_state)
                            temp_dict['action'].append(action)
                            temp_dict['normalized_vol'].append(vol)
                    i += 1
                    temp_df = pd.DataFrame.from_dict(temp_dict)
                    vol_df = vol_df.append(temp_df)

        # ...corrupt agents w/ detach
        if self.detach_interval or self.detach_points:
            for agent_key in self.corr_detach_agents.keys():
                for agent in self.corr_detach_agents[agent_key]:
                    i = 0
                    vol_table = agent.calculate_normalized_volatility()
                    # vol_table: {abstr_state -> {action -> normalized std dev scaled by sqrt population}}
                    temp_dict = {'ensemble_key': [],
                                 'agent_num': [],
                                 'abstr_state': [],
                                 'action': [],
                                 'normalized_vol': []}
                    for abstr_state, action_to_vol in vol_table.items():
                        for action, vol in action_to_vol.items():
                            temp_dict['ensemble_key'].append(agent_key)
                            temp_dict['agent_num'].append(i)
                            temp_dict['abstr_state'].append(abstr_state)
                            temp_dict['action'].append(action)
                            temp_dict['normalized_vol'].append(vol)
                    i += 1
                    temp_df = pd.DataFrame.from_dict(temp_dict)
                    vol_df = vol_df.append(temp_df)

        # Write to file
        vol_df.to_csv(outfile, index=False)


    # -----------------------
    # Visualization functions
    # -----------------------
    def visualize_results(self, infilepath, outdirpath=None, outfilename=None, title=None):
        """
        :param infilepath: the name of the file from which to read the results of the experiment
        :param outdirpath: optional argument to folder where to save the figure generated
        :param outfilename: optional argument for name of saved file
        """
        if outdirpath is None:
            outdirpath = self.results_dir

        exp_res = open(infilepath, "r")
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
        plt.ylabel('Average Proportion of Optimal Policy Captured By Ensemble')
        plt.ylim(0, 1)
        plt.suptitle('Q-Learning Performance in True Abstractions and Ground MDP')
        leg = ax.legend(bbox_to_anchor=(0.6, 0.3), loc='upper left', ncol=1, mode="expand", shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.5)

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
        Graph the results from the corrupted MDPs
        :param infilepath: name of the file with the data to be graphed
        :param outdirpath: optional argument to folder where results will be saved
        """
        if outdirpath is None:
            outdirpath = self.results_dir
        ax = plt.subplot(111)

        # Read in data as dataframe, get 'key' which consists of the abstraction type, the abstraction epsilon,
        #  the corruption value, and the number within that batch
        # infile looks like: key | ep 1 | ep 2 | ep3...
        names = ['key'] + [i for i in range(self.num_episodes)]
        df = pd.read_csv(infilepath, names=names)
        print(infilepath)
        # Turn string representing key into a list of values
        df['key'] = df['key'].str.replace('(', '').str.replace(')','').str.replace(', ',',').str.split(',')

        # This extracts the batch number from the key and parses the key into usable columns
        def remove_batch_num(row):
            return tuple(row['key'][:-1])
        def convert_key_to_tuple(row):
            return tuple(row['key'])
        # key is the unique identifier for a single ensemble on one corrupt MDP
        #print(df.to_string())
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
            plt.ylim(0, 1)
            plt.plot(episodes, list(avg_df.iloc[i]), label="%s" % ([avg_df.index[i][0], avg_df.index[i][3]]))
            if graph_between:
                plt.fill_between(episodes, upper, lower, alpha=0.2)
        leg = plt.legend(bbox_to_anchor=(0.6, 0.25), loc='upper left', fancybox=True)
        plt.suptitle(title)
        plt.xlabel('Episode Number')
        plt.ylabel('Average Proportion of Optimal Policy Captured By Ensemble')
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

            def parse_batch(batch_string):
                batch_list = ast.literal_eval(batch_string)
                abstr_tag = batch_list[0]
                if 'A_STAR' in abstr_tag:
                    abstr_string = 'A* Abstraction'
                elif 'Q_STAR' in abstr_tag:
                    abstr_string = 'Q* Abstraction'
                elif 'PI_STAR' in abstr_tag:
                    abstr_string = 'Pi* Abstraction'
                else:
                    abstr_string = None
                if 'explicit errors' in batch_list[2]:
                    corr_tag = 'Explicit'
                else:
                    corr_tag = batch_list[2]
                mdp_num = batch_list[3]
                return abstr_string, corr_tag, mdp_num

            abstr_string = ''
            if 'a_star' in a_t:
                abstr_string = 'A* Abstraction'
            elif 'pi_star' in a_t:
                abstr_string = 'Pi* Abstraction'
            elif 'q_star' in a_t:
                abstr_string = 'Q* Abstraction'
            else:
                abstr_string = 'Ground MDP'

            # Iterate through all batches and graph them
            for index, row in temp_df.iterrows():
                plt.plot(episodes, row, label="%s" % (index,))
            plt.legend(bbox_to_anchor=(0.75,0.2), loc='upper left', fancybox=True)
            plt.xlabel('Episode Number')
            plt.ylabel('Average Proportion of Optimal Policy Captured By Ensemble')
            plt.title(abstr_string + ', ' + c_p  + ' Error State(s)')

            # This creates a nice file name for the graph
            file_name = str(a_t) + '_' +  str(c_p)
            print(file_name)
            file_name = file_name.replace('.', '')
            if individual_mdp_dir is None:
                individual_mdp_dir = 'corrupted'
            file_name = individual_mdp_dir + '/{}{}'.format(file_name, '.png')
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

    # -----------------
    # Getters & setters
    # -----------------
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

