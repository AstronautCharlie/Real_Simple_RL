"""
Graph the Q-value of states tracked in an experiment, stored in (folder)/q_values.csv. Takes in a
"""
import os
import shutil
import itertools
from util import *
from ast import literal_eval
import numpy as np
import matplotlib.pyplot as plt

class QValueVisualizer():
    def __init__(self,
                 experiment=None,
                 results_dir=None,
                 states_to_track=None):
        if not (experiment or (results_dir and states_to_track)):
            raise ValueError("Q-Visualizer needs an experiment or a results directory and states to track")
        if experiment:
            self.experiment = experiment
            self.results_dir = self.experiment.results_dir
            self.states_to_track = self.experiment.states_to_track
        else:
            self.results_dir = results_dir
            self.states_to_track = states_to_track

        # Read in q-values into a dataframe
        true_q_df = pd.read_csv(os.path.join(self.results_dir, 'true/q_values.csv'), index_col=0)
        corr_q_df = pd.read_csv(os.path.join(self.results_dir, 'corrupted/q_values.csv'), index_col=0)
        self.q_value_df = pd.concat([true_q_df, corr_q_df])
        if os.path.isfile(os.path.join(self.results_dir, 'corrupted_w_detach/q_values.csv')):
            self.detach_q_df = pd.read_csv(os.path.join(self.results_dir, 'corrupted_w_detach/q_values.csv'), index_col=0)
            self.q_value_df = pd.concat([self.q_value_df, self.detach_q_df])
        else:
            print('No file found for detached q values')
            self.detach_q_df = None
        # Convert state to tuple
        def convert_to_tuple(state_string):
            return literal_eval(state_string)
        # Hack-fix
        self.q_value_df = self.q_value_df[self.q_value_df['agent_num'] != 'agent_num']
        self.q_value_df['state'] = self.q_value_df['state'].apply(convert_to_tuple)

        # Parse ensemble_key
        # Parse ensemble_key
        self.q_value_df['abstr_type'] = self.q_value_df['ensemble_key'].apply(get_abstr_type_from_ensemble_key)
        self.q_value_df['abstr_eps'] = self.q_value_df['ensemble_key'].apply(get_abstr_eps_from_ensemble_key)
        self.q_value_df['corr_type'] = self.q_value_df['ensemble_key'].apply(get_corr_type_from_ensemble_key)
        self.q_value_df['corr_prop'] = self.q_value_df['ensemble_key'].apply(get_corr_prop_from_ensemble_key)
        self.q_value_df['mdp_num'] = self.q_value_df['ensemble_key'].apply(get_mdp_num_from_ensemble_key)

        # Make folder to hold graphs
        vis_dir = os.path.join(self.results_dir,'q_value_graphs')
        if not os.path.isdir(vis_dir):
            os.mkdir(vis_dir)


    def graph_q_values(self,
                       graph_true=True,
                       graph_corrupt=True,
                       graph_detach=True,
                       aggregate=True,
                       abstr_subset=None,
                       error_dict_subset=None,
                       mdp_subset=None):
        """
        Graph the q-values for states in experiment.states_to_track.
        If aggregate, average across agents and plot with standard deviation
        If not aggregate, graph each agent individually
        Each subset argument is a list specifying which abstr_types/error_dict_numbers/mdp_nums to focus on
        Graph_true, graph_detach, and graph_corrupt all flag whether or not to graph the ensembles in those environments

        Create q_value folder in results dir
        Iterate through states to track
        Apply state, abstr_subset, error_dict_subset, mdp_subset
        For each abstraction, error_dict_entry, mdp_num,
            if aggregate:
                create graph
            else:
                create folder for abstraction/error_dict_entry/mdp_num
                create graph for each agent in folder
        """
        # Make clean graph folder
        graph_folder = os.path.join(self.results_dir, 'q_value_graphs')
        if not os.path.isdir(graph_folder):
            os.mkdir(graph_folder)
        else:
            shutil.rmtree(graph_folder)
            os.mkdir(graph_folder)

        # Graph each state separately, so iterate one at a time
        for tracked_state in self.states_to_track:
            # Subset for given state
            temp_df = self.q_value_df.loc[self.q_value_df['state'] == tracked_state]

            # Get each combination of abstraction type, error_dict_num, and mdp_num
            abstr_types = temp_df['abstr_type'].dropna().unique()
            corr_type = temp_df['corr_type'].dropna().unique()
            corr_prop = temp_df['corr_prop'].dropna().unique()
            mdp_num = temp_df['mdp_num'].dropna().unique()
            combo_list = list(itertools.product(abstr_types, corr_type, corr_prop, mdp_num))
            # Add in true values (where everything null except abstraction type)
            for type in abstr_types:
                combo_list.append((type, None, None, None, None))

            # Iterate through each combination of values, subset df, and if df is non-empty then get
            #  average/stdev of agents in that combo and graph it
            for combo in combo_list:
                # If ground/true abstraction, match on abstraction type only
                if pd.isna(combo[1]):
                    subset_df = temp_df.loc[(temp_df['abstr_type'] == combo[0])
                                            & temp_df['corr_type'].isna()
                                            & temp_df['corr_prop'].isna()
                                            & temp_df['mdp_num'].isna()]

                # Else match on everything
                else:
                    subset_df = temp_df.loc[(temp_df['abstr_type'] == combo[0])
                                            & (temp_df['corr_type'] == combo[1])
                                            & (temp_df['corr_prop'] == combo[2])
                                            & (temp_df['mdp_num'] == combo[3])]
                if subset_df.empty:
                    continue

                # If aggregate, take mean/std dev across all agents
                if aggregate:
                    # Get mean and stdev of Q-values across agents for each episode
                    agg_df = subset_df.groupby(['episode', 'action'], as_index=False).agg({'q_value': [np.mean, np.std]})

                    # Graph the q-value of each action over time
                    fig, ax = plt.subplots()
                    for action in agg_df['action'].unique():
                        temp = agg_df.loc[agg_df['action'] == action]
                        ax.plot(temp['episode'], temp[('q_value','mean')], alpha=0.75, label=action)
                        ax.fill_between(temp['episode'],
                                        temp[('q_value','mean')] - temp[('q_value', 'std')],
                                        temp[('q_value','mean')] + temp[('q_value', 'std')],
                                        alpha=0.3)
                        ax.legend()
                        fig.suptitle((tracked_state, combo))
                    # Save figure
                    abstr_name = abstr_to_string(combo[0])
                    if pd.isna(combo[1]):
                        fig_name = str(abstr_name) + '_true_mdp' + str(combo[3]) + '_state' + str(tracked_state)
                    else:
                        fig_name = str(abstr_name) + '_corr_errorclass' + str(combo[2]) + '_mdp' + str(combo[3]) + '_state'\
                                   + str(tracked_state)
                    plt.savefig(os.path.join(graph_folder, fig_name + '.jpg'))
                    plt.close()
                # If not aggregate, graph each agent separately
                else:
                    # Graph q-value of each action over time
                    fig, ax = plt.subplots()
                    action_list = subset_df['action'].unique()
                    agent_nums = subset_df['agent_num'].unique()
                    for agent_num in agent_nums:
                        for action in action_list:
                            q_value_data = subset_df.loc[(subset_df['agent_num'] == agent_num)
                                                         & (subset_df['action'] == action)]['q_value']
                            x_axis = np.arange(len(q_value_data))
                            ax.plot(x_axis, q_value_data, label=action)
                            ax.legend()
                            fig.suptitle((tracked_state, combo, agent_num))
                        # Save figure
                        abstr_name = abstr_to_string(combo[0])
                        if pd.isna(combo[1]):
                            fig_name = str(abstr_name) + '_true_mdp' + str(combo[3]) + '_state' + str(tracked_state) +\
                                       '_agent' + str(agent_num)
                        else:
                            fig_name = str(abstr_name) + '_corr_errorclass' + str(combo[2]) + '_mdp' + str(combo[3])\
                                       + '_state' + str(tracked_state) + '_agent' + str(agent_num)
                        plt.savefig(os.path.join(graph_folder, fig_name + '.jpg'))
                        plt.cla()
                    plt.close()

    def get_q_value_df(self):
        return self.q_value_df


# Testing purposes only
if __name__ == '__main__':
    v = QValueVisualizer(results_dir='../exp_output/hot')
    df = v.get_q_value_df()
    #print(df.to_string())

    v.graph_q_values(aggregate=False)


