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
import matplotlib.colors as colors


from MDP.ValueIterationClass import ValueIteration

class QValueVisualizer():
    def __init__(self,
                 experiment=None,
                 results_dir=None,
                 states_to_track=None):
        if not (experiment or (results_dir and states_to_track)):
            raise ValueError("Q-Visualizer needs an experiment or a results directory and states to track")
        """
        if experiment:
            self.experiment = experiment
            self.results_dir = self.experiment.results_dir
            self.states_to_track = self.experiment.states_to_track
        else:
            self.results_dir = results_dir
            self.states_to_track = states_to_track
        """
        self.experiment = experiment
        if self.experiment:
            self.results_dir = self.experiment.results_dir
            self.states_to_track = self.experiment.states_to_track
        else:
            self.results_dir = results_dir
            self.states_to_track = states_to_track

        # Read in q-values into a dataframe
        true_q_df = pd.read_csv(os.path.join(self.results_dir, 'true/q_values.csv'))
        corr_q_df = pd.read_csv(os.path.join(self.results_dir, 'corrupted/q_values.csv'))
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
                       aggregate=True):
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

    def visualize_q_value_error(self,
                                folder,
                                mdp,
                                episodes):
        """
        Create graphs showing the difference in Q-value between the true Q-value (as determined by value iteration)
        and the Q-values learned by the agents corresponding to the ensemble given by 'folder'

        :param folder: string indicating the folder containing the q-values of interest
        :param mdp: MDP (required for value iteration)
        :param episodes: list of episode numbers for which the errors will be calculated
        """
        # Locate file
        if self.experiment:
            q_value_folder = os.path.join(self.experiment.results_dir, folder)
        else:
            q_value_folder = os.path.join(self.results_dir, folder)
        if not os.path.exists(q_value_folder):
            raise ValueError('Q-value file ' + str(q_value_folder) + ' does not exist')

        # Read in dataframe
        q_value_df = pd.read_csv(os.path.join(q_value_folder, 'q_values.csv'), header=0, error_bad_lines=False)

        # Create df holding true q-values from value iteration
        vi = ValueIteration(mdp)
        vi.run_value_iteration()
        true_q_values = vi.get_q_table()
        true_q_value_lists = []
        for (state, action), value in true_q_values.items():
            true_q_value_lists.append([state, action, value])
        names = ['state', 'action', 'true q_value']
        true_q_value_df = pd.DataFrame(true_q_value_lists, columns=names)
        true_q_value_df['state'] = true_q_value_df['state'].astype(str)
        true_q_value_df['action'] = true_q_value_df['action'].astype(str)

        # Join dfs and calculate errors
        joined_df = q_value_df.merge(true_q_value_df, on=['state', 'action'])
        joined_df['error'] = joined_df['q_value'] - joined_df['true q_value']
        #print(joined_df[:10].to_string())
        #print(joined_df['ensemble_key'].unique())

        # Convert state to literal tuple
        def lit_eval(val):
            return ast.literal_eval(val)
        joined_df['state'] = joined_df['state'].apply(lit_eval)

        # Create 2d array of all states
        states = []
        for i in range(11, 0, -1):
            row = []
            for j in range(1, 12):
                row.append((j, i))
            states.append(row)
        #print(states)

        # Graph the errors for each ensemble and episode number given
        for episode in episodes:
            for key in joined_df['ensemble_key'].unique():
                print(key)
                if key != "ground":
                    abstr_type = key.split(',')[0].strip('(')
                    if 'PI' in abstr_type:
                        abstr_type = 'Pi*'
                    elif 'A_STAR' in abstr_type:
                        abstr_type = 'A*'
                    elif 'Q_STAR' in abstr_type:
                        abstr_type = 'Q*'
                    try:
                        num = key.split(',')[1].strip(')')
                    except:
                        print(abstr_type)
                        print(key)
                        quit()
                    title = str(key) + ', episode ' + str(episode)
                else:
                    abstr_type = 'ground'
                    num = ''
                fig, axs = plt.subplots(2, 2)

                #print(key, episode)

                # Subset for the given ensemble/episode
                temp_df = joined_df.loc[(joined_df['episode'] == episode)
                                        & (joined_df['ensemble_key'] == key)]

                # Average error across all ensembles
                temp_df = temp_df[['state', 'action', 'error']]
                temp_df = temp_df.groupby(['state', 'action'], as_index=False).mean()
                #print(temp_df.to_string())

                # This will hold the array mapping action to error-per-state
                error_dict = {}

                # Create 2d-array of errors where position corresponds to square location.
                # This is hacky, but it gets the data into a heatmap-able form
                for i in range(len(states)):
                    row = states[i]
                    up_row = np.array([])
                    down_row = np.array([])
                    left_row = np.array([])
                    right_row = np.array([])
                    for j in range(len(row)):
                        state_df = temp_df.loc[temp_df['state'] == states[i][j]]
                        if state_df.empty:
                            up_row = np.append(up_row, 0)
                            down_row = np.append(down_row, 0)
                            left_row = np.append(left_row, 0)
                            right_row = np.append(right_row, 0)

                        else:
                            up_df = state_df.loc[state_df['action'] == 'Dir.UP']
                            if up_df.empty:
                                up_row = np.append(up_row, 0)
                            else:
                                up_row = np.append(up_row, up_df['error'].values[0])

                            down_df = state_df.loc[state_df['action'] == 'Dir.DOWN']
                            if down_df.empty:
                                down_row = np.append(down_row, 0)
                            else:
                                down_row = np.append(down_row, down_df['error'].values[0])

                            left_df = state_df.loc[state_df['action'] == 'Dir.LEFT']
                            if left_df.empty:
                                left_row = np.append(left_row, 0)
                            else:
                                left_row = np.append(left_row, left_df['error'].values[0])

                            right_df = state_df.loc[state_df['action'] == 'Dir.RIGHT']
                            if right_df.empty:
                                right_row = np.append(right_row, 0)
                            else:
                                right_row = np.append(right_row, right_df['error'].values[0])

                        """
                        else:
                            up_row = np.append(up_row, state_df.loc[state_df['action'] == 'Dir.UP']['error'].values[0])
                            down_row = np.append(down_row, state_df.loc[state_df['action'] == 'Dir.DOWN']['error'].values[0])
                            left_row = np.append(left_row, state_df.loc[state_df['action'] == 'Dir.LEFT']['error'].values[0])
                            right_row = np.append(right_row, state_df.loc[state_df['action'] == 'Dir.RIGHT']['error'].values[0])
                        """

                    if 'Dir.UP' not in error_dict.keys():
                        error_dict['Dir.UP'] = up_row
                    else:
                        try:
                            error_dict['Dir.UP'] = np.vstack([error_dict['Dir.UP'], up_row])
                        except:
                            print('FUCK')
                            print(error_dict['Dir.UP'], up_row)
                            quit()

                    if 'Dir.DOWN' not in error_dict.keys():
                        error_dict['Dir.DOWN'] = down_row
                    else:
                        error_dict['Dir.DOWN'] = np.vstack([error_dict['Dir.DOWN'], down_row])

                    if 'Dir.LEFT' not in error_dict.keys():
                        error_dict['Dir.LEFT'] = left_row
                    else:
                        error_dict['Dir.LEFT'] = np.vstack([error_dict['Dir.LEFT'], left_row])

                    if 'Dir.RIGHT' not in error_dict.keys():
                        error_dict['Dir.RIGHT'] = right_row
                    else:
                        error_dict['Dir.RIGHT'] = np.vstack([error_dict['Dir.RIGHT'], right_row])

                # Graph figures
                fig.suptitle(abstr_type + ', mdp' + num + ', episode ' + str(episode))
                axs[0,0].set_title('Up')
                im = axs[0,0].imshow(error_dict['Dir.UP'], norm=MidpointNormalize(vmin=-1, vmax=0, midpoint=0), cmap=plt.get_cmap('bwr'))
                axs[0,1].set_title('Down')
                im = axs[0,1].imshow(error_dict['Dir.DOWN'], norm=MidpointNormalize(vmin=-1, vmax=0, midpoint=0), cmap=plt.get_cmap('bwr'))
                axs[1,0].set_title('Left')
                im = axs[1,0].imshow(error_dict['Dir.LEFT'], norm=MidpointNormalize(vmin=-1, vmax=0, midpoint=0), cmap=plt.get_cmap('bwr'))
                axs[1,1].set_title('Right')
                im = axs[1,1].imshow(error_dict['Dir.RIGHT'], norm=MidpointNormalize(vmin=-1, vmax=0, midpoint=0), cmap=plt.get_cmap('bwr'))
                cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
                fig.colorbar(im, cax=cbar_ax, cmap='bwr')

                # Save figure
                file_name = os.path.join(q_value_folder, abstr_type[:-1]+'_mdp'+num[1:]+'_ep'+str(episode))
                plt.savefig(file_name)
                fig.clf()

class MidpointNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


# Testing purposes only
if __name__ == '__main__':
    v = QValueVisualizer(results_dir='../exp_output/hot')
    df = v.get_q_value_df()
    #print(df.to_string())

    v.graph_q_values(aggregate=False)


