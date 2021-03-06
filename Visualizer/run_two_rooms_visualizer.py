"""
Run TwoRoomsVisualizer to generate and save roll-outs, state-value gradients, and value heatmaps.

Imports MDP from run_experiment
"""

import os
import pandas as pd
import ast
import pygame
import shutil
import matplotlib.pyplot as plt
from GridWorld.ActionEnums import Dir
from resources.AbstractionTypes import Abstr_type
from resources.CorruptionTypes import Corr_type
from Visualizer.TwoRoomsVisualizer import TwoRoomsVisualizer
from GridWorld.TwoRoomsMDP import TwoRoomsMDP
from run_experiment import MDP # This defines the parameters of the MDP; needed for visualization
# Define parameters of MDP here:
#MDP = TwoRoomsMDP(lower_width=3, lower_height=3, upper_width=3, upper_height=3, hallway_states=[2])

# Set these parameters before running
# Directory containing the output of an experiment
DATA_DIR = '../exp_output/hot'
# True or corrupted MDPs
MDP_TYPE = 'corrupted'
# Where to store the visualizations
OUTPUT_DIR = '../exp_output/hot/visualizations'
RANDOM_OR_EXPLICIT = 'explicit'

if __name__ == '__main__':
    # This is the directory containing the output of an experiment
    data_dir = DATA_DIR

    # Make files if does not exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for filename in os.listdir(OUTPUT_DIR):
        full_file = os.path.join(OUTPUT_DIR, filename)
        if os.path.isdir(full_file):
            shutil.rmtree(full_file)
        else:
            os.remove(full_file)

    # ----------------------
    # Corrupt Visualizations
    # ----------------------
    print('Generating visualizations for corrupt MDPs')
    # Name corrupted data files
    corr_value_file = os.path.join(data_dir, 'corrupted/learned_state_values.csv')
    error_file = os.path.join(data_dir, 'corrupted/error_states.csv')
    corr_abstraction_file = os.path.join(data_dir, 'corrupted/corrupted_abstractions.csv')
    corr_policy_file = os.path.join(data_dir, 'corrupted/learned_policies.csv')

    if os.path.exists(corr_value_file) and os.path.exists(corr_abstraction_file)\
        and os.path.exists(error_file) and os.path.exists(corr_policy_file):

        # Get list of all keys and agent nums in the corrupted files
        names = ['key', 'agent_num', 'dict']
        policy_df = pd.read_csv(corr_policy_file, names=names)
        unique_keys = policy_df['key'].unique()
        agent_nums = policy_df['agent_num'].unique()
        # Parse the strings representing the keys
        parsed_keys = []
        for string_key in unique_keys:
            key = []
            string_list = string_key.split(',')
            if 'PI_STAR' in string_list[0]:
                key.append(Abstr_type.PI_STAR)
            elif 'A_STAR' in string_list[0]:
                key.append(Abstr_type.A_STAR)
            elif 'Q_STAR' in string_list[0]:
                key.append(Abstr_type.Q_STAR)
            key.append(ast.literal_eval(string_list[1][1:]))
            if RANDOM_OR_EXPLICIT == 'explicit':
                key.append("'explicit errors'")
                key.append(ast.literal_eval(string_list[3][1:]))
            elif RANDOM_OR_EXPLICIT == 'random':
                key.append(Corr_type.UNI_RAND)
                key.append(ast.literal_eval(string_list[3][1:]))
            else:
                raise ValueError('RANDOM OR EXPLICIT must be set to \'random\' or \'explicit\'. Is currently ' + RANDOM_OR_EXPLICIT)

            key.append(ast.literal_eval(string_list[4][1:-1]))
            parsed_keys.append(tuple(key))
        #print(parsed_keys)

        # Now go through all the keys and agent numbers present in the files and create roll-out/state-value gradient images
        viz = TwoRoomsVisualizer()
        for key in parsed_keys:
            for agent_num in agent_nums:
                # Create generic file names and paths
                abstr_string = viz.get_abstr_name(key[0])
                folder_path = 'corrupted'
                file_name = 'corr' + str(key[3])+'mdp'+str(key[4]) + abstr_string + str(agent_num) + '.png'

                # Create folder for ensemble rollouts
                ensemble_folder_path = os.path.join('ensemble_rollouts', folder_path)
                if OUTPUT_DIR:
                    ensemble_folder_path = os.path.join(OUTPUT_DIR, ensemble_folder_path)
                if not os.path.exists(ensemble_folder_path):
                    os.makedirs(ensemble_folder_path)
                # Draw ensemble roll-out
                print('Generating roll-out for key', key)
                surface = viz.create_corruption_mdp(MDP,
                                                    key,
                                                    corr_abstraction_file,
                                                    error_file)
                surface = viz.draw_errors(MDP,
                                          surface,
                                          key,
                                          error_file)
                surface = viz.draw_corrupt_ensemble_rollouts(MDP,
                                                             surface,
                                                             key,
                                                             corr_policy_file,
                                                             corr_abstraction_file,
                                                             [agent_num])
                ensemble_file_name = os.path.join(ensemble_folder_path, file_name)
                pygame.image.save(surface, ensemble_file_name)

                # Create folder for state value gradients
                value_folder_path = os.path.join('value_gradients', folder_path)
                if OUTPUT_DIR:
                    value_folder_path = os.path.join(OUTPUT_DIR, value_folder_path)
                if not os.path.exists(value_folder_path):
                    os.makedirs(value_folder_path)
                # Draw state value gradient
                surface = viz.draw_state_value_gradient(MDP,
                                                        key,
                                                        agent_num,
                                                        corr_value_file)
                if error_file is not None:
                    surface = viz.draw_misaggregations(MDP,
                                                       surface,
                                                       key,
                                                       error_file,
                                                       corr_abstraction_file)
                # Save visualization
                value_file_name = os.path.join(value_folder_path, file_name)
                pygame.image.save(surface, value_file_name)

                # Create folder for heatmaps
                heatmap_folder_path = os.path.join('value_heatmaps', folder_path)
                if OUTPUT_DIR:
                    heatmap_folder_path = os.path.join(OUTPUT_DIR, heatmap_folder_path)
                if not os.path.exists(heatmap_folder_path):
                    os.makedirs(heatmap_folder_path)
                # Create heatmaps
                fig = viz.create_value_heatmap(MDP,
                                               key,
                                               agent_num,
                                               corr_value_file)
                # Save visualization
                heatmap_file_name = os.path.join(heatmap_folder_path, file_name)
                #if OUTPUT_DIR:
                #    heatmap_file_name = os.path.join(OUTPUT_DIR, heatmap_file_name)
                plt.savefig(heatmap_file_name)
                plt.close()
    else:
        print("Corruption files not found. Skipping Corrupt MDP visualization")

    # ------------------------------------
    # Corrupt w/ detachment visualizations
    # ------------------------------------
    print('Generating visualizations for corrupt MDPs w/ detachments')
    corr_value_file = os.path.join(data_dir, 'corrupted_w_detach/learned_state_values.csv')
    error_file = os.path.join(data_dir, 'corrupted/error_states.csv')
    corr_abstraction_file = os.path.join(data_dir, 'corrupted/corrupted_abstractions.csv')
    corr_policy_file = os.path.join(data_dir, 'corrupted_w_detach/learned_policies.csv')
    detached_file = os.path.join(data_dir, 'corrupted_w_detach/detached_states.csv')
    final_s_a_file = os.path.join(data_dir, 'corrupted_w_detach/final_s_a.csv')

    # Check that these files actually exist. If not, skip this part
    if (os.path.exists(corr_value_file) and
            os.path.exists(error_file) and
            os.path.exists(corr_abstraction_file) and
            os.path.exists(corr_policy_file) and
            os.path.exists(detached_file) and os.path.exists(final_s_a_file)):
        # Get list of all keys and agent nums in the corrupted files
        names = ['key', 'agent_num', 'dict']
        policy_df = pd.read_csv(corr_policy_file, names=names)
        unique_keys = policy_df['key'].unique()
        agent_nums = policy_df['agent_num'].unique()
        # Parse the strings representing the keys
        parsed_keys = []
        for string_key in unique_keys:
            key = []
            string_list = string_key.split(',')
            if 'PI_STAR' in string_list[0]:
                key.append(Abstr_type.PI_STAR)
            elif 'A_STAR' in string_list[0]:
                key.append(Abstr_type.A_STAR)
            elif 'Q_STAR' in string_list[0]:
                key.append(Abstr_type.Q_STAR)
            key.append(ast.literal_eval(string_list[1][1:]))
            if RANDOM_OR_EXPLICIT == 'explicit':
                key.append("'explicit errors'")
                key.append(ast.literal_eval(string_list[3][1:]))
            elif RANDOM_OR_EXPLICIT == 'random':
                key.append(Corr_type.UNI_RAND)
                key.append(ast.literal_eval(string_list[3][1:]))
            else:
                raise ValueError(
                    'RANDOM OR EXPLICIT must be set to \'random\' or \'explicit\'. Is currently ' + RANDOM_OR_EXPLICIT)
            key.append(ast.literal_eval(string_list[4][1:-1]))
            parsed_keys.append(tuple(key))

        # Now go through all the keys and agent numbers present in the files and create roll-out/state-value gradient images
        viz = TwoRoomsVisualizer()
        for key in parsed_keys:
            for agent_num in agent_nums:
                # Create generic file string
                abstr_string = viz.get_abstr_name(key[0])
                #folder_path = 'corrupted_w_detach/corr' + str(key[3]) + '/mdp' + str(key[4])
                #file_name = abstr_string + str(agent_num) + '.png'
                folder_path = 'corrupted_w_detach'
                corr_num = str(key[3])
                mdp_num = str(key[4])
                file_name = 'corr' + str(key[3])+'mdp'+str(key[4]) + abstr_string + str(agent_num) + '.png'

                # Create folder for ensemble roll-out
                ensemble_folder_path = os.path.join('ensemble_rollouts', folder_path)
                if OUTPUT_DIR:
                    ensemble_folder_path = os.path.join(OUTPUT_DIR, ensemble_folder_path)
                if not os.path.exists(ensemble_folder_path):
                    os.makedirs(ensemble_folder_path)
                # Draw ensemble roll-out
                surface = viz.create_corruption_mdp(MDP,
                                                    key,
                                                    corr_abstraction_file,
                                                    error_file)
                surface = viz.draw_errors(MDP,
                                          surface,
                                          key,
                                          error_file)
                surface = viz.draw_corrupt_ensemble_rollouts(MDP,
                                                             surface,
                                                             key,
                                                             corr_policy_file,
                                                             corr_abstraction_file,
                                                             [agent_num])
                # Save file
                ensemble_file_name = os.path.join(ensemble_folder_path, file_name)
                pygame.image.save(surface, ensemble_file_name)

                # Create folder for value gradients
                value_folder_path = os.path.join('value_gradients', folder_path)
                if OUTPUT_DIR:
                    value_folder_path = os.path.join(OUTPUT_DIR, value_folder_path)
                if not os.path.exists(value_folder_path):
                    os.makedirs(value_folder_path)
                # Draw state value gradient
                surface = viz.draw_state_value_gradient(MDP,
                                                        key,
                                                        agent_num,
                                                        corr_value_file)
                if error_file is not None:
                    surface = viz.draw_misaggregations(MDP,
                                                       surface,
                                                       key,
                                                       error_file,
                                                       corr_abstraction_file)
                # Save visualization
                value_file_name = os.path.join(value_folder_path, file_name)
                pygame.image.save(surface, value_file_name)

                # Create folders for heatmaps
                heatmap_folder_path = os.path.join('value_heatmaps', folder_path)
                if OUTPUT_DIR:
                    heatmap_folder_path = os.path.join(OUTPUT_DIR, heatmap_folder_path)
                if not os.path.exists(heatmap_folder_path):
                    os.makedirs(heatmap_folder_path)
                # Create heatmaps
                fig = viz.create_value_heatmap(MDP,
                                               key,
                                               agent_num,
                                               corr_value_file)
                # Save visualization
                heatmap_file_name = os.path.join(heatmap_folder_path, file_name)
                plt.savefig(heatmap_file_name)
                plt.close()

                # Create folders for detached state visualization
                detach_folder = os.path.join('detach_map', folder_path)
                if OUTPUT_DIR:
                    detach_folder = os.path.join(OUTPUT_DIR, detach_folder)
                if not os.path.exists(detach_folder):
                    os.makedirs(detach_folder)
                # Create detachment visualization
                grid = viz.draw_detached_abstraction(MDP,
                                                     key,
                                                     agent_num,
                                                     corr_abstraction_file,
                                                     final_s_a_file,
                                                     error_file,
                                                     detached_file)
                detach_file_name = os.path.join(detach_folder, file_name)
                pygame.image.save(grid, detach_file_name)

                # Do summary
                if not os.path.exists(os.path.join(data_dir, 'corrupted_w_detach', 'summary')):
                    os.mkdir(os.path.join(data_dir, 'corrupted_w_detach', 'summary'))
                summary_file = os.path.join(data_dir, 'corrupted_w_detach/summary/summary_'
                                            + 'corr' + corr_num + '_' + abstr_string +'_mdp'
                                            + mdp_num + '_agent' + str(agent_num) + '.txt')
                viz.summarize_final_s_a_detachment(key,
                                                   agent_num,
                                                   corr_abstraction_file,
                                                   final_s_a_file,
                                                   error_file,
                                                   detached_file,
                                                   summary_file)

    # ----------------------
    # True Visualizations
    # ----------------------
    print('Generating visualizations for true MDPs')
    # Name true data files
    true_value_file = os.path.join(data_dir, 'true/learned_state_values.csv')
    true_abstraction_file = os.path.join(data_dir, 'true/abstractions.csv')
    true_policy_file = os.path.join(data_dir, 'true/learned_policies.csv')

    # Get list of all keys and agent nums in theklm  corrupted files
    names = ['key', 'agent_num', 'dict']
    policy_df = pd.read_csv(true_policy_file, names=names)
    unique_keys = policy_df['key'].unique()
    agent_nums = policy_df['agent_num'].unique()
    # Parse the strings representing the keys
    parsed_keys = []
    for string_key in unique_keys:
        key = []
        string_list = string_key.split(',')
        #print(string_list)
        if 'PI_STAR' in string_list[0]:
            key.append(Abstr_type.PI_STAR)
        elif 'A_STAR' in string_list[0]:
            key.append(Abstr_type.A_STAR)
        elif 'Q_STAR' in string_list[0]:
            key.append(Abstr_type.Q_STAR)
        else:
            continue
        key.append(ast.literal_eval(string_list[1][1:-1]))
        parsed_keys.append(tuple(key))
    #print('parsed keys', parsed_keys)

        # Now go through all the keys and agent numbers present in the files and create roll-out/state-value gradient images
    viz = TwoRoomsVisualizer()
    for key in parsed_keys:
        for agent_num in agent_nums:
            #print('agent_num', agent_num)
            abstr_string = viz.get_abstr_name(key[0])
            abstr_eps = None
            if len(key) > 1:
                abstr_eps = key[1]

            # Create generic folder path and file name
            folder_path = 'true'
            file_name = abstr_string + str(agent_num) + '.png'

            # Create folder for ensemble roll-out
            ensemble_folder_path = os.path.join('ensemble_rollouts', folder_path)
            if OUTPUT_DIR:
                ensemble_folder_path = os.path.join(OUTPUT_DIR, ensemble_folder_path)
            if not os.path.exists(ensemble_folder_path):
                os.makedirs(ensemble_folder_path)
            ensemble_file_name = os.path.join(ensemble_folder_path, file_name)
            # Draw ensemble roll-out
            grid_mdp = viz.create_abstract_mdp_from_file(MDP,
                                                         true_abstraction_file,
                                                         key)
            surface = viz.draw_true_ensemble_rollouts(MDP,
                                                      grid_mdp,
                                                      key,
                                                      true_policy_file,
                                                      true_abstraction_file,
                                                      [agent_num])
            # Save visualization
            pygame.image.save(surface, ensemble_file_name)

            # Create folder for value gradients
            value_folder_path = os.path.join('value_gradients', folder_path)
            if OUTPUT_DIR:
                value_folder_path = os.path.join(OUTPUT_DIR, value_folder_path)
            if not os.path.exists(value_folder_path):
                os.makedirs(value_folder_path)
            # Draw state value gradient
            surface = viz.draw_state_value_gradient(MDP,
                                                    key,
                                                    agent_num,
                                                    true_value_file)
            # Save visualization
            value_file_name = os.path.join(value_folder_path, file_name)
            pygame.image.save(surface, value_file_name)

            # Create folder for heatmaps
            heatmap_folder_path = os.path.join('value_heatmaps', folder_path)
            if OUTPUT_DIR:
                heatmap_folder_path = os.path.join(OUTPUT_DIR, heatmap_folder_path)
            if not os.path.exists(heatmap_folder_path):
                os.makedirs(heatmap_folder_path)
            # Create heatmaps
            fig = viz.create_value_heatmap(MDP,
                                           key,
                                           agent_num,
                                           true_value_file)
            # Save visualization
            heatmap_file_name = os.path.join(heatmap_folder_path, file_name)
            plt.savefig(heatmap_file_name)
            plt.close()

