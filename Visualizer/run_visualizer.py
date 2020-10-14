"""
This file runs GridWorldVisualizer on the given inputs to produce value gradient and roll-out visualizations and
saves them to the given input

TODO: Expand to work for true values as well as corrupted values
"""
import os
import pandas as pd
import ast
import pygame
from GridWorld.ActionEnums import Dir
from resources.AbstractionTypes import Abstr_type
from resources.CorruptionTypes import Corr_type
from Visualizer.GridWorldVisualizer import GridWorldVisualizer

# Set these parameters before running
# Directory containing the output of an experiment
DATA_DIR = '../exp_output/cold/archive1'
# True or corrupted MDPs
MDP_TYPE = 'corrupted'
# Where to store the visualizations
OUTPUT_DIR = None

if __name__ == '__main__':
    # This is the directory containing the output of an experiment
    data_dir = DATA_DIR

    # Name corrupted data files
    corr_value_file = os.path.join(data_dir, 'corrupted/learned_state_values.csv')
    error_file = os.path.join(data_dir, 'corrupted/error_states.csv')
    corr_abstraction_file = os.path.join(data_dir, 'corrupted/corrupted_abstractions.csv')
    corr_policy_file = os.path.join(data_dir, 'corrupted/learned_policies.csv')

    # Name true data files
    true_value_file = os.path.join(data_dir, 'true/learned_state_values.csv')
    true_abstraction_file = os.path.join(data_dir, 'true/abstractions.csv')
    true_policy_file = os.path.join(data_dir, 'true/learned_policies.csv')

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
        key.append(Corr_type.UNI_RAND)
        key.append(ast.literal_eval(string_list[3][1:]))
        key.append(ast.literal_eval(string_list[4][1:-1]))
        parsed_keys.append(tuple(key))

    # Now go through all the keys and agent numbers present in the files and create roll-out/state-value gradient images
    viz = GridWorldVisualizer()
    for key in parsed_keys:
        for agent_num in agent_nums:
            # Draw ensemble roll-out
            surface = viz.create_corruption_visualization(key,
                                                          corr_abstraction_file,
                                                          error_file)
            surface = viz.draw_errors(surface,
                                      key,
                                      error_file)
            surface = viz.draw_corrupt_ensemble_rollouts(surface,
                                                         key,
                                                         corr_policy_file,
                                                         corr_abstraction_file,
                                                         [agent_num])
            abstr_string = viz.get_abstr_name(key[0])
            folder_path = 'ensemble_rollouts/' + abstr_string + '/' + 'corr' + str(key[3]) + '/' + 'mdp'\
                          + str(key[4])
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_name = os.path.join(folder_path, str(agent_num) + '.png')
            if OUTPUT_DIR:
                file_name = os.path.join(OUTPUT_DIR, file_name)
            pygame.image.save(surface, file_name)

            # Draw state value gradient
