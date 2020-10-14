"""
This file tests relevant functions from the GridWorldVisualizer class
"""

from GridWorld.GridWorldMDPClass import GridWorldMDP
from MDP.ValueIterationClass import ValueIteration
from MDP.AbstractMDPClass import AbstractMDP
from resources.AbstractionMakers import make_abstr
from resources.AbstractionTypes import Abstr_type
from resources.CorruptionTypes import Corr_type
from Visualizer.GridWorldVisualizer import *
import pygame

def test_display_gridworld_mdp():
    """
    Test the 'create_gridworld_mdp' method
    """
    mdp = GridWorldMDP()
    viz = GridWorldVisualizer()
    viz.display_gridworld_mdp(mdp)

def test_display_abstract_gridworld_mdp():
    """
    Create a pi* abstraction on a gridworld and test that it displays
    properly
    """
    # Make abstr MDP
    mdp = GridWorldMDP()
    abstr_mdp = mdp.make_abstr_mdp(Abstr_type.PI_STAR)
    viz = GridWorldVisualizer()
    viz.display_abstract_gridworld_mdp(abstr_mdp)

def test_corruption_visualization():
    """
    Test create_corruption_visualization and display_corruption_visualization
    """
    viz = GridWorldVisualizer()
    key = (Abstr_type.PI_STAR, 0.0, Corr_type.UNI_RAND, 0.1, 0)
    corr_abstr_file = '../exp_output/corrupted/corrupted_abstractions.csv'
    error_file = '../exp_output/corrupted/error_states.csv'
    #viz.create_corruption_visualization(key, corr_abstr_file, error_file)
    viz.display_corrupt_visualization(key, corr_abstr_file, error_file)

def test_generate_gridworld_rollout():
    """
    Test the generation of a rollout for one agent in an ensemble
    """
    policy_file = '../test_exp/corrupted/learned_policies.csv'
    abstr_file = '../test_exp/corrupted/corrupted_abstractions.csv'
    key = (Abstr_type.PI_STAR, 0.0, Corr_type.UNI_RAND, 0.05, 0)
    agent_num = 0
    viz = GridWorldVisualizer()
    print(viz.generate_corrupt_abstract_rollout(policy_file, abstr_file, key, agent_num))

def draw_ensemble_rollouts_test(key, policy_file, abstr_file, error_file, agent_nums, save=False):
    viz = GridWorldVisualizer()
    surface = viz.create_corruption_visualization(key, abstr_file, error_file)
    surface = viz.draw_errors(surface, key, error_file)
    surface = viz.draw_corrupt_ensemble_rollouts(surface, key, policy_file, abstr_file, agent_nums)
    if save:
        abstr_string = viz.get_abstr_name(key[0])
        agent_string = ''
        for agent in agent_nums:
            agent_string += str(agent)
        file_name = 'ensemble_rollout_' + abstr_string + '_' + str(key[3]) + '_' \
                    + str(key[4]) + '_' + agent_string + '.png'
        pygame.image.save(surface, file_name)
        #viz.display_surface(surface)

def draw_true_ensemble_rollouts_test():
    policy_file = '../test_exp2/true/learned_policies.csv'
    abstraction_file = '../test_exp2/true/abstractions.csv'
    agent_num = 5
    viz = GridWorldVisualizer()
    surface = viz.create_abstract_gridworld_mdp()

def generate_true_abstract_rollout_test():
    viz = GridWorldVisualizer()
    key = (Abstr_type.PI_STAR, 0.0)
    policy_file = '../test_exp2/true/learned_policies.csv'
    abstraction_file = '../test_exp2/true/abstractions.csv'
    agent_num = 0
    print(viz.generate_true_abstract_rollout(key, policy_file, abstraction_file, agent_num))

def draw_true_ensemble_rollouts_test(key):
    # draw_true_ensemble_rollouts requires a surface, so first we have to create a an abstract Gridworld surface
    #  from an abstract mdp

    # Define arguments to function to test
    #key = (Abstr_type.PI_STAR, 0.0)
    policy_file = '../test_exp2/true/learned_policies.csv'
    abstraction_file = '../test_exp2/true/abstractions.csv'
    viz = GridWorldVisualizer()
    surface = viz.create_abstract_gridworld_mdp_from_file(abstraction_file,
                                                          key)

    # Run the actual file
    final = viz.draw_true_ensemble_rollouts(surface,
                                    key,
                                    policy_file,
                                    abstraction_file,
                                    5)
    viz.display_surface(final)

def draw_state_value_gradient_test(key, agent_num, value_file, abstraction_file, error_file=None, save=False):
    viz = GridWorldVisualizer()
    surface = viz.draw_state_value_gradient(key, agent_num, value_file)
    if error_file is not None:
        surface = viz.draw_misaggregations(surface, key, error_file, abstraction_file)
    if save:
        file_name = 'state_value_gradient_' + str(key[0]) + '_' + str(key[1]) + '_' + str(key[3]) + '_' \
                    + str(key[4]) + '_' + str(agent_num) + '.png'
        pygame.image.save(surface, file_name)
    if save:
        abstr_string = viz.get_abstr_name(key[0])
        agent_string = str(agent_num)
        file_name = 'value_gradient_' + abstr_string + '_' + str(key[3]) + '_' \
                    + str(key[4]) + '_' + agent_string + '.png'
        pygame.image.save(surface, file_name)
    #viz.display_surface(surface)

if __name__ == '__main__':
    #test_display_gridworld_mdp()
    #test_display_abstract_gridworld_mdp()
    #test_corruption_visualization()
    #test_generate_gridworld_rollout()
    #generate_true_abstract_rollout_test()

    # Draw roll-outs for true MDPs
    # draw_true_ensemble_rollouts_test((Abstr_type.Q_STAR, 0.0))

    # Draw roll-outs for corrupt MDPs


    # Test parse_file_for_dict function
    # Done: true learned state values (2-key, agent_num, dict) for Abstract MDP
    # Done: true learned state values (ground, agent_num, dict) for ground state
    # NOT DONE: true exp results (ground, list of return values); for this to work we'd have to write the rewards
    #          as a string instead of a separate column for each reward
    # NOT DONE: true exp results (2-key, list of return values)
    # DONE: corrupted learned state values (5-key, agent_num, dict)
    #print(viz.parse_file_for_dict(key, value_file, 2))

    # Visualize the gradient with the errors drawn on it
    value_file = '../exp_output/cold/archive1/corrupted/learned_state_values.csv'
    error_file = '../exp_output/cold/archive1/corrupted/error_states.csv'
    abstraction_file = '../exp_output/cold/archive1/corrupted/corrupted_abstractions.csv'
    policy_file= '../exp_output/cold/archive1/corrupted/learned_policies.csv'
    key = (Abstr_type.PI_STAR, 0.0, Corr_type.UNI_RAND, 0.1, 1)
    agent_nums = [0, 1, 2, 3, 4]
    for agent in agent_nums:
        draw_ensemble_rollouts_test(key, policy_file, abstraction_file, error_file, [agent], save=True)
        draw_state_value_gradient_test(key, agent, value_file, abstraction_file, error_file=error_file, save=True)

    quit()
    # Visualize true gradient
    value_file = '../exp_output/hot/true/learned_state_values.csv'
    abstraction_file = '../exp_output/hot/true/abstractions.csv'
    policy_file = '../exp_output/hot/true/learned_policies.csv'
    key = 'ground'
    #key = (Abstr_type.Q_STAR, 0.0)
    draw_state_value_gradient_test(key, 3, value_file, abstraction_file)

