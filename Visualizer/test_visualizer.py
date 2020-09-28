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

def draw_ensemble_rollouts_test(key):
    policy_file = '../test_exp2/corrupted/learned_policies.csv'
    abstr_file = '../test_exp2/corrupted/corrupted_abstractions.csv'
    error_file = '../test_exp2/corrupted/error_states.csv'
    #key = (Abstr_type.A_STAR, 0.0, Corr_type.UNI_RAND, 0.05, 3)
    viz = GridWorldVisualizer()
    surface = viz.create_corruption_visualization(key, abstr_file, error_file)
    surface = viz.draw_errors(surface, key, error_file)
    surface = viz.draw_corrupt_ensemble_rollouts(surface,
                                                 key,
                                                 policy_file,
                                                 abstr_file,
                                                 5)
    viz.display_surface(surface)

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

def visualize_true_abstract_rollout_test():
    viz = GridWorldVisualizer()
    key = (Abstr_type.PI_STAR, 0.0)
    policy_file = '../test_exp2/true/learned_policies.csv'
    abstraction_file = '../test_exp2/true/abstractions.csv'
    agent_num = 1

def draw_true_ensemble_rollouts_test():
    # draw_true_ensemble_rollouts requires a surface, so first we have to create a an abstract Gridworld surface
    #  from an abstract mdp

    # Define arguments to function to test
    key = (Abstr_type.PI_STAR, 0.0)
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

if __name__ == '__main__':
    #test_display_gridworld_mdp()
    #test_display_abstract_gridworld_mdp()
    #test_corruption_visualization()
    #test_generate_gridworld_rollout()
    draw_ensemble_rollouts_test((Abstr_type.Q_STAR, 0.0, Corr_type.UNI_RAND, 0.05, 2))
    #generate_true_abstract_rollout_test()
    #draw_true_ensemble_rollouts_test()


