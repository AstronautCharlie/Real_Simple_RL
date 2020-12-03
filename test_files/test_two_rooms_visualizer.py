"""
Complete!
"""

from GridWorld.TwoRoomsMDP import TwoRoomsMDP
from MDP.ValueIterationClass import ValueIteration
from MDP.AbstractMDPClass import AbstractMDP
from resources.AbstractionMakers import make_abstr
from resources.AbstractionTypes import Abstr_type
from resources.CorruptionTypes import Corr_type
from Visualizer.TwoRoomsVisualizer import *
import pygame

from run_experiment import MDP

ABSTRACTION_FILE = '../exp_output/hot/true/abstractions.csv'
CORR_ABSTR_FILE = '../exp_output/hot/corrupted/corrupted_abstractions.csv'
ERROR_FILE = '../exp_output/hot/corrupted/error_states.csv'
TRUE_POLICY_FILE = '../exp_output/hot/true/learned_policies.csv'
CORR_POLICY_FILE = '../exp_output/hot/corrupted/learned_policies.csv'
TRUE_KEY = (Abstr_type.Q_STAR, 0.0)
CORR_KEY = (Abstr_type.Q_STAR, 0.0, Corr_type.UNI_RAND, 1, 0)
TRUE_VALUE_FILE = '../exp_output/hot/true/learned_state_values.csv'
CORR_VALUE_FILE = '../exp_output/hot/corrupted/learned_state_values.csv'

if __name__ == '__main__':
    viz = TwoRoomsVisualizer()
    TEST_NUM = 7

    # Black-and-white grid
    if TEST_NUM == 1:
        mdp = MDP
        surface = viz.create_grid(mdp)
        viz.display_surface(surface)

    # True abstract MDP w/ roll-outs
    if TEST_NUM == 2:
        mdp = MDP
        surface = viz.create_abstract_mdp_from_file(mdp,
                                                    ABSTRACTION_FILE,
                                                    TRUE_KEY)
        surface = viz.draw_true_ensemble_rollouts(mdp,
                                                  surface,
                                                  TRUE_KEY,
                                                  TRUE_POLICY_FILE,
                                                  ABSTRACTION_FILE,
                                                  [3])
        viz.display_surface(surface)

    # Corrupted abstract MDP w/ roll-outs
    if TEST_NUM == 3:
        mdp = MDP
        surface = viz.create_corruption_mdp(mdp,
                                            CORR_KEY,
                                            CORR_ABSTR_FILE,
                                            ERROR_FILE)

        surface = viz.draw_corrupt_ensemble_rollouts(mdp,
                                                     surface,
                                                     CORR_KEY,
                                                     CORR_POLICY_FILE,
                                                     CORR_ABSTR_FILE,
                                                     [0,1,2,3,4])
        viz.display_surface(surface)

    # True Abstract MDP w/ state_value_gradient
    if TEST_NUM == 4:
        mdp = MDP
        #surface = viz.create_abstract_mdp_from_file(mdp,
        #                                            ABSTRACTION_FILE,
        #                                            TRUE_KEY)
        surface = viz.draw_state_value_gradient(mdp,
                                                TRUE_KEY,
                                                2,
                                                TRUE_VALUE_FILE)
        viz.display_surface(surface)

    # Corrupted abstract MDP w/ state_value_gradients/misaggregations
    if TEST_NUM == 5:
        mdp = MDP
        surface = viz.draw_state_value_gradient(mdp,
                                                CORR_KEY,
                                                3,
                                                CORR_VALUE_FILE)
        surface = viz.draw_misaggregations(mdp,
                                           surface,
                                           CORR_KEY,
                                           ERROR_FILE,
                                           CORR_ABSTR_FILE)
        viz.display_surface(surface)

    # True abstract value heatmaps
    if TEST_NUM == 6:
        mdp = MDP
        surface = viz.create_value_heatmap(mdp,
                                           TRUE_KEY,
                                           1,
                                           TRUE_VALUE_FILE)
        plt.show(surface)

    # Corrupted abstract value heatmaps
    if TEST_NUM == 7:
        mdp = MDP
        surface = viz.create_value_heatmap(mdp,
                                           CORR_KEY,
                                           1,
                                           CORR_VALUE_FILE)
        plt.show(surface)