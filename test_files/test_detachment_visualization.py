"""
Test the detachment visualizations in GridWorldVisualizer and TwoRoomsVisualizer
"""
from Visualizer.GridWorldVisualizer import GridWorldVisualizer
from GridWorld.GridWorldMDPClass import *
from GridWorld.ActionEnums import *
from resources.AbstractionTypes import *

TEST_NUM = 1

if __name__ == '__main__':
    if TEST_NUM == 1:
        viz = GridWorldVisualizer()
        key = (Abstr_type.Q_STAR, 0.0, 'explicit errors', 0, 0)
        agent_num = 1
        error_file = '../exp_output/hot/corrupted/error_states.csv'
        starting_s_a_file = '../exp_output/hot/corrupted/corrupted_abstractions.csv'
        final_s_a_file = '../exp_output/hot/corrupted_w_detach/final_s_a.csv'
        detach_file = '../exp_output/hot/corrupted_w_detach/detached_states.csv'
        grid = viz.draw_detached_abstraction(key,
                                             agent_num,
                                             starting_s_a_file,
                                             final_s_a_file,
                                             error_file,
                                             detach_file)
        viz.display_surface(grid)
