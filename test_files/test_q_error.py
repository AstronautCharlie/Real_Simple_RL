"""
Test the error visualizer from QValueVisualizer
"""

from Visualizer.QValueVisualizer import QValueVisualizer
from GridWorld.GridWorldMDPClass import GridWorldMDP

if __name__ == '__main__':
    mdp = GridWorldMDP()
    v = QValueVisualizer(results_dir='../exp_output/big_test',
                         states_to_track=mdp.get_all_possible_states())
    v.visualize_q_value_error('noisy', mdp, episodes=[i for i in range(50, 1000, 50)])

