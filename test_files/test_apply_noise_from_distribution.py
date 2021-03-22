"""
Test the apply noise from distribution functionality from AbstractionCorrupters
"""

from GridWorld.GridWorldMDPClass import GridWorldMDP
from resources.AbstractionCorrupters import apply_noise_from_distribution
from resources.AbstractionTypes import Abstr_type
from resources.AbstractionMakers import make_abstr
from MDP.ValueIterationClass import ValueIteration
from scipy.stats import skewnorm, norm
import numpy as np

if __name__ == '__main__':
    np.random.seed(1234)
    args = {'loc': 0, 'scale': 0.01}
    mdp = GridWorldMDP()
    abstr_type = Abstr_type.PI_STAR
    corr_abstr_mdp = apply_noise_from_distribution(mdp,
                                                    abstr_type,
                                                    norm,
                                                    args,
                                                    0.005,
                                                    seed=124)

    # Get true abstraction to compare it with
    vi = ValueIteration(mdp)
    vi.run_value_iteration()
    q_table = vi.get_q_table()
    true_abstr = make_abstr(q_table, abstr_type, seed=1234)

    corr_abstr_dict = corr_abstr_mdp.state_abstr.abstr_dict
    true_abstr_dict = true_abstr.abstr_dict
    for state in true_abstr_dict.keys():
        print(state)
        print('True abstr state', true_abstr_dict[state])
        print('Corr abstr state', corr_abstr_dict[state])
