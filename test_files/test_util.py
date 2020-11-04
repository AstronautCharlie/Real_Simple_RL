from resources.util import *
from resources.AbstractionTypes import *
from resources.CorruptionTypes import *

keys = [(Abstr_type.PI_STAR, 0.0, Corr_type.UNI_RAND, 0.1, 0),
        (Abstr_type.A_STAR, 0.0, Corr_type.UNI_RAND, 0.1, 0),
        (Abstr_type.Q_STAR, 0.0, Corr_type.UNI_RAND, 0.1, 0)]
agent_nums = [0, 1, 2, 3, 4]
corruption_file = '../exp_output/cold/archive4/corrupted/corrupted_abstractions.csv'
error_file = '../exp_output/cold/archive4/corrupted/error_states.csv.'
detached_state_file = '../exp_output/cold/archive4/corrupted_w_detach/detached_states.csv'

if __name__ == '__main__':
    for key in keys:
        for agent_num in agent_nums:
            print('Detached state breakdown for', key, agent_num)
            categorize_detached_states(key, agent_num, corruption_file, error_file, detached_state_file)
            print()

