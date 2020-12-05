from resources.util import *
from resources.AbstractionTypes import *
from resources.CorruptionTypes import *

# Explicit errors

keys = [#(Abstr_type.PI_STAR, 0.0, "'explicit errors'", 0, 0),
        #(Abstr_type.A_STAR, 0.0, "'explicit errors'", 0, 0),
        (Abstr_type.A_STAR, 0.0, "'explicit errors'", 0, 0),
        #(Abstr_type.PI_STAR, 0.0, "'explicit errors'", 1, 0),
        #(Abstr_type.A_STAR, 0.0, "'explicit errors'", 1, 0),
        (Abstr_type.A_STAR, 0.0, "'explicit errors'", 1, 0)]#,
        #(Abstr_type.PI_STAR, 0.0, "'explicit errors'", 2, 0),
        #(Abstr_type.A_STAR, 0.0, "'explicit errors'", 2, 0),
        #(Abstr_type.A_STAR, 0.0, "'explicit errors'", 2, 0)]#,
        #(Abstr_type.PI_STAR, 0.0, "'explicit errors'", 3, 0),
        #(Abstr_type.A_STAR, 0.0, "'explicit errors'", 3, 0),
        #(Abstr_type.Q_STAR, 0.0, "'explicit errors'", 3, 0)]


# Random errors
'''
CORRUPTION_LIST = [(Corr_type.UNI_RAND, 0.05), (Corr_type.UNI_RAND, 0.1)]
ABSTR_EPSILON_LIST = [(Abstr_type.A_STAR, 0.0), (Abstr_type.PI_STAR, 0.0), (Abstr_type.Q_STAR, 0.0)]
MDP_NUMS = [0, 1, 2, 3, 4]
keys = []
for abstr in ABSTR_EPSILON_LIST:
    for corr in CORRUPTION_LIST:
        for num in MDP_NUMS:
            keys.append((abstr[0], abstr[1], corr[0], corr[1], num))
'''

agent_nums = [0, 1, 2, 3, 4]
#corruption_file = '../exp_output/cold/archive8-divergent_pair_test/corrupted/corrupted_abstractions.csv'
#error_file = '../exp_output/cold/archive8-divergent_pair_test/corrupted/error_states.csv.'
#detached_state_file = '../exp_output/cold/archive8-divergent_pair_test/corrupted_w_detach/detached_states.csv'
corruption_file = '../exp_output/hot/corrupted/corrupted_abstractions.csv'
error_file = '../exp_output/hot/corrupted/error_states.csv'
detached_state_file = '../exp_output/hot/corrupted_w_detach/detached_states.csv'


if __name__ == '__main__':
    for key in keys:
        for agent_num in agent_nums:
            print('Detached state breakdown for', key, agent_num)
            categorize_detached_states(key, agent_num, corruption_file, error_file, detached_state_file)
            print()

