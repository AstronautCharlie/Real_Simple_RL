from GridWorld.GridWorldMDPClass import GridWorldMDP
from GridWorld.GridWorldStateClass import GridWorldState
from GridWorld.ActionEnums import Dir
from MDP.ValueIterationClass import ValueIteration
from Visualizer.GridWorldVisualizer import GridWorldVisualizer
from GridWorld.TaxiMDPClass import TaxiMDP
#quick test


if __name__ == '__main__':

    # Testing Apra's value iteration on FourRooms
    '''
    grid_mdp_test = GridWorldMDP(height=11, width=11, slip_prob=0.0, gamma=0.99, build_walls=True)
    value_itr = ValueIteration(grid_mdp_test, 0.99, 0.0001)
    value_itr.doValueIteration(10000)
    #print(value_itr)
    result = value_itr.get_q_table()
    for key in result.keys():
        print(key[0], key[1], result[key])
    #viz = GridWorldVisualizer(grid_mdp_test, value_itr)
    #viz.visualizeLearnedPolicy()
    '''

    # Testing VI on TaxiMDP
    mdp = TaxiMDP(slip_prob = 0.0, gamma = 0.99)
    value_itr = ValueIteration(mdp, 0.001)
    value_itr.doValueIteration(10000)
    result = value_itr.get_q_table()
    for key in result.keys():
        print(key[0], key[1], result[key])




#state = GridWorldState(1,1)
# out = grid_mdp_test.next_possible_states(state,Dir.UP)
# print([str(k) for k in out.keys()])
# print(out.values())
#
# all_states = grid_mdp_test.get_all_possible_states()
# print([str(state) for state in all_states])


