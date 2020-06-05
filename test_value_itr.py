from GridWorld.GridWorldMDPClass import GridWorldMDP
from GridWorld.GridWorldStateClass import GridWorldState
from GridWorld.ActionEnums import Dir
from MDP.ValueIterationClass import ValueIteration
from Visualizer.GridWorldVisualizer import GridWorldVisualizer
#quick test


if __name__ == '__main__':

    # Testing Apra's value iteration
    '''
    grid_mdp_test = GridWorldMDP(height=11, width=11, slip_prob=0.2, gamma=0.95, build_walls=True)
    value_itr = ValueIteration(grid_mdp_test, 0.9, 0.001)
    value_itr.doValueIteration(10000)
    print(value_itr)
    #viz = GridWorldVisualizer(grid_mdp_test, value_itr)
    #viz.visualizeLearnedPolicy()
    '''

    # Testing Trevor's value iteration

    mdp = GridWorldMDP(slip_prob=0.2, gamma=0.95, build_walls=True)
    vi = ValueIteration(mdp, 0.9, 0.001)
    vi.run_vi()
    print(vi)


#state = GridWorldState(1,1)
# out = grid_mdp_test.next_possible_states(state,Dir.UP)
# print([str(k) for k in out.keys()])
# print(out.values())
#
# all_states = grid_mdp_test.get_all_possible_states()
# print([str(state) for state in all_states])

