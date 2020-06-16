'''
This file tests that optimal policies in the ground state space of
FourRooms and Taxi are representable in the abstract state space given
by Q* and A* abstractions
'''

from GridWorld.TaxiMDPClass import TaxiMDP
from GridWorld.TaxiStateClass import TaxiState
from GridWorld.GridWorldMDPClass import GridWorldMDP
from MDP.ValueIterationClass import ValueIteration
from resources.AbstractionMakers import make_abstr, Abstr_type

def is_optimal_policy_representable(vi, optimal_policy, state_abstr):
    '''
    Return true if no two states with different optimal actions in the
    optimal policy are grouped together in the state abstraction except in
    the cases when the optimal action is non-unique. In those cases, 
    :param mdp: MDP
    :param optimal_policy: dictionary: action -> (state, value)
    :param state_abstr: dictionary: ground_state -> abstract_state
    :return: boolean
    '''
    # Iterate through all ground states that are grouped together
    abstr_dict = state_abstr.get_abstr_dict()
    for ground_state_1 in abstr_dict.keys():
        for ground_state_2 in abstr_dict.keys():
            # If two states are grouped together, they must have either
            # the same optimal action or overlapping optimal actions
            # (in the case where optimal actions are non-unique)
            abstr_state_1 = abstr_dict[ground_state_1]
            abstr_state_2 = abstr_dict[ground_state_2]
            if ground_state_1 != ground_state_2 and abstr_state_1 == abstr_state_2:
                best_actions_1 = vi.get_all_optimal_actions(ground_state_1)
                best_actions_2 = vi.get_all_optimal_actions(ground_state_2)
                if len(best_actions_1) > 1:
                    print(ground_state_1, ground_state_2, best_actions_1, best_actions_2)
                # Intersection of optimal actions must be non-null
                best_action_intersect = list(set(best_actions_1) & set(best_actions_2))
                if len(best_action_intersect) == 0:
                    return False
    return True

def print_policy(policy):
    '''
    Print the policy
    '''
    for key in policy.keys():
        print(key, policy[key])

if __name__ == '__main__':
    # Test that optimal ground policy for FourRooms is representable in
    # abstaction given by Q*

    # Get optimal ground policy for FourRooms
    four_rooms = GridWorldMDP(slip_prob=0.0, gamma=0.99)
    vi = ValueIteration(four_rooms)
    vi.run_value_iteration()
    optimal_policy = vi.get_optimal_policy()
    #print_policy(optimal_policy)

    # Get Q* abstraction for FourRooms and optimal abstract policy
    abstr = make_abstr(vi.get_q_table(), Abstr_type.A_STAR)

    print(is_optimal_policy_representable(vi, optimal_policy, abstr))

