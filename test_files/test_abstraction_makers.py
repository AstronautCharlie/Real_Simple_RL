'''
This tests the creation of state abstractions from the AbstractionMakers file based on the q-tables
generated by ValueIteration
'''

from MDP.ValueIterationClass import ValueIteration
from GridWorld.TaxiMDPClass import TaxiMDP, Act
from GridWorld.GridWorldMDPClass import GridWorldMDP
from resources.AbstractionMakers import make_abstr
from resources.AbstractionMakers import Abstr_type


if __name__ == '__main__':
    # GridWorld tests
    """
    # Q-star, epsilon = 0
    mdp = GridWorldMDP(slip_prob=0, gamma=0.99)
    vi = ValueIteration(mdp)
    vi.run_value_iteration()
    q_table = vi.get_q_table()
    for key in q_table.keys():
        print(key[0], key[1], q_table[key])
    abstr = make_abstr(q_table, Abstr_type.Q_STAR) #, epsilon=0.01)
    print(abstr)
    # results: 
    # 12 abstract states, 9 have 2 ground states and the other 
    # 3 have 3 
    """

    # Taxi MDP tests

    # Q-star, epsilon = 0
    mdp = TaxiMDP(slip_prob=0.0, gamma=0.99)
    vi = ValueIteration(mdp)
    vi.run_value_iteration()
    q_table = vi.get_q_table()

    for key in q_table.keys():
        print(key[0], key[1], q_table[key])
    abstr = make_abstr(q_table, Abstr_type.Q_STAR)


    # Count the number of states that get abstracted together
    state_count = 0
    states_visited = []
    for key in abstr.get_abstr_dict().keys():
        state_count += 1
    print(state_count)
    print(abstr.get_abstr_dict())

    # Write results of q-table to file
    f = open('test_abstr_results.txt', 'w')
    for key in q_table.keys():
        to_write = str(key[0]) + ' ' + str(key[1]) + ' ' + str(q_table[key]) + '\n'
        f.write(to_write)
    f.close()

    results = abstr.get_abstr_dict()
    for key in results.keys():
        print(key, results[key])
    """
    for val in results.values():
        for key in results.keys():
            for other_key in results.keys():
                key_pass = key.get_passenger_loc()
                key_goal = key.get_goal_loc()
                other_pass = other_key.get_passenger_loc()
                other_goal = other_key.get_goal_loc()
                if key != other_key and results[key] == results[other_key] and key_goal == other_goal and (key_pass == other_pass or key_pass == (0,0) or other_pass == (0,0)):
                    print(key, other_key, results[key])
    """



    print(state_count)
