"""
This file tests that the TwoRoomsMDP class functions as expected
"""
from GridWorld.TwoRoomsMDP import TwoRoomsMDP
from GridWorld.GridWorldStateClass import GridWorldState
from MDP.ValueIterationClass import ValueIteration

if __name__ == '__main__':

    test_num = 8

    # (1) Check that each state-action combination on the default arguments yields the expected results
    if test_num==1:
        mdp = TwoRoomsMDP()
        print('Checking all state-action combos')
        # 5 squares wide, 11 squares tall (including hallway)
        for x in range(1, 20):
            for y in range(1, 20):
                if mdp.is_inside_rooms(GridWorldState(x,y)):
                    for action in mdp.actions:
                        state = GridWorldState(x, y)
                        next_state = mdp.transition(state, action)
                        if state != next_state:
                            print(state, action, next_state)
                    print()

    # (2) Upper offset
    elif test_num==2:
        mdp = TwoRoomsMDP(upper_offset=1)
        for x in range(1,20):
            for y in range(1,20):
                if mdp.is_inside_rooms(GridWorldState(x,y)):
                    for action in mdp.actions:
                        state = GridWorldState(x, y)
                        next_state = mdp.transition(state, action)
                        if state != next_state:
                            print(state, action, next_state)
                    print()

    # (3) Lower offset
    elif test_num==3:
        mdp = TwoRoomsMDP(lower_offset=2)
        for x in range(1,20):
            for y in range(1,20):
                if mdp.is_inside_rooms(GridWorldState(x,y)):
                    for action in mdp.actions:
                        state = GridWorldState(x,y)
                        next_state = mdp.transition(state,action)
                        if state != next_state:
                            print(state, action, next_state)
                    print()

    # Non-default hallway
    elif test_num==4:
        mdp = TwoRoomsMDP(hallway_states=[2,5], hallway_height=3)
        for x in range(1,20):
            for y in range(1,20):
                if mdp.is_inside_rooms(GridWorldState(x,y)):
                    act_count = 0
                    act_list = []
                    for action in mdp.actions:
                        state=GridWorldState(x,y)
                        next_state=mdp.transition(state, action)
                        if state != next_state:
                            act_count += 1
                            act_list.append(action)
                    print(state, act_count, act_list)

    # Non-square options
    elif test_num==5:
        mdp = TwoRoomsMDP(lower_width=4,
                          lower_height=6,
                          upper_height=2,
                          upper_width=7,
                          upper_offset=2)
        for x in range(1,20):
            for y in range(1,20):
                if mdp.is_inside_rooms(GridWorldState(x,y)):
                    act_count = 0
                    act_list = []
                    for action in mdp.actions:
                        state=GridWorldState(x,y)
                        next_state=mdp.transition(state, action)
                        if state != next_state:
                            act_count += 1
                            act_list.append(action)
                    print(state, act_count, act_list)

    # I-option, different goal/start locations
    elif test_num==6:
        mdp = TwoRoomsMDP(lower_width=3,
                          upper_width=3,
                          lower_height=1,
                          upper_height=1,
                          hallway_states=[2],
                          hallway_height=10,
                          goal_location=[(3,1)],
                          init_state=(1,12))
        for x in range(1,20):
            for y in range(1,20):
                if mdp.is_inside_rooms(GridWorldState(x,y)):
                    act_count = 0
                    act_list = []
                    for action in mdp.actions:
                        state=GridWorldState(x,y)
                        next_state=mdp.transition(state, action)
                        if state != next_state:
                            act_count += 1
                            act_list.append(action)
                    print(state, act_count, act_list)

    # One-room option
    elif test_num==7:
        mdp = TwoRoomsMDP(upper_width=0, upper_height=0, hallway_states=[],
                          lower_width=5, lower_height=5)
        for x in range(1, 20):
            for y in range(1, 20):
                if mdp.is_inside_rooms(GridWorldState(x, y)):
                    act_count = 0
                    act_list = []
                    for action in mdp.actions:
                        state = GridWorldState(x, y)
                        next_state = mdp.transition(state, action)
                        if state != next_state:
                            act_count += 1
                            act_list.append(action)
                    print(state, act_count, act_list)

    # VI
    elif test_num==8:
        mdp = TwoRoomsMDP(gamma=0.9,lower_width=1,lower_height=5, upper_width=0, upper_height=0, hallway_states=[])
        vi = ValueIteration(mdp)
        vi.run_value_iteration()
        q_table = vi.get_q_table()
        for key, value in q_table.items():
            print(key[0], key[1], value)

    # get_all_possible_states
    elif test_num==9:
        mdp = TwoRoomsMDP(lower_offset=2,
                          hallway_states=[4],
                          upper_width=10,
                          upper_height=2)
        '''
        for x in range(1,20):
            for y in range(1,20):
                for action in mdp.actions:
                    state = GridWorldState(x,y)
                    next_state = mdp.transition(state, action)
                    if state != next_state:
                        print(state, action, next_state)
                print()
        '''
        for state in mdp.get_all_possible_states():
            print(state)

    # get_next_possible_states
    elif test_num==10:
        mdp = TwoRoomsMDP()
        for x in range(1,20):
            for y in range(1,20):
                state = GridWorldState(x,y)
                if mdp.is_inside_rooms(state):
                    for action in mdp.actions:
                        #try:
                            #print('Succeeded with state, action', state, action)
                            next_states = mdp.get_next_possible_states(state, action)
                            for key, value in next_states.items():
                                if state != key:
                                    print(state, action, key, value)
                    print()
                        #except:
                        #    print('Failed with state, action', state, action)
                        #    quit()
