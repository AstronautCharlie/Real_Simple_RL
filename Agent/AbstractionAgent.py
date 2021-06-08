"""
This type of Agent extends the regular Q-learning agent by adding in an abstraction. This abstraction is a
grouping of ground states to abstract states. When a Q-value is update for a ground state, the agent gets any
other states that are mapped to the same abstract state as the ground state and performs the same update to
their Q-value. If done properly, this should yield the same behavior as an agent operating in an abstract MDP
"""

from Agent.AgentClass import Agent
from GridWorld.ActionEnums import Dir
from collections import defaultdict
import random
from statistics import pstdev


class AbstractionAgent(Agent):
    def __init__(self,
                 mdp,
                 s_a=None,
                 alpha=0.1,
                 epsilon=0.1,
                 decay_exploration=True,
                 consistency_check='abstr',
                 detach_reassignment='individual',
                 seed=None,
                 ground_states=None):
        """
        Create an agent with a state abstraction mapping
        :param mdp: an MDP environment
        :param s_a: StateAbstraction (mapping of ground states to abstract states). If no s_a is provided,
                    then this is equivalent to a regular Q-learning agent
        :param alpha: learning rate
        :param epsilon: exploration rate
        """
        Agent.__init__(self, mdp, alpha, epsilon, decay_exploration=decay_exploration, seed=seed)
        self.s_a = s_a
        if ground_states is not None:
            self.all_possible_states = ground_states
        else:
            self.all_possible_states = mdp.get_all_possible_states()
        if consistency_check not in ('abstr', 'vote'):
            raise ValueError('Consistency Check type must be \'abstr\' or \'vote\'')
        if detach_reassignment not in ('individual', 'group'):
            raise ValueError('Detach Reassignment must be \'individual\' or \'group\'')
        self.consistency_check = consistency_check
        self.detach_reassignment = detach_reassignment
        # Create a dictionary mapping each ground state to all other ground states with the same abstract state
        #  This is done so that we don't have to iterate through all states to check abstract state mapping when
        #  performing an update
        if self.s_a is not None:
            self.group_dict = self.reverse_abstr_dict(self.s_a.abstr_dict)
        else:
            self.group_dict = None
        #if seed:
        #    random.seed(1234)
        #print('Reassignment type is', self.detach_reassignment)

    def reverse_abstr_dict(self, dict):
        """
        Given a dictionary mapping ground states to abstract states, create a dictionary mapping abstract states to
        lists of ground states
        """
        abstr_to_ground = {}
        for state in self.all_possible_states:
            try:
                abstr_state = self.s_a.abstr_dict[state]
            except:
                print(state)
                quit()
            if abstr_state not in abstr_to_ground.keys():
                abstr_to_ground[abstr_state] = [state]
            else:
                abstr_to_ground[abstr_state].append(state)
        return abstr_to_ground

    def update(self, state, action, next_state, reward):
        """
        Update the Agent's internal q-table according to the Bellman equation for all
        states that are mapped to the same abstract state as the ground state
        :param state: current State
        :param action: action taken by Agent
        :param next_state: next State
        :param reward: float
        """
        # Get all states mapped to the same abstract state as the current state
        if self.s_a is not None:
            try:
                states_to_update = self.group_dict[self.s_a.abstr_dict[state]]
            except:
                for key, value in self.group_dict.items():
                    print(key, end = ' ')
                    for val in value:
                        print(val, end = ' ')
                    print()
                for key, value in self.s_a.abstr_dict.items():
                    print(key, value)
                print('AbstractionAgent updating non-existsant state', str(state))
                quit()

        else:
            states_to_update = [state]


        # Update all states in the same abstract state
        old_q_value = self.get_q_value(state, action)
        # This if/then is to keep the Q-values from exploding. As a result, the terminal state will always have
        # value 0
        if not next_state.is_terminal():
            best_next_action_value = self.get_best_action_value(next_state)
        else:
            best_next_action_value = 0.0
        # Calculate the new Q-value
        new_q_value = old_q_value + self._alpha * (reward + self.mdp.gamma * best_next_action_value - old_q_value)
        # Apply the update to all states also mapped to this state
        for equiv_state in states_to_update:
            self._set_q_value(equiv_state, action, new_q_value)

    def detach_state(self, state, reset_q_value=False):
        """
        Change the abstraction mapping such that the given state is now mapped to a new, singleton abstract state.
        Also reset the Q-table for abstract state to 0 for all state-action pairs.

        If the state is already in a singleton abstract state, it will not be reassigned and its Q-value will be
        unchanged.
        :param state: State to be updated
        :return: 0 if a state was detached, 1 if state was already in a singleton abstract state, 2 if state is terminal
                (in which case nothing is done)
        """
        if self.s_a is None or state not in self.s_a.abstr_dict.keys():
            raise ValueError(
                'Attempting to detach state that is not mapped to an abstract state. State is ' + str(state.data))

        # Check if state is already in a singleton abstract state. If so, do nothing
        if not self.s_a.abstr_dict[state]:
            return 1

        # If state is terminal, its abstract state doesn't matter so we remove it
        if state.is_terminal():
            return 2

        # This is a hack. For some reason the above statement wasn't catching the TwoRoomMDP terminal states, so I
        #  did this instead
        try:
            if (state.x, state.y) in self.mdp.goal_location:
                return 2
        except:
            pass
        # First check that this state is not a singleton. If it is, do not detach
        temp = []
        for temp_state, temp_value in self.s_a.abstr_dict.items():
            if temp_value == self.s_a.abstr_dict[state] and temp_state != state:
                temp.append(temp_state)
        if len(temp) == 0:
            print('State', state, 'is not mapped to any other abstract states, so is already a singleton. Not detaching')
            print()
            return 1

        # Set state to its own abstract state
        old_abstr_state = self.s_a.abstr_dict.pop(state, None)
        max_abstr_state = max(self.s_a.abstr_dict.values())
        self.s_a.abstr_dict[state] = max_abstr_state + 1

        # Update abstr -> list of ground state mapping
        self.group_dict[old_abstr_state].remove(state)
        self.group_dict[max_abstr_state + 1] = [state]

        # Reset Q-value for each action in this state by taking the action, finding the max Q-value of the next
        #  state, and assigning the Q-value for the state-action pair to that max Q-value
        if reset_q_value:
            cycle_actions = []
            non_cycle_values = []
            # Take a 1-step roll-out for each action and reassign the Q-value of the action to the reward + gamma *
            #  max Q-value of next state
            for action in self.mdp.actions:
                self.mdp.set_current_state(state)
                next_state = self.mdp.transition(state, action)
                reward = self.mdp.reward(state, action, next_state)
                next_state_q_value = self.get_best_action_value(next_state)
                next_val = reward + self.mdp.gamma * next_state_q_value
                if next_state == state:
                    cycle_actions.append(action)
                else:
                    non_cycle_values.append(next_val)
                    self._set_q_value(state, action, next_val)
            # For any actions that kept the agent in the current state, set the Q-value to gamma * max Q-value of
            #  non-cycle actions
            if len(non_cycle_values) == 0:
                for action in self.mdp.actions:
                    self._set_q_value(state, action, 0)
            for action in cycle_actions:
                try:
                    self._set_q_value(state, action, self.mdp.gamma * max(non_cycle_values))
                except:
                    print('failed with state, action, non_cycle_values', state, (state.x, state.y) in self.mdp.goal_location, action, non_cycle_values)
                    quit()

        return 0

    def detach_group(self, state_group, reset_q_value=False):
        """
        Detach the states in the given state group from their abstract state and reassign them all to one new abstract state
        """
        print(state_group)

        # Check if state group is of length one. If it is and is already a singleton state, do not detach
        if len(state_group) == 1:
            state = state_group[0]
            same_counter = 0
            for temp_state, temp_value in self.s_a.abstr_dict.items():
                if temp_value == self.s_a.abstr_dict[state] and temp_state != state:
                    same_counter += 1
            if same_counter == 0:
                print('State', state, 'is singleton, not detaching')
                return

        # Check if state group makes up the entirety of an abstract state. If so, return without doing anything
        old_abstr_state = self.s_a.abstr_dict[state_group[0]]
        other_state_count = 0
        for s in self.group_dict[old_abstr_state]:
            if s not in state_group:
                other_state_count += 1
        if other_state_count == 0:
            print('Error group makes up entirety of abstract state. Not detaching')
            return

        # Set group to its own abstract state in ground -> abstr mapping
        max_abstr_state = max(self.s_a.abstr_dict.values())
        for state in state_group:
            self.s_a.abstr_dict[state] = max_abstr_state + 1

        # Update abstr -> ground mapping
        for state in state_group:
            self.group_dict[old_abstr_state].remove(state)
        self.group_dict[max_abstr_state + 1] = state_group

        # In this case, take a 1-step roll-out for each constituent state-action pair and average the results together
        #  to get new q-values
        if reset_q_value:
            # This maps actions to list of values (one per state-value pair); will be averaged at the end to get new q-value
            new_action_dict = {}
            for action in self.mdp.actions:
                new_action_dict[action] = []
            for state in state_group:
                cycle_actions = []
                non_cycle_values = []
                for action in self.mdp.actions:
                    self.mdp.set_current_state(state)
                    next_state = self.mdp.transition(state, action)
                    reward = self.mdp.reward(state, action, next_state)
                    next_state_q_value = self.get_best_action_value(next_state)
                    next_val = reward + self.mdp.gamma * next_state_q_value
                    if next_state == state:
                        cycle_actions.append(action)
                    else:
                        non_cycle_values.append(next_val)
                        new_action_dict[action].append(next_val)
                if len(non_cycle_values) == 0:
                    for action in self.mdp.actions:
                        self._set_q_value(state, action, 0)
                else:
                    for action in cycle_actions:
                        try:
                            new_action_dict[action].append(self.mdp.gamma * max(non_cycle_values))
                        except:
                            print('failed with state, action, non_cycle_values', state, (state.x, state.y) in self.mdp.goal_location, action, non_cycle_values)
                            quit()
            # Now set the q-values of all the states in state group to the average of the 1-step roll-out results
            new_q_values = {}
            for key, value in new_action_dict.items():
                if len(value) != 0:
                    new_q_values[key] = sum(value) / len(value)
            for state in state_group:
                if state not in self.mdp.goal_location:
                    for key, value in new_q_values.items():
                        self._set_q_value(state, key, value)
                else:
                    for action in self.mdp.actions:
                        self._set_q_value(state, action, 0)

    def generate_rollout(self, start_from_init=True):
        """
        Create roll-out by following the greedy policy w.r.t. the learned Q-values from the MDP's initial state.
        Roll-out will go until a terminal state is reached or until it enters a cycle.
        This does not update any Q-values.
        :param start_from_init: optional boolean. If True, reset the MDP to its initial state before generating
                                roll-out.
        :return: rollout, a list of States
        """
        if start_from_init:
            self.mdp.reset_to_init()

        rollout = []
        # Dictionary mapping states to actions learned so far
        policy = self.get_learned_policy()
        state = self.get_current_state()
        while state not in rollout:
            rollout.append(state)
            if state.is_terminal():
                break
            action = policy[state]
            next_state = self.mdp.transition(state, action)
            state = next_state
        return rollout

    def check_for_optimal_action_value_next_state(self, state, verbose=False):
        """
        Take every possible action in the given state and return the action that yields the highest reward plus
        discount * max next state Q value
        :param state: a state in the MDP
        :return: optimal action, value of optimal action
        """
        actions = list(self.mdp.actions)
        # Check actions in a random order so that if two actions are equally optimal, we select a random one
        random.shuffle(actions)
        optimal_action = None
        optimal_action_value = float('-inf')
        optimal_next_state = None
        if verbose:
            print('Checking actions for state', state)
        for action in actions:
            self.mdp.set_current_state(state)
            next_state = self.mdp.transition(state, action)
            reward = self.mdp.reward(state, action, next_state)
            next_state_q_value = self.get_best_action_value(next_state)
            next_val = reward + self.mdp.gamma * next_state_q_value
            if verbose:
                print('Q-value for ', action, next_state, reward, round(next_state_q_value, 7), round(next_val, 7))
            if next_val > optimal_action_value:
                optimal_action = action
                optimal_action_value = next_val
                optimal_next_state = next_state
        if verbose:
            print('Best action, action value', optimal_action, optimal_action_value)
        self.mdp.reset_to_init()
        return optimal_action, optimal_action_value, optimal_next_state

    def check_for_optimal_action(self, state):
        action, _, _ = self.check_for_optimal_action_value_next_state(state)
        return action

    def check_for_optimal_action_value(self, state):
        _, val, _ = self.check_for_optimal_action_value_next_state(state)
        return val

    def check_for_optimal_next_state(self, state):
        _, _, next_state = self.check_for_optimal_action_value_next_state(state)
        return next_state

    def check_abstract_state_consistency(self,
                                         abstr_state,
                                         prevent_cycles=False,
                                         verbose=False):
        """
        Check if all constituent ground states in the given abstract state share the same optimal action. Return a
        list of ground states whose optimal actions differ from action learned by abstract state.

        NOTE if self.detach_reassignment = "group", then this returns a list of LISTS
        :param abstr_state: abstract state whose constituent states we are checking
        :param prevent_cycles: If true, states where the optimal action keeps the agent in the state it is in are
                                treated as errors
        :return: a list of states whose actions differ from action learned by abstract state. Return None if
                no such states exist
        """
        error_states = []
        best_action_values = []
        constituent_state_dict = {}
        # Get all ground states mapped to this abstract state. If there is only one constituent state, we
        #  do nothing and return
        constituent_states = self.get_ground_states_from_abstract_state(abstr_state)
        if len(constituent_states) == 1:
            print('FUCK length of constituent states is 1')
            return []

        # If 'abstr', detach states if their optimal action disagrees with action learned for state
        if self.consistency_check == 'abstr':
            # Get best action learned in abstr state. Since all ground states in one abstract state will have the same
            #  learned best action, we just grab the best action from the first constituent state
            if verbose:
                print('Constituent states are', end=' ')
                for c_state in constituent_states:
                    print(c_state, end=' ')
                print()

            best_abstr_action, best_action_value = self.get_best_action_value_pair(constituent_states[0])
            '''
            if verbose:
                for act in self.mdp.actions:
                    print('Q-value for', abstr_state, act, 'is', round(self.get_q_value(constituent_states[0], act), 7))
                    print('Should be same', abstr_state, act, 'is', round(self.get_q_value(constituent_states[1], act), 7))
                print('Best action, value for abstr state', abstr_state, 'is', best_abstr_action, round(best_action_value, 7))
            '''

            # If the states haven't been visited/updated yet, return nothing
            if best_action_value == 0:
                return None

            # Check the optimal action of each constituent state. If it differs from the best abstract action (or keeps
            #   agent in current state if prevent_cycles == True), then add it to error states
            for state in constituent_states:
                best_action, best_action_value, next_state = self.check_for_optimal_action_value_next_state(state,
                                                                                                            verbose=verbose)
                constituent_state_dict[state] = (best_action, best_action_value, next_state)
                if best_action != best_abstr_action:
                    error_states.append(state)
                elif prevent_cycles and next_state == state:
                    error_states.append(state)
                else:
                    best_action_values.append(best_action_value)
            print()
        # In this case, vote; any state whose abstract action differs from majority is detached, ties broken
        #  randomly
        elif self.consistency_check == 'vote':
            # Count number of constituent states for which each action is optimal
            best_action_counter = {}
            for action in self.mdp.actions:
                best_action_counter[action] = 0
            for state in constituent_states:
                best_action, best_action_value, next_state = self.check_for_optimal_action_value_next_state(state)
                constituent_state_dict[state] = (best_action, best_action_value, next_state)
                print('Best action, value for ground state', state, 'is', best_action, round(best_action_value, 3))
                best_action_counter[best_action] += 1
            # 'True' optimal action is the majority (ties broken randomly)
            max_val = 0
            elected_action = None
            actions = self.mdp.actions
            random.shuffle(actions)
            # 'elect' the winner
            for action in actions:
                print('Number of states for which', action, 'is optimal is', best_action_counter[action])
                if best_action_counter[action] > max_val:
                    elected_action = action
                    max_val = best_action_counter[action]
            # Tag each dissenting state as error
            for state in constituent_states:
                if constituent_state_dict[state][0] != elected_action:
                    error_states.append(state)
        else:
            raise ValueError('Consistency check method ' + str(self.consistency_check) + ' is not supported')

        # If detach_reassignment is individual, we just return the list of individual error states

        # If it is group, return a list of lists, where each constituent list is a group of ground states with the
        #  same 1-step optimal roll-out, and therefore will be mapped to the same new abstract state
        #  NOTE if you do this, return type becomes list of lists
        if self.detach_reassignment == 'group':
            temp = []
            for action in self.mdp.actions:
                new_abstr_states = []
                for state in error_states:
                    if constituent_state_dict[state][0] == action:
                        new_abstr_states.append(state)
                if len(new_abstr_states) > 0:
                    temp.append(new_abstr_states)
            print('New abstract state groupings from error states are')
            for i in range(len(temp)):
                print('Group number', i, end = '   ')
                for j in temp[i]:
                    print(j, end = ' ')
            print()
            error_states = temp

        return error_states

    # THIS IS THE MAIN DETACHMENT METHOD
    def detach_inconsistent_states(self,
                                   variance_threshold=None,
                                   prevent_cycles=False,
                                   reset_q_value=False,
                                   verbose=False):
        """
        Iterate through all abstract states, check their consistency, and detach any states that are marked errors
        :param variance_threshold: If provided, states whose optimal action values are more than this many standard
                                    deviations away from the average for the abstract state will be treated as errors
        :param prevent_cycles: If true, states whose optimal actions keep them in their current state are treated
                                as errors
        :return: list of detached states
        """
        print('Checking for inconsistent states, reset q value is', reset_q_value)
        abstr_states = self.get_abstract_states()
        detached_states = []

        for abstr_state in abstr_states:
            error_states = self.check_abstract_state_consistency(abstr_state,
                                                                 variance_threshold=variance_threshold,
                                                                 prevent_cycles=prevent_cycles,
                                                                 verbose=verbose)
            if self.detach_reassignment == 'individual':
                if error_states is not None:
                    detached_states += error_states
                    for state in error_states:
                        result_check = self.detach_state(state, reset_q_value=reset_q_value)
            elif self.detach_reassignment == 'group':
                if error_states is not None:
                    if verbose:
                        print('About to detach group')
                        print('Error states are', end=' ')
                        if isinstance(error_states[0], list):
                            for group in error_states:
                                print('[', end='')
                                for state in group:
                                    print(state, end=' ')
                                print(']')
                        else:
                            for state in error_states:
                                print(state, end=' ')
                    for error_group in error_states:
                        detached_states += error_group
                        print('About to detach group')
                        for state in error_group:
                            print(state, end = ' ')
                        self.detach_group(error_group, reset_q_value=reset_q_value)
            else:
                raise ValueError("detach_reassignment method '" + str(self.detach_reassignment) + " is not supported.")

        print('Finished detaching states, new abstraction dicts are:')
        for key, value in self.group_dict.items():
            print(key, end=' ')
            for val in value:
                print(val, end=' ')
            print()
        for key, value in self.s_a.abstr_dict.items():
            print(key, value, end = '     ')

        if verbose:
            print('Number of detached states', len(detached_states))
            print('Detached states are', end=' ')
            for state in detached_states:
                print(state, end=' ')
            print()
        return detached_states

    # -----------------
    # Getters & setters
    # -----------------

    def get_abstraction_as_string(self):
        """
        Create string representation of the agents abstraction mapping. Takes the form of a list of tuples where first
        element is ground state and second element is abstract state
        :return:
        """
        abstr_list = []
        for g_state, a_state in self.s_a.abstr_dict.items():
            abstr_list.append((g_state.data, a_state))
        return str(abstr_list)

    def get_ground_states_from_abstract_state(self, abstr_state):
        """
        Return a list of ground states mapped to the given abstract state
        :param abstr_state: an abstract state
        :return: list of ground states
        """
        ground_states = []

        for ground, abstr in self.s_a.abstr_dict.items():
            try:
                if abstr == abstr_state:
                    ground_states.append(ground)

            except:
                try:
                    if abstr_state.data.data == abstr:
                        ground_states.append(ground)
                except:
                    try:
                        if abstr == abstr_state.data:
                            ground_states.append(ground)
                    except:
                        data = abstr.data
                        if data == abstr_state:
                            ground_states.append(ground)
        return ground_states

    def get_all_abstract_states(self):
        """
        :return: list of abstract states
        """
        return self.s_a.get_all_abstr_states()

    def get_all_possible_ground_states(self):
        return self.mdp.get_all_possible_states()

    def get_abstr_from_ground(self, ground_state):
        return self.s_a.get_abstr_from_ground(ground_state)