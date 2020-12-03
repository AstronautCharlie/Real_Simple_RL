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
                 decay_exploration=True):
        """
        Create an agent with a state abstraction mapping
        :param mdp: an MDP environment
        :param s_a: StateAbstraction (mapping of ground states to abstract states). If no s_a is provided,
                    then this is equivalent to a regular Q-learning agent
        :param alpha: learning rate
        :param epsilon: exploration rate
        """
        Agent.__init__(self, mdp, alpha, epsilon, decay_exploration=decay_exploration)
        self.s_a = s_a

        # Create a dictionary mapping each ground state to all other ground states with the same abstract state
        #  This is done so that we don't have to iterate through all states to check abstract state mapping when
        #  performing an update
        # NO LONGER USED
        self.group_dict = {}
        if self.s_a is not None:
            for state in s_a.abstr_dict.keys():
                abstr_state = s_a.get_abstr_from_ground(state)
                self.group_dict[state] = []
                for other_state in s_a.abstr_dict.keys():
                    if state != other_state and \
                            s_a.get_abstr_from_ground(other_state) == abstr_state and \
                            not state.is_terminal() and not other_state.is_terminal():
                        self.group_dict[state].append(other_state)

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
        states_to_update = [state]
        '''
        if self.s_a is not None and state in self.group_dict.keys():
            for equiv_state in self.group_dict[state]:
                states_to_update.append(equiv_state)
        '''
        if self.s_a is not None:
            for other_state in self.mdp.get_all_possible_states():
                if self.s_a.abstr_dict[state] == self.s_a.abstr_dict[other_state]:
                    states_to_update.append(other_state)

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
        if self.s_a is None or state not in self.group_dict.keys():
            raise ValueError(
                'Attempting to detach state that is not mapped to an abstract state. State is ' + str(state.data))

        # Check if state is already in a singleton abstract state. If so, do nothing
        if not self.group_dict[state]:
            return 1

        # If state is terminal, its abstract state doesn't matter so we remove it
        if state.is_terminal():
            return 2

        # This is a hack. For some reason the above statement wasn't catching the TwoRoomMDP terminal states, so I
        #  did this instead
        if (state.x, state.y) in self.mdp.goal_location:
            return 2

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
        temp = []
        for temp_state, temp_value in self.s_a.abstr_dict.items():
            if temp_value == old_abstr_state:
                temp.append(temp_state)

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
                print('Somehow entered trapped state', state)
                quit()
            for action in cycle_actions:
                try:
                    self._set_q_value(state, action, self.mdp.gamma * max(non_cycle_values))
                except:
                    print('failed with state, action, non_cycle_values', state, (state.x, state.y) in self.mdp.goal_location, action, non_cycle_values)
                    quit()
            #print()

        return 0

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
                print(action, next_state, reward, next_state_q_value, next_val)
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
                                         variance_threshold=None,
                                         prevent_cycles=False,
                                         verbose=False):
        """
        Check if all constituent ground states in the given abstract state share the same optimal action. Return a
        list of ground states whose optimal actions differ from action learned by abstract state
        :param abstr_state: abstract state whose constituent states we are checking
        :param variance_threshold: If set, states whose optimal action value is more than variance_threshold standard
                                    deviations away from the mean
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
            return None
        # Get best action learned. Since all ground states in one abstract state will have the same
        #  learned best action, we just grab the best action from the first constituent state
        best_abstr_action, best_action_value, _ = self.check_for_optimal_action_value_next_state(constituent_states[0])
        print('Best action, value for abstr state', abstr_state, 'is', best_abstr_action, round(best_action_value, 3))
        # If the states haven't been visited/updated yet, return nothing
        if best_action_value == 0:
            return None

        # Check the optimal action of each constituent state. If it differs from the best abstract action (or keeps
        #   agent in current state if prevent_cycles == True), then add it to error states
        for state in constituent_states:
            best_action, best_action_value, next_state = self.check_for_optimal_action_value_next_state(state)
            print('Best action, value for ground state', state, 'is', best_action, round(best_action_value, 3))
            constituent_state_dict[state] = (best_action, best_action_value, next_state)
            if best_action != best_abstr_action:
                print('Mismatch!')
                error_states.append(state)
            elif prevent_cycles and next_state == state:
                print('Causes cycle!')
                error_states.append(state)
            else:
                best_action_values.append(best_action_value)
        print()
        #if verbose:
            #print(abstr_state, 'best action is', best_abstr_action)
            #for key, value in constituent_state_dict.items():
            #    print(key, value[0], round(value[1], 3), value[2], end='\t')
            #print()

        # If variance threshold is set, mark as errors all states where the value of the optimal action differs
        #  from the mean by more than variance threshold standard deviations
        if variance_threshold is not None:
            action_value_mean = sum(best_action_values) / len(best_action_values)
            action_value_std_dev = pstdev(best_action_values)
            threshold_diff = variance_threshold * action_value_std_dev
            for state in constituent_states:
                if abs(constituent_state_dict[state][1] - action_value_mean) > threshold_diff:
                    error_states.append(state)
        #print('error states are', end = ' ')
        #for state in error_states:
        #    print(state, end = ' ')
        #print()
        return error_states

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
            if error_states is not None:
                detached_states += error_states

                for state in error_states:
                    #print('Detaching state', state)
                    result_check = self.detach_state(state, reset_q_value=reset_q_value)
                    #if result_check == 1:
                        #print('State is already singleton')
                    #elif result_check == 2:
                        #print('State is terminal')

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
            if abstr == abstr_state:
                ground_states.append(ground)
        return ground_states

    def get_abstract_states(self):
        """
        :return: list of abstract states
        """
        return list(set(list(self.s_a.abstr_dict.values())))