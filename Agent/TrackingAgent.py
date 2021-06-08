# TODO Problem is that 1-step roll-out doesn't reliably split
#   off error states, as

"""
This extension of AbstractionAgent also tracks the Q-value updates made to each abstract state-action pair and
calculates the standard deviation of them
"""
from Agent.AbstractionAgent import AbstractionAgent
from GridWorld.TwoRoomsMDP import TwoRoomsMDP
from GridWorld.GridWorldMDPClass import GridWorldMDP
from GridWorld.GridWorldStateClass import GridWorldState
from resources.AbstractionCorrupters import *
from MDP.StateAbstractionClass import StateAbstraction
from resources.AbstractionMakers import make_abstr
from util import *
import time

LIMIT = 10000

class TrackingAgent(AbstractionAgent):
    def __init__(self,
                 mdp,
                 s_a=None,
                 alpha=0.1,
                 epsilon=0.1,
                 decay_exploration=True,
                 consistency_check='abstr',
                 detach_reassignment='group',
                 volatility_threshold=0,
                 seed=1234,
                 ground_states=None):
        start_time = time.time()
        if s_a is None:
            s_a = StateAbstraction()
            s_a.make_trivial_abstraction(mdp)
        print('made trivial abstraction', time.time() - start_time)
        loop = time.time()
        self.volatility_threshold = volatility_threshold

        super().__init__(mdp,
                         s_a=s_a,
                         alpha=alpha,
                         epsilon=epsilon,
                         decay_exploration=decay_exploration,
                         consistency_check=consistency_check,
                         detach_reassignment=detach_reassignment,
                         seed=seed,
                         ground_states=ground_states)
        print('finished call to super', time.time() - loop)

        loop = time.time()
        # Create record to hold Q-value updates to each abstract state-action pair
        self.abstr_update_record = {}

        abstr_mdp = AbstractMDP(mdp, s_a)
        self.abstr_mdp = abstr_mdp
        abstr_states = abstr_mdp.get_all_abstr_states()
        if ground_states is None:
            ground_states = mdp.get_all_possible_states()
        #abstr_states.sort()

        for abstr_state in abstr_mdp.get_all_abstr_states():
            if abstr_state not in self.abstr_update_record.keys():
                self.abstr_update_record[abstr_state] = {}
                for action in self.mdp.actions:
                    self.abstr_update_record[abstr_state][action] = []
        print('break1', time.time() - loop)
        loop = time.time()


        # Record of Q-value updates to each ground state-action pair
        self.ground_update_record = {}
        for ground_state in ground_states:
            if ground_state not in self.ground_update_record.keys():
                self.ground_update_record[ground_state] = {}
                for action in self.mdp.actions:
                    self.ground_update_record[ground_state][action] = []

        self.episode_step_count = 0


        print('break2', time.time() - loop)
        loop = time.time()
        # Create record to hold counts of abstr state occupations

        self.abstr_state_occupancy_record = {}
        #for abstr_state in abstr_mdp.get_all_abstr_states():
        for abstr_state in abstr_states:
            self.abstr_state_occupancy_record[abstr_state] = 0


        # Create record to hold counts of ground state occupations
        self.state_occupancy_record = {}
        for state in ground_states:
            self.state_occupancy_record[state] = 0
        print('break3', time.time() - loop)
        loop = time.time()

        # Create record to hold counts of state-action pairs selected
        self.state_action_pair_counts = {}
        for state in ground_states:
            self.state_action_pair_counts[state] = {}
            for action in self.mdp.actions:
                self.state_action_pair_counts[state][action] = 0


        # Create record to hold counts of abstract-state action pairs selected
        self.abstr_state_action_pair_counts = {}
        for state in ground_states:
            abstr_state = self.get_abstr_from_ground(state)
            if abstr_state not in self.abstr_state_action_pair_counts.keys():
                self.abstr_state_action_pair_counts[abstr_state] = {}
                for action in self.mdp.actions:
                    self.abstr_state_action_pair_counts[abstr_state][action] = 0


        print('break4', time.time() - loop)
        loop = time.time()

        # Record r + gamma * max q-value of next state
        self.ground_reward_record = {}
        #for state in self.get_all_possible_ground_states():
        for state in ground_states:
            self.ground_reward_record[state] = {}
            for action in self.mdp.actions:
                self.ground_reward_record[state][action] = []

        # Minimum number of visits a state must have to be considered
        self.volatility_threshold = volatility_threshold

        # States that are reachable from a given state
        #  Maps state->list of next states that we ever reached from that state
        self.reachable_state_dict = {}
        for state in ground_states:
            self.reachable_state_dict[state] = []
        print('done', time.time() - loop)

    # -------------------
    # Overwritten methods
    # -------------------
    def explore(self):
        """
        Epsilon-greedy exploration including recording of Q-value updates
        """
        # Regular explore function
        current_state = self.get_current_state()
        action = self.epsilon_greedy(current_state)
        next_state, reward = self.mdp.act(action)

        # Calculate Q-value update
        old_q_value = self.get_q_value(current_state, action)
        best_next_action_value = self.get_best_action_value(next_state)
        td_error = reward + self.mdp.gamma * best_next_action_value - old_q_value
        current_abstr_state = self.s_a.get_abstr_from_ground(current_state)

        # Record TD errors in ground and abstr state records, increment state-action pair counter

        try:
            self.abstr_update_record[current_abstr_state][action].append(td_error)
        except:
            print(self.abstr_update_record[current_abstr_state])
            print(current_abstr_state, current_state, action)
            print('failed in TrackingAgent.explore()')
            quit()

        self.ground_update_record[current_state][action].append(td_error)
        self.state_action_pair_counts[current_state][action] += 1
        self.abstr_state_action_pair_counts[current_abstr_state][action] += 1
        try:
            if next_state != current_state and next_state not in self.reachable_state_dict[current_state]:
                self.reachable_state_dict[current_state].append(next_state)
        except:
            print('failed in TrackingAgent', next_state, current_state)
            quit()
        # Apply Q-value update
        super().update(current_state, action, next_state, reward)

        if self.get_q_value(current_state, action) != 0:
            self._update_learning_parameters()
        self._step_counter += 1
        if next_state.is_terminal():
            self._episode_counter += 1

        # Update state occupancy record
        self.state_occupancy_record[current_state] += 1
        abstr_state = self.abstr_mdp.get_abstr_from_ground(current_state)
        if next_state.is_terminal():
            try:
                self.state_occupancy_record[next_state] += 1
            except:
                self.state_occupancy_record[next_state] = 1

        return current_state, action, next_state, reward

    def detach_state(self, state, reset_q_value=False):

        # In this case, reset Q-value to average of neighbor states
        if reset_q_value == 'neighbor':
            super().detach_state(state, reset_q_value=False)
            reachable_states = self.reachable_state_dict[state]
            q_table = self.get_q_table()
            for action in self.mdp.actions:
                action_val = 0
                for next in reachable_states:
                    action_val += self.get_q_value(next, action)
                action_val = action_val / len(reachable_states)
                self._set_q_value(state, action, action_val)
        else:
            super().detach_state(state, reset_q_value=reset_q_value)
        new_abstr_state = self.get_abstr_from_ground(state)
        # Only checking if it's a key in this one record
        if new_abstr_state not in self.abstr_update_record.keys():
            self.abstr_update_record[new_abstr_state] = {}
            self.abstr_state_action_pair_counts[new_abstr_state] = {}
            self.abstr_state_occupancy_record[new_abstr_state] = 0
            for action in self.mdp.actions:
                self.abstr_update_record[new_abstr_state][action] = []
                self.abstr_state_action_pair_counts[new_abstr_state][action] = 0


    # -----------------
    # Tracking-Specific
    # -----------------
    def calculate_normalized_volatility(self, verbose=False):
        """
        For all states that have been visited at least self.volatility_threshold times,
        calculate normalized volatility

        Return result in dictionary {abstr_state -> {action -> normalized std dev scaled by sqrt population}}
        """
        # This holds the final result
        volatility_record = {}

        abstr_states = np.unique(self.get_all_abstract_states())
        for abstr_state in abstr_states:
            volatility_record[abstr_state] = {}
            # Get each state-action pair with enough visits, record # of visits
            ground_states = self.get_ground_states_from_abstract_state(abstr_state)
            for action in self.mdp.actions:
                visited_states = []
                min_occupancy_count = float("inf")
                for ground_state in ground_states:
                    visit_count = self.state_action_pair_counts[ground_state][action]
                    if visit_count > self.volatility_threshold:
                        visited_states.append(ground_state)
                        if visit_count < min_occupancy_count:
                            min_occupancy_count = visit_count

                # Randomly sample from Q-value updates
                random_samples = np.array([])
                for ground_state in visited_states:
                    q_value_updates = self.ground_update_record[ground_state][action]
                    random_samples = np.append(random_samples, np.random.choice(q_value_updates, min_occupancy_count))

                # Calculate normalized standard deviation scaled by sqrt population
                normalized_volatility = np.std(random_samples)

                # Get pair count. If abstr_state is not in the pair count keys, then it is a new
                #  abstract state so create an entry for it
                if abstr_state not in self.abstr_state_action_pair_counts.keys():
                    self.abstr_state_action_pair_counts[abstr_state] = {}
                    for action in self.mdp.actions:
                        self.abstr_state_action_pair_counts[abstr_state][action] = 0
                pair_count = self.abstr_state_action_pair_counts[abstr_state][action]

                volatility_record[abstr_state][action] = np.sqrt(pair_count) * normalized_volatility
                if verbose:
                    print(str(abstr_state).ljust(3), str(action).ljust(12), str(round(normalized_volatility, 4)).ljust(5), str(round(np.sqrt(pair_count) * normalized_volatility, 4)))

        return volatility_record

    def calculate_intervals_for_abstr_state(self, abstr_state):
        """
        Calculate all the confidence intervals for the Q-value updates for
        each ground state in the given abstract state.

        Returns dictionary {ground_state -> {action -> (lower bound, upper bound, count of updates)}}
        """
        # Get ground states associated with abstract state
        ground_states = self.get_ground_states_from_abstract_state(abstr_state)

        result = {}

        for ground_state in ground_states:
            result[ground_state] = {}
            for action in self.mdp.actions:
                updates = self.ground_update_record[ground_state][action]
                lower_bound, upper_bound = calculate_confidence_interval(updates, 0.05)
                result[ground_state][action] = (lower_bound, upper_bound, len(updates))

        return result

    def get_volatility_snapshot(self):
        """
        Return a dictionary mapping abstr_state to max volatility over actions at that state.

        Dictionary is sorted in descending order of value
        """

        # Temp is a dictionary mapping abstr_state -> max volatility over actions
        temp = {}
        volatility_record = self.calculate_normalized_volatility()
        for abstr_state, action_to_vol in volatility_record.items():
            temp[abstr_state] = float("-inf")
            for action, vol in action_to_vol.items():
                if temp[abstr_state] < vol:
                    temp[abstr_state] = vol
        final = {k: v for k, v in sorted(temp.items(), key=lambda item: item[1], reverse=True)}

        return final

    def get_volatile_state_by_rank(self, volatility_record, rank=0):
        """
        Return the abstract state with the highest volatility (take max over all actions)

        Rank indicates i for fetching the i-th most volatile state (0-indexed, descending order)
        """
        vol_state = list(volatility_record.keys())[rank]
        vol_value = list(volatility_record.values())[rank]

        return vol_state, vol_value

    def check_abstract_state_consistency_from_record(self, abstr_state):
        """
        Check the consistency of the abstract state by looking at all the states reachable from each constituent state.
        If no constituent state has reachable states with higher value, detach it
        """
        constituent_states = self.get_ground_states_from_abstract_state(abstr_state)
        if len(constituent_states) == 1:
            print('Length of constituent states is 1. Skipping...')
            return []

        error_states = []

        # Look at all of the states reachable from the constituent state. If none of them have max Q-value higher than
        #  the max Q-value of the constituent state, it is an error
        for state in constituent_states:
            reachable_states = self.reachable_state_dict[state]
            is_error = True
            for next_state in reachable_states:
                if (self.get_best_action_value(next_state) > self.get_best_action_value(state)) or next_state.is_terminal():
                    is_error = False
                    break
            if is_error:
                error_states.append(state)

        return error_states


    def detach_inconsistent_states(self,
                                   variance_threshold=None,
                                   prevent_cycles=False,
                                   reset_q_value=False,
                                   verbose=False):
        """
        Get most volatile state as dictated by volatility snapshot, check it for consistency, and
        detach inconsistent ground states
        :param variance_threshold: ignore
        :param prevent_cycles: If true, states where action results in cycle is treated as error
        :param reset_q_value: boolean, Reset q-value of detached states to 0
        :param verbose: print stuff
        """

        volatility_snapshot = self.get_volatility_snapshot()
        if verbose:
            print('Volatility record is')
            i = 0
            for state, vol in volatility_snapshot.items():
                if i > 5:
                    break
                print(state, vol)
                i += 1

        # self.get_most_volatile_state returns a tuple of a state and volatility record, so we just grab first element
        most_volatile_state = self.get_volatile_state_by_rank(volatility_snapshot, 0)[0]
        if len(self.get_ground_states_from_abstract_state(most_volatile_state)) == 1:
            if verbose:
                print('Most volatile state is singleton, skipping detach')
            self.reset_volatility_record(most_volatile_state)
            return []

        if verbose:
            print('most volatile state is', most_volatile_state, type(most_volatile_state))
            print('Ground state volatilities are')
            for ground_state in self.get_ground_states_from_abstract_state(most_volatile_state):
                for action in self.mdp.actions:
                    if self.state_action_pair_counts[ground_state][action] > 0:
                        update_mean = np.mean(self.ground_update_record[ground_state][action])
                        update_stdev = np.std(self.ground_update_record[ground_state][action])
                        lb, ub = calculate_confidence_interval(self.ground_update_record[ground_state][action],
                                                               alpha=0.05)
                        lb_a, ub_a = calculate_confidence_interval(self.ground_update_record[ground_state][action],
                                                                   alpha=0.1)
                        lb_b, ub_b = calculate_confidence_interval(self.ground_reward_record[ground_state][action],
                                                                   alpha=0.05)

                        print(ground_state, action,
                              'mean:', np.round(update_mean, 4),
                              'std dev:', np.round(update_stdev, 4),
                              '90% conf int:', np.round(lb, 4), np.round(ub, 4),
                              '80% conf int:', np.round(lb_a, 4), np.round(ub_a, 4),
                              '90% reward conf int:', np.round(lb_b, 4), np.round(ub_b, 4),
                              'visit count:', self.state_action_pair_counts[ground_state][action])
                print()

        # Detachment based on local maxes
        error_states = self.check_abstract_state_consistency_from_record(most_volatile_state)

        # If no error states, return immediately
        if error_states == []:
            if verbose:
                print('No error states. Returning immediately...')
            return []

        # Do the actual detachment
        if verbose:
            print('Detaching error states: ', end='')
            for state in error_states:
                print(state, end = '')
            print()
        detached_states = error_states
        for error_state in error_states:
            self.detach_state(error_state, reset_q_value=reset_q_value)

        # Create entries in volatility records for newly created abstract states
        for detached_state in detached_states:
            new_abstr_state = self.get_abstr_from_ground(detached_state)
            # Only checking if it's a key in this one record
            if new_abstr_state not in self.abstr_update_record.keys():
                self.abstr_update_record[new_abstr_state] = {}
                self.abstr_state_action_pair_counts[new_abstr_state] = {}
                self.abstr_state_occupancy_record[new_abstr_state] = 0
                for action in self.mdp.actions:
                    self.abstr_update_record[new_abstr_state][action] = []
                    self.abstr_state_action_pair_counts[new_abstr_state][action] = 0

        # Reset records for abstract state
        self.reset_volatility_record(most_volatile_state)

        # If applicable, reset q-values for each detached state-action pair to the result
        #  of a 1-step rollout
        if reset_q_value:
            for d_state in detached_states:
                cycle_actions = []
                non_cycle_values = []
                # Take 1-step roll-out for each action and reassign Q-value to result
                for action in self.mdp.actions:
                    self.mdp.set_current_state(d_state)
                    next_state = self.mdp.transition(d_state, action)
                    reward = self.mdp.reward(d_state, action, next_state)
                    next_state_q_value = self.get_best_action_value(next_state)
                    next_val = reward + self.mdp.gamma * next_state_q_value
                    if next_state == d_state:
                        cycle_actions.append(action)
                    else:
                        non_cycle_values.append(next_val)
                        self._set_q_value(d_state, action, next_val)
                # For any actions that kept agent in current state, set Q-value to
                #  gamma * max Q-value of non-cycle actions
                if len(non_cycle_values) == 0:
                    for action in self.mdp.actions:
                        self._set_q_value(d_state, action, 0)
                for action in cycle_actions:
                    if len(non_cycle_values) > 0:
                        self._set_q_value(d_state, action, self.mdp.gamma * max(non_cycle_values))
                    else:
                        print('No non-cycle action for', d_state)
            # Reset to initial state
            self.mdp.reset_to_init()


        # Print if applicable
        if verbose:
            print('Finished detaching states, new abstraction dicts are:')
            for key, value in self.group_dict.items():
                print(key, end=' ')
                for val in value:
                    print(val, end=' ')
                print()
            for key, value in self.s_a.abstr_dict.items():
                print(key, value, end = '     ')

        return detached_states

    def reset_volatility_record(self, abstr_state):
        """
        Reset the record of q-value updates for the given abstract state and all its constituent states to
        an empty list
        """
        # Abstr state occupancy record

        for action in self.mdp.actions:
            # TD error record for abstract state
            self.abstr_update_record[abstr_state][action] = []

            # State-action pair counts
            self.abstr_state_action_pair_counts[abstr_state][action] = 0

        ground_states = self.get_ground_states_from_abstract_state(abstr_state)
        for ground_state in ground_states:

            # Ground state occupancy record
            self.state_occupancy_record[ground_state] = 0
            for action in self.mdp.actions:
                # TD error record for ground state
                self.ground_update_record[ground_state][action] = []

                # State-action pair counts
                self.state_action_pair_counts[ground_state][action] = 0

    # -----------------------
    # Make online abstraction
    # -----------------------
    def make_online_abstraction(self,
                                abstr_type,
                                epsilon=1e-12,
                                combine_zeroes=False,
                                seed=None):
        """
        Convert the existing Q-table into the given type of abstraction
        :param abstr_type: type of abstraction to make
        :param epsilon: approximation epsilon for making abstraction
        :param combine_zeroes: if true, all states with value 0 are combined
        :param threshold: minimum threshold for what counts as a 0 state
        :param seed: ignore
        """
        approx_s_a = make_abstr(self._q_table,
                                abstr_type,
                                epsilon=epsilon,
                                combine_zeroes=combine_zeroes,
                                seed=seed)

        q_table = self.get_q_table()

        self.s_a = approx_s_a
        zero_count = 0
        for state in self.s_a.get_all_abstr_states():
            is_zero = True
            for a in self.mdp.actions:
                if self._q_table[(state, a)] != 0:
                    is_zero = False
            if is_zero:
                zero_count += 1
        print('zero count:', zero_count)

        # Add abstract states to transition records
        for abstr_state in self.s_a.get_all_abstr_states():
            ground_states = self.s_a.get_ground_from_abstr(abstr_state)
            self.abstr_state_action_pair_counts[abstr_state] = {}
            for action in self.mdp.actions:
                self.abstr_state_action_pair_counts[abstr_state][action] = 0
                for ground in ground_states:
                    self.abstr_state_action_pair_counts[abstr_state][action] += self.state_action_pair_counts[ground][action]
                    self.abstr_update_record[abstr_state][action].extend(self.ground_update_record[ground][action])
        self.group_dict = self.reverse_abstr_dict(self.s_a.abstr_dict)

    def detach_most_visited_state(self):
        """
        Detach the (non-singleton) ground state with the highest visit count
        """
        visit_counts = list(set(list(self.state_occupancy_record.values())))
        visit_counts.sort(reverse=True)
        i = 0
        while True:
            if i >= len(visit_counts):
                break
            max_visits = visit_counts[i]
            for state, value in self.state_occupancy_record.items():
                abstr_state = self.get_abstr_from_ground(state).data
                group = self.group_dict[abstr_state]
                if value == max_visits:
                    print(state, value, end='    [')
                    abstr_state = self.get_abstr_from_ground(state).data
                    group = self.group_dict[abstr_state]
                    for s in group:
                        print(s, end=', ')
                    print(']')
                    if (len(group) > 1):
                        print('detaching', state)
                        self.detach_state(state)
                        return
            i += 1

    def get_most_visited_grouped_state(self):
        """
        Select the ground state that has been visited most that is also not a singleton state
        :return:
        """
        visit_counts = list(set(list(self.state_occupancy_record.values())))
        visit_counts.sort(reverse=True)
        i = 0
        while True:
            if i >= len(visit_counts):
                break
            max_visits = visit_counts[i]
            for state, value in self.state_occupancy_record.items():
                abstr_state = self.get_abstr_from_ground(state).data
                group = self.group_dict[abstr_state]
                if value == max_visits:
                    abstr_state = self.get_abstr_from_ground(state).data
                    group = self.group_dict[abstr_state]
                    if (len(group) > 1):
                        return state
            i += 1
        return None

    # -------------------------
    # Make temporal abstraction
    # -------------------------
    def make_temporal_abstraction(self, n=1):
        """
        Make a temporal abstraction by repeatedly:
            - randomly selecting a seed state
            - finding all states that are n-neighbors (i.e. are reachable in n-steps from seed state based on observed
                data)
            - grouping those together into an abstract state
        """
        states = self.get_all_possible_ground_states()
        np.random.shuffle(states)

        state_abstr_dict = {}
        abstr_state_counter = 1

        for state in states:
            if state not in state_abstr_dict.keys() and not state.is_terminal():
                # add seed state to abstract state
                state_abstr_dict[state] = abstr_state_counter

                # add n-neighbors to abstract state
                state_queue = [[state]]
                for i in range(n):
                    temp = state_queue.pop(0)
                    reachable_states = []
                    for s in temp:
                        reachable_states.extend(self.reachable_state_dict[s])
                    to_push = []
                    for r_state in reachable_states:
                        if r_state not in state_abstr_dict.keys():
                            state_abstr_dict[r_state] = abstr_state_counter
                            to_push.append(r_state)
                    state_queue.append(to_push)
                    abstr_state_counter += 1

        for state in states:
            if state.is_terminal():
                state_abstr_dict[state] = abstr_state_counter

        self.s_a = StateAbstraction(state_abstr_dict)
        print(self.s_a)

        # Add abstract states to transition records
        for abstr_state in self.s_a.get_all_abstr_states():
            ground_states = self.s_a.get_ground_from_abstr(abstr_state)
            self.abstr_state_action_pair_counts[abstr_state] = {}
            self.abstr_update_record[abstr_state] = {}
            for action in self.mdp.actions:
                self.abstr_state_action_pair_counts[abstr_state][action] = 0
                self.abstr_update_record[abstr_state][action] = []
                for ground in ground_states:
                    self.abstr_state_action_pair_counts[abstr_state][action] += self.state_action_pair_counts[ground][
                        action]
                    self.abstr_update_record[abstr_state][action].extend(self.ground_update_record[ground][action])
        # Add group dict (for detachment)
        self.group_dict = self.reverse_abstr_dict(self.s_a.abstr_dict)

# Testing use only
if __name__ == '__main__':

    # Create environment
    mdp = TwoRoomsMDP(lower_width=3, upper_width=3,
                      lower_height=3, upper_height=3,
                      hallway_states=[3], goal_location=[(1,5)])
    error_dict = {GridWorldState(1,2): GridWorldState(2,5),
                  GridWorldState(3,3): GridWorldState(1,6)}

    ABSTR_TYPE = Abstr_type.Q_STAR
    ERROR_NUM = 6

    mdp = GridWorldMDP()
    if ABSTR_TYPE == Abstr_type.Q_STAR:
        abstr_mdp = mdp.make_abstr_mdp(Abstr_type.Q_STAR)
        if ERROR_NUM == 1:
            error_dict = {GridWorldState(6, 3): GridWorldState(10, 9),
                          GridWorldState(9, 10): GridWorldState(9, 3)}
        elif ERROR_NUM == 2:
            error_dict = {GridWorldState(9, 8): GridWorldState(2, 1),
                          GridWorldState(9, 11): GridWorldState(2, 4)}
        # Lower right room all grouped together
        elif ERROR_NUM == 3:
            error_dict = {GridWorldState(7,1): GridWorldState(11,1),
                          GridWorldState(7,2): GridWorldState(11,1),
                          GridWorldState(7,3): GridWorldState(11,1),
                          GridWorldState(7,4): GridWorldState(11,1),
                          GridWorldState(8, 1): GridWorldState(11, 1),
                          GridWorldState(8, 2): GridWorldState(11, 1),
                          GridWorldState(8, 3): GridWorldState(11, 1),
                          GridWorldState(8, 4): GridWorldState(11, 1),
                          GridWorldState(9, 1): GridWorldState(11, 1),
                          GridWorldState(9, 2): GridWorldState(11, 1),
                          GridWorldState(9, 3): GridWorldState(11, 1),
                          GridWorldState(9, 4): GridWorldState(11, 1),
                          GridWorldState(10, 1): GridWorldState(11, 1),
                          GridWorldState(10, 2): GridWorldState(11, 1),
                          GridWorldState(10, 3): GridWorldState(11, 1),
                          GridWorldState(10, 4): GridWorldState(11, 1),
                          GridWorldState(11, 2): GridWorldState(11, 1),
                          GridWorldState(11, 3): GridWorldState(11, 1),
                          GridWorldState(11, 4): GridWorldState(11, 1)}
        # Goal room all grouped together
        elif ERROR_NUM == 4:
            error_dict = {GridWorldState(7,6): GridWorldState(11,11),
                          GridWorldState(7,7): GridWorldState(11,11),
                          GridWorldState(7,8): GridWorldState(11,11),
                          GridWorldState(7,9): GridWorldState(11,11),
                          GridWorldState(7, 10): GridWorldState(11, 11),
                          GridWorldState(7, 11): GridWorldState(11, 11),
                          GridWorldState(8, 6): GridWorldState(11, 11),
                          GridWorldState(8, 7): GridWorldState(11, 11),
                          GridWorldState(8, 8): GridWorldState(11, 11),
                          GridWorldState(8, 9): GridWorldState(11, 11),
                          GridWorldState(8, 10): GridWorldState(11, 11),
                          GridWorldState(8, 11): GridWorldState(11, 11),
                          GridWorldState(9, 6): GridWorldState(11, 11),
                          GridWorldState(9, 7): GridWorldState(11, 11),
                          GridWorldState(9, 8): GridWorldState(11, 11),
                          GridWorldState(9, 9): GridWorldState(11, 11),
                          GridWorldState(9, 10): GridWorldState(11, 11),
                          GridWorldState(9, 11): GridWorldState(11, 11),
                          GridWorldState(10, 6): GridWorldState(11, 11),
                          GridWorldState(10, 7): GridWorldState(11, 11),
                          GridWorldState(10, 8): GridWorldState(11, 11),
                          GridWorldState(10, 9): GridWorldState(11, 11),
                          GridWorldState(10, 10): GridWorldState(11, 11),
                          GridWorldState(10, 11): GridWorldState(11, 11),
                          GridWorldState(11, 6): GridWorldState(11, 11),
                          GridWorldState(11, 7): GridWorldState(11, 11),
                          GridWorldState(11, 8): GridWorldState(11, 11),
                          GridWorldState(11, 9): GridWorldState(11, 11),
                          GridWorldState(11, 10): GridWorldState(11, 11)}
        elif ERROR_NUM == 5:
            # Upper-left AND lower-right grouped together
            error_dict = {GridWorldState(7,1): GridWorldState(11,1),
                          GridWorldState(7,2): GridWorldState(11,1),
                          GridWorldState(7,3): GridWorldState(11,1),
                          GridWorldState(7,4): GridWorldState(11,1),
                          GridWorldState(8, 1): GridWorldState(11, 1),
                          GridWorldState(8, 2): GridWorldState(11, 1),
                          GridWorldState(8, 3): GridWorldState(11, 1),
                          GridWorldState(8, 4): GridWorldState(11, 1),
                          GridWorldState(9, 1): GridWorldState(11, 1),
                          GridWorldState(9, 2): GridWorldState(11, 1),
                          GridWorldState(9, 3): GridWorldState(11, 1),
                          GridWorldState(9, 4): GridWorldState(11, 1),
                          GridWorldState(10, 1): GridWorldState(11, 1),
                          GridWorldState(10, 2): GridWorldState(11, 1),
                          GridWorldState(10, 3): GridWorldState(11, 1),
                          GridWorldState(10, 4): GridWorldState(11, 1),
                          GridWorldState(11, 2): GridWorldState(11, 1),
                          GridWorldState(11, 3): GridWorldState(11, 1),
                          GridWorldState(11, 4): GridWorldState(11, 1),
                          # Upper-left starts here
                          GridWorldState(1,7): GridWorldState(1,11),
                          GridWorldState(1,8): GridWorldState(1,11),
                          GridWorldState(1,9): GridWorldState(1,11),
                          GridWorldState(1,10): GridWorldState(1,11),
                          GridWorldState(2, 7): GridWorldState(1, 11),
                          GridWorldState(2, 8): GridWorldState(1, 11),
                          GridWorldState(2, 9): GridWorldState(1, 11),
                          GridWorldState(2, 10): GridWorldState(1, 11),
                          GridWorldState(2,11): GridWorldState(1,11),
                          GridWorldState(3, 7): GridWorldState(1, 11),
                          GridWorldState(3, 8): GridWorldState(1, 11),
                          GridWorldState(3, 9): GridWorldState(1, 11),
                          GridWorldState(3, 10): GridWorldState(1, 11),
                          GridWorldState(3, 11): GridWorldState(1, 11),
                          GridWorldState(4, 7): GridWorldState(1, 11),
                          GridWorldState(4, 8): GridWorldState(1, 11),
                          GridWorldState(4, 9): GridWorldState(1, 11),
                          GridWorldState(4, 10): GridWorldState(1, 11),
                          GridWorldState(4, 11): GridWorldState(1, 11),
                          GridWorldState(5, 7): GridWorldState(1, 11),
                          GridWorldState(5, 8): GridWorldState(1, 11),
                          GridWorldState(5, 9): GridWorldState(1, 11),
                          GridWorldState(5, 10): GridWorldState(1, 11),
                          GridWorldState(5, 11): GridWorldState(1, 11)
                          }
        elif ERROR_NUM == 6:
            # Starting room all grouped together
            error_dict = {GridWorldState(1,2): GridWorldState(1,1),
                          GridWorldState(1,3): GridWorldState(1,1),
                          GridWorldState(1,4): GridWorldState(1,1),
                          GridWorldState(1,5): GridWorldState(1,1),
                          GridWorldState(2, 1): GridWorldState(1,1),
                          GridWorldState(2, 2): GridWorldState(1, 1),
                          GridWorldState(2, 3): GridWorldState(1, 1),
                          GridWorldState(2, 4): GridWorldState(1, 1),
                          GridWorldState(2, 5): GridWorldState(1, 1),
                          GridWorldState(3, 1): GridWorldState(1, 1),
                          GridWorldState(3, 2): GridWorldState(1, 1),
                          GridWorldState(3, 3): GridWorldState(1, 1),
                          GridWorldState(3, 4): GridWorldState(1, 1),
                          GridWorldState(3, 5): GridWorldState(1, 1),
                          GridWorldState(4, 1): GridWorldState(1, 1),
                          GridWorldState(4, 2): GridWorldState(1, 1),
                          GridWorldState(4, 3): GridWorldState(1, 1),
                          GridWorldState(4, 4): GridWorldState(1, 1),
                          GridWorldState(4, 5): GridWorldState(1, 1),
                          GridWorldState(5, 1): GridWorldState(1, 1),
                          GridWorldState(5, 2): GridWorldState(1, 1),
                          GridWorldState(5, 3): GridWorldState(1, 1),
                          GridWorldState(5, 4): GridWorldState(1, 1),
                          GridWorldState(5, 5): GridWorldState(1, 1)}

    elif ABSTR_TYPE == Abstr_type.A_STAR:
        abstr_mdp = mdp.make_abstr_mdp(Abstr_type.A_STAR)
        if ERROR_NUM == 1:
            error_dict = {GridWorldState(4, 9): GridWorldState(9, 10),
                          GridWorldState(2, 11): GridWorldState(7, 10)}
        elif ERROR_NUM == 2:
            error_dict = {GridWorldState(2, 11): GridWorldState(7, 10)}
        # Reconstructed from thesis work
        elif ERROR_NUM == 3:
            error_dict = {GridWorldState(3,1): GridWorldState(5,5),
                          GridWorldState(8,2): GridWorldState(9,10),
                          GridWorldState(4,1): GridWorldState(8,7),
                          GridWorldState(4,3): GridWorldState(5,3),
                          GridWorldState(4,9): GridWorldState(9,9),
                          GridWorldState(1,3): GridWorldState(7,11),
                          GridWorldState(7,11): GridWorldState(9,10),
                          GridWorldState(3,7): GridWorldState(9,11),
                          GridWorldState(8,10): GridWorldState(7,1)}
    elif ABSTR_TYPE == Abstr_type.PI_STAR:
        abstr_mdp = mdp.make_abstr_mdp(Abstr_type.PI_STAR)
        if ERROR_NUM == 1:
            error_dict = {GridWorldState(9, 11): GridWorldState(2, 3),
                          GridWorldState(7, 8): GridWorldState(1, 8),
                          GridWorldState(6, 3): GridWorldState(5, 2)}
        # Reconstructed from random
        if ERROR_NUM == 2:
            error_dict = {GridWorldState(11,6): GridWorldState(5,11),
                          GridWorldState(10,11): GridWorldState(1,4),
                          GridWorldState(7,4): GridWorldState(9,8),
                          GridWorldState(3,10): GridWorldState(7,7),
                          GridWorldState(11,4): GridWorldState(9,8)}

    if error_dict:
        c_s_a = make_corruption(abstr_mdp, reassignment_dict=error_dict)
        abstr_mdp = AbstractMDP(mdp, c_s_a)

    # Make agent
    agent = TrackingAgent(mdp, s_a=abstr_mdp.state_abstr)

    # Explore
    i = 0
    while agent._episode_counter < 100:
        _, _, next_state, _ = agent.explore()
        if next_state.is_terminal():
            print('Finished episode', i, 'step count is', agent.episode_step_count)
            agent.episode_step_count = 0
            i += 1

    '''
    print('\nState, action, stdev of updates')
    for key, value in agent.abstr_update_record.items():
        for key2, val2 in value.items():
            if len(val2) > LIMIT:
                print(key, key2, np.std(val2), len(val2))
    '''

    for key, value in agent.state_occupancy_record.items():
        print(key, value)

    for key, value in agent.abstr_state_occupancy_record.items():
        print(key, value)

    print('\nNormalized volatility')
    for abstr_state, action_to_volatility in agent.calculate_normalized_volatility().items():
        for action, volatility in action_to_volatility.items():
            tag_string = str(abstr_state) + ' ' + str(action) + ': '
            #pair_count = agent.abstr_state_action_pair_counts[abstr_state][action]
            #print(tag_string.ljust(15), str(round(volatility, 4)).ljust(6), str(pair_count).ljust(6), round(np.sqrt(pair_count) * volatility, 4))
            print(tag_string.ljust(15), str(round(volatility, 4)))

    '''
    print('\nError and corrupted states')
    for key, value in error_dict.items():
        print('Error state', key, 'mapped to abstr state', abstr_mdp.get_abstr_from_ground(value))
        print('Corr state is', end = ' ')
        error_group = abstr_mdp.get_ground_from_abstr(abstr_mdp.get_abstr_from_ground(value))
        for error in error_group:
            print(error, end = ' ')
        print()
    '''

    vol_record = agent.rank_volatility_record()
    for abstr_state, vol in vol_record.items():
        print(abstr_state, vol)

    vol_state, vol_val = agent.get_volatile_state_by_rank()
    print('most volatile state = ', vol_state, 'value =', vol_val)

    vol_state, vol_val = agent.get_volatile_state_by_rank(rank=1)
    print('2nd most volatile state = ', vol_state, 'value =', vol_val)

    vol_state, vol_val = agent.get_volatile_state_by_rank(rank=2)
    print('3rd most volatile state = ', vol_state, 'value =', vol_val)
    quit()

    print('\nConfidence intervals on Q-updates for corrupted abstract states')
    for err_state, corr_state in error_dict.items():
        corr_abstr_state = agent.get_abstr_from_ground(corr_state)
        print('Abstract state is', corr_abstr_state)
        bound_dict = agent.calculate_intervals_for_abstr_state(corr_abstr_state)
        for ground_state, action_to_bounds in bound_dict.items():
            for action, bounds in action_to_bounds.items():
                lb, ub, count = bounds
                tag_string = str(ground_state) + ' ' + str(action) + ': '
                print(tag_string.ljust(20), end = '')
                if lb or ub:
                    bound_string = '[' + str(round(lb, 4)) + ', ' + str(round(ub, 4)) + '] '
                    print(bound_string.ljust(20), str(count).ljust(6), sep='')
                else:
                    bound_string = '[' + str(lb) + ', ' + str(ub) + '] '
                    print(bound_string.ljust(20), count, sep='')
        print()

