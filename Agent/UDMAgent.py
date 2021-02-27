"""
This extends AbstractionAgent by replacing the original consistency check process with a modification of Andrew
McCallum's UDM algorithm.

The agent records the abstract state observed, the action taken, and the discounted reward received for each time-step.
Then it periodically checks, for each state, whether the returns from all incoming trajectories that share an outgoing
action are 'close enough' to each other. If not, it splits the states according to which incoming trajectories overlap

#TODO: If two incoming_states are grouped together with one outgoing action, but with another outgoing action one state
#TODO  doesn't have enough transitions to be counted, then these states are split
"""

from Agent.AbstractionAgent import AbstractionAgent
from GridWorld.TwoRoomsMDP import TwoRoomsMDP
from GridWorld.GridWorldStateClass import GridWorldState
from GridWorld.GridWorldMDPClass import GridWorldMDP
from resources.AbstractionTypes import Abstr_type
from resources.AbstractionCorrupters import *
from MDP.AbstractMDPClass import AbstractMDP
from GridWorld.GridWorldMDPClass import GridWorldMDP
import itertools

SPLORE_RANGE = 10000
EPISODE_COUNT = 100
SEED = 1234
TRANSITION_THRESHOLD = 5
NUM_TESTS = 1
import numpy as np
from scipy import stats

def split_record_to_additional_states(split_record):
    """
    Given a split record {abstr_state -> {outgoing_action -> list of groups}}, create a dictionary mapping abstract
    states to a list of incoming states where memory is required (if any)

    E.g. if we 'split' state 1 based on whether we come from (2,1) or (1,2) (but not (3,2) or (2,3)),
    dictionary will contain {1 -> [(2,1), (1,2)]}
    """
    memory_record = {}
    for abstr_state, action_to_group in split_record.items():
        memory_record[abstr_state] = []
        temp_record = []
        print('Abstr state is', abstr_state)
        for action, groups in action_to_group.items():
            # If the length of the grouping is greater than 1, pull out states that aren't in every
            #  group. The remaining states are the ones we need memory for.
            if len(groups) > 1:
                print('action, groups are', action, end=' ')
                for group in groups:
                    print('(', end = '')
                    for state in group:
                        print(state, end = ' ')
                    print(')', end = '  ')
                all_states = []
                for group in groups:
                    for state in group:
                        all_states.append(state)
                all_states = list(set(all_states))
                for state in all_states:
                    common = True
                    for group in groups:
                        if state not in group:
                            common = False
                    if common:
                        for group in groups:
                            group.remove(state)
                for group in groups:
                    temp_record.append(list(set(group)))
                print()

        if len(temp_record) > 0:
            print('Temp record is', end =' ')
            for lst in temp_record:
                for state in lst:
                    print(state, end = ' ')
                print('|', end = ' ')
            print()
        '''
        if len(temp_record) > 0:
            print('Unique temp record is', end = ' ')
            temp_unique = list(set(temp_record))
            for lst in temp_unique:
                for state in lst:
                    print(state, end = ' ')
                print('|', end = ' ')
            print()
        '''
        temp_record = refine_memory_record_strict(list(k for k, _ in itertools.groupby(temp_record)))
        temp_record.sort()
        # printing for debugging
        '''
        for tuple in temp_record:
            print('[', end='')
            for state in tuple:
                print(state, end=' ')
            print(']', end = '  ')
        print()
        '''

        memory_record[abstr_state] = temp_record

        #TODO: Make sure temp_record is correct up to this point. I think it is but who the hell knows.
        #TODO: Okay it's probably right except for duplicates
        # Go through and perform a refinement. Duplicates are removed but I need to refine everything
        '''
        if len(temp_record) > 0:
            print('Abstr State is', abstr_state)
        for i in range(len(temp_record)):
            lst = temp_record[i]
            if len(lst) > 0:
                print(i)
                for item in lst:
                    print(item, end = ' ')
                print()
            if len(lst) > 0:
                print()

        # At this point, groups are disjoint. Now we need to compare groups across actions and
        #  see if we need to further refine.

        # For each state, create a dictionary mapping it to a list containing every other state
        #  Go through the groups, and if the key state appears, remove all value states that aren't
        #  included in the group. At the end, the values will only be those states always mapped
        #  together
        all_states = [state for group in temp_record for state in group]
        states_grouped = []
        aggregated_states = []
        for state in all_states:
            if state not in aggregated_states:
                temp_list = [state for group in temp_record for state in group]
                #print('temp list is', end = ' ')
                #for t in temp_list:
                #    print(t, end = ' ')
                #print()
                to_remove = []
                for group in temp_record:
                    if state in group:
                        for other_state in temp_list:
                            if other_state not in group:
                                to_remove.append(other_state)
                to_remove = list(set(to_remove))
                for removed_state in to_remove:
                    #print('Removing state', removed_state)
                    try:
                        temp_list.remove(removed_state)
                    except:
                        print('Tried to remove', removed_state, 'but it isn\'t there')
                        quit()
                #print('Final group is', end = ' ')
                #for s in temp_list:
                #    print(s, end = ' ')
                aggregated_states.append(temp_list)
                for s in temp_list:
                    states_grouped.append(s)
        print('Abstr state is', abstr_state)
        for final_group in aggregated_states:
            #print("final group is", final_group)
            final_group = list(set(final_group))
            print('final group is: ', end = ' ')
            for state in final_group:
                print(state, end = ' ')
            print(type(final_group))
            if tuple(final_group) not in memory_record[abstr_state]:
                print('adding group')
                memory_record[abstr_state].append(tuple(final_group))
            else:
                print('SKIP')
        '''
    return memory_record

def refine_memory_record_strict(tuple_list):
    """
    Given a grouping (list of lists, where each sub-list is a group of states), refine it. Assumes no duplicates.

    STRICT assumption: e.g. if we have a (2,4) for action a1 and (2,4), (3,5) for action a2, this will split
    (2,4) and (3,5) separately
    """
    # TODO: Remove groups that are strict subsets of other groups
    for tup in tuple_list:
        is_subset = False

    # Split overlapping groups
    for i in range(len(tuple_list)):
        tup = tuple_list[i]
        if len(tup) > 1:
            # Go through each state in a tuple and see if it appears elsewhere. If so, record which other tuple(s) it
            #  goes in. Then group all states that appear in the same other tuples together
            dup_record = {}
            for state in tup:
                for j in range(len(tuple_list)):
                    if i != j:
                        other_tup = tuple_list[j]
                        if state in other_tup:
                            if state in dup_record.keys():
                                dup_record[state].append(j)
                            else:
                                dup_record[state] = [j]
            dup_states = list(dup_record.keys())
            # Remove states with duplicates from current group
            for dup_state in dup_states:
                tup.remove(dup_state)
            # Group duplicate states according to which other tuples they belong to.
            #TODO: don't split if group is superset of another group
            new_groups = []
            grouped_states = []
            for dup_state in dup_states:
                if dup_state not in grouped_states:
                    new_group = [dup_state]
                    grouped_states.append(new_group)
                    for other_dup_state in dup_states:
                        if dup_state != other_dup_state and dup_record[dup_state] == dup_record[other_dup_state]:
                            new_group.append(other_dup_state)
                    new_groups.append(new_group)
            '''
            print('New groups')
            for group in new_groups:
                print('Refined group, ', end = ' ')
                for state in group:
                    print(state, end = ' ')
                print()
            '''

    # Remove any empty tuples
    final_tuple_list = []
    for tup in tuple_list:
        temp_tup = []
        for i in range(len(tup)):
            if tup[i]:
                temp_tup.append(tup[i])
        if len(temp_tup) > 0:
            final_tuple_list.append(tuple(temp_tup))
        #if len(tup) != 0 and tup[0] is not None:
        #    final_tuple_list.append(tup)
    #final_tuple_list = final_tuple_list

    #return tuple_list
    return list(set(final_tuple_list))

def print_memory_record(memory_record):
    """
    Print dictionary {abstr_state -> list of tuples of states grouped together}
    """
    print(memory_record)
    for abstr_state, tup_list in memory_record.items():
        if len(tup_list) > 0:
            print('Abstr state is', abstr_state)
        for i in range(len(tup_list)):
            tup = tup_list[i]
            print(i, end = ': ')
            for state in tup:
                print(state, end = ' ')
            print()
        """
        for i in range(len(tup_list)):
            print(i, end = ': ')
            for tup in tup_list:
                print('(', end='')
                for state in tup:
                    print(state, end = ', ')
                print(')', end='  ')
        """
        #print()

def match_interval_to_group(interval, group):
    """
    Test whether the interval matches the group, i.e. whether the interval overlaps with every interval in group.
    Group is a list of tuples.
    Returns true or false
    """
    for other_interval in group:
        if (interval[1] - interval[0] > 0):
            if interval[0] >= other_interval[1] or interval[1] <= other_interval[0]:
                return False
        else:
            if interval[0] > other_interval[1] or interval[1] < other_interval[0]:
                return False
    return True

def count_number_of_splits(split_record, verbose=False):
    """
    Given a split record ({abstr_state -> {outgoing_action -> list of lists of states grouped together}}), count the
    number of splits (which is length of list of lists per outgoing_action, abstr_state, and overall)
    """
    overall_split_count = 0
    unique_state_splits = 0
    for abstr_state, group_dict in split_record.items():
        if verbose:
            print('Abstract state', abstr_state)
        abstr_state_split_count = 0
        for outgoing_action, list_of_lists in group_dict.items():
            if len(list_of_lists) > 0:
                if verbose:
                    print(outgoing_action, end=' ')
                action_split_count = len(list_of_lists) - 1
                if verbose:
                    print('split count', action_split_count)
                abstr_state_split_count += action_split_count
        if verbose:
            print(abstr_state, 'split count', abstr_state_split_count)
        overall_split_count += abstr_state_split_count
        if abstr_state_split_count >= 1:
            unique_state_splits += 1
        if verbose:
            print()
    if verbose:
        end = '\n##########\n##########\n##########\n'
    else:
        end='\n\n'
    print('Number of new states', overall_split_count,
          'Abstract states split', unique_state_splits, end=end)

class UDMAgent(AbstractionAgent):
    def __init__(self,
                 mdp,
                 s_a=None,
                 alpha=0.1,
                 epsilon=0.1,
                 decay_exploration=True,
                 confidence_interval_alpha=0.05,
                 transition_threshold=5,
                 seed=None,
                 episode_buffer=0):
        """
        Create a UDM agent on the given MDP with the given State Abstraction (s_a)
        """
        # Consistency_check and detach_reassignment shouldn't matter
        super().__init__(mdp,
                         s_a=s_a,
                         epsilon=epsilon,
                         alpha=alpha,
                         decay_exploration=decay_exploration,
                         consistency_check='abstr',
                         detach_reassignment='group',
                         seed=seed)

        # Alpha parameter used for Student's T statistic for confidence interval
        self.confidence_interval_alpha = confidence_interval_alpha

        # Transition threshold governs how many visits a state-action pair must have to be considered for split
        self.transition_threshold = transition_threshold

        # These hold the experience
        self.action_record = []
        self.reward_record = []
        self.ground_state_record = []
        self.abstr_state_record = []
        self.end_of_episode_record = []
        self.episode_count = 0
        self.step_count_record = []
        self.episode_step_count = 0
        self.episode_buffer = episode_buffer

    # --------------------
    # UDM-specific methods
    # --------------------
    def get_state_transition_returns(self, state):
        """
        Get returns for given state, indexed by incoming transition and outgoing action.
        Returns results in a dictionary: {outgoing_action -> {incoming_ground_state -> list of returns}}
        Assumes input is abstract state.
        """
        # This holds final result
        transition_records = {}

        # Calculate returns
        returns = self.get_returns()

        # Iterate through state record and grab all transitions to given state
        for i in range(1, len(self.abstr_state_record)):
            if self.abstr_state_record[i] == state:
                # If last action ended episode, there is no incoming state
                if len(self.end_of_episode_record) > 0 and self.end_of_episode_record[i-1] == 1:
                    incoming_state = None
                else:
                    incoming_state = self.ground_state_record[i-1]
                outgoing_action = self.action_record[i]
                # If this is the first instance of this outgoing action, add to keys
                if outgoing_action not in transition_records.keys():
                    transition_records[outgoing_action] = {}
                # If this is the first time we're seeing this incoming state (for this outgoing action)
                #  add to dictionary
                if incoming_state not in transition_records[outgoing_action].keys():
                    transition_records[outgoing_action][incoming_state] = [returns[i]]
                # Else add to existing record
                else:
                    transition_records[outgoing_action][incoming_state].append(returns[i])

        return transition_records

    def get_all_transition_returns(self):
        """
        Get all returns, indexed by abstract state, outgoing action, and incoming ground state
        Returns {abstract_state -> {outgoing_action -> {incoming_ground_state -> list of returns}}}
        """
        transition_records = {}
        for state in self.s_a.get_all_abstr_states():
            temp_record = self.get_state_transition_returns(state)
            transition_records[state] = temp_record
        return transition_records

    def calculate_confidence_interval(self, return_list):
        """
        Given a single list of returns, calculate the confidence interval according to equations 5 and 6 in
        McCallum's UDM paper.

        Returns lower_bound, upper_bound of confidence interval
        """
        if len(return_list) == 1:
            return return_list[0], return_list[0]

        # Sample mean
        mean = np.mean(return_list)

        # Sample standard deviation
        n = len(return_list)
        numer = n * np.sum(np.square(return_list)) - np.square(np.sum(return_list))
        denom = n * (n-1)
        #if (n-1 == 0) or (n == 0):
        #    print('Failure with', return_list)
        #    quit()
        stdev = np.sqrt(numer / denom)

        # Student's T function
        t = stats.t.ppf(1 - self.confidence_interval_alpha / 2, n - 1)

        # Calculate confidence interval bounds
        lower_bound = mean - t * stdev / np.sqrt(n)
        upper_bound = mean + t * stdev / np.sqrt(n)

        return lower_bound, upper_bound

    def group_overlapping_intervals(self, returns_per_state, verbose=False):
        """
        Given incoming state-indexed returns ({incoming_ground_state -> list of returns}), group
        incoming_ground_states such that states from different groups have disjoint confidence intervals. Technically
        this is NP-hard, so taking a lazy greedy approach. To be added to a group, an interval must overlap with
        EVERY other interval in a group.

        Returns a list of lists, where each sub-list is a set of states whose confidence intervals overlap. If the
        length of main list is 1, then all previous states have overlapping confidence intervals.
        """
        thresholded_returns_per_state = {}
        for incoming_state, return_list in returns_per_state.items():
            if not incoming_state:
                continue
            if verbose:
                print(len(return_list), end=' ')
                if len(return_list) > TRANSITION_THRESHOLD:
                    print('COUNTS', end = '')
            lb, ub = self.calculate_confidence_interval(return_list)
            lb = round(lb, 4)
            ub = round(ub, 4)
            if verbose:
                print(incoming_state, ' [', lb, ', ', ub, '], ', sep='', end = '  ')
            if len(return_list) > self.transition_threshold:
                thresholded_returns_per_state[incoming_state] = return_list
        if verbose:
            print()
        #print('\nGrouping Overlapping Intervals')
        #for incoming_state, return_list in thresholded_returns_per_state.items():
        #    print(incoming_state, return_list)

        result = []

        # This will hold the confidence intervals. Will be a list of lists, each sub-list being a set of tuples
        #  of the form (lower-bound, upper-bound)
        interval_container = []

        # First pass at clustering states
        for incoming_state, return_list in thresholded_returns_per_state.items():
            # Get confidence interval for given state
            lower_bound, upper_bound = self.calculate_confidence_interval(return_list)
            interval = (lower_bound, upper_bound)
            # See if confidence interval matches any existing group
            match = False
            for i in range(len(result)):
                group = interval_container[i]
                if match_interval_to_group(interval, group):
                    interval_container[i].append(interval)
                    result[i].append(incoming_state)
                    match = True
            if not match:
                interval_container.append([interval])
                result.append([incoming_state])

        # Go back over the list and match any missing intervals
        for incoming_state, return_list in returns_per_state.items():
            if len(return_list) > self.transition_threshold:
                interval = self.calculate_confidence_interval(return_list)
                for i in range(len(result)):
                    group = interval_container[i]
                    if match_interval_to_group(interval, group) and incoming_state not in result[i]:
                        interval_container[i].append(interval)
                        result[i].append(incoming_state)

        return result

    def test_state_for_split(self, state_transition_record, verbose=False):
        """
        Given a transition record ({outgoing_action -> {previous_ground_state -> list of returns}}) for a particular
        state, group previous_ground_states based on whether or not their confidence intervals overlap

        Return {outgoing_action -> list of lists of previous_ground_states}, where sublist contains states that overlap
        """
        result = {}

        for outgoing_action, returns_per_state in state_transition_record.items():
            if verbose:
                print('Outgoing action is', outgoing_action, end = ': ')
            result[outgoing_action] = self.group_overlapping_intervals(returns_per_state, verbose=verbose)
            #if verbose:
                #print()

        return result

    def test_all_states_for_split(self, transition_record, verbose=False):
        """
        Given a transition record ({abstract_state -> {outgoing_action -> {previous_ground_state -> list of returns}}}),
        find the groups of previous_ground_states that overlap in their confidence intervals on returns

        Return {abstract_state -> {outgoing_action -> list of lists previous_ground_states}}
        """
        temp_result = {}

        for a_s, s_transition_record in transition_record.items():
            temp_result[a_s] = self.test_state_for_split(s_transition_record, verbose=verbose)

        return temp_result

    '''
    def calculate_transition_confidence_intervals(self, transition_record):
        """
        Given a transition record ({abstract_state -> {previous_ground_state -> {outgoing_action -> list of returns}}}),
        calculate the overlapping confidence intervals.
        Returns ({abstract_state -> {outgoing_action ->
                                     list of previous_ground_states w/ overlapping confidence intervals}})
        """
        overlap_record = {} 
        for key in transition_record.keys(): 
            overlap_record[key] = {} 
    '''


    # --------------------------
    # Overwritten Parent Methods
    # --------------------------
    def explore(self):
        """
        Epsilon-greedy exploration. Record current state, action taken, and reward observed.
        """
        current_state, action, next_state, reward = super().explore()
        abstr_state = self.get_abstr_from_ground(current_state)
        # If the the episode count is at least the episode buffer, then record the experience
        if self.episode_count >= self.episode_buffer:
            self.action_record.append(action)
            self.ground_state_record.append(current_state)
            self.abstr_state_record.append(abstr_state)
            self.reward_record.append(reward)
        self.end_of_episode_record.append(next_state.is_terminal())
        self.episode_step_count += 1
        if next_state.is_terminal():
            self.episode_count += 1
            self.step_count_record.append(self.episode_step_count)
            self.episode_step_count = 0
        return current_state, action, next_state, reward

    # --------------------------
    # Getters & Setters, Utility
    # --------------------------
    def get_abstr_from_ground(self, ground_state):
        return self.s_a.get_abstr_from_ground(ground_state)

    def get_returns(self):
        """
        v2: Calculate returns from rewards: return[t] = reward[t] + gamma * return[t+1] within a given episode
        v1: Calculate returns from rewards: return[t] = reward[t] + gamma * return[t+1], return[N] = reward[N]
        """
        # v2
        returns = [0] * len(self.reward_record)
        returns[len(self.reward_record) - 1] = self.reward_record[-1]
        for i in range(len(self.reward_record) - 2, -1, -1):
            if self.end_of_episode_record[i]:
                returns[i] = self.reward_record[i]
            else:
                returns[i] = self.reward_record[i] + self.mdp.gamma * returns[i+1]
        return returns

        # v1
        '''
        returns = [0] * len(self.reward_record)
        returns[len(self.reward_record) - 1] = self.reward_record[-1]
        for i in range(len(self.reward_record) - 2, -1, -1):
            returns[i] = self.reward_record[i] + self.mdp.gamma * returns[i+1]
        return returns
        '''

    def print_state_transition_record(self, state_transition_record):
        """
        Given transition records for an individual state, print the record
        Record is {outgoing_action -> {incoming_ground_state -> returns}}
        """
        for key, value in state_transition_record.items():
            print(key)
            for key2, val2 in value.items():
                print(key2, ': ', val2)
            print()

    def print_all_transition_records(self, transition_records):
        """
        Print given transition record
        """
        for key, value in transition_records.items():
            if len(value) > 0:
                print(key)
                self.print_state_transition_record(value)

    def print_action_to_state_list(self, action_to_states):
        for action, list_of_lists in action_to_states.items():
            print(action, end=':\n')
            for sub_list in list_of_lists:
                for state in sub_list:
                    print(state, end=' ')
                print()
            print()
        print()

def test_udm(mdp,
             abstr_type,
             num_episodes,
             error_dict=None,
             episode_buffer=0):
    """
    Test how UDM performs on the given MDP
    """
    # Make abstract MDP
    abstr_mdp = mdp.make_abstr_mdp(abstr_type)
    # Apply corruption if argument is provided
    if error_dict:
        c_s_a = make_corruption(abstr_mdp, reassignment_dict=error_dict)
        abstr_mdp = AbstractMDP(mdp, c_s_a)
    agent = UDMAgent(mdp,
                     s_a=abstr_mdp.state_abstr,
                     transition_threshold=TRANSITION_THRESHOLD,
                     episode_buffer=episode_buffer)
    # Print abstraction
    print('State abstraction is')
    print(agent.get_abstraction_as_string())

    # Explore the number of episodes
    while agent.episode_count < num_episodes:
        agent.explore()
        if agent.episode_count > 0 and agent.end_of_episode_record[-1] == 1:
            print('On episode', agent.episode_count, agent.step_count_record[agent.episode_count - 1])

    # Calculate state splits based on UDM algorithm
    transition_records = agent.get_all_transition_returns()
    split_record = agent.test_all_states_for_split(transition_records)

    abstr_states_seen = []
    for ground_state in agent.mdp.get_all_possible_states():
        abstr_state = agent.get_abstr_from_ground(ground_state)

        if abstr_state not in abstr_states_seen:
            print('Overlapping states for', abstr_state, end = ' ')
            ground_states = agent.get_ground_states_from_abstract_state(abstr_state)
            for ground in ground_states:
                print(ground, end=' ')
            print()
            agent.print_action_to_state_list(agent.test_state_for_split(transition_records[abstr_state]))
            abstr_states_seen.append(abstr_state)
    print("About to print")
    #count_number_of_splits(split_record)
    print(split_record)
    memory_record = split_record_to_additional_states(split_record)
    print_memory_record(memory_record)

    if error_dict:
        for key, value in error_dict.items():
            print('Error state', key, 'mapped to', value, 'Abstr state', abstr_mdp.get_abstr_from_ground(value))
            print('States in corr abstr state are', end = ' ')
            group = abstr_mdp.get_ground_from_abstr(abstr_mdp.get_abstr_from_ground(value))
            for state in group:
                print(state, end = ' ')

# Main method for testing purposes only
if __name__ == '__main__':

    '''
    mdp = TwoRoomsMDP()
    abstr_mdp = mdp.make_abstr_mdp(Abstr_type.Q_STAR)
    agent = UDMAgent(mdp, s_a=abstr_mdp.state_abstr)
    print('State Abstraction is')
    print(agent.get_abstraction_as_string())
    print()
    '''

    # Testing confidence interval function
    '''
    returns = [0.5, 1, 0.75, 2, 0.6, 0.7]
    lower_bound, upper_bound = agent.calculate_confidence_interval(returns)
    print(lower_bound, upper_bound)
    '''

    # Testing group_overlapping_intervals ({incoming_state -> returns})
    '''
    test_dict = {}
    test_dict['state1'] = [1, 1, 1.5, 0.7]
    test_dict['state2'] = [1.5, 1.4, 1.6]
    test_dict['state3'] = [2, 2, 1.9, 2.1]
    test_dict['state4'] = [0, 2, 1]
    test_dict['state5'] = [5, 5, 5]
    test_dict['state6'] = [-0.1, 0, -0.2, -1]
    test_dict['state7'] = [3, 3, 3]
    test_dict['state8'] = [0.54, 0.54, 0.53]
    for state in test_dict.keys():
        print(state, agent.calculate_confidence_interval(test_dict[state]))
    print()

    overlapping_groups = agent.group_overlapping_intervals(test_dict)
    print(overlapping_groups)

    quit()
    '''

    # Test explore and record
    """
    print('Exploring')
    for i in range(SPLORE_RANGE):
        current_state, action, next_state, reward = agent.explore()

    returns = agent.get_returns()
    print('State, action, return')
    for i in range(SPLORE_RANGE):
        print(agent.ground_state_record[i], agent.abstr_state_record[i], agent.action_record[i], returns[i], agent.end_of_episode_record[i])
    print()
    """

    # Get returns for GridWorldState(2,2)
    '''
    abstr_state = agent.get_abstr_from_ground(GridWorldState(2,2))
    print('Abstract state for (2,2) is', abstr_state)
    print('Transition record for', abstr_state)
    transition_record = agent.get_state_transition_returns(abstr_state)
    for key, value in transition_record.items():
        if len(value) != 0:
            print(key, end=': ')
            for key2, value2 in value.items():
                print(key2, ':', value2, end='  ')
    '''

    # Get returns for all abstract states, indexed by incoming ground states and outgoing action
    """
    transition_records = agent.get_all_transition_returns()
    agent.print_all_transition_records(transition_records)
    """

    # Test that the interval calculations are working
    """
    for ground_state in agent.mdp.get_all_possible_states():
        abstr_state = agent.get_abstr_from_ground(ground_state)
        print('Abstr state for', ground_state, 'is', abstr_state)
        print('Overlapping intervals for', abstr_state)
        agent.print_action_to_state_list(agent.test_state_for_split(transition_records[abstr_state]))
    """

    # Check for how many states would be split by this process
    """
    split_record = agent.test_all_states_for_split(transition_records)
    count_number_of_splits(split_record)
    """

    mdp = TwoRoomsMDP(lower_width=3,
                      lower_height=3,
                      upper_width=3,
                      upper_height=3,
                      hallway_states=[3],
                      goal_location=[(1, 5)])
    # True Q-Star abstraction
    '''
    test_udm(mdp, Abstr_type.Q_STAR, EPISODE_COUNT, seed=SEED)
    '''

    # True A-Star abstraction
    '''
    test_udm(mdp, Abstr_type.A_STAR, EPISODE_COUNT, seed=SEED)
    quit()
    '''

    # True Pi* Abstraction
    '''
    test_udm(mdp, Abstr_type.PI_STAR, EPISODE_COUNT)
    quit()
    '''

    # True Q-star with episode buffer
    '''
    test_udm(mdp, Abstr_type.Q_STAR, EPISODE_COUNT, episode_buffer=1)
    quit()
    '''

    # True A-star with episode buffer
    '''
    test_udm(mdp, Abstr_type.A_STAR, EPISODE_COUNT, episode_buffer=10)
    quit()          
    '''

    # True Pi-Star with episode buffer
    '''
    test_udm(mdp, Abstr_type.PI_STAR, EPISODE_COUNT, episode_buffer=20)
    quit()
    '''

    # Bad error 1
    error_dict = {GridWorldState(1, 2): GridWorldState(2, 5)}

    # Q-Star with bad error 1
    '''
    test_udm(mdp,
             Abstr_type.Q_STAR,
             EPISODE_COUNT,
             error_dict=error_dict,
             episode_buffer=10)
    quit()
    '''

    # A-star with bad error 1
    '''
    test_udm(mdp, Abstr_type.A_STAR, EPISODE_COUNT, error_dict=error_dict)
    quit() 
    '''

    # Pi-Star with bad error 1
    '''
    test_udm(mdp, Abstr_type.PI_STAR, EPISODE_COUNT, error_dict=error_dict)
    quit() 
    '''

    # Mild error 1
    mild_error_1 = {GridWorldState(3,4): GridWorldState(3,5)}

    # Q-Star with mild error 1
    '''
    test_udm(mdp, Abstr_type.Q_STAR, EPISODE_COUNT, error_dict=mild_error_1)
    quit()
    '''

    # A-star with mild error 1
    '''
    test_udm(mdp, Abstr_type.A_STAR, EPISODE_COUNT, error_dict=mild_error_1)
    '''

    # Pi-star with mild error 1
    '''
    test_udm(mdp, Abstr_type.PI_STAR, EPISODE_COUNT, error_dict=mild_error_1)
    '''

    # Large MDP
    mdp = GridWorldMDP(goal_location=[(7,11)])

    # Large bad error
    large_bad_error = {GridWorldState(2,1): GridWorldState(8,11),
                       GridWorldState(1,2): GridWorldState(7,10)}

    # Test Q-Star with large MDP, bad error
    test_udm(mdp, Abstr_type.Q_STAR, EPISODE_COUNT, error_dict=large_bad_error)

    """
    # Run the split test 5 times on a true Q-star abstraction to see how much it splits
    for i in range(NUM_TESTS):
        mdp = TwoRoomsMDP(lower_width=3,
                          lower_height=3,
                          upper_width=3,
                          upper_height=3,
                          hallway_states=[3],
                          goal_location=[(1,5)])
        #mdp = GridWorldMDP()
        abstr_mdp = mdp.make_abstr_mdp(Abstr_type.Q_STAR, seed=SEED)
        agent = UDMAgent(mdp, s_a=abstr_mdp.state_abstr, seed=SEED, transition_threshold=TRANSITION_THRESHOLD)
        if i == 0:
            print('State abstraction is')
            print(agent.get_abstraction_as_string())
        #for j in range(SPLORE_RANGE):
        #    agent.explore()
        while sum(agent.end_of_episode_record) < EPISODE_COUNT:
            agent.explore()
            ep_count = sum(agent.end_of_episode_record)
            if ep_count > 0 and agent.end_of_episode_record[-1] == 1:
                print('On episode', ep_count)

        transition_records = agent.get_all_transition_returns()
        split_record = agent.test_all_states_for_split(transition_records)

        abstr_states_seen = []

        for ground_state in agent.mdp.get_all_possible_states():
            abstr_state = agent.get_abstr_from_ground(ground_state)

            if abstr_state not in abstr_states_seen:
                print('Overlapping states for', abstr_state, end=' ')
                for ground in agent.get_ground_states_from_abstract_state(abstr_state):
                    print(ground, end=' ')
                print()
                agent.print_action_to_state_list(agent.test_state_for_split(transition_records[abstr_state], verbose=True))
                abstr_states_seen.append(abstr_state)


        count_number_of_splits(split_record)
        print(sum(agent.end_of_episode_record))
        print('State abstraction is')
        print(agent.get_abstraction_as_string())
        #print(split_record)
        memory_record = split_record_to_additional_states(split_record)
        #print('About to print memory record')
        print_memory_record(memory_record)
    """
    '''
    # Run the split test on a messed-up Q-star abstraction
    for i in range(NUM_TESTS):
        if i % 10000 == 0:
            print(i)
        mdp = TwoRoomsMDP(lower_width=3,
                          lower_height=3,
                          upper_width=3,
                          upper_height=3,
                          hallway_states=[3],
                          goal_location=[(1,5)])
        error_dict = {GridWorldState(1,2): GridWorldState(2,5)}
        abstr_mdp = mdp.make_abstr_mdp(Abstr_type.A_STAR, seed=SEED)
        c_s_a = make_corruption(abstr_mdp, reassignment_dict=error_dict)
        corrupt_mdp = AbstractMDP(mdp, c_s_a)
        agent = UDMAgent(mdp, s_a=corrupt_mdp.state_abstr, seed=SEED, transition_threshold=TRANSITION_THRESHOLD)
        if i == 0:
            print('State abstraction is')
            print(agent.get_abstraction_as_string())
        #for j in range(SPLORE_RANGE):
        #    agent.explore()
        while agent.episode_count < EPISODE_COUNT:
            agent.explore()
            if agent.episode_count > 0 and agent.end_of_episode_record[-1] == 1:
                print('On episode', agent.episode_count)

        transition_records = agent.get_all_transition_returns()
        split_record = agent.test_all_states_for_split(transition_records)

        abstr_states_seen = []
        for ground_state in agent.mdp.get_all_possible_states():
            abstr_state = agent.get_abstr_from_ground(ground_state)

            if abstr_state not in abstr_states_seen:
                print('Overlapping states for', abstr_state, end=' ')
                ground_states = agent.get_ground_states_from_abstract_state(abstr_state)
                for ground in ground_states:
                    print(ground, end=' ')
                print()
                agent.print_action_to_state_list(agent.test_state_for_split(transition_records[abstr_state]))
                abstr_states_seen.append(abstr_state)

        count_number_of_splits(split_record)
        print('State abstraction is')
        print(agent.get_abstraction_as_string())
        print(split_record)
        memory_record = split_record_to_additional_states(split_record)
        print_memory_record(memory_record)

    # Run the split test on a messed-up A-star abstraction
    for i in range(NUM_TESTS):
        if i % 10000 == 0:
            print(i)
        mdp = TwoRoomsMDP(lower_width=3,
                          lower_height=3,
                          upper_width=3,
                          upper_height=3,
                          hallway_states=[3],
                          goal_location=[(1,5)])
        error_dict = {GridWorldState(1,2): GridWorldState(2,5)}
        abstr_mdp = mdp.make_abstr_mdp(Abstr_type.PI_STAR, seed=SEED)
        c_s_a = make_corruption(abstr_mdp, reassignment_dict=error_dict)
        corrupt_mdp = AbstractMDP(mdp, c_s_a)
        agent = UDMAgent(mdp, s_a=corrupt_mdp.state_abstr, seed=SEED, transition_threshold=TRANSITION_THRESHOLD)
        if i == 0:
            print('State abstraction is')
            print(agent.get_abstraction_as_string())
        #for j in range(SPLORE_RANGE):
        #    agent.explore()
        #for i in range(len(EPISODE_COUNT)):
        while agent.episode_count < EPISODE_COUNT:
            agent.explore()
            if agent.episode_count > 0 and agent.end_of_episode_record[-1] == 1:
                print('On episode', agent.episode_count, agent.step_count_record[i])

        transition_records = agent.get_all_transition_returns()
        split_record = agent.test_all_states_for_split(transition_records)

        abstr_states_seen = []
        for ground_state in agent.mdp.get_all_possible_states():
            abstr_state = agent.get_abstr_from_ground(ground_state)

            if abstr_state not in abstr_states_seen:
                print('Overlapping states for', abstr_state, end=' ')
                ground_states = agent.get_ground_states_from_abstract_state(abstr_state)
                for ground in ground_states:
                    print(ground, end=' ')
                print()
                agent.print_action_to_state_list(agent.test_state_for_split(transition_records[abstr_state]))
                abstr_states_seen.append(abstr_state)

        count_number_of_splits(split_record)
        print('State abstraction is')
        print(agent.get_abstraction_as_string())
        print(split_record)
        memory_record = split_record_to_additional_states(split_record)
        print_memory_record(memory_record)
    '''

