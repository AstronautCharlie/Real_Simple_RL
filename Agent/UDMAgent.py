"""
This extends AbstractionAgent by replacing the original consistency check process with a modification of Andrew
McCallum's UDM algorithm.

The agent records the abstract state observed, the action taken, and the discounted reward received for each time-step.
Then it periodically checks, for each state, whether the returns from all incoming trajectories that share an outgoing
action are 'close enough' to each other. If not, it splits the states according to which incoming trajectories overlap
"""

from Agent.AbstractionAgent import AbstractionAgent
from GridWorld.TwoRoomsMDP import TwoRoomsMDP
from GridWorld.GridWorldStateClass import GridWorldState
from resources.AbstractionTypes import Abstr_type

SPLORE_RANGE = 100
import numpy as np
from scipy import stats

class UDMAgent(AbstractionAgent):
    def __init__(self,
                 mdp,
                 s_a=None,
                 alpha=0.1,
                 epsilon=0.1,
                 decay_exploration=True,
                 confidence_interval_alpha=0.05
                 ):
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
                         detach_reassignment='group')

        # Alpha parameter used for Student's T statistic for confidence interval
        self.confidence_interval_alpha = confidence_interval_alpha

        # These hold the experience
        self.action_record = []
        self.reward_record = []
        self.ground_state_record = []
        self.abstr_state_record = []

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
        # Sample mean
        mean = np.mean(return_list)

        # Sample standard deviation
        n = len(return_list)
        numer = n * np.sum(np.square(return_list)) - np.square(np.sum(return_list))
        denom = n * (n-1)
        stdev = np.sqrt(numer / denom)

        # Student's T function
        t = stats.t.ppf(1 - self.confidence_interval_alpha / 2, n - 1)

        # Calculate confidence interval bounds
        lower_bound = mean - t * stdev / np.sqrt(n)
        upper_bound = mean + t * stdev / np.sqrt(n)

        return lower_bound, upper_bound

    def group_overlapping_intervals(self, list_of_lists):
        """

        """

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
        self.action_record.append(action)
        self.ground_state_record.append(current_state)
        self.abstr_state_record.append(abstr_state)
        self.reward_record.append(reward)
        return current_state, action, next_state, reward

    # --------------------------
    # Getters & Setters, Utility
    # --------------------------
    def get_abstr_from_ground(self, ground_state):
        return self.s_a.get_abstr_from_ground(ground_state)

    def get_returns(self):
        """
        Calculate returns from rewards: return[t] = reward[t] + gamma * return[t+1], return[N] = reward[N]
        """
        returns = [0] * len(self.reward_record)
        returns[len(self.reward_record) - 1] = self.reward_record[-1]
        for i in range(len(self.reward_record) - 2, -1, -1):
            returns[i] = self.reward_record[i] + self.mdp.gamma * returns[i+1]
        return returns

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


# Main method for testing purposes only
if __name__ == '__main__':
    mdp = TwoRoomsMDP()
    abstr_mdp = mdp.make_abstr_mdp(Abstr_type.A_STAR)
    agent = UDMAgent(mdp, s_a=abstr_mdp.state_abstr)
    print('State Abstraction is')
    print(agent.get_abstraction_as_string())
    print()

    # Testing confidence interval function
    '''
    returns = [0.5, 1, 0.75, 2, 0.6, 0.7]
    lower_bound, upper_bound = agent.calculate_confidence_interval(returns)
    print(lower_bound, upper_bound)
    '''

    print('Exploring')
    for i in range(SPLORE_RANGE):
        current_state, action, next_state, reward = agent.explore()

    returns = agent.get_returns()
    print('State, action, return')
    for i in range(SPLORE_RANGE):
        print(agent.ground_state_record[i], agent.abstr_state_record[i], agent.action_record[i], returns[i])
    print()

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
    transition_records = agent.get_all_transition_returns()
    agent.print_all_transition_records(transition_records)

    '''
    for key, value in transition_records.items():
        if len(value) > 0:
            print(key, end=': ')
            for key2, val2 in value.items():
                print(key2)
                for key3, val3 in val2.items():
                    print(key3, val3)
            print()
    '''