"""
This type of Agent extends the regular Q-learning agent by adding in an abstraction. This abstraction is a
grouping of ground states to abstract states. When a Q-value is update for a ground state, the agent gets any
other states that are mapped to the same abstract state as the ground state and performs the same update to
their Q-value. If done properly, this should yield the same behavior as an agent operating in an abstract MDP
"""

from Agent.AgentClass import Agent
from GridWorld.ActionEnums import Dir
from collections import defaultdict


class AbstractionAgent(Agent):
    def __init__(self,
                 mdp,
                 s_a=None,
                 alpha=0.1,
                 epsilon=0.1):
        """
        Create an agent with a state abstraction mapping
        :param mdp: an MDP environment
        :param s_a: StateAbstraction (mapping of ground states to abstract states). If no s_a is provided,
                    then this is equivalent to a regular Q-learning agent
        :param alpha: learning rate
        :param epsilon: exploration rate
        """
        Agent.__init__(self, mdp, alpha, epsilon)
        self.s_a = s_a

        # Create a dictionary mapping each ground state to all other ground states with the same abstract state
        #  This is done so that we don't have to iterate through all states to check abstract state mapping when
        #  performing an update
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
        if self.s_a is not None and state in self.group_dict.keys():
            for equiv_state in self.group_dict[state]:
                states_to_update.append(equiv_state)

        # Update all states in the same abstract state
        old_q_value = self.get_q_value(state, action)
        if not next_state.is_terminal():
            best_next_action_value = self.get_best_action_value(next_state)
        else:
            best_next_action_value = 0.0
        new_q_value = old_q_value + self._alpha * (reward + self.mdp.gamma * best_next_action_value - old_q_value)
        for equiv_state in states_to_update:
            self._set_q_value(equiv_state, action, new_q_value)

    def detach_state(self, state):
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

        # Set state to its own abstract state
        self.s_a.abstr_dict.pop(state, None)
        self.group_dict.pop(state, None)
        max_abstr_state = max(self.s_a.abstr_dict.values())
        self.s_a.abstr_dict[state] = max_abstr_state + 1
        self.group_dict[state] = []

        # Reset Q-table to 0 for this state
        for action in self.mdp.actions:
            self._set_q_value(state, action, 0)
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
        #while not state.is_terminal() and state not in rollout:
        while state not in rollout:
            rollout.append(state)
            if state.is_terminal():
                break
            action = policy[state]
            next_state = self.mdp.transition(state, action)
            state = next_state
        return rollout

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

