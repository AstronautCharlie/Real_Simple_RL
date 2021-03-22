'''
This class represents an abstract MDP by combining an MDP with a state abstraction.
Must support all the MDP operations that allow an agent to learn. Handles all MDP related
functions by passing them to its MDP.

Written so that the Experiment class can stamp out abstract MDPs without knowing the
type of the MDP
'''

#from SimpleMDP import SimpleMDP
import copy
from MDP.StateAbstractionClass import StateAbstraction

class AbstractMDP():
	def __init__(self, mdp, state_abstr):
		self.mdp = mdp
		self.actions = mdp.actions
		self.init_state = self.mdp.init_state
		if state_abstr is None:
			print('Making trivial abstraction')
			s_a = StateAbstraction()
			s_a.make_trivial_abstraction(mdp)
			self.state_abstr = s_a
		else:
			self.state_abstr = state_abstr
		self.gamma = mdp.gamma

		self.abstr_type = self.state_abstr.abstr_type
		self.abstr_epsilon = self.state_abstr.epsilon
		# Need to maintain the current state at this level for ExperimentClass to work
		self.current_state = mdp.current_state

	def copy(self):
		# Super hacky work-around
		#if isinstance(self.mdp, SimpleMDP.SimpleMDP):
		#	new_mdp = SimpleMDP.SimpleMDP()
		#	s_a = copy.copy(self.state_abstr)
		#	copied_mdp = AbstractMDP(new_mdp, s_a)
		#	return copied_mdp

		#else:
		new_mdp = self.mdp.copy()
		state_abstr = copy.copy(self.state_abstr)
		copied_mdp = AbstractMDP(new_mdp, state_abstr)
		return copied_mdp

	def __str__(self):
		result = str(self.mdp)
		result += '\n' + "State abstraction is:"
		result += '\n' + str(self.state_abstr)
		return result

	# -------------
	# MDP functions
	# These functions simply pass the arguments to the underlying MDP
	# -------------

	def reward(self, state, action, next_state):
		return self.mdp.reward(state, action, next_state)

	def transition(self, state, action):
		return self.mdp.transition(state, action)

	def reset_to_init(self):
		self.mdp.reset_to_init()
		self.current_state = self.mdp.current_state

	def is_goal_state(self, state):
		return self.mdp.is_goal_state(state)

	def act(self, action):
		state = self.mdp.current_state
		next_state = self.transition(state, action)
		reward = self.reward(state, action, next_state)

		# Update current state to the result of the transition
		self.set_current_state(next_state)
		self.current_state = next_state

		# If the next state is terminal, set current_state
		# to initial state. Still returns next state
		if state.is_terminal():
			self.reset_to_init()

		next_state = self.state_abstr.get_abstr_from_ground(next_state)

		return next_state, reward

	def get_all_possible_states(self):
		return self.mdp.get_all_possible_states()

	def get_all_abstr_states(self):
		abstr_states = []
		states = self.get_all_possible_states()
		for state in states:
			abstr_state = self.get_abstr_from_ground(state)
			if abstr_state not in abstr_states:
				abstr_states.append(abstr_state)
		return abstr_states

	# -----------------
	# Getters & setters
	# -----------------
	def get_current_state(self):
		return self.state_abstr.get_abstr_from_ground(self.mdp.current_state)

	def set_current_state(self, new_state):
		self.mdp.set_current_state(new_state)

	def get_state_abstr(self):
		return self.state_abstr

	def get_abstr_from_ground(self, ground_state):
		return self.state_abstr.get_abstr_from_ground(ground_state)

	def get_ground_from_abstr(self, abstr_state):
		"""
		Return a list of all ground states in the given abstr_state
		"""
		states = self.get_all_possible_states()
		group = []
		for state in states:
			if self.get_abstr_from_ground(state) == abstr_state:
				group.append(state)
		return group

	# -------
	# Utility
	# -------
	def abstr_to_string(self):
		"""
		Write the state abstraction to a string. This is used in the experiment class to write the state abstraction
		to a file
		:return: str_rep: string representation of the state abstraction
		"""
		# Write the state abstraction as a list of 2-tuples where first value is the state information and the second
		#  value is the
		str_rep = '['
		for key, value in self.state_abstr.abstr_dict.items():
			str_rep += '(' + str(key) + ', ' + str(value) + '), '
		# Trim the trailing comma
		str_rep = str_rep[:-2] + ']'
		return str_rep

	# ------------
	# MDP-specific
	#
	# These functions are specific to types of MDPs. If you call one of
	# these functions from an abstractMDP that does not have a ground MDP
	# of the appropriate type, you will get an error
	# ------------

	# Gridworld
	def get_width(self):
		return self.mdp.get_width()

	def get_height(self):
		return self.mdp.get_height()

	def compute_walls(self):
		return self.mdp.compute_walls()