'''
This class represents an abstract MDP by combining an MDP with a state abstraction.
Must support all the MDP operations that allow an agent to learn. Handles all MDP related
functions by passing them to its MDP.

Written so that the Experiment class can stamp out abstract MDPs without knowing the
type of the MDP
'''

import copy

class AbstractMDP():
	def __init__(self, mdp, state_abstr):
		self.mdp = mdp
		self.actions = mdp.actions
		self.init_state = self.mdp.init_state
		self.state_abstr = state_abstr
		self.gamma = mdp.gamma
		self.abstr_type = state_abstr.abstr_type
		self.abstr_epsilon = state_abstr.epsilon
		# Need to maintain the current state at this level for ExperimentClass to work
		self.current_state = mdp.current_state

	def copy(self):
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