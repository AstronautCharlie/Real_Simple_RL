'''
This class represents an abstract MDP by combining an MDP with a state abstraction.
Must support all the MDP operations that allow an agent to learn. Handles all MDP related
functions by passing them to its MDP.

Written so that the Experiment class can stamp out abstract MDPs without knowing the
type of the MDP
'''

class AbstractMDP():
	def __init__(self, mdp, state_abstr):
		self.mdp = mdp
		self.actions = mdp.actions
		self.init_state = self.mdp.init_state
		self.state_abstr = state_abstr
		self.gamma = mdp.gamma

	def copy(self):
		new_mdp =

	# -------------
	# MDP functions
	# -------------

	def reward(self, state, action, next_state):
		return self.mdp.reward(state, action, next_state)

	def transition(self, state, action):
		return self.mdp.transition(state, action)

	def reset_to_init(self):
		self.set_current_state(self.init_state)

	def is_goal_state(self, state):
		return self.mdp.is_goal_state(state)

	def act(self, action):
		state = self.mdp.current_state
		next_state = self.transition(state, action)
		reward = self.reward(state, action, next_state)

		# Update current state to the result of the transition
		self.set_current_state(next_state)

		# If the next state is terminal, set current_state
		# to initial state. Still returns next state
		if state.is_terminal():
			self.reset_to_init()

		next_state = self.state_abstr.get_abstr_from_ground(next_state)

		#print("At state", state, "took action", action, "got to next state", next_state, "got reward", reward)
		return next_state, reward

	# -----------------
	# Getters & setters
	# -----------------
	def get_current_state(self):
		return self.state_abstr.get_abstr_from_ground(self.mdp.current_state)

	def set_current_state(self, new_state):
		self.mdp.set_current_state(new_state)
