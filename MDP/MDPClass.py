'''
Defines the MDP class, which is a container for the actions, 
transition function, reward function, and state info 
'''

import abc 

class MDP():
	def __init__(self,
				 actions, 
				 init_state,
				 gamma):
		self.actions = actions 
		self.init_state = init_state
		self.current_state = init_state
		self.gamma = gamma 
		'''
		Parameters: 
			actions: Enum 
			transition_func: TransitionFunction
			reward_func: RewardFunction
			init_state: State
			gamma: float 
		'''

	# -----------------
	# Getters & setters
	# -----------------
	def get_init_state(self):
		return self.init_state

	def get_current_state(self):
		return self.current_state

	def set_current_state(self, state):
		'''
		:param state:State
		'''
		self.current_state = state

	#@abc.abstractmethod
	#def transition(self, state, action):

	# --------------
	# Main functions 
	# --------------
	
	@abc.abstractmethod 
	def transition(self, state, action):
		'''
		Parameters:
			state:State
			action:Enum

		Returns:
			next_state:State
		'''

	@abc.abstractmethod 
	def reward(self, state, action, next_state):
		'''
		Parameters:
			state:State
			action:Enum
			next_state:State

		Returns:
			reward:float
		'''

	@abc.abstractmethod
	def act(self, action):
		'''
		Take in an action, apply the transition function to the action
		and current state, update the current state to the resulting
		next state, and return the next state and reward
		:param action:Enum
		:return: next_state:State (specific to MDP)
		:return: reward:float
		'''
	@abc.abstractmethod
	def next_possible_states(self,state,action):
		"""
		Given a state, return a dictionary containing all states that can result from taking the given action on it along
		with the probability of ending up in that state
		:param state: State
		:param action: Enum
		:return: dictionary of State -> Float (probability, so needs to be less than one)
		"""

	@abc.abstractmethod
	def get_all_possible_states(self):
		"""
		Returns a list containing all the possible states in this MDP
		:return: List of States
		"""

	@abc.abstractmethod
	def copy(self):
		"""
		Returns a deep copy of self
		:return: a deep copy of self
		"""