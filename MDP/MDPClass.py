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

	# -------
	# Getters 
	# -------
	def get_init_state(self):
		return self.init_state 

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