'''
Contains the Transition and Reward function abstract classes.
'''

import abc

class Transition():
	@abc.abstractmethod
	def __call__(self, state, action, mdp):
		pass 

class Reward():

	@abc.abstractmethod
	def __call__(self, state, action, next_state, mdp):
		pass 


		