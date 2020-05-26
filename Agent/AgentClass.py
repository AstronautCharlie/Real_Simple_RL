'''
This class is a reinforcement learning agent that will interact 
with its MDP, store the results of the interaction, and update
its performance 
'''

# Used to initialize q-table 
from collections import defaultdict
import random 
import numpy as np 
import copy 

class Agent():
	def __init__(self, 
				 mdp,
				 alpha=1.0,
				 epsilon=1.0):
		'''
		Parameters:
			mdp: MDP
			policy: Policy 
			alpha:float (learning rate)

		Notes:
			collections.defaultdict(dict) creates an empty nested
			dictionary where values are themselves empty 
			dictionaries; similar to what David used
		'''
		self.mdp = mdp
		self._q_table = defaultdict(lambda : 0.0)
		self._alpha = alpha
		self._init_alpha = alpha 
		self._epsilon = epsilon
		self._init_epsilon = epsilon 

	# ---------------------
	# Exploration functions
	# ---------------------
	def epsilon_greedy(self, state):
		'''
		Take best action with probability 1-epsilon and random action
		with probability epsilon. 
		Parameters:
			state:GridWorldState

		Returns:
			action:Enum
		'''
		action = None 

		# Flip a 'coin'. If it comes up less than epsilon, take
		# random action. Otherwise pick best action 
		if random.random() < self._epsilon:
			action = np.random.choice(self.mdp.actions)
		else: 
			action = self.get_best_action(state)

		return action 

	def _update_learning_parameters(self):
		'''
		Update the self._epsilon and self._alpha parameters after 
		taking an epsilon-greedy step 

		""" This is a stub! """
		'''
		self._epsilon *= 1.0
		self._alpha *= 0.999

	# --------------
	# Main functions
	# --------------

	def explore(self):
		'''
		Apply epsilon-greedy exploration and update q-table with 
		with the resulting data 

		Returns: 
			current_state:GridWorldState, the state the agent was in 
				before taking action 
			action:Enum, the action taken
			next_state:GridWorldState, the state the agent ended up in
				after taking the action 
			reward:float, the reward received  
		'''
		# Get current state, apply action, and query MDP for result 
		# of applying action on state 
		current_state = self.get_current_state()
		action = self.epsilon_greedy(current_state)
		next_state, reward = self.mdp.act(action)

		# Update q-table, current_state, and learning parameters
		self.update(current_state, action, next_state, reward)

		if self.get_q_value(current_state, action) != 0:
			self._update_learning_parameters()

		return current_state, action, next_state, reward 

	def apply_best_action(self):
		'''
		Apply the best action based on the agent's current q-table
		values. Does not update the agent's q-table 
		'''	
		current_state = self.get_current_state() 
		best_action = self.get_best_action(current_state)
		next_state, reward = self.mdp.act(best_action)
		return current_state, best_action, next_state


	def apply_action(self, action):
		'''
		Apply the given action to agent's current state, get 
		the reward and next state from the mdp, and update the 
		agent's q-table with the results 

		Parameters:
			action:Enum
		'''
		next_state, reward = self.mdp.act(action)
		self.update(current_state, action, next_state, reward)
		return current_state, action, next_state


	def update(self, state, action, next_state, reward):
		'''
		Update the Agent's internal q-table with the state-action-
		next state-reward info according to the Bellman Equation: 
			q(s,a) <- q(s,a) + alpha * [r + gamma * max_a(s',a) - q(s,a))]
	
		Parameters: 
			state: State
			action: Enum
			next_state: State
		'''
		old_q_value = self.get_q_value(state, action)
		best_next_action_value = self.get_best_action_value(next_state)
		new_q_value = old_q_value + self._alpha * (reward + self.mdp.gamma * best_next_action_value - old_q_value)
		self._set_q_value(state, action, new_q_value)
		

	# --------------------------
	# Getters, setters & utility
	# --------------------------

	def reset_to_init(self):
		'''
		Reset the agent's current state to the initial state in the 
		mdp 
		'''
		self.mdp.reset_to_init()


	# Seems like Agent shouldn't have the ability to set the state 
	# of the MDP 
	#def set_current_state(self, new_state):
		'''
		Set current state of agent to given state

		Parameters:
			new_state:State
		'''
	#	self.mdp.set_current_state(new_state)

	def get_current_state(self):
		'''
		Get current state of agent 
		'''
		return self.mdp.get_current_state()

	def get_best_action_value_pair(self, state):
		'''
		Get the action with the best q-value and the q-value associated
		with that action 

		Parameters:
			state:State

		Returns:
			best_action:Enum
			max_val:float 
		'''
		# Initialize best action to be a random choice (in case no )
		max_val = float("-inf")
		best_action = None 

		# Iterate through actions and find action with highest q-value
		# in q-table. Shuffle actions so that if best actions have 
		# the same value, a random one is chosen
		shuffled_actions = self.mdp.actions.copy()
		np.random.shuffle(shuffled_actions)
		for action in shuffled_actions:
			q_value = self.get_q_value(state, action)
			if q_value > max_val:
				max_val = q_value 
				best_action = action 
		return best_action, max_val 

	def get_best_action(self, state):
		'''
		Return the action with the max q-value for the given state

		Parameters:
			state:State

		Returns:
			best_action:Enum
		'''
		best_action, _ = self.get_best_action_value_pair(state)
		return best_action 

	def get_best_action_value(self, state):
		'''
		Return the q-value of the action with the max q-value for
		the given state 

		Parameters: 
			state:State

		Returns:
			reward:float 
		'''
		_, reward = self.get_best_action_value_pair(state)
		return reward 

	def get_action_values(self, state):
		'''
		Get all the action-value pairs for the given state and
		return them as a list of tuples

		Parameters:
			state:State
		
		Returns:
			action_value_list:list 
		'''
		action_value_list = [] 
		for action in self.mdp.actions:
			pair = tuple([action, self.get_q_value(state, action)])
			action_value_list.append(pair)
		return action_value_list

	def get_q_value(self, state, action):
		'''
		Query the q-table for the value of the given state-action 
		pair 

		Parameters:
			state:State
			action:Enum

		returns:
			q-value:float
		'''
		return self._q_table[(state, action)]

	def get_mdp(self):
		return self.mdp

	def _set_q_value(self, state, action, new_value):
		'''
		Set the q-value of the given state-action pair to the new
		value

		Parameters:
			state:State
			action:Enum
			new_value:float
		'''
		self._q_table[(state, action)] = new_value 
		
