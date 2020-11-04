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
from resources.AbstractionMakers import make_abstr
from resources.AbstractionTypes import Abstr_type
from MDP.StateClass import State
from MDP.AbstractMDPClass import AbstractMDP


class Agent():
	def __init__(self,
				 mdp,
				 alpha=0.1,
				 epsilon=0.1,
				 decay_exploration=True):
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
		#self._q_table = {}
		self._alpha = alpha
		self._init_alpha = alpha 
		self._epsilon = epsilon
		self._init_epsilon = epsilon 
		self._step_counter = 0
		self._episode_counter = 0
		self.decay_exploration = decay_exploration

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
			# if(state.__eq__(GridWorldState(1,1))):
			# 	print(action)

		return action 

	def _update_learning_parameters(self):
		'''
		Update the self._epsilon and self._alpha parameters after 
		taking an epsilon-greedy step 

		Currently taken from David Abel's _anneal function, assumes 
		episode number is always 1 
		'''
		if self.decay_exploration:
			self._epsilon = self._init_epsilon / (1.0 + (self._step_counter / 2000000))
		self._alpha = self._init_alpha / (1.0 + (self._step_counter / 2000000))

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
		self._step_counter += 1
		if next_state.is_terminal():
			self._episode_counter += 1

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

		Only ever used for testing

		Parameters:
			action:Enum
		Returns:
			next_state:State
			reward:float
		'''
		next_state, reward = self.mdp.act(action)
		return next_state, reward

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

	# Can probably be deprecated since we aren't going to make online abstractions in this way
	def make_abstraction(self, abstr_type, epsilon, ignore_zeroes=False, threshold=1e-6):
		"""
		Create an abstraction out of the current q-table of the given type with given epsilon
		:return: new_abstr_mdp, a new abstract MDP made from the current q-table, with q-values informed by current
					q-table
		"""
		# Create a state abstraction based on the current q-table
		curr_q_table = self.get_q_table()
		new_abstr = make_abstr(curr_q_table, abstr_type, epsilon=epsilon, ignore_zeroes=ignore_zeroes, threshold=threshold)

		# Update agent's q-table for the new abstract states
		# For each new abstract state, average the state-action values of the constituent states and
		#  make that average the state-action value for the new abstract state
		new_q_table = defaultdict(lambda : 0.0)

		#  All old values are the same
		for key, value in curr_q_table.items():
			new_q_table[key] = value

		# Get possible states of MDP
		possible_states = self.mdp.get_all_possible_states()

		# Make guess at new values for abstract states by averaging state-action values of constituent states
		#  Iterate through all abstract states
		for abstr_state in new_abstr.abstr_dict.values():
			# For each action...
			for action in self.mdp.actions:
				action_val = 0
				map_count = 0
				# ...Get the states that are grouped together and average their state-action values for that action
				for ground_state in possible_states:
					if new_abstr.get_abstr_from_ground(ground_state).data == abstr_state:
						action_val += curr_q_table[(ground_state, action)]
						map_count += 1
				if map_count != 0:
					# Since abstr_state is just an integer, we have to make a State out of it
					new_q_table[(State(data=abstr_state, is_terminal=False), action)] = action_val / map_count

		# Assign this updated q-table to the agent's q-table
		self._q_table = new_q_table

		# Update the agent's MDP to be the AbstractMDP generated by combining the state abstraction with the current
		#  MDP
		new_abstr_mdp = AbstractMDP(self.mdp, new_abstr)
		self.mdp = new_abstr_mdp

		# Return number of abstract states and number of ground states mapped to abstract states
		unique_abstr_states = []
		ground_states = []
		for key in new_abstr.abstr_dict.keys():
			if key not in ground_states:
				ground_states.append(key)
			if new_abstr.abstr_dict[key] not in unique_abstr_states:
				unique_abstr_states.append(new_abstr.abstr_dict[key])
		return len(unique_abstr_states), len(ground_states)

	# --------------------------
	# Getters, setters & utility
	# --------------------------

	def __str__(self):
		try:
			result = 'Agent on ' + str(self.mdp)
			return result
		except:
			print('error')
			print(self.mdp.get_current_state())
			print(type(self.mdp.get_current_state()))
			print()
			return('')

	def reset_mdp(self):
		'''
		Reset the MDP to its initial state
		:return: None
		'''
		self.mdp.reset_to_init()

	def reset_to_init(self):
		'''
		Reset the agent's current state to the initial state in the 
		mdp 
		'''
		self.mdp.reset_to_init()
		self._alpha = self._init_alpha
		self._epsilon = self._init_epsilon

	# Seems like Agent shouldn't have the ability to set the state 
	# of the MDP 
	def set_current_state(self, new_state):
		'''
		Set current state of agent to given state

		Parameters:
			new_state:State
		'''
		self.mdp.set_current_state(new_state)

	def get_current_state(self):
		'''
		Get current state of agent 
		'''
		curr_state = self.mdp.get_current_state()
		return curr_state

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
		shuffled_actions = list(copy.copy(self.mdp.actions))
		random.shuffle(shuffled_actions)
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

	def get_q_table(self):
		return self._q_table

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

	def get_learned_policy_as_string(self):
		"""
		Generate a dictionary (States -> Actions) of the learned policy
		:return: policy_dict
		"""
		policy_dict = {}

		# If the MDP is a abstract MDP, we can't query the ground states directly
		if isinstance(self.mdp, AbstractMDP):
			for state in self.mdp.get_all_possible_states():
				abstr_state = self.mdp.get_state_abstr().get_abstr_from_ground(state)
				policy_dict[str(abstr_state)] = str(self.get_best_action(abstr_state))
		else:
			for state in self.mdp.get_all_possible_states():
				policy_dict[str(state)] = str(self.get_best_action(state))

		return policy_dict

	def get_learned_policy(self):
		"""
		Get the policy learned by the agent as a dictionary mapping states to actions
		:return: policy_dict, a dictionary mapping states to actions
		"""
		policy_dict = {}

		# If the MDP is a abstract MDP, we can't query the ground states directly
		if isinstance(self.mdp, AbstractMDP):
			for state in self.mdp.get_all_possible_states():
				abstr_state = self.mdp.get_state_abstr().get_abstr_from_ground(state)
				policy_dict[abstr_state] = self.get_best_action(abstr_state)
		else:
			for state in self.mdp.get_all_possible_states():
				policy_dict[state] = self.get_best_action(state)

		return policy_dict

	def get_learned_state_values(self):
		"""
		Get the state values learned by the agent as a dictionary mapping states to values. This is done by taking
		the max q-value as the state value.
		:return: value_dict, a dictionary mapping states to values
		"""
		value_dict = {}

		# If the MDP is an abstract MDP, can't query ground states directly
		if isinstance(self.mdp, AbstractMDP):
			for state in self.mdp.get_all_possible_states():
				abstr_state = self.mdp.get_state_abstr().get_abstr_from_ground(state)
				best_action = self.get_best_action(abstr_state)
				if isinstance(abstr_state, State):
					state_tag = abstr_state.data
				else:
					state_tag = abstr_state
				value_dict[state_tag] = self.get_q_value(abstr_state, best_action)
		# If not, query ground states directly
		else:
			for state in self.mdp.get_all_possible_states():
				best_action = self.get_best_action(state)
				value_dict[tuple(state.data)] = self.get_q_value(state, best_action)

		return value_dict




		
