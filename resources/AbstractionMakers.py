'''
This file defines functions to create Q*, a*, and pi* 
StateAbstractions. Assumes that all states in the q_table 
have a value assigned for all actions 
'''
from MDP.StateAbstractionClass import StateAbstraction 
from resources.AbstractionTypes import Abstr_type 
import numpy as np
from MDP.StateClass import State
import random 

def make_abstr(q_table, abstr_type, epsilon=1e-10, combine_zeroes=False, threshold=1e-10, seed=None):
	"""
	:param q_table: dictionary((state,action):float)
	:param abstr_type:Enum(Abstr_type)
	:param ignore_zeroes:boolean
	:param threshold: float
	:return: q_star:StateAbstraction

	Given a q_table of state-action -> value mappings, create a Q*
	StateAbstraction with error tolerance epsilon (If, for each action, the action-values between the
	two states are within epsilon of each other, the states are abstracted together
	"""
	if seed:
		np.random.seed(seed)

	abstr_dict = {} 
	abstr_counter = 1 

	# Get all the states and actions. This will be useful in the 
	# next loop 
	temp_states = [] 
	temp_actions = []
	for key in q_table.keys():
		(state, action) = key
		temp_states.append(state)
		temp_actions.append(action)

	states = [] 
	actions = [] 
	for state in temp_states:
		if state not in states: 
			states.append(state)
	for action in temp_actions: 
		if action not in actions:
			actions.append(action)

	# Separate out states with 0 value
	zero_states = []
	for state in states:
		is_zero = True
		for action in actions:
			if (state, action) not in q_table.keys():
				q_table[(state, action)] = 0
			try:
				if abs(q_table[(state, action)]) > threshold:
					is_zero = False
			except:
				print(state, action)
				print('failed in make_abstr')
				print(threshold, type(threshold))
				quit()
		if is_zero:
			#print('Found zero state', state)
			zero_states.append(state)

	for state in zero_states:
		states.remove(state)

	# Shuffle states to resolve the non-unqiue cluster problem
	# Point of this is that if we're using an approximate grouping, we don't want to always
	# get the same cluster
	#if epsilon > 1e-12:
	np.random.shuffle(states)

	# Iterate through all non-zero states
	for state in states:

		# If this state has already been mapped to an abstract
		# state, so will all the other states it would be 
		# mapped to, so skip it 
		if state in abstr_dict.keys():
			continue

		# This flag is used to increment the abstr_counter 
		# if at least two states are mapped together 
		incr_counter = False 

		# Iterate through the other states and compare the 
		# state-action values 
		for other_state in states:

			# Ignore self-comparison 
			if state == other_state:
				continue

			# If other state is already mapped, skip it to avoid
			# non-unique abstraction problem 
			if other_state in abstr_dict.keys():
				continue

			is_match = True

			# If q-star, compare state-action value pairs
			# across all actions. If all are within epsilon 
			# of each other, the two states are mapped together
			if abstr_type == Abstr_type.Q_STAR:
				state_action, _ = get_best_action_value_pair(q_table, state, actions)
				other_action, _ = get_best_action_value_pair(q_table, other_state, actions)
				if state_action != other_action:
					is_match = False
				for action in actions: 
					if abs(q_table[(state, action)] - q_table[(other_state, action)]) > epsilon:
						is_match = False 

			# If a-star, compare state-action value pairs for 
			# action with highest value. If these values are 
			# the same and the actions are the same, then 
			# the two states are mapped together 
			elif abstr_type == Abstr_type.A_STAR:
				state_action, state_val = get_best_action_value_pair(q_table, state, actions)
				other_action, other_val = get_best_action_value_pair(q_table, other_state, actions)
				if state_action != other_action or abs(state_val - other_val) > epsilon:
					is_match = False

			# If pi-star, get the action with max state-action
			# value for each state. If these are the same, 
			# then the two states are mapped together
			elif abstr_type == Abstr_type.PI_STAR:
				state_action, _ = get_best_action_value_pair(q_table, state, actions)
				other_action, _ = get_best_action_value_pair(q_table, other_state, actions)
				if state_action != other_action:
					is_match = False 
			else:
				raise ValueError("Abstraction type not supported: " + str(abstr_type))


			# If they are a match, map them both to the same 
			#  unique abstract stated, identified by a
			#  number
			if is_match:
				#print('Match!', state, other_state)
				abstr_dict[other_state] = abstr_counter
		
		# If at least two states got mapped together, 
		# increment the abstract state counter 
		abstr_dict[state] = abstr_counter#State(abstr_counter)
		abstr_counter += 1

	# Handle zero states
	for state in zero_states:
		#print('fixing zero state', state)
		abstr_dict[state] = abstr_counter #State(abstr_counter)
		if not combine_zeroes:
			abstr_counter += 1

	abstr = StateAbstraction(abstr_dict, abstr_type, epsilon)
	print('State abstraction is', str(abstr))
	#for key, value in q_table.items():
#		print(key[0], key[1], value)
	return abstr

def get_best_action_value_pair(q_table, state, actions):
	'''
	Helper function to make_abstr. Given a state, a q_table 
	and a list of actions, return the action with highest value
	and the value of that action 
	'''
	max_val = float("-inf")
	best_action = None 
	for action in actions:
		if q_table[(state, action)] > max_val: 
			max_val = q_table[(state, action)]
			best_action = action
	return best_action, max_val 