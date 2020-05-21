'''
This class is a reinforcement learning agent that will interact 
with its MDP, store the results of the interaction, and update
its performance 
'''

# Used to initialize q-table 
from collections import defaultdict

class Agent():
	def __init__(self, 
				 mdp,
				 policy,
				 alpha=0.5):
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
		self._current_state = mdp.get_init_state()
		self.policy = policy
		# This will store q-values associated with (state, action) pairs 
		self.q_table = defaultdict(lambda : 0)

	# ---------------------
	# Exploration functions
	# ---------------------
	#def epsilon_greedy(self, state):

	#def 

	# ---------------------------
	# Main act & update functions
	# ---------------------------

	def act(self):
		'''
		Apply the agent's policy to its current state and return the 
		action dictated by that policy

		Parameters: 
			None

		Returns: 
			action: Enum 
		'''
		current_state = self.get_current_state()
		print("Currently at:", str(current_state))

		action = self.policy(current_state)
		print("Policy dictates:", str(action))

		next_state, reward = self.mdp.act(current_state, action)
		print("Next state is:", str(next_state))
		print("Reward:", reward)

		self.set_current_state(next_state)
		print()


	def update(self, state, action, next_state, reward):
		'''
		Update the Agent's internal q-table with the new info based
		on Bellman Equation: 
			q(s,a) <- q(s,a) + alpha * [r + gamma * max_a(s',a) - q(s,a))]

		Parameters: 
			state: State
			action: Enum
			next_state: State
		'''
		

	# -------------------
	# Getters and setters  
	# -------------------

	def set_current_state(self, next_state):
		self._current_state = next_state 

	def get_current_state(self):
		return self._current_state

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
		max_val = float("-inf")
		best_action = None 
		for action in self.mdp.actions:
			q_value = self.q_table[(state, action)]
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
		best_action, _ = self.get_best_action_value_pair(state)[0]
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
		_, reward = self.get_best_action_value_pair(state)[1]
		return reward 
