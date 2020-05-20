'''
This class is a reinforcement learning agent that will interact 
with its MDP, store the results of the interaction, and update
its performance 
'''

class Agent():
	def __init__(self, 
				 mdp,
				 policy):
		'''
		Let's go ahead and attach the agent to an MDP since I can't
		foresee an instance where we'd want a free-floating agent 

		Parameters:
			mdp: MDP
			policy: Policy 

		Notes:
			collections.defaultdict(dict) creates an empty nested
			dictionary where values are themselves empty 
			dictionaries; similar to what David used
		'''
		self.mdp = mdp
		self._current_state = mdp.get_init_state()
		self.policy = policy
		self.q_table = {}

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


	def update(self, state, action, next_state):
		'''
		Update the Agent's internal q-table with the new info 

		Parameters: 
			state: State
			action: Enum
			next_state: State
		'''
		return 

	# -------------------
	# Getters and setters  
	# -------------------

	def set_current_state(self, next_state):
		self._current_state = next_state 

	def get_current_state(self):
		return self._current_state
