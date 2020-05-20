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
		self.current_state = mdp.get_init_state()
		self.policy = policy
		self.q_table = collections.defaultdict(dict)

	def act(self, state):
		'''
		Apply the agent's policy to the given state and return the 
		action dictated by that policy

		Parameters: 
			state: State

		Returns: 
			action: Enum 
		'''

	def update(self, state, action, next_state):
		'''
		Update the Agent's internal q-table with the new info 

		Parameters: 
			state: State
			action: Enum
			next_state: State
		'''
