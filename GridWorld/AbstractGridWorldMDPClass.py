'''
This class represents an abstract GridWorldMDP by combining an MDP with a 
StateAbstraction, which maps states to abstact states 
'''
from GridWorld.GridWorldMDPClass import GridWorldMDP 
from GridWorld.GridWorldStateClass import GridWorldState 
from GridWorld.ActionEnums import Dir 

class AbstractGridWorldMDP(GridWorldMDP):
	def __init__(self, 
				 	height=11,
				 	width=11,
				 	init_state=(1,1),
				 	gamma=0.99,
				 	slip_prob=0.05,
				 	goal_location=None,
				 	goal_value=1.0,
				 	build_walls=True,
				 	state_abstr=None 
				 ):
		super().__init__(height=height,
							width=width,
							init_state=init_state,
							gamma=gamma,
							slip_prob=slip_prob,
							goal_location=None,
							goal_value=goal_value,
							build_walls=build_walls)
		self.state_abstr = state_abstr 

	# -----------------
	# Getters & setters
	# -----------------
	def get_current_state(self):
		'''
		Overrides from parent class 

		Because this is an abstract MDP, this will return the abstract
		state corresponding to the current ground state 
		'''
		return self.state_abstr.get_abstr_from_ground(self.current_state)

	def get_abstr_from_ground(self, state):
		'''
		Get the abstr state corresponding to the given ground state 
		'''
		self.state_abstr.get_abstr_from_ground(state) 

	# -----------------
	# Main act function 
	# -----------------
	def act(self, action):
		'''
		Overrides from parent class; returns abstract state associated with 
		next_state instead of next_state directly 

		Given an action supplied by the agent, apply the action to the current state
		via the transition function, update the current state to the next state,
		and return the abstract state associated with the next state and 
		the reward gotten from that next state 

		If the agent reaches the goal state, reset to initial state

		Parameters:
			state:GridWorldState
			action:Enum

		Returns:
			next_state:
			reward:float 
		'''
		state = self.current_state
		next_state = self.transition(state, action)
		reward = self.reward(state, action, next_state)

		# Update current state to the result of the transition
		self.set_current_state(next_state)

		# If the next state is in the goal locaton, set current_state 
		# to initial state. Still returns next state  
		if self.is_goal_state(next_state):
			self.reset_to_init()

		next_state = self.state_abstr.get_abstr_from_ground(next_state)

		return next_state, reward
