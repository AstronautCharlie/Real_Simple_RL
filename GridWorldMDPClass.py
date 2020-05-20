'''
This class extends MDP to be specific to GridWorld, including a 
goal-location, specific actions, dimensions, and walls 
'''

from MDPClass import MDP 
from ActionEnums import Dir
from GridWorldStateClass import GridWorldState
from Functions import GridWorldTransition, GridWorldReward 

class GridWorldMDP(MDP):
	def __init__(self, 
				 #transition_func,
				 #reward_func,
				 height=11,
				 width=11,
				 init_state=(1,1),
				 gamma=0.99,
				 slip_prob=0.05,
				 goal_location=[(0,0)],
				 goal_value=1.0
				 ):
		'''
		Do we want to explicitly set the walls or have functions
		to calculate the walls? 
		'''
		transition_func = GridWorldTransition(slip_prob)
		reward_func = GridWorldReward(goal_value)

		super().__init__(actions=Dir, 
			   			 transition_func=transition_func,
			   			 reward_func=reward_func,
			   			 init_state=GridWorldState(init_state[0], init_state[1]),
			   			 gamma=gamma)
		self.height=height
		self.width=width
		self.goal_location=goal_location
		self.walls=[]

	# -------------------------------
	# Transition and reward functions 
	# -------------------------------

	def apply_transition(self, state, action):
		return self.transition_func(state, action, self)

	def get_reward(self, state, action, next_state):
		return self.reward_func(state, action, next_state, self)

	def act(self, state, action):
		'''
		Given a state and an action (both supplied by the Agent), return the 
		next state and the reward gotten from that next state 

		Parameters:
			state:State
			action:Enum

		Returns:
			next_state:State
			reward:float 
		'''
		next_state = self.apply_transition(state, action)
		reward = self.get_reward(state, action, next_state)

		return next_state, reward 