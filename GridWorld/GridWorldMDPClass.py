'''
This class extends MDP to be specific to GridWorld, including a 
goal-location, specific actions, dimensions, and walls 
'''

from MDP.MDPClass import MDP
from GridWorld.ActionEnums import Dir
from GridWorld.GridWorldStateClass import GridWorldState
#from GridWorld.GridWorldFunctions import GridWorldTransition, GridWorldReward
import random 
import math 

class GridWorldMDP(MDP):
	def __init__(self, 
				 #transition_func,
				 #reward_func,
				 height=11,
				 width=11,
				 init_state=(1,1),
				 gamma=0.99,
				 slip_prob=0.05,
				 goal_location=[(11,11)],
				 goal_value=1.0,
				 build_walls=True
				 ):
		'''
		Do we want to explicitly set the walls or have functions
		to calculate the walls? 
		'''
		#transition_func = GridWorldTransition(slip_prob)
		#reward_func = GridWorldReward(goal_value)

		super().__init__(actions=Dir, 
						 #transition_func=transition_func,
						 #reward_func=reward_func,
						 init_state=GridWorldState(init_state[0], init_state[1]),
						 gamma=gamma)
		self.height = height
		self.width = width
		self.slip_prob = slip_prob
		self.goal_location = goal_location
		self.goal_value = goal_value
		self.walls = []

		if build_walls:
			self.walls = self._compute_walls()

	# -----------------
	# Utility functions 
	# -----------------

	def is_goal_state(self, state):
		'''
		Checks if state is in goal location

		Parameters:
			state:State

		Returns:
			boolean
		'''
		return (state.x, state.y) in self.goal_location

	def _compute_walls(self):
		'''
		Calculate the locations of walls; taken from David Abel's
		simple_rl package 
		'''
		walls = [] 

		half_width = math.ceil(self.width / 2)
		half_height = math.ceil(self.height / 2)

		# Calculate left-to-right walls 
		for i in range(1, self.width+1):
			if i == half_width:
				half_height -= 1 
			if i+1 == math.ceil(self.width / 3) or i == math.ceil((2 * self.width) / 3):
				continue
			walls.append((i, half_height))

		# Calculate top-to-bottom walls 
		for j in range(1, self.height+1):
			if j+1 == math.ceil(self.height / 3) or j == math.ceil((2 * self.height) / 3):
				continue
			walls.append((half_width, j))

		return walls 

	# -------------------------------
	# Transition and reward functions 
	# -------------------------------

	def transition(self, state, action):
		'''
		Parameters:
			state:GridWorldState
			action:Enum
			mdp:GridWorldMDP

		Returns:
			state:GridWorldState
		'''
		print("transitioning from state", str(state), 
				"with action", action)
		next_state = state
		# If terminal, do nothing
		if state.is_terminal():
			return self.init_state

		# Apply slip probability and change action if applicable
		if random.random() < self.slip_prob:
			if action in [Dir.UP, Dir.DOWN]:
				action = random.choice([Dir.LEFT, Dir.RIGHT])
			elif action in [Dir.LEFT, Dir.RIGHT]:
				action = random.choice([Dir.UP, Dir.DOWN])

		# Calculate next state based on action
		if action == Dir.UP and state.y < self.height and (state.x, state.y + 1) not in self.walls:
			next_state = GridWorldState(state.x, state.y + 1)
		if action == Dir.DOWN and state.y > 1 and (state.x, state.y - 1) not in self.walls:
			next_state = GridWorldState(state.x, state.y - 1)
		if action == Dir.LEFT and state.x > 1 and (state.x - 1, state.y) not in self.walls:
			next_state = GridWorldState(state.x - 1, state.y)
		if action == Dir.RIGHT and state.x < self.width and (state.x + 1, state.y) not in self.walls:
			next_state = GridWorldState(state.x + 1, state.y)

		# If the next state takes the agent into the goal location, 
		# return initial state 
		if (next_state.x, next_state.y) in self.goal_location:
			next_state.set_terminal(True)

		return next_state

	def reward(self, state, action, next_state):
		'''
		Parameters:
			state:State
			action:Enum
			next_state:State

		Returns:
			reward:float
		'''
		if (next_state.x, next_state.y) in self.goal_location:
			return self.goal_value
		else:
			return 0.0

	#def apply_transition(self, state, action):
		'''
		Query self's transition function for the result of the given 
		state/value pair 

		Parameters:
			state:State
			action:Enum
		'''
		#return self.transition_func(state, action, self)

	#def get_reward(self, state, action, next_state):
		'''
		Query self's reward function for the result of the given
		state/value/next-state tuple 

		Parameters:
			state:State
			action:Enum
			next_state:State
		'''
		#return self.reward_func(state, action, next_state, self)

	# -----------------
	# Main act function
	# -----------------

	def act(self, state, action):
		'''
		Given a state and an action (both supplied by the Agent), return the 
		next state and the reward gotten from that next state 

		If the agent reaches the goal state, reset to initial state

		Parameters:
			state:State
			action:Enum

		Returns:
			next_state:State
			reward:float 
		'''
		# Apply the transition and reward functions
		next_state = self.transition(state, action)
		reward = self.reward(state, action, next_state)

		# If the next state is in the goal locaton, set next_state
		# to initial state and return that 
		#if self.is_goal_state(next_state):
		#	next_state = self.init_state

		return next_state, reward 