'''
This class extends MDP to be specific to GridWorld, including a 
goal-location, specific actions, dimensions, and walls 
'''

from MDP.MDPClass import MDP
from GridWorld.ActionEnums import Dir
from GridWorld.GridWorldStateClass import GridWorldState
import random 
import math 

class GridWorldMDP(MDP):
	def __init__(self, 
				 height=11,
				 width=11,
				 init_state=(1,1),
				 gamma=0.99,
				 slip_prob=0.05,
				 goal_location=None,
				 goal_value=1.0,
				 build_walls=True
				 ):
		super().__init__(actions=list(Dir),
						 init_state=GridWorldState(init_state[0], init_state[1]),
						 gamma=gamma)
		self.height = height
		self.width = width
		self.slip_prob = slip_prob
		if goal_location is None:
			self.goal_location = [(width, height)]
		else:
			self.goal_location = goal_location
		self.goal_value = goal_value
		self.walls = []

		if build_walls:
			self.walls = self._compute_walls()

	# -----------------
	# Getters & setters 
	# -----------------
	def get_height(self):
		return self.height

	def get_width(self):
		return self.width

	def get_init_state(self):
		'''
		Return GridWorldState with data = self.init_state
		'''
		return self.init_state

	def get_current_state(self):
		return self.current_state

	def set_current_state(self, state):
		self.current_state = state

	# -----------------
	# Utility functions 
	# -----------------

	def is_goal_state(self, state):
		'''
		Checks if state is in goal location

		Parameters:
			state:GridWorldState

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

	def reset_to_init(self):
		'''
		Reset current state to initial state 
		'''
		self.set_current_state(self.get_init_state())

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
		next_state = state

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
			state:GridWorldState
			action:Enum
			next_state:GridWorldState

		Returns:
			reward:float
		'''
		if (next_state.x, next_state.y) in self.goal_location:
			return self.goal_value
		else:
			return 0.0

	# -----------------
	# Main act function
	# -----------------

	def act(self, action):
		'''
		Given an action supplied by the agent, apply the action to the current state
		via the transition function, update the current state to the next state,
		and return the next state and the reward gotten from that next state 

		If the agent reaches the goal state, reset to initial state

		Parameters:
			state:GridWorldState
			action:Enum

		Returns:
			next_state:GridWorldState
			reward:float 
		'''
		# Apply the transition and reward functions
		state = self.current_state
		next_state = self.transition(state, action)
		reward = self.reward(state, action, next_state)

		# Update current state to the result of the transition
		self.set_current_state(next_state)

		# If the next state is in the goal locaton, set current_state 
		# to initial state. Still returns next state  
		if self.is_goal_state(next_state):
			self.reset_to_init()

		return next_state, reward
