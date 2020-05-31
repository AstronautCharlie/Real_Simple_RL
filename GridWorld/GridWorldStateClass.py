'''
This class extends the State class to be specific to the
GridWorldMDP
'''

from MDP.StateClass import State

class GridWorldState(State):
	def __init__(self, x, y, is_terminal=False):
		'''
		Represents state as grid point (x,y) with (0,0) being
		the lower-left hand corner 

		Parameters:
			x:int 
			y:int
		'''
		super().__init__([x,y], is_terminal)
		self.x = x
		self.y = y

	def __str__(self):
		return '(' + str(self.x) + ',' + str(self.y) + ')'

	def __hash__(self):
		return hash(tuple(self.data))

	def __eq__(self, other):
		return isinstance(other, GridWorldState) and self.x == other.x and self.y == other.y

	def __lt__(self, other):
		'''
		Required to for numpy.unique() to work. Arbitrarily defined
		as that one state is 'less' than another if the x coordinate
		is smaller or if the x coordinates are the same and the y 
		coordinate is smaller 
		'''
		return self.x < other.x or (self.x == other.x and self.y < other.y)
	