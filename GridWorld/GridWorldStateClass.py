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

	def __eq__(self, other):
		return isinstance(other, GridWorldState) and self.x == other.x and self.y == other.y
