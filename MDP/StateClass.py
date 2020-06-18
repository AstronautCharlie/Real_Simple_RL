'''
This is the State class, which must be subclassed to be applied
to specific MDPs
'''

class State():
	def __init__(self, data=[], is_terminal=False):
		'''
		Data is stored as a list 
		'''
		self.data = data
		self._is_terminal = is_terminal 

	def __str__(self):
		result = str(self.data)
		return result

	# -----------------
	# Getters & Setters 
	# -----------------
	def is_terminal(self):
		return self._is_terminal

	def get_data(self):
		return self.data

	def set_terminal(self, value):
		self._is_terminal = value  