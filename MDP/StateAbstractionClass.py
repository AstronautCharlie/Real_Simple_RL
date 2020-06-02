'''
This class represents a state abstraction from set of ground
states to a set of abstract states
'''
class StateAbstraction():
	def __init__(self, abstr_dict=None):#, ground_states=[]):
		'''
		Parameters:
			abstr_dict:dictionary(States:States)
			ground_states:list of states 

		The abstr_dict is a mapping of the ground states to abstract
		states. If no abstr_dict is provided, assume the trivial 
		abstraction of mapping each state to itself 
		'''
		#if abstr_dict is None:
		#	abstr_dict = {state: state for state in ground_states}
		self.abstr_dict = abstr_dict 

	def get_abstr_from_ground(self, state):
		'''
		Parameters:
			state:State

		returns:
			abstr_state, the abstract state corresponding to this 
			ground state 

		Get the abstract state corresponding to the given state. 
		If the given state does not occur in abstr_dict, return
		the state itself 
		'''
		if self.abstr_dict is not None and state in self.abstr_dict.keys():
			return self.abstr_dict[state]
		else:
			return state 

	def __str__(self):
		abstr_states_temp = list(self.abstr_dict.values())

		abstr_states = [] 
		for state in abstr_states_temp:
			if state not in abstr_states:
				abstr_states.append(state)

		result = "" 

		abstr_states.sort()
		for state in abstr_states:
			for key in self.abstr_dict.keys():
				if self.abstr_dict[key] == state:
					result += 'ground -> abstr: ' + str(key) + ' -> ' + str(self.abstr_dict[key])
					result += '\n'
		return result 
		'''
		for state in abstr_dict.keys():
			if 
		result = "" 
		for key in self.abstr_dict.keys():
			result += "ground -> abstr: " + str(key) + ' -> ' + str(self.abstr_dict[key])
			result += "\n"
		return result
		'''