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
		if state in self.abstr_dict.keys():
			return self.abstr_dict[state]
		else:
			return state 