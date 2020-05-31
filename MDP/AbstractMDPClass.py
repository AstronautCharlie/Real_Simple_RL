'''
This class represents an abstract GridWorldMDP by combining an MDP with a 
StateAbstraction, which maps states to abstact states 
'''
from GridWorldMDPClass import GridWorldMDP 
from GridWorldStateClass import GridWorldState 
from ActionEnums import Dir 

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

	