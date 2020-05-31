'''
This Enum dictates the types of state abstraction supported 
by AbstractionMaker 
'''
from enum import Enum 

class Abstr_type(Enum):
	Q_STAR = 1
	A_STAR = 2 
	PI_STAR = 3