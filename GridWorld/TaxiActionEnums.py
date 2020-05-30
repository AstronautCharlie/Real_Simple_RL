'''
This enum lists the number of actions that can be
taken in the Taxi MDP
'''

from enum import Enum

class Act(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    PICKUP = 5
    DROPOFF = 6
