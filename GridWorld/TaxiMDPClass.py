'''
This class extends the MDP class to the classic 5x5 taxi domain. 4 possible
starting locations for the passenger, 4 possible drop-off locations

The taxi is initialized in a random location on the grid below.
l represents a wall. _ and RGBY represent valid states.

Grid:
5  R _l_ _ G
4  _ _ _ _ _
3  _ _ _ _ _
2  _l_ _l_ _
1  Yl_ _lB _
   1 2 3 4 5

The passenger is randomly initialized in one of the R, G, B or Y and
wants to go to one of those states (possibly being the state in which
it started)
'''

from MDP.MDPClass import MDP
from GridWorld.GridWorldStateClass import GridWorldState
from GridWorld.TaxiActionEnums import Act
import random

class TaxiMDP(MDP):
    def __init__(self,
                 passenger_goal=None,
                 passenger_init=None,
                 gamma=0.99):
        # Calculate random location for taxi to
        start_x = random.randint(1,5)
        start_y = random.randint(1,5)
        super().__init__(actions=Act,
                         init_state=GridWorldState(start_x, start_y),
                         gamma=gamma)
        # Possible starting and drop-off points for passenger
        rgby = [(1,1), (1,5), (1,4), (5,5)]

        # Passenger goal and init data
        if passenger_goal is None:
            passenger_goal = random.choice(rgby)
        self.passenger_goal = passenger_goal

        if passenger_init is None:
            passenger_init = random.choice(rgby)
        self.passenger_init = passenger_init

        self._has_passenger = False

    def __str__(self):
        print("Taxi at:", self.get_current_state())
        if self.has_passenger():
            print("Passenger in taxi")
        else:
            print("Passenger at:", self.passenger_init)
        print("Drop off at", self.passenger_goal)

        print()

    # -----------------
    # Getters & setters
    # -----------------
    def has_passenger(self):
        return self._has_passenger




