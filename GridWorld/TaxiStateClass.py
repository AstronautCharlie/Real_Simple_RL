'''
This extends the State class to be specific to the Taxi environment
by adding a 'has_passenger' flag to indicate whether or not the
passenger is aboard
'''

from MDP.StateClass import State
import random


class TaxiState(State):
    '''
    Taxi_loc, passenger_loc, and goal_loc are all assumed to be tuples
    where the first element is the x coordinate and the second element
    is the y coordinate (assuming they are not null)

    passenger_loc can be in [(1,1), (1,5), (4,1), (5,5) and (0,0)],
    where the last state indicates that the passenger is in the car.
    This was done to avoid having to update the passenger_position along
    with with the taxi location when the passenger is in the taxi 
    '''

    def __init__(self,
                 taxi_loc,
                 passenger_loc=None,
                 goal_loc=None,
                 is_terminal=False,
                 goals=None,
                 passenger_locs=None):
        '''
        taxi_loc, passenger_loc, goal_loc are assumed to be tuples if
        they are not None. If any one is None, initialize a random
        location from the valid options
        :param taxi_loc: array-like
        :param passenger_loc: array-like
        :param goal_loc: array-like
        :param is_terminal: boolean
        '''

        if goals is None:
            goals = [(1, 1), (1, 5), (4, 1), (5, 5)]
        if passenger_locs is None:
            passenger_locs = goals
        '''  
        if taxi_loc is None:
            taxi_x = random.randint(1, 5)
            taxi_y = random.randint(1, 5)
            taxi_loc = (taxi_x, taxi_y)
        else:
            if taxi_loc[0] not in range(1, 6) or taxi_loc[1] not in range(1, 6):
                raise ValueError('Taxi location ' + str(taxi_loc) + ' not in valid range')
            
            self._taxi_loc = (taxi_loc[0], taxi_loc[1])
        '''
        self._taxi_loc = (taxi_loc[0], taxi_loc[1])

        # Check passenger location validity, picking a random valid
        # choice if no input is given
        if passenger_loc is None:
            passenger_loc = random.choice(passenger_locs)

        '''
        if passenger_loc not in goals and (passenger_loc[0] != 0 or passenger_loc[1] != 0):
            raise ValueError('Passenger location ' + str(passenger_loc) + ' is an invalid passenger location')
        else:
        '''
        self._passenger_loc = passenger_loc

        # Check goal location validity, picking a random valid choice
        # if no input is given
        if goal_loc is None:
            goal_loc = random.choice(goals)

        if goal_loc not in goals:
            raise ValueError('Goal location ' + str(goal_loc) + ' is not a valid goal location')
        else:
            self._goal_loc = goal_loc

        super().__init__([taxi_loc, passenger_loc, goal_loc], is_terminal)

    def __str__(self):
        """
        Represent a TaxiMDP state as a tuple of (Taxi loc, passenger loc, goal loc)
        with a tag indicating terminal if it is a terminal state
        """
        #result = 'Taxi, Passenger, Goal : '
        result = str(self.get_taxi_loc()) + ' '
        result += str(self.get_passenger_loc()) + ' '
        result += str(self.get_goal_loc())
        if self.is_terminal():
            result += '; terminal'
        return result

    def __eq__(self, other):
        '''
        States are equivalent if taxi locations, passenger locations, and
        goal locations are all equivalent
        :param other: TaxiState
        :return: boolean
        '''
        #return self.get_taxi_loc() == other.get_taxi_loc() and \
        #       self.get_passenger_loc() == other.get_passenger_loc() and \
        #       self.get_goal_loc() == other.get_goal_loc() and self.is_terminal() == other.is_terminal()
        return self.data == other.data and self.is_terminal() == other.is_terminal()

    def __hash__(self):
        return hash(tuple(self.data))

    # -----------------
    # Getters & setters
    # -----------------
    def get_taxi_loc(self):
        return self._taxi_loc[0], self._taxi_loc[1]

    def get_passenger_loc(self):
        return self._passenger_loc

    def set_passenger_loc(self, new_loc):
        rgby = [(1, 1), (1, 5), (4, 1), (5, 5)]

        if new_loc not in rgby and (new_loc[0] != 0 or new_loc[1] != 0):
            raise ValueError(new_loc + ' is not a valid passenger location')

    def get_goal_loc(self):
        return self._goal_loc
