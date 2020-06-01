'''
This class extends the MDP class to the classic 5x5 taxi domain. 4 possible
starting locations for the passenger, 4 possible drop-off locations

The taxi is initialized in a random location on the grid below.
l represents a wall. _ and RGBY represent valid states.

Grid:
5  R _l_ _ G
4  _ _l_ _ _
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
from GridWorld.TaxiStateClass import TaxiState
import random

class TaxiMDP(MDP):
    def __init__(self,
                 goal=None,
                 passenger_init=None,
                 gamma=0.99,
                 slip_prob=0.05):
        # Create initial state, randomly selecting valid passenger
        # and goal locations if none are provided
        rgby = [(1,1), (1,5), (4,1), (5,5)]

        # Random taxi location
        taxi_x = random.randint(1,5)
        taxi_y = random.randint(1,5)

        # Random passenger location (if none provided)
        if goal is None:
            goal = random.choice(rgby)

        if passenger_init is None:
            passenger_init = random.choice(rgby)

        super().__init__(Act, TaxiState(taxi_loc = (taxi_x, taxi_y),
                                        passenger_loc = passenger_init,
                                        goal_loc = goal), gamma)

        # MDP-specific fields, including passenger init and goal states
        # for bookkeeping
        self._slip_prob = slip_prob
        self._passenger_init = passenger_init
        self._goal = goal

    # -----------------
    # Getters & setters
    # -----------------
    def has_passenger(self):
        return self._current_state.get_passenger_loc() == (0,0)

    # --------------
    # Main functions
    # --------------
    def transition(self, state, action):
        '''
        Transition function for this MDP. Standard gridworld action,
        takes given action with probability = 1-self.slip_prob, takes
        perpindicular action with probability slip_prob/2

        Since walls are not stored as a field, this function handles
        walls. If an action bumps the taxi into a wall, position
        doesn't change.

        Walls are between (1,1)-(2,1), (1,2)-(2,2),
                            (3,1)-(4,1), (3,2)-(4,2)
                            (2,5)-(3,5), (2,4)-(3,4)
        :param state: GridWorldState
        :param action: Enum
        :return: next_state: GridWorldState
        '''

        # Handle directional actions first
        if action in [Act.UP, Act.DOWN, Act.LEFT, Act.RIGHT]:

            # Assign perpendicular action with probability slip_prob
            if random.random() < self.get_slip_prob():
                if action in [Act.UP, Act.DOWN]:
                    action = random.choice([Act.LEFT, Act.RIGHT])
                else:
                    action = random.choice([Act.UP, Act.DOWN])

            # UP
           # if action == Act.UP:



    def reward(self, state, action, next_state):
        '''
        Reward function for this MDP
        :param state: GridWorldState
        :param action: Enum
        :param next_state: GridWorldState
        :return: reward: float
        '''

    def act(self, action):
        '''
        Apply the given action to the MDPs current state, update the
        current state to the result, and return the next state and
        reward
        :param action: Enum
        :return: next_state: GridWorldState
        :return: reward: float
        '''
        # Apply the transition and reward functions
        state = self.current_state
        next_state = self.transition(state, action)
        reward = self.reward(state, action, next_state)

        # Update current state to the result of the transition
        self.set_current_state(next_state)

        # If the next state is in the goal locaton, set current_state
        # to initial state. Still returns next state
        if self.is_goal_state(next_state):
            self.reset_to_init()

        return next_state, reward


    # -------
    # Utility
    # -------
    def __str__(self):
        result = ''
        result += "Taxi at: " + str(self.get_current_state()) + '\n'
        if self.has_passenger():
            result += "Passenger in taxi" + '\n'
        else:
            result += "Passenger at: " + str(self.passenger_init) + '\n'
        result += "Drop off at " + str(self.passenger_goal)

        return result



