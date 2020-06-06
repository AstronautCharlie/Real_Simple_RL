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

"""NOTE""":
how to handle slip probability?
'''

from MDP.MDPClass import MDP
from GridWorld.TaxiActionEnums import Act
from GridWorld.TaxiStateClass import TaxiState
import random
from queue import Queue

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
        self.slip_prob = slip_prob
        self.passenger_init = passenger_init
        self.goal = goal

    # -----------------
    # Getters & setters
    # -----------------
    def has_passenger(self):
        return self.current_state.get_passenger_loc() == (0,0)

    def get_taxi_loc(self):
        return self.get_current_state().get_taxi_loc()

    def get_passenger_loc(self):
        return self.get_current_state().get_passenger_loc()

    def get_goal_loc(self):
        return self.get_current_state().get_goal_loc()

    # --------------
    # Main functions
    # --------------
    def transition(self, state, action):
        '''
        Transition function for this MDP. Standard gridworld action,
        takes given action with probability = 1-self.slip_prob, takes
        random action with probability self.slip_prob

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
        # These are the states where there is a wall to the right
        blocked_right = [(1,1), (1,2), (3,1), (3,2), (2,5), (2,4)]
        # These are the states where there is a wall to the left
        blocked_left = [(2,1), (2,2), (4,1), (4,2), (3,5), (3,4)]

        # Get state info for easier access
        taxi_loc = state.get_taxi_loc()
        passenger_loc = state.get_passenger_loc()
        goal_loc = state.get_goal_loc()

        # With probability self.slip_prob, pick a random action instead of
        # the selected action. Since both pickup and drop-off would, if
        # randomly chosen, land the agent in the same state, we pick
        # those with half the probability we pick a random direction
        # action
        coin_flip = random.random()
        if coin_flip < self.slip_prob:
            if coin_flip < self.slip_prob / 5:
                action = random.choice([Act.PICKUP, Act.DROPOFF])
            else:
                action = random.choice([Act.RIGHT,
                                        Act.LEFT,
                                        Act.UP,
                                        Act.DOWN])

        # If the state is terminal, then any action keeps it in the terminal state
        if state.is_terminal():
            return state

        # Handle directional actions first
        if action in [Act.UP, Act.DOWN, Act.LEFT, Act.RIGHT]:

            # If action is up or down, walls aren't a problem. Move the
            # state taxi if it isn't running into a border, and leave
            # it unchanged if it is.
            if action == Act.UP:
                if taxi_loc[1] < 5:
                    return TaxiState((taxi_loc[0], taxi_loc[1] + 1),
                                      passenger_loc,
                                      goal_loc)
                else:
                    return state
            if action == Act.DOWN:
                if taxi_loc[1] > 1:
                    return TaxiState((taxi_loc[0], taxi_loc[1] - 1),
                                      passenger_loc,
                                      goal_loc)
                else:
                    return state

            # If action is left or right, we have to check the walls as
            # well as the border. If it runs into a wall or border,
            # leave taxi position unchanged.
            if action == Act.RIGHT:
                if taxi_loc[0] < 5 and taxi_loc not in blocked_right:
                    return TaxiState((taxi_loc[0] + 1, taxi_loc[1]),
                                     passenger_loc,
                                     goal_loc)
                else:
                    return state

            if action == Act.LEFT:
                if taxi_loc[0] > 1 and taxi_loc not in blocked_left:
                    return TaxiState((taxi_loc[0] - 1, taxi_loc[1]),
                                     passenger_loc,
                                     goal_loc)
                else:
                    return state
        # Handle passenger pick-up and drop-off
        else:
            # If the taxi is in the same location as the passenger,
            # move the passenger to (0,0) (the "in the taxi" state)
            # Else, no change occurs
            if action == Act.PICKUP:
                if taxi_loc == passenger_loc:
                    return TaxiState(taxi_loc,
                                     (0,0),
                                     goal_loc)
                else:
                    return state

            # If taxi has the passenger and is at the goal location,
            # drop off the passenger and make next state terminal
            # Otherwise, no change
            elif action == Act.DROPOFF:
                if passenger_loc == (0,0) and taxi_loc == goal_loc:
                    return TaxiState(taxi_loc,
                                     goal_loc,
                                     goal_loc,
                                     is_terminal=True)
                else:
                    return state

            else:
                ValueError('Illegal action argument ' + str(action))



    def reward(self, state, action, next_state):
        '''
        Reward is -1 for all actions except successful drop off of
        passenger (+20 instead) or illegally trying to pick up or drop
        off passenger (-10 instead). Illegal drop off defined as performing
        the drop off action anywhere but the goal; illegal pick up
        defined as performing the pick up action anywhere but the
        passenger location, (includes the case when the passenger is
        already in the car)
        :param state: TaxiState
        :param action: Enum
        :param next_state: TaxiState
        :return: reward: float
        '''
        taxi_loc = state.get_taxi_loc()
        passenger_loc = state.get_passenger_loc()
        goal_loc = state.get_goal_loc()

        next_passenger = next_state.get_passenger_loc()
        next_goal = next_state.get_goal_loc()

        if state.is_terminal():
            return 0.0

        # Handle case of drop-off
        if action == Act.DROPOFF:
            # If passenger is aboard, the taxi is in the goal location, and the passenger gets to the
            # the goal, return +20. Otherwise, -10 for illegal drop off
            if passenger_loc == (0, 0) and taxi_loc == goal_loc and next_passenger == next_goal:
                return 20.0
            else:
                return -10.0

        # Handle case of pick-up
        elif action == Act.PICKUP:
            # If taxi is in the same location as passenger, return
            # normal -1. Otherwise, -10 for illegal pickup
            if passenger_loc == taxi_loc:
                return -1.0
            else:
                return -10.0

        # In all other cases, normal -1.0
        else:
            return -1.0


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

        if state.is_terminal():
            self.reset_to_init()

        return next_state, reward

    # ----------------------------
    # Inherited abstract functions
    # ----------------------------
    def next_possible_states(self, state, action):
        """
        Given a state, return a dictionary containing all states that can result from taking the given action on it along
        with the probability of ending up in that state
        :param state: TaxiState
        :param action: Enum
        :return: dictionary of State -> Float (probability, so needs to be less than one)
        """
        # If the state is terminal, any action it takes keeps it in its current state,
        # so no change happens
        if state.is_terminal():
            return {state: 1.0}

        # State information
        taxi_loc = state.get_taxi_loc()
        passenger_loc = state.get_passenger_loc()
        goal_loc = state.get_goal_loc()

        # These are the states where there is a wall to the right
        blocked_right = [(1,1), (1,2), (3,1), (3,2), (2,5), (2,4)]
        # These are the states where there is a wall to the left
        blocked_left = [(2,1), (2,2), (4,1), (4,2), (3,5), (3,4)]

        # Booleans indicating whether or not the taxi can move in the given directions
        can_left = taxi_loc[0] > 1 and taxi_loc not in blocked_left
        can_right = taxi_loc[0] < 5 and taxi_loc not in blocked_right
        can_up = taxi_loc[1] < 5
        can_down = taxi_loc[1] > 1
        # Note this doesn't cover the terminal state. Must be handled elsewhere
        can_pickup = taxi_loc == passenger_loc
        can_dropoff = taxi_loc == goal_loc and passenger_loc == (0,0)

        # Define next states after each action is taken

        # Next state after left. If the taxi can move left, the next state
        # will be one to the left of it. Otherwise, the next state
        # will be the state it's already in
        if can_left:
            next_left = TaxiState((taxi_loc[0] - 1, taxi_loc[1]),
                                  passenger_loc,
                                  goal_loc)
        else:
            next_left = state

        # Next state after right
        if can_right:
            next_right = TaxiState((taxi_loc[0] + 1, taxi_loc[1]),
                                   passenger_loc,
                                   goal_loc)
        else:
            next_right = state

        # Next state after up
        if can_up:
            next_up = TaxiState((taxi_loc[0], taxi_loc[1] + 1),
                                passenger_loc,
                                goal_loc)
        else:
            next_up = state

        # Next state after down
        if can_down:
            next_down = TaxiState((taxi_loc[0], taxi_loc[1] - 1),
                                  passenger_loc,
                                  goal_loc)
        else:
            next_down = state

        # Next state after pickup. NOTE: this does not handle the terminal case. Must be handled
        # elsewhere
        if can_pickup:
            next_pickup = TaxiState(taxi_loc,
                                    (0,0),
                                    goal_loc)
        else:
            next_pickup = state


        # Next state after pickup. If the Taxi can drop off the passenger, then the next
        # state is the goal location and the result should be terminal
        if can_dropoff:
            next_dropoff = TaxiState(taxi_loc,
                                     passenger_loc=taxi_loc,
                                     goal_loc=goal_loc,
                                     is_terminal=True)
        else:
            next_dropoff = state

        # This is the final result
        next_states = {next_left: 0.0,
                       next_right: 0.0,
                       next_up: 0.0,
                       next_down: 0.0,
                       next_pickup: 0.0,
                       next_dropoff: 0.0}

        # Each action has (slip_probability / 6) probability of being
        # selected. Doing += so that if two of the next_ states are the
        # same, the probability is added
        eps_soft_prob = self.slip_prob / 5
        next_states[next_left] += eps_soft_prob
        next_states[next_right] += eps_soft_prob
        next_states[next_up] += eps_soft_prob
        next_states[next_down] += eps_soft_prob
        next_states[next_pickup] += eps_soft_prob / 2
        next_states[next_dropoff] += eps_soft_prob / 2

        # The action actually selected being performed is 1-slip_probability
        if action == Act.LEFT:
            next_states[next_left] += 1 - self.slip_prob
        elif action == Act.RIGHT:
            next_states[next_right] += 1 - self.slip_prob
        elif action == Act.UP:
            next_states[next_up] += 1 - self.slip_prob
        elif action == Act.DOWN:
            next_states[next_down] += 1 - self.slip_prob
        elif action == Act.PICKUP:
            next_states[next_pickup] += 1 - self.slip_prob
        elif action == Act.DROPOFF:
            next_states[next_dropoff] += 1 - self.slip_prob
        else:
            raise ValueError('Illegal action argument:', action)

        return next_states

    def get_all_possible_states(self):
        """
        Returns a list containing all the possible states in this MDP
        :return: List of States
        """
        possible_states = []
        rgby = [(1, 1), (1, 5), (4, 1), (5, 5)]
        for x in range(1, 6):
            for y in range(1, 6):
                for passenger in rgby + [(0, 0)]:
                    for goal in rgby:
                        state = TaxiState((x, y),
                                          passenger,
                                          goal)
                        possible_states.append(state)
        return possible_states



    # -------
    # Utility
    # -------
    def __str__(self):
        result = str(self.get_current_state())
        return result

    def is_goal_state(self, curr_state):
        '''
        If in the current state the passenger is in the taxi and the
        taxi_loc is in the goal state, and the action is the drop off
        action, then the goal has been achieved and return True,
        otherwise return False

        Note that we can't simply check if the passenger is in the
        goal location because it's possible that the passenger spawned
        in the goal location. In that case, the taxi still has to pick
        up and drop off the passenger
        :param curr_state: TaxiState
        :param action: Enum
        :return: boolean
        '''
        taxi_loc = curr_state.get_taxi_loc()
        passenger_loc = curr_state.get_passenger_loc()
        goal_loc = curr_state.get_goal_loc()

        return taxi_loc == goal_loc and passenger_loc == (0,0) and curr_state.is_terminal()

    def reset_to_init(self):
        rgby = [(1,1), (1,5), (4,1), (5,5)]
        passenger_init = random.choice(rgby)
        goal = random.choice(rgby)
        taxi_x = random.randint(1,5)
        taxi_y = random.randint(1,5)
        self.set_current_state(TaxiState((taxi_x, taxi_y),
                                         passenger_init,
                                         goal))



