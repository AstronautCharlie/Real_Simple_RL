'''
Extend MDP class to a 10x10 taxi domain.

10 R _ _|_ _ G _ _|C _
9  _ _ _|_ _ _ _ _|_ _
8  _ _ _|_ _ _|_ _|_ _
7  _ _ _|W _ _|_ _|_ _
6  _ _ _ _ _ _|M _ _ _
5  _ _ _ _ _ _|_ _ _ _
4  _|_ _ _|_ _ _ _|_ _
3  _|_ _ _|_ _ _ _|_ _
2  Y|_ _ _|_ _ _ _|_ _
1  _|_ _ _|B _ _ _|_ P
   1 2 3 4 5 6 7 8 9 10
'''
'''
blocked_right = [(1,1), (1,2), (1,3), (1,4),
                 (3,7), (3,8), (3,9), (3,10),
                 (4,1), (4,2), (4,3), (4,4),
                 (6,5), (6,6), (6,7), (6,8),
                 (8,1), (8,2), (8,3), (8,4),
                 (8,7), (8,8), (8,9), (8,10)]

blocked_left = [(tup[0]+1, tup[1]) for tup in blocked_right]
'''
blocked_right = []
blocked_left = []
starter_goals = [(1, 2), (1, 10), (4, 7), (5, 1), (6, 10), (7, 6), (9, 10), (10, 1)]

from MDP.MDPClass import MDP
from GridWorld.TaxiStateClass import TaxiState
from GridWorld.TaxiActionEnums import Act
import random

class LargeTaxiMDP(MDP):
    def __init__(self,
                 goal=None,
                 passenger_init=None,
                 gamma=0.99,
                 slip_prob=0.0,
                 same_goal=False):
        #goals = [(1,2), (1,10), (4,7), (5,1), (6,10), (7,6), (9,10), (10,1)]
        self.same_goal = same_goal
        if self.same_goal:
            goals = [(10,1)]
        else:
            goals = starter_goals

        self.goals = goals

        taxi_x = random.randint(1,10)
        taxi_y = random.randint(1,10)

        if passenger_init is None:
            passenger_init = random.choice(goals)

        super().__init__(Act, TaxiState(taxi_loc=(taxi_x, taxi_y),
                                        passenger_loc=passenger_init,
                                        goal_loc=goal,
                                        goals=self.goals,
                                        passenger_locs=starter_goals), gamma)

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
        Return next state for given state and action. Applies splip probability
        '''

        # If state is terminal, return immediately
        if state.is_terminal():
            return state

        taxi_loc = state.get_taxi_loc()
        passenger_loc = state.get_passenger_loc()
        goal_loc = state.get_goal_loc()

        # Apply slip probability. Probability of randomly selecting pickup/dropoff
        #  is half that of randomly selecting movement
        coin_flip = random.random()
        if coin_flip < self.slip_prob:
            if coin_flip < self.slip_prob / 5:
                action = random.choice([Act.PICKUP, Act.DROPOFF])
            else:
                action = random.choice([Act.RIGHT,
                                        Act.LEFT,
                                        Act.UP,
                                        Act.DOWN])

        # Handle individual actions
        if action == Act.UP and taxi_loc[1] < 10:
            return TaxiState((taxi_loc[0], taxi_loc[1]+1),
                              passenger_loc=passenger_loc,
                              goal_loc=goal_loc,
                              goals=self.goals,
                             passenger_locs=starter_goals)
        elif action == Act.DOWN and taxi_loc[1] > 1:
            return TaxiState((taxi_loc[0], taxi_loc[1]-1),
                              passenger_loc=passenger_loc,
                              goal_loc=goal_loc,
                              goals=self.goals,
                             passenger_locs=starter_goals)
        elif action == Act.RIGHT and taxi_loc[0] < 10 and taxi_loc not in blocked_right:
            return TaxiState((taxi_loc[0]+1, taxi_loc[1]),
                              passenger_loc=passenger_loc,
                              goal_loc=goal_loc,
                              goals=self.goals,
                             passenger_locs=starter_goals)
        elif action == Act.LEFT and taxi_loc[0] > 1 and taxi_loc not in blocked_left:
            return TaxiState((taxi_loc[0]-1, taxi_loc[1]),
                              passenger_loc=passenger_loc,
                              goal_loc=goal_loc,
                              goals=self.goals,
                             passenger_locs=starter_goals)
        elif action == Act.PICKUP and taxi_loc == passenger_loc:
            return TaxiState(taxi_loc,
                             passenger_loc=(0,0),
                             goal_loc=goal_loc,
                             goals=self.goals,
                             passenger_locs=starter_goals)
        elif action == Act.DROPOFF and passenger_loc == (0,0) and taxi_loc == goal_loc:
            return TaxiState(taxi_loc, passenger_loc=goal_loc, goal_loc=goal_loc, is_terminal=True, goals=self.goals,
                             passenger_locs=starter_goals)
        else:
            return state

    def reward(self, state, action, next_state):
        '''
        -1 for all actions except successful drop-off of passenger (+20)
        or illegally trying to pick up or drop off (-10)
        '''
        taxi_loc = state.get_taxi_loc()
        passenger_loc = state.get_passenger_loc()
        goal_loc = state.get_goal_loc()

        next_passenger = next_state.get_passenger_loc()
        next_goal = next_state.get_goal_loc()

        if state.is_terminal():
            return 0.0

        # Handle drop-off case
        if action == Act.DROPOFF:
            if passenger_loc == (0,0) and taxi_loc == goal_loc and next_passenger == next_goal:
                return 20.0
            else:
                return -10.0
        # Handle case of pickup
        elif action == Act.PICKUP:
            # Normal -1 for valid pickup, -10 otherwise
            if passenger_loc == taxi_loc:
                #return -1.0
                return 0.0
            else:
                return -10.0

        # Handle all other cases
        #return -1.0
        return 0.0

    def act(self, action):
        """
        Apply given action to MDP's current state, update current state to
        result, and return next state and reward
        """
        # Apply transition/reward functions
        state = self.current_state
        next_state = self.transition(state, action)
        reward = self.reward(state, action, next_state)

        # Update current state to result
        self.set_current_state(next_state)

        # If next state is goal location, reset MDP
        if state.is_terminal():
            self.reset_to_init()

        return next_state, reward

    # ----------------------------
    # Inherited abstract functions
    # ----------------------------
    def get_next_possible_states(self, state, action):
        """
        Return a dictionary mapping possible next states to probability
        of reaching that next state given state and action

        Needed to run VI
        """
        if state.is_terminal():
            return {state: 1.0}

        taxi_loc = state.get_taxi_loc()
        passenger_loc = state.get_passenger_loc()
        goal_loc = state.get_goal_loc()

        # Flags indicating available movement
        can_left = taxi_loc[0] > 1 and taxi_loc not in blocked_left
        can_right = taxi_loc[0] < 10 and taxi_loc not in blocked_right
        can_up = taxi_loc[1] < 10
        can_down = taxi_loc[1] > 1
        can_pickup = taxi_loc == passenger_loc
        can_dropoff = taxi_loc == goal_loc and passenger_loc == (0,0)

        # Define next states after each action
        if can_left:
            next_left = TaxiState((taxi_loc[0]-1, taxi_loc[1]),
                                  passenger_loc=passenger_loc,
                                  goal_loc=goal_loc,
                                  goals=self.goals,
                                  passenger_locs=starter_goals)
        else:
            next_left = state

        if can_right:
            next_right = TaxiState((taxi_loc[0]+1, taxi_loc[1]),
                                   passenger_loc=passenger_loc,
                                   goal_loc=goal_loc,
                                   goals=self.goals,
                                   passenger_locs=starter_goals)
        else:
            next_right = state

        if can_up:
            next_up = TaxiState((taxi_loc[0], taxi_loc[1]+1),
                                passenger_loc=passenger_loc,
                                goal_loc=goal_loc,
                                goals=self.goals,
                                passenger_locs=starter_goals)
        else:
            next_up = state

        if can_down:
            next_down = TaxiState((taxi_loc[0], taxi_loc[1]-1),
                                  passenger_loc=passenger_loc,
                                  goal_loc=goal_loc,
                                  goals=self.goals,
                                  passenger_locs=starter_goals)
        else:
            next_down = state

        if can_pickup:
            next_pickup = TaxiState(taxi_loc, passenger_loc=(0,0), goal_loc=goal_loc, goals=self.goals,
                                    passenger_locs=starter_goals)
        else:
            next_pickup = state

        if can_dropoff:
            next_dropoff = TaxiState(taxi_loc,
                                     passenger_loc=taxi_loc,
                                     goal_loc=goal_loc,
                                     is_terminal=True,
                                     goals=self.goals,
                                     passenger_locs=starter_goals)
        else:
            next_dropoff = state

        # This will hold final results
        next_states = {next_left: 0.0,
                       next_right: 0.0,
                       next_up: 0.0,
                       next_down: 0.0,
                       next_pickup: 0.0,
                       next_dropoff: 0.0}

        # Each action has slip_prob/6 probability of being selected
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
        goal_states = starter_goals
        if self.same_goal:
            goal_states = [(10,1)]
        for x in range(1, 11):
            for y in range(1, 11):
                for passenger in starter_goals + [(0, 0)]:
                    for goal in goal_states:
                        state = TaxiState((x, y),
                                          passenger_loc=passenger,
                                          goal_loc=goal,
                                          goals=self.goals,
                                          passenger_locs=starter_goals)
                        possible_states.append(state)
                        # Add possible goal state
                        if (x,y) == goal and (x,y) == passenger:
                            goal_state = TaxiState((x,y), passenger_loc=passenger, goal_loc=goal, is_terminal=True, goals=self.goals,
                                                   passenger_locs=starter_goals)
                            possible_states.append(goal_state)
        return possible_states

    # -------
    # Utility
    # -------
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
        '''
        taxi_loc = curr_state.get_taxi_loc()
        passenger_loc = curr_state.get_passenger_loc()
        goal_loc = curr_state.get_goal_loc()

        return taxi_loc == goal_loc and passenger_loc == (0,0) and curr_state.is_terminal()

    def reset_to_init(self):
        passenger_init = random.choice(starter_goals)
        if not self.same_goal:
            goal = random.choice(self.goals)
        else:
            goal = (10,1)
        taxi_x = random.randint(1,10)
        taxi_y = random.randint(1,10)
        self.set_current_state(TaxiState((taxi_x, taxi_y),
                                         passenger_loc=passenger_init,
                                         goal_loc=goal,
                                         goals=self.goals,
                                         passenger_locs=starter_goals))

    def copy(self):
        copy = LargeTaxiMDP(self.goal,
                       self.passenger_init,
                       self.gamma,
                       self.slip_prob,
                        same_goal=self.same_goal)

        return copy


# For testing purposes only
if __name__ == '__main__':
    print(blocked_right)
    print(blocked_left)

    mdp = LargeTaxiMDP()
    states = mdp.get_all_possible_states()
    for state in states[:100]:
        for action in mdp.actions:
            next_states = mdp.get_next_possible_states(state, action)
            for key, value in next_states.items():
                if value > 0:
                    print(state, action, key, value)