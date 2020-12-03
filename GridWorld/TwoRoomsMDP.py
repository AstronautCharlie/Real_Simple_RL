"""
This file defines an MDP consisting of 2 rooms of variable sizes, divided by a wall with a 'hallway' square (or
squares)
"""
from MDP.MDPClass import MDP
from GridWorld.GridWorldStateClass import GridWorldState
from GridWorld.ActionEnums import Dir
import random

class TwoRoomsMDP(MDP):
    def __init__(self,
                 upper_width=5,
                 upper_height=5,
                 lower_width=5,
                 lower_height=5,
                 upper_offset=0,
                 lower_offset=0,
                 init_state=(1,1),
                 goal_location=None,
                 slip_prob=0.0,
                 goal_value=1.0,
                 hallway_states=[3],
                 hallway_height=1,
                 gamma=0.99):
        """
        :param upper_width: width (x-coordinate) of upper room
        :param upper_height: height (y-coordinate) of upper room
        :param lower_width: width of lower room
        :param lower_height: height of upper room
        :param upper_offset: shift upper room to the right by this value
        :param lower_offset: shift lower room to the right by this value
        :param init_state: starting state (x,y)
        :param goal_location: goal state (x,y)
        :param slip_prob: probably of random action instead of selected action
        :param goal_value: reward on reaching goal
        :param hallway_states: tuple of states through which the agent can move to get from
                one room to another
        :param hallway_height: length of the hallway states
        :param gamma: discount factor
        """
        super().__init__(actions=list(Dir),
                         init_state=GridWorldState(init_state[0], init_state[1]),
                         gamma=gamma)
        lower_bound = min(upper_offset, lower_offset)
        upper_offset = upper_offset - lower_bound
        lower_offset = lower_offset - lower_bound
        self.upper_width = upper_width
        self.upper_height = upper_height
        self.lower_width = lower_width
        self.lower_height = lower_height
        self.upper_offset = upper_offset
        self.lower_offset = lower_offset
        self.goal_location = goal_location
        self.goal_value = goal_value
        self.slip_prob = slip_prob
        self.hallway_states = hallway_states
        self.hallway_height = hallway_height

        # Hallway states shouldn't be wider than either room
        #if max(self.hallway_states) > min(self.upper_width + self.upper_offset, self.lower_width + self.lower_offset) \
        #        or min(self.hallway_states) < min(self.upper_offset, self.lower_offset):
        #    raise ValueError('Hallway states extend beyond room widths ' + str(self.hallway_states) )

        # Some useful values
        self.total_height = self.lower_height + self.upper_height + self.hallway_height
        #print(self.lower_height, self.upper_height, self.hallway_height)
        self.total_width = max(self.lower_offset + self.lower_width, self.upper_offset + self.upper_width)
        self.upper_start_height = self.lower_height + self.hallway_height + 1
        #print('In MDP. total_width, total_height =', self.total_width, self.total_height)

        # If no goal location given, make goal location be the upper right hand corner of the upper room; if there is
        #  no upper room, make it upper-right hand corner of lower room
        if self.goal_location is None:
            if self.upper_width > 0 and self.upper_height > 0:
                self.goal_location = [(self.upper_width + self.upper_offset, self.total_height)]
            else:
                self.goal_location = [(self.lower_offset + self.lower_width, self.lower_height)]
        # If goal location is outside rooms, raise value error
        for loc in self.goal_location:
            if not self.is_inside_rooms(GridWorldState(loc[0], loc[1])):
                raise ValueError('Goal location is outside rooms ' + str([loc for loc in self.goal_location]))

    # -----------------
    # Utility functions
    # -----------------
    def is_goal_state(self, state):
        return (state.x, state.y) in self.goal_location

    def copy(self):
        copy = TwoRoomsMDP(upper_width=self.upper_width,
                           upper_height=self.upper_height,
                           lower_width=self.lower_width,
                           lower_height=self.lower_height,
                           upper_offset=self.upper_offset,
                           lower_offset=self.lower_offset,
                           init_state=(self.init_state.x, self.init_state.y),
                           goal_location=self.goal_location,
                           slip_prob=self.slip_prob,
                           goal_value=self.goal_value,
                           hallway_states=self.hallway_states,
                           hallway_height=self.hallway_height,
                           gamma=self.gamma)
        copy.current_state = self.current_state
        return copy

    def is_inside_rooms(self, state):
        # Order of cases: lower room (top, bottom, left right), upper room, hallways
        return (0 < state.y <= self.lower_height and self.lower_offset < state.x <= self.lower_offset + self.lower_width) \
                    or (self.upper_start_height <= state.y <= self.total_height and self.upper_offset < state.x <= self.upper_offset + self.upper_width) \
                    or (self.lower_height < state.y < self.upper_start_height and state.x in self.hallway_states)

    def get_width(self):
        return self.total_width

    def get_height(self):
        return self.total_height

    def get_params(self):
        """
        Return a string listing the parameters
        """
        param_string = 'upper width: '.ljust(20) + str(self.upper_width)\
                       + '\nupper height: '.ljust(20) + str(self.upper_height)\
                       + '\nupper offset: '.ljust(20) + str(self.upper_offset)\
                       + '\nlower width: '.ljust(20) + str(self.lower_width)\
                       + '\nlower height: '.ljust(20) + str(self.lower_height) \
                       + '\nlower offset: '.ljust(20) + str(self.lower_offset)\
                       + '\nhallway states: '.ljust(20) + str(self.hallway_states)\
                       + '\nhallway height: '.ljust(20) + str(self.hallway_height)\
                       + '\nstart location: '.ljust(20) + str(self.init_state)\
                       + '\ngoal location: '.ljust(20) + str(self.goal_location)\
                       + '\nslip probability: '.ljust(20) + str(self.slip_prob)
        return param_string

    # -----------------------------
    # Transition & reward functions
    # -----------------------------
    def transition(self, state, action):

        # If in goal state, no actions available
        if self.is_goal_state(state):
            return state

        # Apply slip probability
        if random.random() < self.slip_prob:
            if action in [Dir.UP, Dir.DOWN]:
                action = random.choice([Dir.LEFT, Dir.RIGHT])
            else:
                action = random.choice([Dir.UP, Dir.DOWN])

        # Start by assigning next_state to current_state. This way we only have to check for cases where action
        #  successfully changes states below
        next_state = state

        # Check if state is outside of the two rooms; if so action should have no effect
        if not self.is_inside_rooms(state):
            return next_state

        # Calculate next state for cases where action changes state; add +1 to upper_height to account for
        #  wall
        if action == Dir.UP:
            # If in lower room not against wall, or in lower room under hallway state, or in upper room
            #  not against wall, or in hallway
            if state.y < self.lower_height \
                    or (state.y == self.lower_height and state.x in self.hallway_states) \
                    or (self.upper_start_height <= state.y < self.total_height) \
                    or (self.lower_height < state.y < self.upper_start_height and state.x in self.hallway_states):
                next_state = GridWorldState(state.x, state.y + 1)
        elif action == Dir.DOWN:
            # In upper room not against wall, in upper room above hallway, or in lower room not against wall, or in
            #  hallway
            if (state.y > self.upper_start_height) \
                    or (state.y == self.upper_start_height and state.x in self.hallway_states) \
                    or (1 < state.y <= self.lower_height) \
                    or (self.lower_height < state.y < self.upper_start_height and state.x in self.hallway_states):
                next_state = GridWorldState(state.x, state.y - 1)
        elif action == Dir.LEFT:
            # In lower room not against wall, or upper room not against wall
            if (state.y <= self.lower_height and state.x > max(self.lower_offset + 1, 1)) \
                    or (state.y >= self.upper_start_height and state.x > max(self.upper_offset + 1, 1)):
                next_state = GridWorldState(state.x - 1, state.y)
        elif action == Dir.RIGHT:
            # In lower room not against wall, or upper room not against wall
            if (state.y <= self.lower_height and state.x < self.lower_width + self.lower_offset) \
                    or (state.y >= self.upper_start_height and state.x < self.upper_width + self.upper_offset):
                next_state = GridWorldState(state.x + 1, state.y)

        # If agent enters goal state, make next state terminal
        if (next_state.x, next_state.y) in self.goal_location:
            next_state.set_terminal(True)
            #print('Got to goal location', (next_state.x, next_state.y), next_state.is_terminal())
        return next_state

    def reward(self, state, action, next_state):
        if (state.x, state.y) not in self.goal_location and (next_state.x, next_state.y) in self.goal_location:
            return self.goal_value
        else:
            return 0.0

    # -----------------
    # Main Act function
    # -----------------
    def act(self, action):
        """
        Apply the given action to the current state, update current state to resulting next state, and return
        the next state/rewards. If next state is goal state, reset to initial state
        :param action:
        :return:
        """
        state = self.current_state
        next_state = self.transition(state, action)
        reward = self.reward(state, action, next_state)

        self.set_current_state(next_state)

        if self.is_goal_state(next_state):
            self.reset_to_init()

        return next_state, reward

    # -------------------------
    # Value Iteration functions
    # -------------------------
    def get_all_possible_states(self):
        """
        Create a list of all possible states in the MDP
        """
        state_list = []
        for x in range(1, self.total_width + 1):
            for y in range(1, self.total_height + 1):
                #print('Checking if', x, y, 'is a state')
                state = GridWorldState(x, y)
                if self.is_inside_rooms(state):
                    state_list.append(state)
            #print()
        return state_list

    def get_next_possible_states(self, state, action):
        """
        Get a dictionary (States -> floats), mapping states to the probability that that state
        is reached by the given (state, action) pair
        """
        next_state_probs = {}

        if self.is_goal_state(state):
            next_state_probs[state] = 1
            return next_state_probs

        up_state = GridWorldState(state.x, state.y + 1)
        down_state = GridWorldState(state.x, state.y - 1)
        left_state = GridWorldState(state.x - 1, state.y)
        right_state = GridWorldState(state.x + 1, state.y)
        # can the agent move left?
        left_cond = self.is_inside_rooms(GridWorldState(state.x-1, state.y))
        # can the agent move right?
        right_cond = self.is_inside_rooms(GridWorldState(state.x+1, state.y))
        # can the agent move down?
        down_cond = self.is_inside_rooms(GridWorldState(state.x,state.y-1))
        # can the agent move up
        up_cond = self.is_inside_rooms(GridWorldState(state.x,state.y+1))

        # Set next_state_probs for current state so it can be incremented later
        next_state_probs[state] = 0

        # I'm sure there's a cleaner way to do this but what the hell
        if action == Dir.UP:
            if (up_cond):
                next_state_probs[up_state] = 1 - self.slip_prob
            else:
                next_state_probs[state] += (1 - self.slip_prob)
            # what if it slips?: it would either slip right or left
            if (left_cond):
                next_state_probs[left_state] = self.slip_prob / 2
            else:
                next_state_probs[state] += self.slip_prob / 2
            if (right_cond):
                next_state_probs[right_state] = self.slip_prob / 2
            else:
                next_state_probs[state] += self.slip_prob / 2
        elif action == Dir.DOWN:
            if (down_cond):
                next_state_probs[down_state] = (1 - self.slip_prob)
            else:
                next_state_probs[state] += (1 - self.slip_prob)
            # what if it slips?: it would either slip right or left
            if (left_cond):
                next_state_probs[left_state] = self.slip_prob / 2
            else:
                next_state_probs[state] += self.slip_prob / 2
            if (right_cond):
                next_state_probs[right_state] = self.slip_prob / 2
            else:
                next_state_probs[state] += self.slip_prob / 2
        elif action == Dir.LEFT:
            if (left_cond):
                next_state_probs[left_state] = (1 - self.slip_prob)
            else:
                next_state_probs[state] += (1 - self.slip_prob)
            # what if it slips?: it would either slip up or down
            if (up_cond):
                next_state_probs[up_state] = self.slip_prob / 2
            else:
                next_state_probs[state] += self.slip_prob / 2
            if (down_cond):
                next_state_probs[down_state] = self.slip_prob / 2
            else:
                next_state_probs[state] += self.slip_prob / 2
        elif action == Dir.RIGHT:
            if (right_cond):
                next_state_probs[right_state] = 1 - self.slip_prob
            else:
                next_state_probs[state] += (1 - self.slip_prob)
            # what if it slips?: it would either slip up or down
            if (up_cond):
                next_state_probs[up_state] = self.slip_prob / 2
            else:
                next_state_probs[state] += self.slip_prob / 2
            if (down_cond):
                next_state_probs[down_state] = self.slip_prob / 2
            else:
                next_state_probs[state] += self.slip_prob / 2

        # In the end remove keys whose value is 0
        next_state_probs = {k: v for k, v in next_state_probs.items() if v}
        return next_state_probs