
from GridWorld.ActionEnums import Dir
import random
from GridWorld.GridWorldStateClass import GridWorldState
from MDP.Functions import Transition
from MDP.Functions import Reward

class GridWorldTransition(Transition):
    def __init__(self, slip_prob=0.0):
        self.slip_prob = slip_prob

    def __call__(self, state, action, mdp):
        '''
        This needs access to the MDP parameters

        Parameters:
            state:GridWorldState
            action:Enum
            mdp:GridWorldMDP

        Returns:
            state:GridWorldState
        '''
        next_state = state

        # If terminal, do nothing
        if state.is_terminal():
            return next_state

        # Apply slip probability and change action if applicable
        if random.random() < self.slip_prob:
            if action in [Dir.UP, Dir.DOWN]:
                action = random.choice([Dir.LEFT, Dir.RIGHT])
            elif action in [Dir.LEFT, Dir.RIGHT]:
                action = random.choice([Dir.UP, Dir.DOWN])

        # Calculate next state based on action
        if action == Dir.UP and state.y < mdp.height and (state.x, state.y + 1) not in mdp.walls:
            next_state = GridWorldState(state.x, state.y + 1)
        if action == Dir.DOWN and state.y > 1 and (state.x, state.y - 1) not in mdp.walls:
            next_state = GridWorldState(state.x, state.y - 1)
        if action == Dir.LEFT and state.x > 1 and (state.x - 1, state.y) not in mdp.walls:
            next_state = GridWorldState(state.x - 1, state.y)
        if action == Dir.RIGHT and state.x < mdp.width and (state.x + 1, state.y) not in mdp.walls:
            next_state = GridWorldState(state.x + 1, state.y)

        if (next_state.x, next_state.y) in mdp.goal_location:
            next_state.set_terminal(True)

        return next_state


class GridWorldReward(Reward):
    def __init__(self, goal_value=1.0):
        self.goal_value = goal_value

    def __call__(self, state, action, next_state, mdp):
        if (next_state.x, next_state.y) in mdp.goal_location:
            return self.goal_value
        else:
            return 0.0
