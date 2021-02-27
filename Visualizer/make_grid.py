from Visualizer.TwoRoomsVisualizer import *
from GridWorld.TwoRoomsMDP import *

if __name__ == '__main__':
    mdp = TwoRoomsMDP(upper_height=3,
                      lower_height=3,
                      hallway_states=[2],
                      lower_width=3,
                      upper_width=3)

    v = TwoRoomsVisualizer()
    grid = v.create_grid(mdp)
    v.display_surface(grid)