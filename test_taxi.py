'''
Test the TaxiMDP class
'''

from GridWorld.TaxiMDPClass import TaxiMDP
from GridWorld.TaxiStateClass import TaxiState
from GridWorld.TaxiActionEnums import Act

taxi = TaxiMDP(slip_prob=0.0)

# Setting taxi location to (3,3) for test purposes
taxi.set_current_state(TaxiState((3,3),
                                 taxi.get_passenger_loc(),
                                 taxi.get_goal_loc()))
print(taxi)
print()

print('Testing illegal pickup and drop off (passenger not aboard). Should do nothing')
print(taxi.transition(taxi.get_current_state(), Act.PICKUP))
print(taxi.transition(taxi.get_current_state(), Act.DROPOFF))
print()

print('Testing legal pickup')
taxi.set_current_state(TaxiState(taxi.get_passenger_loc(),
                                 taxi.get_passenger_loc(),
                                 taxi.get_goal_loc()))
print(taxi)
print(taxi.transition(taxi.get_current_state(), Act.PICKUP))
print()

print('Testing illegal drop off with passenger aboard')
taxi.set_current_state(TaxiState((3,3), (0,0), taxi.get_goal_loc()))
print(taxi)
print(taxi.transition(taxi.get_current_state(), Act.DROPOFF))
print()

print('Testing illegal pick up with passenger already aboard. Should do nothing')
taxi.set_current_state(TaxiState((1,1), (0,0), (5,5)))
print(taxi)
print(taxi.transition(taxi.get_current_state(), Act.PICKUP))
print()

# Testing motion
'''
print("Testing movement from (3,3). Should be unimpeded.")
print("Moving left, right, up, down")
print(taxi.transition(taxi.get_current_state(), Act.LEFT))
print(taxi.transition(taxi.get_current_state(), Act.RIGHT))
print(taxi.transition(taxi.get_current_state(), Act.UP))
print(taxi.transition(taxi.get_current_state(), Act.DOWN))
print()
print("Testing movement from (1,1). Everything but up should be impeded")
taxi.set_current_state(TaxiState((1,1),
                                 taxi.get_passenger_loc(),
                                 taxi.get_goal_loc()))
print("Moving left, right, up, down")
print(taxi.transition(taxi.get_current_state(), Act.LEFT))
print(taxi.transition(taxi.get_current_state(), Act.RIGHT))
print(taxi.transition(taxi.get_current_state(), Act.UP))
print(taxi.transition(taxi.get_current_state(), Act.DOWN))
print()

print("Testing movement from (3,5). Left and up should be impeded")
taxi.set_current_state(TaxiState((3,5),
                                 taxi.get_passenger_loc(),
                                 taxi.get_goal_loc()))
print(taxi.transition(taxi.get_current_state(), Act.LEFT))
print(taxi.transition(taxi.get_current_state(), Act.RIGHT))
print(taxi.transition(taxi.get_current_state(), Act.UP))
print(taxi.transition(taxi.get_current_state(), Act.DOWN))
print()

print("Testing movement from (2,2). Left should be impeded")
taxi.set_current_state(TaxiState((2,2),
                                 taxi.get_passenger_loc(),
                                 taxi.get_goal_loc()))
print(taxi.transition(taxi.get_current_state(), Act.LEFT))
print(taxi.transition(taxi.get_current_state(), Act.RIGHT))
print(taxi.transition(taxi.get_current_state(), Act.UP))
print(taxi.transition(taxi.get_current_state(), Act.DOWN))
print()

print("Testing movement from (2,4). Right should be impeded")
taxi.set_current_state(TaxiState((2,4),
                                 taxi.get_passenger_loc(),
                                 taxi.get_goal_loc()))
print(taxi.transition(taxi.get_current_state(), Act.LEFT))
print(taxi.transition(taxi.get_current_state(), Act.RIGHT))
print(taxi.transition(taxi.get_current_state(), Act.UP))
print(taxi.transition(taxi.get_current_state(), Act.DOWN))
print()
'''
