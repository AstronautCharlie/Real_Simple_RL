"""
Test that the TrackingAgent functionality works with the Discretization Abstraction
"""

import gym
from Agent.TrackingAgentV2 import TrackingAgent

env = gym.make('MountainCar-v0_save')
params = {'abstraction_type': 'discretization', 'starting_mesh': 5, 'finest_mesh': 15}
agent = TrackingAgent(env, **params)

# Make and print out discretization abstraction
print(agent.params['s_a'].space.low, agent.params['s_a'].space.high, agent.params['s_a'].bucket_sizes)

done = False
while not done:
    _, done = agent.explore()


for key, value in agent.abstr_state_occupancy_record.items():
    print(key, value)
