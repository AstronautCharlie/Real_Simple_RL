"""
New and improved Experiment class. Assumes the MDP adheres to the Env
interface laid out by OpenAI Gym

Experiment needs to take in:
- Agent types
- # of agents per ensemble
- # of episodes

Needs to do:
- Copying of environments (need 1 per agent)
- Run all the agents (AgentWrapper will handle the detachment, anything that has to happen during the loop)
- Collect results
- Create step graphs

"""
import gym
import gym_classics

class Experiment():
    def __init__(self, env, **kwargs):
        self.env = env
        self.params = kwargs
        required_args = ['agent_params', 'num_agents', 'num_episodes']
        for req in required_args:
            if req not in self.params.keys():
                raise ValueError('Missing required argument: {}'.format(req))

if __name__ == '__main__':
    agent_params = [{}]
    kwargs = {'agent_params': agent_params, 'num_agents': 10, 'num_episodes': 100}
    env = gym.make('MountainCar-v0')
    exp = Experiment(env, **kwargs)
    for key, value in exp.params.items():
        print(key, value)
    observation = exp.env.reset()
    observation, reward, done, info = env.step(exp.env.action_space.sample())
    print('Observation is', observation)

