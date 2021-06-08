"""
Wrapper for TrackingAgent which runs the agent for an episode and returns
the number of steps and the cumulative reward for the episode

AgentWrapper has to
- Run agent until end of episode, performing detach if required
- Return, at the end of the episode, the number of steps and the cumulative reward
"""
from Agent.TrackingAgentV2 import TrackingAgent


class AgentWrapper:
    def __init__(self, env, **kwargs):
        self.agent = TrackingAgent(env)
        self.params = kwargs
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0

    def run_episode(self):
        """
        Run the agent in the environment for 1 episode. If applicable, do detach at the end
        of the episode.
        :return: (step count, cumulative_reward)
        """
