"""
Wrapper for TrackingAgent which runs the agent for an episode and returns
the number of steps and the cumulative reward for the episode

AgentWrapper has to
- Run agent until end of episode, performing detach if required
- Return, at the end of the episode, the number of steps and the cumulative reward

Parameters:
- 'make_abstraction': List of ints (when the abstractions happen)
- 'do_detach': list of ints (where detach happens)
- 'reset_q_value': True, False, 'neighbor', or 'rollout'
"""
from Agent.TrackingAgentV2 import TrackingAgent


class AgentWrapper:
    def __init__(self, env, **kwargs):
        self.agent = TrackingAgent(env, **kwargs)
        self.params = kwargs
        self.episode_count = 0
        if 'reset_q_value' not in self.params.keys() and 'do_detach' in self.params.keys():
            raise ValueError('AgentWrapper: "reset_q_value" parameter required if "do_detach" parameter is set')

    def run_episode(self):
        """
        Run the agent in the environment for 1 episode, returning the
        cumulative step count and reward
        :return: (step count, cumulative_reward)
        """
        # Reset environment
        self.agent.env.reset()
        done = False
        step_count = 0
        total_reward = 0

        while not done:
            reward, done = self.agent.explore()
            step_count += 1
            if step_count % 100 == 0:
                print('step count {}'.format(step_count))
            total_reward += self.agent.params['gamma']**step_count * reward
        return step_count, total_reward

    def run_all_episodes(self, episode_count):
        """
        Run the agent for as many episodes as parameters dictate, and
        return arrays containing per-episode step counts and rewards,
        online abstraction made (empty list if no such abstraction is made),
        and list of detached states (if any)

        Detach state or create abstraction if those parameters are set
        :return: step count array, total reward array, online abstraction made,
                and detached states
        """
        # Holds final result
        step_arr = []
        reward_arr = []
        new_abstr = {}
        detached_states = []

        if 'abstraction_type' in self.params.keys() and self.params['abstraction_type'] == 'discretization':
            self.agent.make_abstraction()

        while self.episode_count < episode_count:
            # Run episode, record results
            steps, reward = self.run_episode()
            step_arr.append(steps)
            reward_arr.append(reward)
            self.episode_count += 1
            if self.episode_count % 1 == 0:
                print('Episode {} finished with step count {}'.format(self.episode_count, steps))

            # Create temporal abstraction if applicable
            if 'make_abstraction' in self.params.keys() and self.episode_count in self.params['make_abstraction']:
                self.agent.make_abstraction()
                new_abstr = self.agent.params['s_a'].abstr_dict

            # Detach states if applicable
            if 'refine_abstraction' in self.params.keys() and self.episode_count in self.params['refine_abstraction']:
                newly_detached = self.agent.refine_abstraction()
                detached_states.extend(newly_detached)
        print('final abstraction')
        for i in range(len(self.agent.params['s_a'].cell_to_abstract_cell)):
            for key, value in self.agent.params['s_a'].cell_to_abstract_cell[i].items():
                print(key, value)

        return step_arr, reward_arr, new_abstr, detached_states
