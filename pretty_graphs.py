"""
This script takes in a set of file locations, makes detailed graphs of the per-agent rewards in the given files,
all on one graph
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ast import literal_eval

folder = 'exp_output/hot'#graphs/taxi/n=1/training_eps=15/'

reward_files = ['true/per_agent_rewards.csv',
         'online/per_agent_rewards.csv']
#files = ['exp_output/graphs/stochastic/11x11_n=1/true/per_agent_rewards.csv',
#         'exp_output/graphs/stochastic/11x11_n=1/online/per_agent_rewards.csv']

step_files = ['true/per_agent_steps.csv',
              'online/per_agent_steps.csv']


labels = ['Ground MDP',
          'Neighbor Abstraction w/ Detachment']

if __name__ == '__main__':

    # Reward graph
    plt.style.use('seaborn-whitegrid')
    columns = ['ensemble_key', 'agent_num', 'reward_list']
    fig, ax = plt.subplots()
    label_counter = 0
    for file in reward_files:
        file_path = os.path.join(folder, file)
        # Read in DF
        rewards = None
        df = pd.read_csv(file_path, names=columns)
        for i in range(len(df)):
            if rewards is None:
                rewards = np.array(literal_eval(df.iloc[i]['reward_list']))
            else:
                rewards = np.vstack((rewards, literal_eval(df.iloc[i]['reward_list'])))

        # Calculate data
        mean_rewards = rewards.mean(axis=0)
        #for i in range(len(mean_rewards)):
        #    if i > 30:
        #        mean_rewards[i] = mean_rewards[i-30:i].mean()
        std_rewards = rewards.std(axis=0)
        lower_bound = np.clip(mean_rewards - std_rewards, a_min=float("-inf"), a_max=float("inf"))
        upper_bound = np.clip(mean_rewards + std_rewards, a_min=float("-inf"), a_max=float("inf"))
        episodes = [i for i in range(len(mean_rewards))]

        # Graph it
        #print(labels[label_counter])#, labels[i])
        ax.plot(episodes, mean_rewards, label=labels[label_counter])
        ax.fill_between(episodes, lower_bound, upper_bound, alpha=0.4)
        ax.legend(loc='lower right')
        ax.set_ylim(bottom=-50, top=None)
        plt.title('Performance of Q-learning Ensembles')
        plt.xlabel('Episode')
        plt.ylabel('Proportion of Value of Optimal Value Captured')
        label_counter += 1
    plt.savefig(os.path.join(folder, 'comparison'))
    #plt.show()
    plt.close(fig)

    # Step graph
    fig, ax = plt.subplots()
    label_counter = 0
    columns = ['ensemble_key', 'agent_num', 'steps']
    for file in step_files:
        file_path = os.path.join(folder, file)
        steps = None
        df = pd.read_csv(file_path, names=columns)
        for i in range(len(df)):
            if steps is None:
                steps = np.array(literal_eval(df.iloc[i]['steps']))
            else:
                steps = np.vstack((steps, literal_eval(df.iloc[i]['steps'])))

        # Calculate data
        mean_steps = steps.mean(axis=0)

        std_steps = steps.std(axis=0)
        lower_bound = mean_steps - std_steps
        upper_bound = mean_steps + std_steps
        #lower_bound = np.clip(mean_steps - std_steps, a_min=0, a_max=250)
        #upper_bound = np.clip(mean_steps + std_steps, a_min=0, a_max=250)
        episodes = [i for i in range(len(mean_steps))]

        # Graph it
        ax.plot(episodes, mean_steps, label=labels[label_counter])
        ax.fill_between(episodes, lower_bound, upper_bound, alpha=0.4)
        ax.legend(loc='upper right')
        ax.set_ylim(bottom=0, top=500)
        plt.title('Step Counts of Q-learning Ensembles')
        plt.xlabel('Episode')
        plt.ylabel('Step Counts')
        label_counter += 1
    plt.savefig(os.path.join(folder, 'step_count_comparison'))
    plt.show()
    plt.close(fig)