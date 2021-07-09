"""
Visualize the results of the Experiment in the given location;
creates graphs of the steps and rewards per episode
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def vis_results(output_dir, agent_names, value_name):
    """
    Create graphs comparing performance of given agent names for particular value and
    save it to the output_dir
    :param output_dir: output directory of the experiment
    :param agent_names: List of agent names in the experiment
    :param value_name: str, either 'steps' or 'rewards'
    :return: None
    """
    fig, ax = plt.subplots()

    for name in agent_names:
        path_to_data = os.path.join(output_dir, name, value_name + '.csv')
        df = pd.read_csv(path_to_data, index_col=0)
        data = df.to_numpy()
        mean_data = np.mean(data, axis=0)
        std_data = np.std(data, axis=0)
        ep_values = np.arange(len(data[0]))
        lower_bound = mean_data - std_data
        upper_bound = mean_data + std_data
        ax.plot(ep_values, mean_data, label=name)
        ax.fill_between(ep_values, lower_bound, upper_bound, alpha=0.3)
    title_string = value_name.capitalize() + ' Comparison'
    ax.set_xlabel('Episode')
    ax.set_ylabel(value_name.capitalize())
    fig.suptitle(title_string)
    ax.legend()
    fig_name = os.path.join(output_dir, value_name) + '.png'
    plt.savefig(fig_name)


if __name__ == '__main__':
    output = '../experiment_output-v2/MountainCar-v0_save'
    agents = ['abstr', 'baseline']
    values = ['steps', 'rewards']

    for value in values:
        vis_results(output, agents, value)
