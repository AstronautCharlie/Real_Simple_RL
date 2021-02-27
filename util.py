"""
Utility functions that are generally useful to have
"""
import ast
import string
import pandas as pd
from resources.AbstractionTypes import Abstr_type
import numpy as np
from scipy import stats
def parse_file_for_dict(key, file, agent_num=None):
    """
    Parse the given file and return the value associated with the key and batch num. Will work on any file where
    key is a string all together and batch num is its own column.
    :param key: a 2-tuple for a true abstraction, a 5-tuple for a corrupt abstraction, or the string "ground"
    :param file: a csv file mapping keys and batch numbers to saved values of some kind
    :param agent_num: optional integer indicating a particular agent. If provided, we'll look at a particular
                        agent within the given key
    :return: the value from the file matching the key and batch num
    """
    if agent_num is not None:
        names = ['key', 'agent_num', 'dict']
    else:
        names = ['key', 'dict']
    df = pd.read_csv(file, names=names)
    # In this case we're in the ground state and the key is just 'ground'
    if key == "ground":
        df = df.loc[df['key'] == key]
        if agent_num is not None:
            df = df.loc[df['agent_num'] == agent_num]
        value = ast.literal_eval(df['dict'].values[0])
    # In this case we're in a true state abstraction, key is (Abstr_type, abstr_eps)
    elif len(key) == 2:
        # Split key into abstr_type and abstr_eps
        # print(df['key'])
        df['abstr_type'], df['abstr_eps'] = df['key'].str.split(', ').str
        # Fill in abstr_eps field (for ground)
        df['abstr_eps'] = df['abstr_eps'].fillna('0.0')
        df['abstr_type'] = df['abstr_type'].map(lambda x: x.strip('(<: 1234>'))
        df['abstr_eps'] = df['abstr_eps'].map(lambda x: x.strip('(<: >)'))
        df = df.loc[(df['abstr_type'] == str(key[0]))
                    & (df['abstr_eps'] == str(key[1]))]
        if agent_num is not None:
            df = df.loc[df['agent_num'] == agent_num]
        value = ast.literal_eval(df['dict'].values[0])
    # In this case we're in a corrupt abstraction, key is (Abstr_type, abstr_eps, corr_type, corr_prop, mdp_num)
    elif len(key) == 5:
        #print(df.to_string())
        df['abstr_type'], df['abstr_eps'], df['corr_type'], df['corr_prop'], df['batch_num'] = df['key'].str.split(
            ', ').str
        df['abstr_type'] = df['abstr_type'].map(lambda x: x.strip('(<: 1234>'))
        df['corr_type'] = df['corr_type'].map(lambda x: x.strip('<>: 1234'))
        df['batch_num'] = df['batch_num'].map(lambda x: x.strip(')'))
        df = df.loc[(df['abstr_type'] == str(key[0]))
                    & (df['abstr_eps'].astype(float) == key[1])
                    & (df['corr_type'] == str(key[2]))
                    & (df['corr_prop'].astype(float) == key[3])
                    & (df['batch_num'].astype(int) == key[4])]
        if agent_num is not None:
            df = df.loc[df['agent_num'] == agent_num]
        value = ast.literal_eval(df['dict'].values[0])
    else:
        raise ValueError('Key provided is not of valid type (either "ground", 2-tuple, or 5-tuple)')

    return value

def categorize_detached_states(key, agent_num, corrupted_abstr_file, error_file, detached_state_file):
    """
    Categorize the states detached during an experiment as error, corrupted, or non-error.
    :param key: an ensemble key
    :param error_file: csv file with error states by key
    :param detached_state_file: csv file with detached states by key and episode
    """
    # Read in data and create lists of error states, corrupted states, and non-error states
    abstraction = parse_file_for_dict(key, corrupted_abstr_file)
    error_tuples = parse_file_for_dict(key, error_file)
    detached_states = parse_file_for_dict(key, detached_state_file, agent_num=agent_num)
    #print(abstraction)
    #print(error_tuples)
    #print(detached_states)

    # Convert abstraction from list into dictionary
    abstr_dict = {}
    for tup in abstraction:
        abstr_dict[tup[0]] = tup[1]
    #print('Abstraction dictionary:', abstr_dict)

    # Label all error states
    error_states = [tup[0] for tup in error_tuples]

    # Go through all corrupted abstract states and get the corrupted ground states
    corrupted_states = []
    for tup in error_tuples:
        corrupted_abstr_state = tup[2]
        for key, value in abstr_dict.items():
            if value == corrupted_abstr_state:
                corrupted_states.append(key)

    # Label all non-error states
    non_error_states = []
    for key in abstr_dict.keys():
        if key not in error_states and key not in corrupted_states:
            non_error_states.append(key)

    # Get error states

    # Remove duplicates
    non_error_states = list(dict.fromkeys(non_error_states))
    error_states = list(dict.fromkeys(error_states))
    corrupted_states = list(dict.fromkeys(corrupted_states))


    #print('Error states', error_states)
    #print('Corrupted states', corrupted_states)
    #print('Non-error states', non_error_states)

    # Categorize states
    counter_dict = {'error': 0, 'corrupted': 0, 'non-error': 0}
    sum_error = []
    sum_corrupted = []
    sum_nonerror = []
    for state in detached_states:
        if state in error_states:
            counter_dict['error'] += 1
            sum_error.append(state)
        elif state in corrupted_states:
            counter_dict['corrupted'] += 1
            sum_corrupted.append(state)
        elif state in non_error_states:
            counter_dict['non-error'] += 1
            sum_nonerror.append(state)

    # Print results
    print('Error states:', error_states)
    print('Corrupted states:', corrupted_states)
    print(counter_dict)
    if len(sum_error) > 0:
        print('Error states detached:', end=' ')
        for state in sum_error:
            print(state, end=' ')
        print()
    if len(sum_corrupted) > 0:
        print('Corrupted states detached:', end=' ')
        for state in sum_corrupted:
            print(state, end=' ')
        print()
    if len(sum_nonerror) > 0:
        print('Non-error states detached:', end=' ')
        for state in sum_nonerror:
            print(state, end=' ')
        print()
    print()
    # Go through detached states and see if they are in the error file

    return sum_error, sum_corrupted, sum_nonerror

def parse_ensemble_key(ensemble_key):
    """
    Parse ensemble_key string, either of the form
    "ground",
    "(Abstr_type, abstr_epsilon)",
    "(Abstr_type, Abstr_Epsilon, Corr_Type, proportion, MDP_number)",
    "(Abstr_type, Abstr_Epsilon, 'explicit errors', error_dict_number, MDP_number)"
    Returns tuple, each element containing one of these items
    """
    if ensemble_key == "ground":
        return "ground", None, None, None, None
    else:
        split_string = ensemble_key.split(",")

        # Parse abstraction type
        abstr_type = None
        if 'PI_STAR' in split_string[0]:
            abstr_type = Abstr_type.PI_STAR
        elif 'Q_STAR' in split_string[0]:
            abstr_type = Abstr_type.Q_STAR
        elif 'A_STAR' in split_string[0]:
            abstr_type = Abstr_type.A_STAR
        else:
            raise ValueError('Abstraction type of ensemble key being parsed is not supported')

        # Parse epsilon
        abstr_epsilon = float(split_string[1].translate(str.maketrans('', '', string.punctuation)))

        # Return parsed key if that's all the elements in the list
        if len(split_string) == 2:
            return abstr_type, abstr_epsilon, None, None, None

        # Parse corruption type, number of states/error dict number, and MDP num
        corr_type = split_string[2]
        prop_or_dict_num = int(split_string[3])
        mdp_num = int(split_string[4].translate(str.maketrans('', '', string.punctuation)))

        return abstr_type, abstr_epsilon, corr_type, prop_or_dict_num, mdp_num

def get_abstr_type_from_ensemble_key(ensemble_key):
    abstr_type, _, _, _, _ = parse_ensemble_key(ensemble_key)
    return abstr_type

def get_abstr_eps_from_ensemble_key(ensemble_key):
    _, abstr_eps, _, _, _ = parse_ensemble_key(ensemble_key)
    return abstr_eps

def get_corr_type_from_ensemble_key(ensemble_key):
    _, _, corr_type, _, _ = parse_ensemble_key(ensemble_key)
    return corr_type

def get_corr_prop_from_ensemble_key(ensemble_key):
    _, _, _, corr_prop, _ = parse_ensemble_key(ensemble_key)
    return corr_prop

def get_mdp_num_from_ensemble_key(ensemble_key):
    _, _, _, _, mdp_num = parse_ensemble_key(ensemble_key)
    return mdp_num

def abstr_to_string(abstraction_type):
    if abstraction_type == "ground":
        return "ground"
    elif abstraction_type == Abstr_type.Q_STAR:
        return 'q'
    elif abstraction_type == Abstr_type.A_STAR:
        return 'a'
    else:
        return 'pi'

def calculate_confidence_interval(value_list, alpha):
    """
    Calculate the upper and lower bounds of the confidence interval, where
    confidence degree is dictated by alpha
    """
    # Edge cases of length 0 or 1
    if len(value_list) == 0:
        #raise ValueError('Called calculate_confidence_interval on empty list' + str(value_list))
        return None, None
    if len(value_list) == 1:
        return value_list[0], value_list[0]

    mean = np.mean(value_list)
    n = len(value_list)
    numer = n * np.sum(np.square(value_list)) - np.square(np.sum(value_list))
    denom = n * (n-1)
    stdev = np.sqrt(numer/denom)

    # Student's T function
    t = stats.t.ppf(1 - alpha / 2, n - 1)

    # Calculate confidence interval bounds
    lower_bound = mean - t * stdev / np.sqrt(n)
    upper_bound = mean + t * stdev / np.sqrt(n)

    return lower_bound, upper_bound