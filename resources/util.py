"""
Utility functions that are generally useful to have
"""
import ast
import pandas as pd

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
        if key not in error_tuples and key not in corrupted_states:
            non_error_states.append(key)

    # Get error states
    error_states = [tup[0] for tup in error_tuples]

    print('Error states', error_states)
    print('Corrupted states', corrupted_states)
    print('Non-error states', non_error_states)

    counter_dict = {'error': 0, 'corrupted': 0, 'non-error': 0}
    for state in detached_states:
        if state in error_states:
            counter_dict['error'] += 1
        elif state in corrupted_states:
            counter_dict['corrupted'] += 1
        elif state in non_error_states:
            counter_dict['non-error'] += 1
    print(counter_dict)
    # Go through detached states and see if they are in the error file