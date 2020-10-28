"""
Utility functions that are generally useful to have
"""
import ast
import pandas as pd

def parse_file_for_dict(self, key, file, agent_num=None):
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