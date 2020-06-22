'''
Tests q-learning in an abstract MDP
'''

from GridWorld.GridWorldMDPClass import GridWorldMDP
from GridWorld.AbstractGridWorldMDPClass import AbstractGridWorldMDP
from MDP.ValueIterationClass import ValueIteration
from GridWorld.TaxiMDPClass import TaxiMDP
from MDP.AbstractMDPClass import AbstractMDP
from Agent.AgentClass import Agent
from MDP.StateClass import State
from resources.AbstractionMakers import make_abstr
from resources.AbstractionTypes import Abstr_type

def count_nonzero_dict_entries(dct):
    nonzero_count = 0
    for key in dct.keys():
        if dct[key] != 0.0:
            nonzero_count += 1
    return nonzero_count

def print_q_table(q_table):
    for key in q_table:
        print(key[0], key[1], q_table[key])

if __name__ == '__main__':

    # GridWorld

    # Make ground MDP
    mdp = GridWorldMDP(slip_prob=0.0)
    # Run VI to get q-table
    vi = ValueIteration(mdp)
    vi.run_value_iteration()
    q_table = vi.get_q_table()
    # Make state abstractions
    q_star_abstr = make_abstr(q_table, Abstr_type.Q_STAR)
    a_star_abstr = make_abstr(q_table, Abstr_type.A_STAR)
    pi_star_abstr = make_abstr(q_table, Abstr_type.PI_STAR)
    # Make abstract MDPs
    q_mdp = AbstractGridWorldMDP(state_abstr=q_star_abstr)
    a_mdp = AbstractGridWorldMDP(state_abstr=a_star_abstr)
    pi_mdp = AbstractGridWorldMDP(state_abstr=pi_star_abstr)

    q2_mdp = AbstractMDP(mdp, state_abstr=q_star_abstr)

    print("VALUE OF OPTIMAL POLICY")
    print_q_table(q_table)

    print("\n\n\nQ* ABSTR")
    print(q_star_abstr)
    #print(a_star_abstr)
    #print(pi_star_abstr)

    # Create agents on each of these MDPs
    ground_agent = Agent(mdp)
    q_agent = Agent(q_mdp)
    q2_agent = Agent(q2_mdp)
    a_agent = Agent(a_mdp)
    pi_agent = Agent(pi_mdp)
    # Train agents and print rewards
    for i in range(100000):
        ground_agent.explore()
        q_agent.explore()
        q2_agent.explore()
        #print_q_table(q2_agent.get_q_table())
        #print()
        #a_agent.explore()
        #pi_agent.explore()
    print("\n\n\nGROUND AGENT")
    ground_q_table = ground_agent.get_q_table()
    print_q_table(ground_q_table)



    print("\n\n\nQ* ABSTR AGENT")
    q_q_table = q_agent.get_q_table()
    print_q_table(q_q_table)

    print("\n\n\nQ*2 ABSTR AGENT")
    q2_q_table = q2_agent.get_q_table()
    print_q_table(q2_q_table)
    print()

    print(count_nonzero_dict_entries(ground_q_table))
    print(count_nonzero_dict_entries(q_q_table))
    print(count_nonzero_dict_entries((q2_q_table)))

    #print(len(q2_q_table.keys()))
    #for key in q2_q_table:
    #    print(key, key[0])

    #print("\n\n\nA* ABSTR AGENT")
    #a_q_table = a_agent.get_q_table()
    #print_q_table(a_q_table)

    #print("\n\n\nPI* ABSTR AGENT")
    #pi_q_table = pi_agent.get_q_table()
    #print_q_table(pi_q_table)

    '''
    # Test TaxiMDP learning
    mdp = TaxiMDP()
    # Run VI to get q-table
    vi = ValueIteration(mdp)
    vi.run_value_iteration()
    q_table = vi.get_q_table()
    # Make state abstractions
    q_star_abstr = make_abstr(q_table, Abstr_type.Q_STAR)
    # Make abstract MDPs
    q_mdp = AbstractGridWorldMDP(state_abstr=q_star_abstr)
    q2_mdp = AbstractMDP(mdp, state_abstr=q_star_abstr)


    # Create agents on each of these MDPs
    ground_agent = Agent(mdp)
    q_agent = Agent(q_mdp)
    q2_agent = Agent(q2_mdp)

    # Train agents and print rewards
    for i in range(100000):
        #ground_agent.explore()
        q_agent.explore()
        q2_agent.explore()
        #a_agent.explore()
        #pi_agent.explore()
    #print("\n\n\nGROUND AGENT")
    #ground_q_table = ground_agent.get_q_table()
    #print_q_table(ground_q_table)

    print("\n\n\nQ* ABSTR AGENT")
    q_q_table = q_agent.get_q_table()
    print_q_table(q_q_table)

    print("\n\n\nQ*2 ABSTR AGENT")
    q2_q_table = q2_agent.get_q_table()
    print_q_table(q2_q_table)

    #print("\n\n\nA* ABSTR AGENT")
    #a_q_table = a_agent.get_q_table()
    #print_q_table(a_q_table)

    #print("\n\n\nPI* ABSTR AGENT")
    #pi_q_table = pi_agent.get_q_table()
    #print_q_table(pi_q_table)
    '''





