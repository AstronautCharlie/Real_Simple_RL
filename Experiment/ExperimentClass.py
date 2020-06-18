'''
This class creates and runs a test of q-learning on the given MDP for the given abstraction
types and epsilon values, and compares the value of each trajectory to the value of the
optimal ground-state trajectory from that point
'''
from MDP.MDPClass import MDP
from MDP.StateAbstractionClass import StateAbstraction
from MDP.ValueIterationClass import ValueIteration
from MDP.AbstractMDPClass import AbstractMDP
from Agent.AgentClass import Agent
from resources.AbstractionMakers import make_abstr
from resources.AbstractionTypes import Abstr_type

out_path = '/results/test_abstr_q_learning/'

class Experiment():
    def __init__(self, mdp, abstr_epsilon_list, num_agents=10):
        '''
        Create an experiment, which will hold the ground MDP, the abstract MDPs (parameters dictated by abstr_epsilon_list),
        and an ensemble of (num_agents) q-learning agents on each MDP.
        :param mdp: MDP
        :param abstr_epsilon_list: list of tuples, where first element is abstraction type and
        second is the epsilon
        '''
        self.ground_mdp = mdp
        for val in abstr_epsilon_list:
            if val[0] not in Abstr_type or val[1] < 0 or val[1] > 1:
                raise ValueError('Abstraction Epsilon List is invalid', abstr_epsilon_list)
        self.abstr_epsilon_list = abstr_epsilon_list
        self.num_agents = num_agents

        # Run Value Iteration to get q-table for abstractions
        vi = ValueIteration(mdp)
        vi.run_value_iteration()
        q_table = vi.get_q_table()

        # Create abstract MDPs for element of abstr_epsilon_list:
        self.abstract_mdps = []
        for val in abstr_epsilon_list:
            state_abstr = make_abstr(q_table, val[0], val[1])
            self.abstract_mdps.append(AbstractMDP(mdp, state_abstr))

        # Create an ensemble of agents on these MDPs
        self.agents = []

        # Create agents on ground mdp
        ground_agents = []
        for i in range(self.num_agents):
            mdp = self.ground_mdp.copy()
            agent = Agent(mdp)
            ground_agents.append(agent)
        self.agents.append(ground_agents)

        # Create agents on abstract MDPs
        for abstract_mdp in self.abstract_mdps:
            abstract_mdp_ensemble = []
            for i in range(self.num_agents):
                mdp = abstract_mdp.copy()
                agent = Agent(mdp)
                abstract_mdp_ensemble.append(agent)
            self.agents.append(abstract_mdp_ensemble)

    def run_trajectory(self, agent):
        '''
        Run an agent on the MDP until it reaches a terminal state. Record the discounted rewards achieved along the way
        and the starting state
        :param: agent: Q-learning agent
        :return: starting_state: State
        :return: reward: sum of discounted rewards gotten over the course of the trajectory
        '''
        # Bypass abstraction function to get current ground state
        starting_state = agent.mdp.current_state

        current_state = starting_state

        # This will track the rewards accumulated along the trajectory
        sum_rewards = 0
        discount = 1

        while not current_state.is_terminal():
            _, action, next_state, reward = agent.explore()
            sum_rewards += discount * reward
            current_state = next_state
            discount *= agent.mdp.gamma
            print(_, action, next_state, reward)


    '''
    For every MDP: 
        Run each agent on its MDP for one trajectory, recording the discounted rewards accumulated along the way 
            Get the starting position of the agent (in FourRooms it'll always be the same) 
            Write the value of the optimal policy for this starting position 
            Run the agent until it reaches a terminal state, recording the sum of discounted rewards accumulated along the way 
            Get the fraction of the optimal value the agent achieved
            Write this to a file, one row per agent, one column per trajectory 
        
    '''



