import numpy as np
from collections import defaultdict
import random

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        
        self.alpha = 0.1
        self.gamma = 0.9999

        self.eps_start = 1.0
        self.eps_decay=.9999
        self.eps_min=0.0001
        self.epsilon = self.eps_start

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        def epsilon_greedy(Q, state, nA, eps):
            """Selects epsilon-greedy action for supplied state.

            Params
            ======
                Q (dictionary): action-value function
                state (int): current state
                nA (int): number actions in the environment
                eps (float): epsilon
            """
            if random.random() > eps: # select greedy action with probability epsilon
                return np.argmax(Q[state])
            else:                     # otherwise, select an action randomly
                return random.choice(np.arange(nA))

        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

        return epsilon_greedy(self.Q, state, self.nA, self.epsilon)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        

        def epsilon_greedy_prob_return(nA, epsilon, Q, s_1):
            all_prob = np.ones(nA) * epsilon/nA
            prob_above_epsilon = (1 - epsilon) + (epsilon/nA) ## greedy choice


            greedy_action = np.argmax(Q[s_1])
            all_prob[greedy_action] = prob_above_epsilon

            # prob_adj_value_best = prob_above_epsilon * Q[s_1][greedy_action]
            # prob_adj_value_rest = np.dot(Q[s_1], prob_below_epsilon )
            
            # print(prob_adj_value_best)
            # print(prob_adj_value_rest)

            return np.dot(Q[s_1], all_prob )
            

        def update_expected_Q(alpha, gamma, Q, s_0, a_0, r_1, epsilon, nA, s_1 = None):
            # new_Q = copy.deepcopy(Q)
            Qs_next  = epsilon_greedy_prob_return(nA, epsilon, Q, s_1) if s_1 is not None else 0  

            current_return = Q[s_0][a_0]
            target_return = r_1 + (gamma * Qs_next)
            
            difference = (target_return - current_return)
            
            # new_Q[s_0][a_0] = current_return + (alpha * difference)
            # return new_Q
            return current_return + (alpha * difference)

        self.Q[state][action] = update_expected_Q(self.alpha, self.gamma, self.Q, state, action, reward, self.epsilon, self.nA, next_state )
        
        
