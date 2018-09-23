import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return np.random.choice(self.nA, p=self.get_action_prob(i_episode, state))

    def step(self, state, action, reward, next_state, done, i_episode, alpha=1, gamma=1.0):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        self.Q[state][action] = Agent.update_Q(self.Q[state][action], np.dot(self.Q[next_state],\
                                self.get_action_prob(i_episode, state)), reward, alpha, gamma)

    @staticmethod
    def update_Q(Qsa, Qsa_next, reward, alpha, gamma):
        return Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))

    def get_action_prob(self, i_episode, state):
        epsilon = 1.0 / i_episode
        action_prob = np.ones(self.nA) * epsilon / self.nA
        action_prob[np.argmax(self.Q[state])] = 1 - epsilon + (epsilon / self.nA)
        return action_prob