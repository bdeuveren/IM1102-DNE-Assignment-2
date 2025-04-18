"""Replay Memory Implementation."""

from collections import deque
import random
from collections import namedtuple


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    """Replay Memory.

    Replay Memory is a technique used in Deep Reinforcement Learning to store and reuse past experiences (i.e., state, action, reward, next state tuples).
    It is critical in stabilizing and improving the learning process of RL agents—especially when using deep neural networks (like in DQN - Deep Q-Networks).
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        """Store experience in replay memory.

        Parameters
        ----------
            state: The current state of the agent.
            action: The action taken by the agent.
            reward: The reward received after taking the action.
            next_state: The next state of the agent after taking the action.
            done: A boolean indicating if the episode has ended.
        """
        self.buffer.append(Transition(*args))

    def sample(self, batch_size) -> Transition:
        """Sample a batch of experiences from replay memory.

        Returns
        -------
            Returns a tuple of (states, actions, rewards, next_states, dones).
            - states: The current states of the agent.
            - actions: The actions taken by the agent.
            - rewards: The rewards received after taking the actions.
            - next_states: The next states of the agent after taking the actions.
            - dones: A boolean indicating if the episode has ended.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Return the current size of replay memory."""
        return len(self.buffer)
