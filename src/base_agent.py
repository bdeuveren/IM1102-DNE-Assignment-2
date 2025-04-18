"""Agent class for reinforcement learning."""

import abc
import gymnasium as gym


class BaseAgent(abc.ABC):
    """Abstract class for the Agent."""

    def __init__(self, env: gym.Env):
        """Initialize the agent."""
        self.env = env

    @abc.abstractmethod
    def act(self, state: any, epsilon=0.01) -> int:
        """Select an action based on the current state and epsilon."""
        pass

    @abc.abstractmethod
    def step(self, action: int):
        """Perform a step in the environment."""
        pass

    @abc.abstractmethod
    def train(self, num_episodes: int):
        """Train the agent."""
        pass

    @abc.abstractmethod
    def run(self):
        """Train the agent."""
        pass
