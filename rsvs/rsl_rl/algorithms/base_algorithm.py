from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import torch


class BaseAlgorithm(ABC):
    """Abstract base class for RL algorithms.
    
    All algorithms must implement the core interface for:
    - Acting: act(), test_mode(), train_mode()
    - Learning: update(), process_env_step(), compute_returns()
    """
    
    @abstractmethod
    def __init__(self) -> None:
        """Initialize the algorithm."""
        pass
    
    @abstractmethod
    def test_mode(self) -> None:
        """Set the algorithm to evaluation mode."""
        pass
    
    @abstractmethod
    def train_mode(self) -> None:
        """Set the algorithm to training mode."""
        pass
    
    @abstractmethod
    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        """Output actions based on observations.
        
        Args:
            obs: Actor observations tensor. Shape: [num_envs, obs_dim]
            critic_obs: Critic observations tensor. Shape: [num_envs, critic_obs_dim]
            
        Returns:
            actions: Action tensor. Shape: [num_envs, action_dim]
        """
        pass

    @abstractmethod
    def update(self) -> Tuple[float, ...]:
        """Update the policy using collected experiences.
        
        Returns:
            Tuple of loss values (varies by algorithm).
        """
        pass
    
    @abstractmethod
    def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor, infos: Dict[str, Any]) -> None:
        """Process the environment step, including rewards and done signals.
        
        Args:
            rewards: Reward tensor. Shape: [num_envs]
            dones: Done flags tensor. Shape: [num_envs]
            infos: Dictionary with additional info (e.g., 'time_outs')
        """
        pass
    
    @abstractmethod
    def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
        """Compute the returns for the collected experiences.
        
        Args:
            last_critic_obs: Final critic observation for bootstrapping. Shape: [num_envs, critic_obs_dim]
        """
        pass