"""Privacy accounting for differential privacy in federated learning."""

import numpy as np
from typing import Tuple, List
import math


class PrivacyAccountant:
    """Tracks and computes privacy budget for DP-FL."""
    
    def __init__(
        self,
        noise_multiplier: float,
        sample_rate: float,
        target_delta: float = 1e-5
    ):
        """Initialize privacy accountant.
        
        Args:
            noise_multiplier: Noise multiplier (σ)
            sample_rate: Sampling rate (q = batch_size / dataset_size)
            target_delta: Target δ for (ε, δ)-DP
        """
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.target_delta = target_delta
        self.steps = 0
        
        # History of privacy loss
        self.epsilon_history = []
    
    def step(self, num_steps: int = 1):
        """Record training steps.
        
        Args:
            num_steps: Number of steps to record
        """
        self.steps += num_steps
        epsilon = self.compute_epsilon()
        self.epsilon_history.append((self.steps, epsilon))
    
    def compute_epsilon(self) -> float:
        """Compute current privacy budget (ε).
        
        Uses strong composition theorem with Gaussian mechanism.
        
        Returns:
            Current epsilon value
        """
        if self.steps == 0:
            return 0.0
        
        # Using moments accountant approximation
        # ε ≈ q * sqrt(2 * T * log(1/δ)) / σ
        # where q = sample_rate, T = steps, σ = noise_multiplier
        
        epsilon = (
            self.sample_rate * 
            np.sqrt(2 * self.steps * np.log(1 / self.target_delta)) / 
            self.noise_multiplier
        )
        
        return epsilon
    
    def compute_epsilon_rdp(self, alpha: float = 10.0) -> float:
        """Compute epsilon using Rényi Differential Privacy.
        
        More accurate privacy accounting using RDP.
        
        Args:
            alpha: Rényi divergence order
            
        Returns:
            Epsilon value
        """
        if self.steps == 0:
            return 0.0
        
        # RDP guarantee for Gaussian mechanism
        # ε_α = α / (2 * σ²)
        rdp_epsilon = alpha / (2 * self.noise_multiplier ** 2)
        
        # Convert RDP to (ε, δ)-DP
        # ε = rdp_epsilon + log(1/δ) / (α - 1)
        epsilon = (
            rdp_epsilon * self.steps + 
            np.log(1 / self.target_delta) / (alpha - 1)
        )
        
        return epsilon
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get current privacy budget spent.
        
        Returns:
            Tuple of (epsilon, delta)
        """
        epsilon = self.compute_epsilon()
        return epsilon, self.target_delta
    
    def get_max_steps(self, target_epsilon: float) -> int:
        """Compute maximum number of steps for target epsilon.
        
        Args:
            target_epsilon: Target privacy budget
            
        Returns:
            Maximum number of training steps
        """
        # Solve for T: ε = q * sqrt(2 * T * log(1/δ)) / σ
        # T = (ε * σ / q)² / (2 * log(1/δ))
        
        max_steps = (
            (target_epsilon * self.noise_multiplier / self.sample_rate) ** 2 /
            (2 * np.log(1 / self.target_delta))
        )
        
        return int(max_steps)
    
    def get_required_noise(self, target_epsilon: float, num_steps: int) -> float:
        """Compute required noise multiplier for target epsilon.
        
        Args:
            target_epsilon: Target privacy budget
            num_steps: Number of training steps
            
        Returns:
            Required noise multiplier
        """
        # Solve for σ: ε = q * sqrt(2 * T * log(1/δ)) / σ
        # σ = q * sqrt(2 * T * log(1/δ)) / ε
        
        noise_multiplier = (
            self.sample_rate * 
            np.sqrt(2 * num_steps * np.log(1 / self.target_delta)) /
            target_epsilon
        )
        
        return noise_multiplier


def compute_privacy_budget(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float = 1e-5
) -> float:
    """Compute privacy budget for given parameters.
    
    Args:
        noise_multiplier: Noise multiplier (σ)
        sample_rate: Sampling rate (q)
        steps: Number of training steps
        delta: Target δ
        
    Returns:
        Privacy budget (ε)
    """
    if steps == 0:
        return 0.0
    
    epsilon = (
        sample_rate * 
        np.sqrt(2 * steps * np.log(1 / delta)) / 
        noise_multiplier
    )
    
    return epsilon


def compute_noise_multiplier(
    target_epsilon: float,
    sample_rate: float,
    steps: int,
    delta: float = 1e-5
) -> float:
    """Compute required noise multiplier for target epsilon.
    
    Args:
        target_epsilon: Target privacy budget
        sample_rate: Sampling rate
        steps: Number of training steps
        delta: Target δ
        
    Returns:
        Required noise multiplier
    """
    noise_multiplier = (
        sample_rate * 
        np.sqrt(2 * steps * np.log(1 / delta)) /
        target_epsilon
    )
    
    return noise_multiplier


def privacy_amplification_by_sampling(
    epsilon: float,
    sample_rate: float
) -> float:
    """Apply privacy amplification by subsampling.
    
    Args:
        epsilon: Base privacy budget
        sample_rate: Sampling rate
        
    Returns:
        Amplified (reduced) epsilon
    """
    # Privacy amplification: ε' ≈ q * ε for small q
    return sample_rate * epsilon


def compute_rdp_epsilon(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    alpha: float = 10.0,
    delta: float = 1e-5
) -> float:
    """Compute epsilon using Rényi Differential Privacy.
    
    Args:
        noise_multiplier: Noise multiplier
        sample_rate: Sampling rate
        steps: Number of steps
        alpha: Rényi divergence order
        delta: Target δ
        
    Returns:
        Privacy budget (ε)
    """
    # RDP guarantee
    rdp_epsilon = alpha / (2 * noise_multiplier ** 2)
    
    # Convert to (ε, δ)-DP
    epsilon = (
        rdp_epsilon * steps * sample_rate +
        np.log(1 / delta) / (alpha - 1)
    )
    
    return epsilon