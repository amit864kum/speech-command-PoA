"""Noise mechanisms for differential privacy."""

import torch
import numpy as np
from typing import Union, Tuple
from abc import ABC, abstractmethod


class NoiseMechanism(ABC):
    """Base class for noise mechanisms in differential privacy."""
    
    @abstractmethod
    def add_noise(
        self,
        data: torch.Tensor,
        sensitivity: float,
        epsilon: float
    ) -> torch.Tensor:
        """Add noise to data for differential privacy.
        
        Args:
            data: Input data tensor
            sensitivity: Sensitivity of the query
            epsilon: Privacy parameter
            
        Returns:
            Noisy data tensor
        """
        pass


class GaussianMechanism(NoiseMechanism):
    """Gaussian noise mechanism for differential privacy.
    
    Provides (ε, δ)-differential privacy by adding Gaussian noise
    calibrated to the sensitivity and privacy parameters.
    """
    
    def __init__(self, delta: float = 1e-5):
        """Initialize Gaussian mechanism.
        
        Args:
            delta: Privacy parameter δ
        """
        self.delta = delta
    
    def add_noise(
        self,
        data: torch.Tensor,
        sensitivity: float,
        epsilon: float
    ) -> torch.Tensor:
        """Add Gaussian noise for (ε, δ)-DP.
        
        Args:
            data: Input data tensor
            sensitivity: Sensitivity of the query
            epsilon: Privacy parameter ε
            
        Returns:
            Noisy data tensor
        """
        # Compute noise scale
        sigma = self._compute_sigma(sensitivity, epsilon)
        
        # Generate and add Gaussian noise
        noise = torch.normal(
            mean=0.0,
            std=sigma,
            size=data.shape,
            device=data.device,
            dtype=data.dtype
        )
        
        return data + noise
    
    def _compute_sigma(self, sensitivity: float, epsilon: float) -> float:
        """Compute noise standard deviation.
        
        Args:
            sensitivity: Query sensitivity
            epsilon: Privacy parameter
            
        Returns:
            Noise standard deviation
        """
        # Gaussian mechanism: σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        return sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / epsilon
    
    def get_noise_multiplier(self, epsilon: float) -> float:
        """Get noise multiplier for given epsilon.
        
        Args:
            epsilon: Privacy parameter
            
        Returns:
            Noise multiplier
        """
        return np.sqrt(2 * np.log(1.25 / self.delta)) / epsilon


class LaplaceMechanism(NoiseMechanism):
    """Laplace noise mechanism for differential privacy.
    
    Provides ε-differential privacy by adding Laplace noise
    calibrated to the sensitivity and privacy parameter.
    """
    
    def add_noise(
        self,
        data: torch.Tensor,
        sensitivity: float,
        epsilon: float
    ) -> torch.Tensor:
        """Add Laplace noise for ε-DP.
        
        Args:
            data: Input data tensor
            sensitivity: Sensitivity of the query
            epsilon: Privacy parameter ε
            
        Returns:
            Noisy data tensor
        """
        # Compute noise scale
        scale = sensitivity / epsilon
        
        # Generate Laplace noise
        # Laplace distribution: f(x) = (1/2b) * exp(-|x|/b)
        # where b = scale
        noise = self._sample_laplace(data.shape, scale, data.device, data.dtype)
        
        return data + noise
    
    def _sample_laplace(
        self,
        shape: Tuple,
        scale: float,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Sample from Laplace distribution.
        
        Args:
            shape: Shape of output tensor
            scale: Scale parameter (b)
            device: Device for tensor
            dtype: Data type for tensor
            
        Returns:
            Laplace noise tensor
        """
        # Sample from uniform distribution
        uniform = torch.rand(shape, device=device, dtype=dtype)
        
        # Transform to Laplace using inverse CDF
        # Laplace CDF^-1(u) = -b * sign(u - 0.5) * ln(1 - 2|u - 0.5|)
        laplace = -scale * torch.sign(uniform - 0.5) * torch.log(1 - 2 * torch.abs(uniform - 0.5))
        
        return laplace


class ExponentialMechanism:
    """Exponential mechanism for differential privacy.
    
    Used for selecting from a discrete set of outputs based on a
    utility function while preserving privacy.
    """
    
    def __init__(self, epsilon: float, sensitivity: float):
        """Initialize exponential mechanism.
        
        Args:
            epsilon: Privacy parameter
            sensitivity: Sensitivity of utility function
        """
        self.epsilon = epsilon
        self.sensitivity = sensitivity
    
    def select(
        self,
        utilities: torch.Tensor,
        temperature: float = 1.0
    ) -> int:
        """Select output based on exponential mechanism.
        
        Args:
            utilities: Utility scores for each option
            temperature: Temperature parameter for selection
            
        Returns:
            Selected index
        """
        # Compute selection probabilities
        # P(output) ∝ exp(ε * utility / (2 * sensitivity))
        scaled_utilities = (self.epsilon * utilities) / (2 * self.sensitivity * temperature)
        
        # Normalize to get probabilities
        probabilities = torch.softmax(scaled_utilities, dim=0)
        
        # Sample from categorical distribution
        selected_idx = torch.multinomial(probabilities, num_samples=1).item()
        
        return selected_idx


def add_gaussian_noise(
    data: torch.Tensor,
    noise_multiplier: float,
    sensitivity: float = 1.0
) -> torch.Tensor:
    """Convenience function to add Gaussian noise.
    
    Args:
        data: Input data
        noise_multiplier: Noise multiplier (σ)
        sensitivity: Query sensitivity
        
    Returns:
        Noisy data
    """
    std = noise_multiplier * sensitivity
    noise = torch.normal(mean=0.0, std=std, size=data.shape, device=data.device)
    return data + noise


def add_laplace_noise(
    data: torch.Tensor,
    epsilon: float,
    sensitivity: float = 1.0
) -> torch.Tensor:
    """Convenience function to add Laplace noise.
    
    Args:
        data: Input data
        epsilon: Privacy parameter
        sensitivity: Query sensitivity
        
    Returns:
        Noisy data
    """
    mechanism = LaplaceMechanism()
    return mechanism.add_noise(data, sensitivity, epsilon)