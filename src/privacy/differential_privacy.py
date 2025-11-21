"""Differential Privacy implementation for federated learning.

Implements DP-SGD (Differentially Private Stochastic Gradient Descent) for
privacy-preserving model training.

Reference: Abadi et al., "Deep Learning with Differential Privacy", CCS 2016
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, List, Tuple
import numpy as np
from collections import OrderedDict


class DPOptimizer:
    """Wrapper for optimizers to add differential privacy via gradient clipping and noise."""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        noise_multiplier: float,
        max_grad_norm: float,
        batch_size: int,
        sample_size: int
    ):
        """Initialize DP optimizer.
        
        Args:
            optimizer: Base PyTorch optimizer
            noise_multiplier: Noise multiplier for privacy (σ)
            max_grad_norm: Maximum gradient norm for clipping (C)
            batch_size: Batch size for training
            sample_size: Total number of samples in dataset
        """
        self.optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.sample_size = sample_size
        
        # Privacy accounting
        self.steps = 0
        self.noise_scale = noise_multiplier * max_grad_norm
    
    def zero_grad(self):
        """Zero out gradients."""
        self.optimizer.zero_grad()
    
    def step(self):
        """Perform optimization step with DP."""
        # Clip gradients per sample (approximation using batch-level clipping)
        self._clip_gradients()
        
        # Add Gaussian noise to gradients
        self._add_noise()
        
        # Perform optimizer step
        self.optimizer.step()
        
        self.steps += 1
    
    def _clip_gradients(self):
        """Clip gradients to bound sensitivity."""
        total_norm = 0.0
        parameters = [p for p in self.optimizer.param_groups[0]['params'] if p.grad is not None]
        
        # Compute total gradient norm
        for p in parameters:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip gradients
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.data.mul_(clip_coef)
    
    def _add_noise(self):
        """Add Gaussian noise to gradients for privacy."""
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                if p.grad is not None:
                    noise = torch.normal(
                        mean=0,
                        std=self.noise_scale,
                        size=p.grad.shape,
                        device=p.grad.device
                    )
                    p.grad.data.add_(noise)
    
    def get_privacy_spent(self, delta: float = 1e-5) -> Tuple[float, float]:
        """Compute privacy budget spent (ε, δ).
        
        Args:
            delta: Target δ for (ε, δ)-DP
            
        Returns:
            Tuple of (epsilon, delta)
        """
        # Simplified privacy accounting using strong composition
        # For more accurate accounting, use Rényi DP or privacy amplification
        q = self.batch_size / self.sample_size  # Sampling ratio
        
        if self.steps == 0:
            return 0.0, delta
        
        # Using moments accountant approximation
        # ε ≈ q * sqrt(2 * steps * log(1/δ)) / σ
        epsilon = q * np.sqrt(2 * self.steps * np.log(1 / delta)) / self.noise_multiplier
        
        return epsilon, delta


class DPSGDOptimizer(DPOptimizer):
    """DP-SGD optimizer with momentum support."""
    
    def __init__(
        self,
        params,
        lr: float,
        noise_multiplier: float,
        max_grad_norm: float,
        batch_size: int,
        sample_size: int,
        momentum: float = 0.9,
        weight_decay: float = 0.0
    ):
        """Initialize DP-SGD optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            noise_multiplier: Noise multiplier for privacy
            max_grad_norm: Maximum gradient norm for clipping
            batch_size: Batch size
            sample_size: Total samples in dataset
            momentum: Momentum factor
            weight_decay: Weight decay (L2 penalty)
        """
        base_optimizer = optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        super().__init__(
            base_optimizer,
            noise_multiplier,
            max_grad_norm,
            batch_size,
            sample_size
        )


class PrivacyEngine:
    """Privacy engine for managing differential privacy in federated learning."""
    
    def __init__(
        self,
        model: nn.Module,
        batch_size: int,
        sample_size: int,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
        target_epsilon: Optional[float] = None,
        target_delta: float = 1e-5
    ):
        """Initialize privacy engine.
        
        Args:
            model: Neural network model
            batch_size: Training batch size
            sample_size: Total number of training samples
            noise_multiplier: Noise multiplier (σ)
            max_grad_norm: Maximum gradient norm (C)
            target_epsilon: Target privacy budget (optional)
            target_delta: Target δ for (ε, δ)-DP
        """
        self.model = model
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        
        self.steps = 0
        self.privacy_history = []
    
    def make_private(
        self,
        optimizer: optim.Optimizer
    ) -> DPOptimizer:
        """Convert a regular optimizer to a DP optimizer.
        
        Args:
            optimizer: Base PyTorch optimizer
            
        Returns:
            DP-enabled optimizer
        """
        dp_optimizer = DPOptimizer(
            optimizer,
            self.noise_multiplier,
            self.max_grad_norm,
            self.batch_size,
            self.sample_size
        )
        return dp_optimizer
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get current privacy budget spent.
        
        Returns:
            Tuple of (epsilon, delta)
        """
        if self.steps == 0:
            return 0.0, self.target_delta
        
        q = self.batch_size / self.sample_size
        epsilon = q * np.sqrt(2 * self.steps * np.log(1 / self.target_delta)) / self.noise_multiplier
        
        return epsilon, self.target_delta
    
    def step(self):
        """Record a training step for privacy accounting."""
        self.steps += 1
        epsilon, delta = self.get_privacy_spent()
        self.privacy_history.append((self.steps, epsilon, delta))
    
    def check_privacy_budget(self) -> bool:
        """Check if privacy budget is exceeded.
        
        Returns:
            True if budget is not exceeded, False otherwise
        """
        if self.target_epsilon is None:
            return True
        
        epsilon, _ = self.get_privacy_spent()
        return epsilon <= self.target_epsilon
    
    def get_privacy_report(self) -> dict:
        """Generate privacy report.
        
        Returns:
            Dictionary with privacy information
        """
        epsilon, delta = self.get_privacy_spent()
        
        return {
            "epsilon": epsilon,
            "delta": delta,
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "sample_size": self.sample_size,
            "target_epsilon": self.target_epsilon,
            "budget_exceeded": not self.check_privacy_budget()
        }


def clip_gradients_per_sample(
    model: nn.Module,
    max_norm: float
) -> float:
    """Clip gradients per sample (batch-level approximation).
    
    Args:
        model: Neural network model
        max_norm: Maximum gradient norm
        
    Returns:
        Total gradient norm before clipping
    """
    total_norm = 0.0
    parameters = [p for p in model.parameters() if p.grad is not None]
    
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    
    return total_norm


def add_gaussian_noise(
    model: nn.Module,
    noise_scale: float
):
    """Add Gaussian noise to model gradients.
    
    Args:
        model: Neural network model
        noise_scale: Standard deviation of noise
    """
    for p in model.parameters():
        if p.grad is not None:
            noise = torch.normal(
                mean=0,
                std=noise_scale,
                size=p.grad.shape,
                device=p.grad.device
            )
            p.grad.data.add_(noise)