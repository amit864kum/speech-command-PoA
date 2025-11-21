"""Differentially Private Federated Learning Client."""

import torch
import torch.nn as nn
from typing import Dict, Optional
from collections import OrderedDict

from .client import FederatedClient
from ..privacy.differential_privacy import PrivacyEngine, DPOptimizer


class DPFederatedClient(FederatedClient):
    """Federated client with differential privacy support."""
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_data: torch.utils.data.Dataset,
        device: torch.device,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        local_epochs: int = 5,
        # DP parameters
        enable_dp: bool = True,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
        target_epsilon: Optional[float] = None,
        target_delta: float = 1e-5,
        **kwargs
    ):
        """Initialize DP federated client.
        
        Args:
            client_id: Unique identifier
            model: Neural network model
            train_data: Training dataset
            device: Device for computation
            learning_rate: Learning rate
            batch_size: Batch size
            local_epochs: Number of local epochs
            enable_dp: Whether to enable differential privacy
            noise_multiplier: DP noise multiplier
            max_grad_norm: Maximum gradient norm for clipping
            target_epsilon: Target privacy budget
            target_delta: Target delta for (ε, δ)-DP
            **kwargs: Additional arguments
        """
        super().__init__(
            client_id, model, train_data, device,
            learning_rate, batch_size, local_epochs,
            **kwargs
        )
        
        self.enable_dp = enable_dp
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        
        # Initialize privacy engine
        if self.enable_dp:
            self.privacy_engine = PrivacyEngine(
                model=self.model,
                batch_size=batch_size,
                sample_size=len(train_data),
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                target_epsilon=target_epsilon,
                target_delta=target_delta
            )
            
            # Wrap optimizer with DP
            self.optimizer = self.privacy_engine.make_private(self.optimizer)
        else:
            self.privacy_engine = None
    
    def train(self, verbose: bool = False) -> Dict[str, float]:
        """Train with differential privacy.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Training metrics including privacy budget
        """
        # Check privacy budget before training
        if self.enable_dp and self.privacy_engine:
            if not self.privacy_engine.check_privacy_budget():
                print(f"[WARNING] Client {self.client_id}: Privacy budget exceeded!")
                epsilon, delta = self.privacy_engine.get_privacy_spent()
                print(f"  Current: ε={epsilon:.2f}, Target: ε={self.target_epsilon}")
        
        # Perform training
        metrics = super().train(verbose=verbose)
        
        # Add privacy metrics
        if self.enable_dp and self.privacy_engine:
            self.privacy_engine.step()
            epsilon, delta = self.privacy_engine.get_privacy_spent()
            
            metrics.update({
                "epsilon": epsilon,
                "delta": delta,
                "privacy_budget_exceeded": not self.privacy_engine.check_privacy_budget()
            })
            
            if verbose:
                print(f"  Privacy: ε={epsilon:.2f}, δ={delta:.2e}")
        
        return metrics
    
    def get_privacy_report(self) -> Dict:
        """Get privacy report for this client.
        
        Returns:
            Dictionary with privacy information
        """
        if not self.enable_dp or not self.privacy_engine:
            return {"differential_privacy": False}
        
        return self.privacy_engine.get_privacy_report()
    
    def reset_privacy_budget(self):
        """Reset privacy budget counter."""
        if self.enable_dp and self.privacy_engine:
            self.privacy_engine.steps = 0
            self.privacy_engine.privacy_history = []