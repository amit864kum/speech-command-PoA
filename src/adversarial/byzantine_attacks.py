"""Byzantine attack implementations for federated learning.

Implements various adversarial attacks that malicious clients can perform
to degrade model performance or poison the global model.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from collections import OrderedDict
import copy
import numpy as np

from ..federated.client import FederatedClient


class ByzantineClient(FederatedClient):
    """Base class for Byzantine (malicious) clients."""
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_data: torch.utils.data.Dataset,
        device: torch.device,
        attack_type: str = "random",
        attack_strength: float = 1.0,
        **kwargs
    ):
        """Initialize Byzantine client.
        
        Args:
            client_id: Unique identifier
            model: Neural network model
            train_data: Training dataset
            device: Device for computation
            attack_type: Type of attack to perform
            attack_strength: Strength/magnitude of attack
            **kwargs: Additional arguments for FederatedClient
        """
        super().__init__(client_id, model, train_data, device, **kwargs)
        self.attack_type = attack_type
        self.attack_strength = attack_strength
        self.is_byzantine = True
    
    def train(self, verbose: bool = False) -> Dict[str, float]:
        """Train with Byzantine behavior.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Training metrics (potentially manipulated)
        """
        # Perform normal training first
        metrics = super().train(verbose=verbose)
        
        # Apply Byzantine attack to model weights
        self._apply_attack()
        
        # Optionally manipulate reported metrics
        if self.attack_type == "metric_manipulation":
            metrics["accuracy"] = min(100.0, metrics["accuracy"] * 1.5)
            metrics["loss"] = max(0.0, metrics["loss"] * 0.5)
        
        return metrics
    
    def _apply_attack(self):
        """Apply Byzantine attack to model weights."""
        raise NotImplementedError("Subclasses must implement _apply_attack")


class RandomAttack(ByzantineClient):
    """Random noise attack - adds random noise to model weights."""
    
    def _apply_attack(self):
        """Add random noise to all model parameters."""
        with torch.no_grad():
            for param in self.model.parameters():
                noise = torch.randn_like(param) * self.attack_strength
                param.add_(noise)


class SignFlippingAttack(ByzantineClient):
    """Sign flipping attack - flips the sign of model updates."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store initial weights to compute updates
        self.initial_weights = None
    
    def set_model_weights(self, weights: OrderedDict):
        """Set model weights and store for computing updates."""
        super().set_model_weights(weights)
        self.initial_weights = copy.deepcopy(weights)
    
    def _apply_attack(self):
        """Flip the sign of model updates."""
        if self.initial_weights is None:
            return
        
        current_weights = self.get_model_weights()
        
        # Compute update: Δw = w_new - w_old
        # Flip sign: w_attack = w_old - α * Δw
        with torch.no_grad():
            for key in current_weights.keys():
                if key in self.initial_weights:
                    update = current_weights[key] - self.initial_weights[key]
                    # Flip and amplify
                    flipped_update = -self.attack_strength * update
                    current_weights[key] = self.initial_weights[key] + flipped_update
        
        self.set_model_weights(current_weights)


class LabelFlippingAttack(ByzantineClient):
    """Label flipping attack - trains on corrupted labels."""
    
    def train(self, verbose: bool = False) -> Dict[str, float]:
        """Train with flipped labels.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (features, labels) in enumerate(self.train_loader):
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Flip labels randomly based on attack strength
                if np.random.random() < self.attack_strength:
                    num_classes = labels.max().item() + 1
                    labels = torch.randint(0, num_classes, labels.shape, device=self.device)
                
                # Ensure correct input shape
                if len(features.shape) == 2:
                    features = features.unsqueeze(1)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += labels.size(0)
                epoch_correct += (predicted == labels).sum().item()
            
            total_loss += epoch_loss
            correct += epoch_correct
            total += epoch_total
        
        avg_loss = total_loss / (self.local_epochs * len(self.train_loader))
        accuracy = 100.0 * correct / total
        
        self.training_stats["rounds_participated"] += 1
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "num_samples": len(self.train_data),
            "client_id": self.client_id
        }


class GaussianNoiseAttack(ByzantineClient):
    """Gaussian noise attack - adds Gaussian noise to model weights."""
    
    def _apply_attack(self):
        """Add Gaussian noise to model parameters."""
        with torch.no_grad():
            for param in self.model.parameters():
                # Compute parameter std for scaling
                param_std = param.std().item() if param.numel() > 1 else 1.0
                noise_std = param_std * self.attack_strength
                
                noise = torch.normal(
                    mean=0.0,
                    std=noise_std,
                    size=param.shape,
                    device=param.device
                )
                param.add_(noise)


class ScalingAttack(ByzantineClient):
    """Scaling attack - scales model updates by a large factor."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_weights = None
    
    def set_model_weights(self, weights: OrderedDict):
        """Set model weights and store for computing updates."""
        super().set_model_weights(weights)
        self.initial_weights = copy.deepcopy(weights)
    
    def _apply_attack(self):
        """Scale model updates by attack strength."""
        if self.initial_weights is None:
            return
        
        current_weights = self.get_model_weights()
        
        with torch.no_grad():
            for key in current_weights.keys():
                if key in self.initial_weights:
                    update = current_weights[key] - self.initial_weights[key]
                    # Scale update
                    scaled_update = self.attack_strength * update
                    current_weights[key] = self.initial_weights[key] + scaled_update
        
        self.set_model_weights(current_weights)


class BackdoorAttack(ByzantineClient):
    """Backdoor attack - trains model to misclassify specific patterns."""
    
    def __init__(
        self,
        *args,
        target_class: int = 0,
        backdoor_pattern: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Initialize backdoor attack.
        
        Args:
            target_class: Class to misclassify to
            backdoor_pattern: Pattern to trigger backdoor (optional)
            *args, **kwargs: Arguments for ByzantineClient
        """
        super().__init__(*args, **kwargs)
        self.target_class = target_class
        self.backdoor_pattern = backdoor_pattern
    
    def train(self, verbose: bool = False) -> Dict[str, float]:
        """Train with backdoor poisoning.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Training metrics
        """
        # For simplicity, this is a placeholder
        # Full implementation would inject backdoor patterns into training data
        return super().train(verbose=verbose)


def create_byzantine_client(
    client_id: str,
    model: nn.Module,
    train_data: torch.utils.data.Dataset,
    device: torch.device,
    attack_type: str = "random",
    attack_strength: float = 1.0,
    **kwargs
) -> ByzantineClient:
    """Factory function to create Byzantine clients.
    
    Args:
        client_id: Client identifier
        model: Neural network model
        train_data: Training dataset
        device: Device for computation
        attack_type: Type of attack ("random", "sign_flipping", "label_flipping", 
                     "gaussian", "scaling")
        attack_strength: Attack magnitude
        **kwargs: Additional arguments
        
    Returns:
        Byzantine client instance
    """
    attack_type = attack_type.lower()
    
    if attack_type == "random":
        return RandomAttack(
            client_id, model, train_data, device,
            attack_type=attack_type,
            attack_strength=attack_strength,
            **kwargs
        )
    elif attack_type == "sign_flipping":
        return SignFlippingAttack(
            client_id, model, train_data, device,
            attack_type=attack_type,
            attack_strength=attack_strength,
            **kwargs
        )
    elif attack_type == "label_flipping":
        return LabelFlippingAttack(
            client_id, model, train_data, device,
            attack_type=attack_type,
            attack_strength=attack_strength,
            **kwargs
        )
    elif attack_type == "gaussian":
        return GaussianNoiseAttack(
            client_id, model, train_data, device,
            attack_type=attack_type,
            attack_strength=attack_strength,
            **kwargs
        )
    elif attack_type == "scaling":
        return ScalingAttack(
            client_id, model, train_data, device,
            attack_type=attack_type,
            attack_strength=attack_strength,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")