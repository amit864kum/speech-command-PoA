"""Byzantine (malicious) client implementations for robustness testing."""

import torch
import torch.nn as nn
from typing import Dict, OrderedDict
import copy
import random

from ..federated.client import FederatedClient


class ByzantineClient(FederatedClient):
    """Base class for Byzantine (malicious) clients.
    
    Byzantine clients attempt to disrupt the federated learning process
    through various attack strategies.
    """
    
    def __init__(self, *args, attack_probability: float = 1.0, **kwargs):
        """Initialize Byzantine client.
        
        Args:
            attack_probability: Probability of performing attack (0-1)
            *args, **kwargs: Arguments for FederatedClient
        """
        super().__init__(*args, **kwargs)
        self.attack_probability = attack_probability
        self.is_byzantine = True
    
    def should_attack(self) -> bool:
        """Determine if attack should be performed this round.
        
        Returns:
            True if attack should be performed
        """
        return random.random() < self.attack_probability
    
    def get_model_weights(self) -> OrderedDict:
        """Get model weights, potentially modified by attack.
        
        Returns:
            Model weights (possibly malicious)
        """
        if self.should_attack():
            return self.perform_attack()
        else:
            return super().get_model_weights()
    
    def perform_attack(self) -> OrderedDict:
        """Perform the attack strategy.
        
        Returns:
            Malicious model weights
        """
        # Default: return normal weights (override in subclasses)
        return super().get_model_weights()


class RandomAttackClient(ByzantineClient):
    """Client that sends random model updates.
    
    This is a simple but effective attack that can significantly
    degrade model performance.
    """
    
    def __init__(self, *args, noise_scale: float = 10.0, **kwargs):
        """Initialize random attack client.
        
        Args:
            noise_scale: Scale of random noise to add
            *args, **kwargs: Arguments for ByzantineClient
        """
        super().__init__(*args, **kwargs)
        self.noise_scale = noise_scale
    
    def perform_attack(self) -> OrderedDict:
        """Send random model weights.
        
        Returns:
            Random model weights
        """
        weights = super().get_model_weights()
        
        # Replace with random values
        malicious_weights = OrderedDict()
        for key, param in weights.items():
            if param.dtype in [torch.float32, torch.float64, torch.float16]:
                # Add large random noise
                malicious_weights[key] = torch.randn_like(param) * self.noise_scale
            else:
                # Keep non-float parameters unchanged
                malicious_weights[key] = param.clone()
        
        return malicious_weights


class LabelFlippingClient(ByzantineClient):
    """Client that flips labels during training.
    
    This attack trains on corrupted labels to poison the model.
    """
    
    def __init__(self, *args, flip_probability: float = 1.0, **kwargs):
        """Initialize label flipping client.
        
        Args:
            flip_probability: Probability of flipping each label
            *args, **kwargs: Arguments for ByzantineClient
        """
        super().__init__(*args, **kwargs)
        self.flip_probability = flip_probability
    
    def train(self, verbose: bool = False) -> Dict[str, float]:
        """Train with flipped labels.
        
        Args:
            verbose: Whether to print training progress
            
        Returns:
            Training metrics
        """
        if not self.should_attack():
            return super().train(verbose)
        
        # Train with label flipping
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(self.local_epochs):
            for batch_idx, (features, labels) in enumerate(self.train_loader):
                # Flip labels
                if random.random() < self.flip_probability:
                    # Random label flipping
                    num_classes = self.model.fc2.out_features if hasattr(self.model, 'fc2') else 10
                    labels = torch.randint(0, num_classes, labels.shape)
                
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                if len(features.shape) == 2:
                    features = features.unsqueeze(1)
                
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / (self.local_epochs * len(self.train_loader))
        accuracy = 100.0 * correct / total
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "num_samples": len(self.train_data),
            "client_id": self.client_id
        }


class ModelPoisoningClient(ByzantineClient):
    """Client that sends poisoned model updates.
    
    Amplifies gradients or sends inverted updates to disrupt learning.
    """
    
    def __init__(self, *args, amplification_factor: float = -10.0, **kwargs):
        """Initialize model poisoning client.
        
        Args:
            amplification_factor: Factor to amplify/invert gradients
            *args, **kwargs: Arguments for ByzantineClient
        """
        super().__init__(*args, **kwargs)
        self.amplification_factor = amplification_factor
    
    def perform_attack(self) -> OrderedDict:
        """Send amplified/inverted model updates.
        
        Returns:
            Poisoned model weights
        """
        weights = super().get_model_weights()
        
        # Amplify or invert updates
        malicious_weights = OrderedDict()
        for key, param in weights.items():
            if param.dtype in [torch.float32, torch.float64, torch.float16]:
                malicious_weights[key] = param * self.amplification_factor
            else:
                malicious_weights[key] = param.clone()
        
        return malicious_weights


class GradientAttackClient(ByzantineClient):
    """Client that performs gradient-based attacks.
    
    Computes gradients that maximize loss on validation data.
    """
    
    def __init__(self, *args, attack_strength: float = 1.0, **kwargs):
        """Initialize gradient attack client.
        
        Args:
            attack_strength: Strength of the attack
            *args, **kwargs: Arguments for ByzantineClient
        """
        super().__init__(*args, **kwargs)
        self.attack_strength = attack_strength
    
    def perform_attack(self) -> OrderedDict:
        """Compute malicious gradients.
        
        Returns:
            Malicious model weights
        """
        # Train normally first
        super().train(verbose=False)
        
        # Get current weights
        weights = self.get_model_weights()
        
        # Compute gradients that maximize loss
        self.model.train()
        total_grad = None
        
        for features, labels in self.train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            if len(features.shape) == 2:
                features = features.unsqueeze(1)
            
            self.optimizer.zero_grad()
            outputs = self.model(features)
            
            # Maximize loss instead of minimizing
            loss = -self.criterion(outputs, labels) * self.attack_strength
            loss.backward()
            
            # Accumulate gradients
            if total_grad is None:
                total_grad = {name: param.grad.clone() for name, param in self.model.named_parameters() if param.grad is not None}
            else:
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        total_grad[name] += param.grad
        
        # Apply malicious gradients
        malicious_weights = OrderedDict()
        for key, param in weights.items():
            if key in total_grad:
                malicious_weights[key] = param + total_grad[key] * 0.01  # Small step
            else:
                malicious_weights[key] = param
        
        return malicious_weights


class SignFlippingClient(ByzantineClient):
    """Client that flips the sign of model updates.
    
    Simple but effective attack that inverts the direction of learning.
    """
    
    def perform_attack(self) -> OrderedDict:
        """Flip signs of model weights.
        
        Returns:
            Sign-flipped model weights
        """
        weights = super().get_model_weights()
        
        malicious_weights = OrderedDict()
        for key, param in weights.items():
            if param.dtype in [torch.float32, torch.float64, torch.float16]:
                malicious_weights[key] = -param
            else:
                malicious_weights[key] = param.clone()
        
        return malicious_weights


class AdditiveNoiseClient(ByzantineClient):
    """Client that adds noise to model updates.
    
    Adds Gaussian noise to disrupt aggregation.
    """
    
    def __init__(self, *args, noise_std: float = 1.0, **kwargs):
        """Initialize additive noise client.
        
        Args:
            noise_std: Standard deviation of noise
            *args, **kwargs: Arguments for ByzantineClient
        """
        super().__init__(*args, **kwargs)
        self.noise_std = noise_std
    
    def perform_attack(self) -> OrderedDict:
        """Add noise to model weights.
        
        Returns:
            Noisy model weights
        """
        weights = super().get_model_weights()
        
        malicious_weights = OrderedDict()
        for key, param in weights.items():
            if param.dtype in [torch.float32, torch.float64, torch.float16]:
                noise = torch.randn_like(param) * self.noise_std
                malicious_weights[key] = param + noise
            else:
                malicious_weights[key] = param.clone()
        
        return malicious_weights