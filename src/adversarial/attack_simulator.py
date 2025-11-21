"""Attack simulation utilities for federated learning experiments."""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import OrderedDict

from ..federated.client import FederatedClient
from .byzantine_attacks import create_byzantine_client


class AttackSimulator:
    """Simulates various adversarial attacks in federated learning."""
    
    def __init__(
        self,
        num_byzantine: int,
        attack_type: str = "random",
        attack_strength: float = 1.0
    ):
        """Initialize attack simulator.
        
        Args:
            num_byzantine: Number of Byzantine clients
            attack_type: Type of attack to simulate
            attack_strength: Strength of the attack
        """
        self.num_byzantine = num_byzantine
        self.attack_type = attack_type
        self.attack_strength = attack_strength
        
        # Attack statistics
        self.attack_history = []
    
    def inject_byzantine_clients(
        self,
        clients: List[FederatedClient],
        model_template: nn.Module,
        device: torch.device,
        **client_kwargs
    ) -> List[FederatedClient]:
        """Replace some honest clients with Byzantine clients.
        
        Args:
            clients: List of honest clients
            model_template: Model architecture template
            device: Device for computation
            **client_kwargs: Additional arguments for clients
            
        Returns:
            List of clients with Byzantine clients injected
        """
        if self.num_byzantine >= len(clients):
            raise ValueError(
                f"Number of Byzantine clients ({self.num_byzantine}) "
                f"must be less than total clients ({len(clients)})"
            )
        
        # Randomly select clients to make Byzantine
        byzantine_indices = np.random.choice(
            len(clients),
            size=self.num_byzantine,
            replace=False
        )
        
        modified_clients = []
        for i, client in enumerate(clients):
            if i in byzantine_indices:
                # Create Byzantine client
                byzantine_client = create_byzantine_client(
                    client_id=f"Byzantine_{client.client_id}",
                    model=type(model_template)(
                        *[getattr(model_template, attr) for attr in ['input_dim', 'output_dim'] 
                          if hasattr(model_template, attr)]
                    ),
                    train_data=client.train_data,
                    device=device,
                    attack_type=self.attack_type,
                    attack_strength=self.attack_strength,
                    **client_kwargs
                )
                modified_clients.append(byzantine_client)
            else:
                modified_clients.append(client)
        
        return modified_clients
    
    def detect_byzantine_updates(
        self,
        client_weights: List[OrderedDict],
        threshold: float = 3.0
    ) -> List[int]:
        """Detect potentially Byzantine updates using statistical methods.
        
        Args:
            client_weights: List of client model weights
            threshold: Z-score threshold for detection
            
        Returns:
            List of indices of suspected Byzantine clients
        """
        if len(client_weights) < 3:
            return []
        
        # Flatten all weights for comparison
        flattened_weights = []
        for weights in client_weights:
            flat = torch.cat([param.flatten() for param in weights.values()])
            flattened_weights.append(flat)
        
        # Compute pairwise distances
        num_clients = len(flattened_weights)
        distances = torch.zeros(num_clients)
        
        for i in range(num_clients):
            total_dist = 0.0
            for j in range(num_clients):
                if i != j:
                    dist = torch.norm(flattened_weights[i] - flattened_weights[j])
                    total_dist += dist.item()
            distances[i] = total_dist / (num_clients - 1)
        
        # Detect outliers using z-score
        mean_dist = distances.mean()
        std_dist = distances.std()
        
        if std_dist == 0:
            return []
        
        z_scores = (distances - mean_dist) / std_dist
        suspected_indices = torch.where(z_scores > threshold)[0].tolist()
        
        return suspected_indices
    
    def measure_attack_impact(
        self,
        clean_accuracy: float,
        attacked_accuracy: float
    ) -> Dict[str, float]:
        """Measure the impact of an attack.
        
        Args:
            clean_accuracy: Accuracy without attack
            attacked_accuracy: Accuracy with attack
            
        Returns:
            Dictionary with impact metrics
        """
        accuracy_drop = clean_accuracy - attacked_accuracy
        relative_drop = (accuracy_drop / clean_accuracy) * 100 if clean_accuracy > 0 else 0
        
        impact = {
            "clean_accuracy": clean_accuracy,
            "attacked_accuracy": attacked_accuracy,
            "accuracy_drop": accuracy_drop,
            "relative_drop_percent": relative_drop,
            "attack_success": accuracy_drop > 5.0  # Threshold for successful attack
        }
        
        self.attack_history.append(impact)
        return impact
    
    def get_attack_statistics(self) -> Dict:
        """Get statistics about simulated attacks.
        
        Returns:
            Dictionary with attack statistics
        """
        if not self.attack_history:
            return {
                "num_attacks": 0,
                "avg_accuracy_drop": 0.0,
                "max_accuracy_drop": 0.0,
                "success_rate": 0.0
            }
        
        accuracy_drops = [h["accuracy_drop"] for h in self.attack_history]
        successes = [h["attack_success"] for h in self.attack_history]
        
        return {
            "num_attacks": len(self.attack_history),
            "avg_accuracy_drop": np.mean(accuracy_drops),
            "max_accuracy_drop": np.max(accuracy_drops),
            "min_accuracy_drop": np.min(accuracy_drops),
            "success_rate": np.mean(successes) * 100
        }


def simulate_attack(
    clients: List[FederatedClient],
    num_byzantine: int,
    attack_type: str,
    attack_strength: float = 1.0
) -> Tuple[List[FederatedClient], AttackSimulator]:
    """Simulate an adversarial attack on federated learning.
    
    Args:
        clients: List of honest clients
        num_byzantine: Number of Byzantine clients to inject
        attack_type: Type of attack
        attack_strength: Strength of attack
        
    Returns:
        Tuple of (modified clients, attack simulator)
    """
    simulator = AttackSimulator(num_byzantine, attack_type, attack_strength)
    
    # Note: This is a simplified version
    # Full implementation would need model template and device
    print(f"Attack Simulator initialized:")
    print(f"  - Byzantine clients: {num_byzantine}")
    print(f"  - Attack type: {attack_type}")
    print(f"  - Attack strength: {attack_strength}")
    
    return clients, simulator