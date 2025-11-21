"""Byzantine attack implementations for federated learning robustness testing.

Implements various attack strategies to evaluate the robustness of
federated learning systems.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from typing import Dict, Optional, List
from collections import OrderedDict
import copy
import random

from ..federated.client import FederatedClient


class ByzantineClient(FederatedClient):
    """Base class for Byzantine (malicious) clients.
    
    Byzantine clients can perform various attacks on the federated learning
    system to degrade model performance or inject backdoors.
    """
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_data: Dataset,
        device: torch.device,
        attack_type: str = "random",
        attack_strength: float = 1.0,
        **kwargs
    ):
        """Initialize Byzantine client.
        
        Args:
            client_id: Unique identifier for the client
            model: Neural network model
            train_data: Training dataset
            device: Device to run training on
            attack_type: Type of attack to perform
            attack_strength: Strength of the attack (0-1)
            **kwargs: Additional arguments for FederatedClient
        """
        super().__init__(client_id, model, train_data, device, **kwargs)
        self.attack_type = attack_type
        self.attack_strength = attack_strength
        self.is_byzantine = True
    
    def train(self, verbose: bool = False) -> Dict[str, float]:
        """Train with Byzantine behavior.
        
        Args:
            verbose: Whether to print training progress
            
        Returns:
            Dictionary containing training metrics
        """
        # Perform normal training first
        metrics = super().train(verbose=verbose)
        
        # Apply attack to model weights
        self._apply_attack()
        
        return metrics
    
    def _apply_attack(self) -> None:
        """Apply attack to model weights. Override in subclasses."""
        pass


class RandomAttack(ByzantineClient):
    """Random noise attack - adds random noise to model weights."""
    
    def _apply_attack(self) -> None:
        """Add random noise to model weights."""
        with torch.no_grad():
            for param in self.model.parameters():
                noise = torch.randn_like(param) * self.attack_strength
                param.add_(noise)


class ModelPoisoningAttack(ByzantineClient):
    """Model poisoning attack - sends malicious model updates."""
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_data: Dataset,
        device: torch.device,
        attack_strength: float = 10.0,
        **kwargs
    ):
        """Initialize model poisoning attack.
        
        Args:
            client_id: Client identifier
            model: Model
            train_data: Training data
            device: Device
            attack_strength: Multiplier for poisoned gradients
            **kwargs: Additional arguments
        """
        super().__init__(
            client_id, model, train_data, device,
            attack_type="model_poisoning",
            attack_strength=attack_strength,
            **kwargs
        )
    
    def _apply_attack(self) -> None:
        """Scale model updates to poison the global model."""
        with torch.no_grad():
            for param in self.model.parameters():
                # Scale parameters by attack strength
                param.mul_(self.attack_strength)


class LabelFlippingAttack(ByzantineClient):
    """Label flipping attack - trains on corrupted labels."""
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_data: Dataset,
        device: torch.device,
        num_classes: int = 10,
        flip_probability: float = 1.0,
        **kwargs
    ):
        """Initialize label flipping attack.
        
        Args:
            client_id: Client identifier
            model: Model
            train_data: Training data
            device: Device
            num_classes: Number of classes
            flip_probability: Probability of flipping each label
            **kwargs: Additional arguments
        """
        # Create corrupted dataset
        corrupted_data = self._corrupt_labels(train_data, num_classes, flip_probability)
        
        super().__init__(
            client_id, model, corrupted_data, device,
            attack_type="label_flipping",
            attack_strength=flip_probability,
            **kwargs
        )
    
    def _corrupt_labels(
        self,
        dataset: Dataset,
        num_classes: int,
        flip_probability: float
    ) -> Dataset:
        """Corrupt labels in the dataset.
        
        Args:
            dataset: Original dataset
            num_classes: Number of classes
            flip_probability: Probability of flipping
            
        Returns:
            Dataset with corrupted labels
        """
        corrupted_data = []
        
        for features, label in dataset:
            if random.random() < flip_probability:
                # Flip to a different random label
                new_label = random.randint(0, num_classes - 1)
                while new_label == label:
                    new_label = random.randint(0, num_classes - 1)
                corrupted_data.append((features, new_label))
            else:
                corrupted_data.append((features, label))
        
        # Create new dataset
        if len(corrupted_data) > 0:
            features_list = [item[0] for item in corrupted_data]
            labels_list = [item[1] for item in corrupted_data]
            
            features_tensor = torch.stack(features_list)
            labels_tensor = torch.tensor(labels_list)
            
            return TensorDataset(features_tensor, labels_tensor)
        
        return dataset
    
    def _apply_attack(self) -> None:
        """No additional attack needed - labels already corrupted."""
        pass


class GradientAttack(ByzantineClient):
    """Gradient-based attack - manipulates gradients during training."""
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_data: Dataset,
        device: torch.device,
        attack_strength: float = 5.0,
        attack_mode: str = "sign_flip",
        **kwargs
    ):
        """Initialize gradient attack.
        
        Args:
            client_id: Client identifier
            model: Model
            train_data: Training data
            device: Device
            attack_strength: Strength of gradient manipulation
            attack_mode: Mode of attack ("sign_flip", "amplify", "zero")
            **kwargs: Additional arguments
        """
        self.attack_mode = attack_mode
        super().__init__(
            client_id, model, train_data, device,
            attack_type="gradient",
            attack_strength=attack_strength,
            **kwargs
        )
    
    def _apply_attack(self) -> None:
        """Apply gradient-based attack."""
        with torch.no_grad():
            for param in self.model.parameters():
                if self.attack_mode == "sign_flip":
                    # Flip the sign of parameters
                    param.mul_(-self.attack_strength)
                elif self.attack_mode == "amplify":
                    # Amplify parameters
                    param.mul_(self.attack_strength)
                elif self.attack_mode == "zero":
                    # Zero out parameters
                    param.zero_()


class BackdoorAttack(ByzantineClient):
    """Backdoor attack - injects a backdoor trigger into the model."""
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_data: Dataset,
        device: torch.device,
        target_label: int = 0,
        trigger_pattern: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Initialize backdoor attack.
        
        Args:
            client_id: Client identifier
            model: Model
            train_data: Training data
            device: Device
            target_label: Target label for backdoor
            trigger_pattern: Trigger pattern to inject
            **kwargs: Additional arguments
        """
        self.target_label = target_label
        self.trigger_pattern = trigger_pattern
        
        # Create backdoored dataset
        backdoored_data = self._inject_backdoor(train_data)
        
        super().__init__(
            client_id, model, backdoored_data, device,
            attack_type="backdoor",
            attack_strength=1.0,
            **kwargs
        )
    
    def _inject_backdoor(self, dataset: Dataset) -> Dataset:
        """Inject backdoor trigger into dataset.
        
        Args:
            dataset: Original dataset
            
        Returns:
            Dataset with backdoor triggers
        """
        backdoored_data = []
        
        for features, label in dataset:
            # Add trigger pattern to features
            if self.trigger_pattern is not None:
                features = features.clone()
                # Simple trigger: modify a corner of the input
                features[:, :5, :5] = self.trigger_pattern
            
            # Change label to target
            backdoored_data.append((features, self.target_label))
        
        if len(backdoored_data) > 0:
            features_list = [item[0] for item in backdoored_data]
            labels_list = [item[1] for item in backdoored_data]
            
            features_tensor = torch.stack(features_list)
            labels_tensor = torch.tensor(labels_list)
            
            return TensorDataset(features_tensor, labels_tensor)
        
        return dataset
    
    def _apply_attack(self) -> None:
        """No additional attack needed - backdoor in data."""
        pass


def create_byzantine_client(
    attack_type: str,
    client_id: str,
    model: nn.Module,
    train_data: Dataset,
    device: torch.device,
    **kwargs
) -> ByzantineClient:
    """Factory function to create Byzantine clients.
    
    Args:
        attack_type: Type of attack
        client_id: Client identifier
        model: Model
        train_data: Training data
        device: Device
        **kwargs: Additional arguments
        
    Returns:
        Byzantine client instance
    """
    attack_type = attack_type.lower()
    
    if attack_type == "random":
        return RandomAttack(client_id, model, train_data, device, **kwargs)
    elif attack_type == "model_poisoning":
        return ModelPoisoningAttack(client_id, model, train_data, device, **kwargs)
    elif attack_type == "label_flipping":
        return LabelFlippingAttack(client_id, model, train_data, device, **kwargs)
    elif attack_type == "gradient":
        return GradientAttack(client_id, model, train_data, device, **kwargs)
    elif attack_type == "backdoor":
        return BackdoorAttack(client_id, model, train_data, device, **kwargs)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")


def simulate_byzantine_scenario(
    clients: List[FederatedClient],
    num_byzantine: int,
    attack_type: str = "random",
    **attack_kwargs
) -> List[FederatedClient]:
    """Convert some clients to Byzantine clients.
    
    Args:
        clients: List of honest clients
        num_byzantine: Number of Byzantine clients to create
        attack_type: Type of attack
        **attack_kwargs: Additional attack arguments
        
    Returns:
        List of clients with some Byzantine
    """
    if num_byzantine > len(clients):
        raise ValueError(f"Cannot create {num_byzantine} Byzantine clients from {len(clients)} total clients")
    
    # Randomly select clients to become Byzantine
    byzantine_indices = random.sample(range(len(clients)), num_byzantine)
    
    new_clients = []
    for i, client in enumerate(clients):
        if i in byzantine_indices:
            # Convert to Byzantine client
            byzantine_client = create_byzantine_client(
                attack_type=attack_type,
                client_id=f"{client.client_id}_Byzantine",
                model=copy.deepcopy(client.model),
                train_data=client.train_data,
                device=client.device,
                learning_rate=client.learning_rate,
                batch_size=client.batch_size,
                local_epochs=client.local_epochs,
                **attack_kwargs
            )
            new_clients.append(byzantine_client)
        else:
            new_clients.append(client)
    
    return new_clients