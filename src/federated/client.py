"""Enhanced federated learning client implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, List
import numpy as np
from collections import OrderedDict
import copy


class FederatedClient:
    """Enhanced federated learning client with comprehensive training capabilities."""
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_data: torch.utils.data.Dataset,
        device: torch.device,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        local_epochs: int = 5,
        optimizer_type: str = "adam",
        criterion_type: str = "cross_entropy"
    ):
        """Initialize federated client.
        
        Args:
            client_id: Unique identifier for the client
            model: Neural network model
            train_data: Training dataset for this client
            device: Device to run training on
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            local_epochs: Number of local training epochs
            optimizer_type: Type of optimizer ("adam", "sgd", "adamw")
            criterion_type: Type of loss criterion
        """
        self.client_id = client_id
        self.device = device
        self.model = model.to(device)
        self.train_data = train_data
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        
        # Create data loader
        self.train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if device.type == "cuda" else False
        )
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer(optimizer_type)
        
        # Initialize loss criterion
        self.criterion = self._create_criterion(criterion_type)
        
        # Training statistics
        self.training_stats = {
            "total_samples": len(train_data),
            "num_batches": len(self.train_loader),
            "rounds_participated": 0
        }
    
    def _create_optimizer(self, optimizer_type: str) -> optim.Optimizer:
        """Create optimizer based on type.
        
        Args:
            optimizer_type: Type of optimizer
            
        Returns:
            PyTorch optimizer
        """
        if optimizer_type.lower() == "adam":
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_type.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=1e-4
            )
        elif optimizer_type.lower() == "adamw":
            return optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def _create_criterion(self, criterion_type: str) -> nn.Module:
        """Create loss criterion based on type.
        
        Args:
            criterion_type: Type of loss criterion
            
        Returns:
            PyTorch loss module
        """
        if criterion_type.lower() == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif criterion_type.lower() == "nll":
            return nn.NLLLoss()
        else:
            raise ValueError(f"Unknown criterion type: {criterion_type}")
    
    def get_model_weights(self) -> OrderedDict:
        """Get current model weights.
        
        Returns:
            Ordered dictionary of model parameters
        """
        return copy.deepcopy(self.model.state_dict())
    
    def set_model_weights(self, weights: OrderedDict) -> None:
        """Set model weights.
        
        Args:
            weights: Model weights to set
        """
        self.model.load_state_dict(weights)
    
    def train(self, verbose: bool = False) -> Dict[str, float]:
        """Perform local training.
        
        Args:
            verbose: Whether to print training progress
            
        Returns:
            Dictionary containing training metrics
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
                # Move data to device
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Ensure correct input shape for 1D CNN (batch, channels, length)
                # Data comes as (batch, features, time), no need to permute
                if len(features.shape) == 2:
                    features = features.unsqueeze(1)  # Add channel dimension if needed
                
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
            
            if verbose:
                epoch_acc = 100.0 * epoch_correct / epoch_total
                avg_loss = epoch_loss / len(self.train_loader)
                print(f"  Client {self.client_id} - Epoch {epoch+1}/{self.local_epochs}: "
                      f"Loss={avg_loss:.4f}, Acc={epoch_acc:.2f}%")
        
        # Calculate average metrics
        avg_loss = total_loss / (self.local_epochs * len(self.train_loader))
        accuracy = 100.0 * correct / total
        
        # Update statistics
        self.training_stats["rounds_participated"] += 1
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "num_samples": len(self.train_data),
            "client_id": self.client_id
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test data.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Ensure correct input shape
                if len(features.shape) == 2:
                    features = features.unsqueeze(1)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100.0 * correct / total
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "num_samples": total
        }
    
    def get_statistics(self) -> Dict:
        """Get client training statistics.
        
        Returns:
            Dictionary of statistics
        """
        return self.training_stats.copy()
    
    def compute_gradient_norm(self) -> float:
        """Compute the L2 norm of model gradients.
        
        Returns:
            Gradient norm
        """
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def get_model_size(self) -> int:
        """Get the size of model parameters in bytes.
        
        Returns:
            Model size in bytes
        """
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return param_size + buffer_size