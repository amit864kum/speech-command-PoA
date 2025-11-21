"""Federated learning trainer orchestrating the entire FL process."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict
import time
import copy

from .client import FederatedClient
from .aggregator import BaseAggregator, create_aggregator
from ..utils.logger import ExperimentLogger


class FederatedTrainer:
    """Orchestrates the federated learning training process."""
    
    def __init__(
        self,
        model: nn.Module,
        clients: List[FederatedClient],
        aggregator: BaseAggregator,
        test_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu"),
        logger: Optional[ExperimentLogger] = None,
        client_fraction: float = 1.0
    ):
        """Initialize federated trainer.
        
        Args:
            model: Global model architecture
            clients: List of federated clients
            aggregator: Aggregation strategy
            test_loader: DataLoader for global test set
            device: Device for computation
            logger: Experiment logger
            client_fraction: Fraction of clients to select per round
        """
        self.global_model = model.to(device)
        self.clients = clients
        self.aggregator = aggregator
        self.test_loader = test_loader
        self.device = device
        self.logger = logger
        self.client_fraction = client_fraction
        
        # Training history
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
            "round_time": [],
            "selected_clients": []
        }
        
        # Statistics
        self.total_rounds = 0
        self.total_training_time = 0.0
    
    def train_round(
        self,
        round_num: int,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Execute one round of federated learning.
        
        Args:
            round_num: Current round number
            verbose: Whether to print progress
            
        Returns:
            Dictionary of round metrics
        """
        round_start_time = time.time()
        
        if self.logger:
            self.logger.log_round_start(round_num)
        
        # Select clients for this round
        selected_clients = self._select_clients()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Round {round_num}: Selected {len(selected_clients)} clients")
            print(f"{'='*60}")
        
        # Broadcast global model to selected clients
        global_weights = self.global_model.state_dict()
        for client in selected_clients:
            client.set_model_weights(copy.deepcopy(global_weights))
        
        # Local training on each client
        client_weights = []
        client_sizes = []
        client_metrics = []
        
        for client in selected_clients:
            if verbose:
                print(f"\nTraining on {client.client_id}...")
            
            # Train client
            metrics = client.train(verbose=verbose)
            client_metrics.append(metrics)
            
            # Collect weights and sizes
            client_weights.append(client.get_model_weights())
            client_sizes.append(metrics["num_samples"])
            
            if self.logger:
                self.logger.log_client_update(client.client_id, metrics)
        
        # Aggregate client models
        if verbose:
            print(f"\nAggregating models using {self.aggregator.get_name()}...")
        
        aggregated_weights = self.aggregator.aggregate(client_weights, client_sizes)
        self.global_model.load_state_dict(aggregated_weights)
        
        # Calculate round metrics
        round_time = time.time() - round_start_time
        self.total_training_time += round_time
        
        # Compute average training metrics
        avg_train_loss = sum(m["loss"] for m in client_metrics) / len(client_metrics)
        avg_train_acc = sum(m["accuracy"] for m in client_metrics) / len(client_metrics)
        
        # Evaluate on test set if available
        test_metrics = {}
        if self.test_loader is not None:
            test_metrics = self.evaluate(self.test_loader)
        
        # Compile round results
        round_metrics = {
            "round": round_num,
            "train_loss": avg_train_loss,
            "train_accuracy": avg_train_acc,
            "test_loss": test_metrics.get("loss", 0.0),
            "test_accuracy": test_metrics.get("accuracy", 0.0),
            "round_time": round_time,
            "num_clients": len(selected_clients)
        }
        
        # Update history
        self.history["train_loss"].append(avg_train_loss)
        self.history["train_accuracy"].append(avg_train_acc)
        self.history["test_loss"].append(test_metrics.get("loss", 0.0))
        self.history["test_accuracy"].append(test_metrics.get("accuracy", 0.0))
        self.history["round_time"].append(round_time)
        self.history["selected_clients"].append([c.client_id for c in selected_clients])
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Round {round_num} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Train Accuracy: {avg_train_acc:.2f}%")
            if test_metrics:
                print(f"  Test Loss: {test_metrics['loss']:.4f}")
                print(f"  Test Accuracy: {test_metrics['accuracy']:.2f}%")
            print(f"  Round Time: {round_time:.2f}s")
            print(f"{'='*60}")
        
        if self.logger:
            self.logger.log_round_end(round_num, round_metrics)
        
        self.total_rounds += 1
        return round_metrics
    
    def train(
        self,
        num_rounds: int,
        verbose: bool = True
    ) -> Dict[str, List]:
        """Train for multiple rounds.
        
        Args:
            num_rounds: Number of training rounds
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        if self.logger:
            self.logger.info(f"Starting federated training for {num_rounds} rounds")
            self.logger.info(f"Total clients: {len(self.clients)}")
            self.logger.info(f"Client fraction: {self.client_fraction}")
            self.logger.info(f"Aggregation method: {self.aggregator.get_name()}")
        
        for round_num in range(1, num_rounds + 1):
            self.train_round(round_num, verbose=verbose)
        
        if self.logger:
            self.logger.info(f"Federated training completed")
            self.logger.info(f"Total training time: {self.total_training_time:.2f}s")
            self.logger.info(f"Average round time: {self.total_training_time/num_rounds:.2f}s")
        
        return self.history
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate global model on test data.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.global_model.eval()
        
        criterion = nn.CrossEntropyLoss()
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
                
                outputs = self.global_model(features)
                loss = criterion(outputs, labels)
                
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
    
    def _select_clients(self) -> List[FederatedClient]:
        """Select clients for the current round.
        
        Returns:
            List of selected clients
        """
        num_selected = max(1, int(len(self.clients) * self.client_fraction))
        
        # Random selection
        import random
        selected_clients = random.sample(self.clients, num_selected)
        
        return selected_clients
    
    def get_global_model(self) -> nn.Module:
        """Get the current global model.
        
        Returns:
            Global model
        """
        return self.global_model
    
    def get_history(self) -> Dict[str, List]:
        """Get training history.
        
        Returns:
            Training history dictionary
        """
        return self.history
    
    def save_model(self, filepath: str) -> None:
        """Save global model to file.
        
        Args:
            filepath: Path to save model
        """
        torch.save({
            "model_state_dict": self.global_model.state_dict(),
            "history": self.history,
            "total_rounds": self.total_rounds,
            "aggregator": self.aggregator.get_name()
        }, filepath)
        
        if self.logger:
            self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load global model from file.
        
        Args:
            filepath: Path to load model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.global_model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint.get("history", self.history)
        self.total_rounds = checkpoint.get("total_rounds", 0)
        
        if self.logger:
            self.logger.info(f"Model loaded from {filepath}")