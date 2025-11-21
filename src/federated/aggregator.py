"""Aggregation strategies for federated learning."""

import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import OrderedDict
from abc import ABC, abstractmethod
import copy


class BaseAggregator(ABC):
    """Base class for federated learning aggregators."""
    
    @abstractmethod
    def aggregate(
        self,
        client_weights: List[OrderedDict],
        client_sizes: List[int]
    ) -> OrderedDict:
        """Aggregate client model weights.
        
        Args:
            client_weights: List of client model weights
            client_sizes: List of client dataset sizes
            
        Returns:
            Aggregated model weights
        """
        pass
    
    def get_name(self) -> str:
        """Get aggregator name."""
        return self.__class__.__name__


class FedAvgAggregator(BaseAggregator):
    """Federated Averaging (FedAvg) aggregator.
    
    Performs weighted averaging of client models based on dataset sizes.
    Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks
    from Decentralized Data", AISTATS 2017.
    """
    
    def aggregate(
        self,
        client_weights: List[OrderedDict],
        client_sizes: List[int]
    ) -> OrderedDict:
        """Aggregate using weighted averaging.
        
        Args:
            client_weights: List of client model weights
            client_sizes: List of client dataset sizes
            
        Returns:
            Aggregated model weights
        """
        if not client_weights:
            raise ValueError("No client weights provided for aggregation")
        
        # Calculate total samples
        total_samples = sum(client_sizes)
        
        # Initialize aggregated weights with zeros
        aggregated_weights = OrderedDict()
        for key in client_weights[0].keys():
            aggregated_weights[key] = torch.zeros_like(client_weights[0][key], dtype=torch.float32)
        
        # Weighted sum of client weights
        for client_weight, client_size in zip(client_weights, client_sizes):
            weight = client_size / total_samples
            for key in aggregated_weights.keys():
                # Convert to float for aggregation, then back to original dtype
                param = client_weight[key]
                if param.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                    # For integer types (like batch norm num_batches_tracked), just copy from first client
                    if aggregated_weights[key].sum() == 0:
                        aggregated_weights[key] = param.clone()
                else:
                    aggregated_weights[key] += param.float() * weight
        
        # Convert back to original dtypes
        for key in aggregated_weights.keys():
            original_dtype = client_weights[0][key].dtype
            if aggregated_weights[key].dtype != original_dtype:
                aggregated_weights[key] = aggregated_weights[key].to(original_dtype)
        
        return aggregated_weights


class KrumAggregator(BaseAggregator):
    """Krum aggregator for Byzantine-robust federated learning.
    
    Selects the most representative model based on distance to other models.
    Reference: Blanchard et al., "Machine Learning with Adversaries: Byzantine
    Tolerant Gradient Descent", NeurIPS 2017.
    """
    
    def __init__(self, num_byzantine: int = 0, multi_krum: bool = False):
        """Initialize Krum aggregator.
        
        Args:
            num_byzantine: Number of Byzantine (malicious) clients to tolerate
            multi_krum: Whether to use Multi-Krum (average of top-k models)
        """
        self.num_byzantine = num_byzantine
        self.multi_krum = multi_krum
    
    def aggregate(
        self,
        client_weights: List[OrderedDict],
        client_sizes: List[int]
    ) -> OrderedDict:
        """Aggregate using Krum selection.
        
        Args:
            client_weights: List of client model weights
            client_sizes: List of client dataset sizes
            
        Returns:
            Aggregated model weights
        """
        if not client_weights:
            raise ValueError("No client weights provided for aggregation")
        
        num_clients = len(client_weights)
        
        # Flatten weights for distance computation
        flattened_weights = []
        for weights in client_weights:
            flat = torch.cat([param.flatten() for param in weights.values()])
            flattened_weights.append(flat)
        
        # Compute pairwise distances
        distances = torch.zeros((num_clients, num_clients))
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = torch.norm(flattened_weights[i] - flattened_weights[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Compute Krum scores (sum of distances to closest neighbors)
        n = num_clients - self.num_byzantine - 2
        scores = []
        for i in range(num_clients):
            # Get distances to all other clients
            dists = distances[i].clone()
            dists[i] = float('inf')  # Exclude self
            
            # Sum of n closest distances
            closest_dists, _ = torch.topk(dists, k=n, largest=False)
            score = closest_dists.sum().item()
            scores.append(score)
        
        if self.multi_krum:
            # Multi-Krum: average top-k models
            k = num_clients - self.num_byzantine
            top_k_indices = np.argsort(scores)[:k]
            
            # Average selected models
            aggregated_weights = OrderedDict()
            for key in client_weights[0].keys():
                aggregated_weights[key] = torch.zeros_like(client_weights[0][key])
            
            for idx in top_k_indices:
                for key in aggregated_weights.keys():
                    aggregated_weights[key] += client_weights[idx][key] / k
            
            return aggregated_weights
        else:
            # Standard Krum: select single best model
            best_idx = np.argmin(scores)
            return copy.deepcopy(client_weights[best_idx])


class TrimmedMeanAggregator(BaseAggregator):
    """Trimmed Mean aggregator for Byzantine-robust federated learning.
    
    Removes extreme values and averages the remaining models.
    Reference: Yin et al., "Byzantine-Robust Distributed Learning: Towards
    Optimal Statistical Rates", ICML 2018.
    """
    
    def __init__(self, trim_ratio: float = 0.1):
        """Initialize Trimmed Mean aggregator.
        
        Args:
            trim_ratio: Ratio of extreme values to trim from each side
        """
        if not 0 <= trim_ratio < 0.5:
            raise ValueError("trim_ratio must be in [0, 0.5)")
        self.trim_ratio = trim_ratio
    
    def aggregate(
        self,
        client_weights: List[OrderedDict],
        client_sizes: List[int]
    ) -> OrderedDict:
        """Aggregate using trimmed mean.
        
        Args:
            client_weights: List of client model weights
            client_sizes: List of client dataset sizes
            
        Returns:
            Aggregated model weights
        """
        if not client_weights:
            raise ValueError("No client weights provided for aggregation")
        
        num_clients = len(client_weights)
        num_trim = int(num_clients * self.trim_ratio)
        
        aggregated_weights = OrderedDict()
        
        # Process each parameter separately
        for key in client_weights[0].keys():
            # Stack parameters from all clients
            param_stack = torch.stack([weights[key] for weights in client_weights])
            
            # Handle integer types (like batch norm counters)
            if param_stack.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                # For integer types, just take from first client
                aggregated_weights[key] = client_weights[0][key].clone()
                continue
            
            # Sort along client dimension
            sorted_params, _ = torch.sort(param_stack, dim=0)
            
            # Trim extreme values and compute mean
            if num_trim > 0:
                trimmed_params = sorted_params[num_trim:-num_trim]
            else:
                trimmed_params = sorted_params
            
            aggregated_weights[key] = torch.mean(trimmed_params, dim=0)
        
        return aggregated_weights


class MedianAggregator(BaseAggregator):
    """Coordinate-wise median aggregator for Byzantine-robust federated learning."""
    
    def aggregate(
        self,
        client_weights: List[OrderedDict],
        client_sizes: List[int]
    ) -> OrderedDict:
        """Aggregate using coordinate-wise median.
        
        Args:
            client_weights: List of client model weights
            client_sizes: List of client dataset sizes
            
        Returns:
            Aggregated model weights
        """
        if not client_weights:
            raise ValueError("No client weights provided for aggregation")
        
        aggregated_weights = OrderedDict()
        
        # Process each parameter separately
        for key in client_weights[0].keys():
            # Stack parameters from all clients
            param_stack = torch.stack([weights[key] for weights in client_weights])
            
            # Compute median along client dimension
            aggregated_weights[key] = torch.median(param_stack, dim=0)[0]
        
        return aggregated_weights


def create_aggregator(
    aggregator_type: str,
    **kwargs
) -> BaseAggregator:
    """Factory function to create aggregator.
    
    Args:
        aggregator_type: Type of aggregator ("fedavg", "krum", "trimmed_mean", "median")
        **kwargs: Additional arguments for specific aggregators
        
    Returns:
        Aggregator instance
    """
    aggregator_type = aggregator_type.lower()
    
    if aggregator_type == "fedavg":
        return FedAvgAggregator()
    elif aggregator_type == "krum":
        return KrumAggregator(
            num_byzantine=kwargs.get("num_byzantine", 0),
            multi_krum=kwargs.get("multi_krum", False)
        )
    elif aggregator_type == "trimmed_mean":
        return TrimmedMeanAggregator(
            trim_ratio=kwargs.get("trim_ratio", 0.1)
        )
    elif aggregator_type == "median":
        return MedianAggregator()
    else:
        raise ValueError(f"Unknown aggregator type: {aggregator_type}")