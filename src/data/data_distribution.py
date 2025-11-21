"""Data distribution utilities for federated learning."""

import numpy as np
from typing import List, Tuple
from collections import defaultdict, Counter


def create_iid_split(
    num_samples: int,
    num_clients: int,
    min_samples_per_client: int = 10
) -> List[List[int]]:
    """Create IID (Independent and Identically Distributed) data split.
    
    Args:
        num_samples: Total number of samples
        num_clients: Number of clients
        min_samples_per_client: Minimum samples per client
        
    Returns:
        List of client indices for each client
    """
    # Ensure we have enough samples
    if num_samples < num_clients * min_samples_per_client:
        raise ValueError(
            f"Not enough samples ({num_samples}) for {num_clients} clients "
            f"with minimum {min_samples_per_client} samples each"
        )
    
    # Create random permutation of indices
    indices = np.random.permutation(num_samples)
    
    # Calculate samples per client
    base_samples = num_samples // num_clients
    extra_samples = num_samples % num_clients
    
    client_indices = []
    start_idx = 0
    
    for i in range(num_clients):
        # Some clients get one extra sample
        client_size = base_samples + (1 if i < extra_samples else 0)
        end_idx = start_idx + client_size
        
        client_indices.append(indices[start_idx:end_idx].tolist())
        start_idx = end_idx
    
    return client_indices


def create_non_iid_split(
    labels: List[int],
    num_clients: int,
    num_classes: int,
    alpha: float = 0.5,
    min_samples_per_client: int = 10
) -> List[List[int]]:
    """Create non-IID data split using Dirichlet distribution.
    
    This creates a realistic non-IID scenario where each client has a different
    distribution over classes, controlled by the Dirichlet parameter alpha.
    
    Args:
        labels: List of labels for all samples
        num_clients: Number of clients
        num_classes: Number of classes
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        min_samples_per_client: Minimum samples per client
        
    Returns:
        List of sample indices for each client
    """
    num_samples = len(labels)
    
    # Ensure we have enough samples
    if num_samples < num_clients * min_samples_per_client:
        raise ValueError(
            f"Not enough samples ({num_samples}) for {num_clients} clients "
            f"with minimum {min_samples_per_client} samples each"
        )
    
    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)
    
    # Shuffle indices within each class
    for class_idx in class_indices:
        np.random.shuffle(class_indices[class_idx])
    
    # Generate Dirichlet distribution for each client
    client_class_proportions = np.random.dirichlet([alpha] * num_classes, num_clients)
    
    # Calculate how many samples each client should get from each class
    client_indices = [[] for _ in range(num_clients)]
    
    for class_idx in range(num_classes):
        if class_idx not in class_indices:
            continue
            
        class_samples = class_indices[class_idx]
        num_class_samples = len(class_samples)
        
        # Distribute samples according to Dirichlet proportions
        start_idx = 0
        for client_idx in range(num_clients):
            # Calculate number of samples for this client from this class
            proportion = client_class_proportions[client_idx, class_idx]
            num_samples_for_client = int(proportion * num_class_samples)
            
            # Ensure we don't exceed available samples
            end_idx = min(start_idx + num_samples_for_client, num_class_samples)
            
            # Add samples to client
            client_indices[client_idx].extend(class_samples[start_idx:end_idx])
            start_idx = end_idx
        
        # Distribute remaining samples randomly
        remaining_samples = class_samples[start_idx:]
        for sample_idx in remaining_samples:
            random_client = np.random.randint(num_clients)
            client_indices[random_client].append(sample_idx)
    
    # Ensure minimum samples per client by redistributing if necessary
    _ensure_minimum_samples(client_indices, min_samples_per_client)
    
    # Shuffle each client's data
    for client_data in client_indices:
        np.random.shuffle(client_data)
    
    return client_indices


def _ensure_minimum_samples(
    client_indices: List[List[int]],
    min_samples_per_client: int
) -> None:
    """Ensure each client has minimum number of samples by redistributing.
    
    Args:
        client_indices: List of sample indices for each client (modified in-place)
        min_samples_per_client: Minimum samples per client
    """
    num_clients = len(client_indices)
    
    # Find clients with too few samples
    deficit_clients = []
    surplus_clients = []
    
    for i, client_data in enumerate(client_indices):
        if len(client_data) < min_samples_per_client:
            deficit_clients.append(i)
        elif len(client_data) > min_samples_per_client:
            surplus_clients.append(i)
    
    # Redistribute samples from surplus to deficit clients
    for deficit_client in deficit_clients:
        needed = min_samples_per_client - len(client_indices[deficit_client])
        
        for surplus_client in surplus_clients:
            if needed <= 0:
                break
                
            surplus_size = len(client_indices[surplus_client])
            if surplus_size > min_samples_per_client:
                # Transfer samples
                transfer_count = min(needed, surplus_size - min_samples_per_client)
                transferred_samples = client_indices[surplus_client][-transfer_count:]
                client_indices[surplus_client] = client_indices[surplus_client][:-transfer_count]
                client_indices[deficit_client].extend(transferred_samples)
                needed -= transfer_count


def analyze_data_distribution(
    client_indices: List[List[int]],
    labels: List[int],
    num_classes: int
) -> dict:
    """Analyze the data distribution across clients.
    
    Args:
        client_indices: List of sample indices for each client
        labels: List of labels for all samples
        num_classes: Number of classes
        
    Returns:
        Dictionary with distribution statistics
    """
    num_clients = len(client_indices)
    
    # Calculate class distribution for each client
    client_class_counts = []
    for client_data in client_indices:
        class_counts = Counter([labels[idx] for idx in client_data])
        # Ensure all classes are represented (with 0 if not present)
        full_counts = [class_counts.get(i, 0) for i in range(num_classes)]
        client_class_counts.append(full_counts)
    
    client_class_counts = np.array(client_class_counts)
    
    # Calculate statistics
    total_samples = sum(len(client_data) for client_data in client_indices)
    samples_per_client = [len(client_data) for client_data in client_indices]
    
    # Calculate Jensen-Shannon divergence as a measure of non-IIDness
    global_distribution = np.sum(client_class_counts, axis=0) / total_samples
    js_divergences = []
    
    for client_counts in client_class_counts:
        if np.sum(client_counts) > 0:
            client_dist = client_counts / np.sum(client_counts)
            js_div = _jensen_shannon_divergence(client_dist, global_distribution)
            js_divergences.append(js_div)
    
    return {
        "total_samples": total_samples,
        "num_clients": num_clients,
        "samples_per_client": samples_per_client,
        "min_samples": min(samples_per_client),
        "max_samples": max(samples_per_client),
        "mean_samples": np.mean(samples_per_client),
        "std_samples": np.std(samples_per_client),
        "client_class_counts": client_class_counts.tolist(),
        "global_class_distribution": global_distribution.tolist(),
        "js_divergences": js_divergences,
        "mean_js_divergence": np.mean(js_divergences) if js_divergences else 0.0,
        "non_iid_score": np.mean(js_divergences) if js_divergences else 0.0
    }


def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Jensen-Shannon divergence between two probability distributions.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        
    Returns:
        Jensen-Shannon divergence
    """
    # Ensure distributions are normalized
    p = p / np.sum(p) if np.sum(p) > 0 else p
    q = q / np.sum(q) if np.sum(q) > 0 else q
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    
    # Calculate M = (P + Q) / 2
    m = (p + q) / 2
    
    # Calculate KL divergences
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    
    # Jensen-Shannon divergence
    js_div = 0.5 * kl_pm + 0.5 * kl_qm
    
    return js_div