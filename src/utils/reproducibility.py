"""Utilities for ensuring reproducible experiments."""

import os
import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(device_preference: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device_preference: Device preference ("cpu", "cuda", "auto")
        
    Returns:
        PyTorch device
    """
    if device_preference == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    return device


def ensure_reproducible_dataloader(dataloader, seed: Optional[int] = None) -> None:
    """Ensure DataLoader produces reproducible results.
    
    Args:
        dataloader: PyTorch DataLoader
        seed: Random seed for worker initialization
    """
    if seed is not None:
        def worker_init_fn(worker_id):
            np.random.seed(seed + worker_id)
            random.seed(seed + worker_id)
        
        dataloader.worker_init_fn = worker_init_fn


class ReproducibilityManager:
    """Context manager for reproducible experiments."""
    
    def __init__(self, seed: int = 42, device: str = "auto"):
        """Initialize reproducibility manager.
        
        Args:
            seed: Random seed
            device: Device preference
        """
        self.seed = seed
        self.device_preference = device
        self.device = None
    
    def __enter__(self):
        """Enter context and set up reproducible environment."""
        set_seed(self.seed)
        self.device = get_device(self.device_preference)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        pass
    
    def get_device(self) -> torch.device:
        """Get the configured device."""
        return self.device