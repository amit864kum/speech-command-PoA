"""Enhanced Speech Commands dataset loader with better data distribution support."""

import os
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import random_split, Dataset, Subset
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np
from pathlib import Path

from .data_distribution import create_non_iid_split, create_iid_split


class SpeechCommandsDataset(Dataset):
    """Custom dataset wrapper for Speech Commands with preprocessing."""
    
    def __init__(self, data_list: List[Tuple[torch.Tensor, int]]):
        """Initialize dataset with preprocessed data.
        
        Args:
            data_list: List of (features, label) tuples
        """
        self.data = data_list
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.data[idx]


class SpeechCommandsDataLoader:
    """Enhanced Speech Commands data loader with support for different data distributions."""
    
    def __init__(
        self,
        num_clients: int,
        download_dir: str = "Speech_command/data",
        version: str = "v0.02",
        target_words: Optional[List[str]] = None,
        sample_rate: int = 16000,
        n_mfcc: int = 64,
        n_mels: int = 64,
        data_distribution: str = "iid",
        alpha: float = 0.5,
        min_samples_per_client: int = 10
    ):
        """Initialize the enhanced data loader.
        
        Args:
            num_clients: Number of federated learning clients
            download_dir: Directory to download/store data
            version: Speech Commands version ("v0.01" or "v0.02")
            target_words: List of target words to classify
            sample_rate: Audio sample rate
            n_mfcc: Number of MFCC coefficients
            n_mels: Number of mel filterbanks
            data_distribution: Data distribution type ("iid" or "non_iid")
            alpha: Dirichlet parameter for non-IID distribution
            min_samples_per_client: Minimum samples per client
        """
        self.num_clients = num_clients
        self.download_dir = Path(download_dir)
        self.version = version
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.data_distribution = data_distribution
        self.alpha = alpha
        self.min_samples_per_client = min_samples_per_client
        
        # Default target words (10 classes)
        if target_words is None:
            self.target_words = [
                'yes', 'no', 'up', 'down', 'left', 
                'right', 'on', 'off', 'stop', 'go'
            ]
        else:
            self.target_words = target_words
        
        self.num_classes = len(self.target_words)
        self.label_map = {word: i for i, word in enumerate(self.target_words)}
        
        # Load and process dataset
        self.dataset = self._load_dataset()
        self.max_mfcc_len = self._get_max_mfcc_length()
        print(f"Max MFCC length found: {self.max_mfcc_len}")
        
        self.processed_dataset = self._process_data()
        self.client_data = self._split_data()
        
        # Statistics
        self._print_data_statistics()
    
    def _load_dataset(self) -> SPEECHCOMMANDS:
        """Load the Speech Commands dataset."""
        if not self.download_dir.exists():
            self.download_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading Speech Commands dataset {self.version}...")
        
        # Map version to URL if needed
        if self.version == "v0.01":
            url = "speech_commands_v0.01"
        else:
            url = "speech_commands_v0.02"
        
        dataset = SPEECHCOMMANDS(root=str(self.download_dir), download=True, url=url)
        print("Dataset loaded successfully.")
        return dataset
    
    def _get_max_mfcc_length(self) -> int:
        """Calculate the maximum MFCC sequence length in the dataset."""
        max_len = 0
        feature_extractor = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={'n_mels': self.n_mels}
        )
        
        sample_count = 0
        for waveform, sample_rate, label, speaker_id, utterance_number in self.dataset:
            if label in self.target_words:
                mfccs = feature_extractor(waveform)
                if mfccs.shape[2] > max_len:
                    max_len = mfccs.shape[2]
                
                sample_count += 1
                if sample_count >= 1000:  # Sample first 1000 for efficiency
                    break
        
        return max_len
    
    def _process_data(self) -> List[Tuple[torch.Tensor, int]]:
        """Process raw audio data into MFCC features."""
        processed_data = []
        feature_extractor = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={'n_mels': self.n_mels}
        )
        
        print("Processing audio data...")
        for waveform, sample_rate, label, speaker_id, utterance_number in self.dataset:
            if label in self.target_words:
                # Extract MFCC features
                mfccs = feature_extractor(waveform)
                
                # Pad to consistent length
                pad_amount = self.max_mfcc_len - mfccs.shape[2]
                if pad_amount > 0:
                    padded_mfccs = F.pad(mfccs, (0, pad_amount))
                else:
                    padded_mfccs = mfccs[:, :, :self.max_mfcc_len]
                
                # Remove batch dimension and get label
                features = padded_mfccs.squeeze(0)  # Shape: [n_mfcc, time]
                label_idx = self.label_map[label]
                
                processed_data.append((features, label_idx))
        
        print(f"Processed {len(processed_data)} samples")
        return processed_data
    
    def _split_data(self) -> List[SpeechCommandsDataset]:
        """Split data among clients based on the specified distribution."""
        if self.data_distribution == "iid":
            client_indices = create_iid_split(
                len(self.processed_dataset),
                self.num_clients,
                self.min_samples_per_client
            )
        elif self.data_distribution == "non_iid":
            # Extract labels for non-IID split
            labels = [item[1] for item in self.processed_dataset]
            client_indices = create_non_iid_split(
                labels,
                self.num_clients,
                self.num_classes,
                self.alpha,
                self.min_samples_per_client
            )
        else:
            raise ValueError(f"Unknown data distribution: {self.data_distribution}")
        
        # Create client datasets
        client_datasets = []
        for indices in client_indices:
            client_data = [self.processed_dataset[i] for i in indices]
            client_datasets.append(SpeechCommandsDataset(client_data))
        
        return client_datasets
    
    def _print_data_statistics(self) -> None:
        """Print statistics about the data distribution."""
        print("\n" + "="*50)
        print("DATA DISTRIBUTION STATISTICS")
        print("="*50)
        print(f"Total samples: {len(self.processed_dataset)}")
        print(f"Number of clients: {self.num_clients}")
        print(f"Distribution type: {self.data_distribution}")
        if self.data_distribution == "non_iid":
            print(f"Dirichlet alpha: {self.alpha}")
        
        print("\nSamples per client:")
        for i, client_dataset in enumerate(self.client_data):
            print(f"  Client {i}: {len(client_dataset)} samples")
        
        print("\nClass distribution per client:")
        for i, client_dataset in enumerate(self.client_data):
            class_counts = {}
            for _, label in client_dataset:
                class_counts[label] = class_counts.get(label, 0) + 1
            
            print(f"  Client {i}: {dict(sorted(class_counts.items()))}")
        
        print("="*50)
    
    def get_client_data(self, client_id: int) -> SpeechCommandsDataset:
        """Get data for a specific client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Dataset for the specified client
        """
        if client_id >= self.num_clients:
            raise ValueError(f"Client ID {client_id} >= num_clients {self.num_clients}")
        
        return self.client_data[client_id]
    
    def get_test_data(self, test_fraction: float = 0.1) -> SpeechCommandsDataset:
        """Get a test dataset for global evaluation.
        
        Args:
            test_fraction: Fraction of data to use for testing
            
        Returns:
            Test dataset
        """
        test_size = int(len(self.processed_dataset) * test_fraction)
        test_indices = np.random.choice(
            len(self.processed_dataset), 
            size=test_size, 
            replace=False
        )
        
        test_data = [self.processed_dataset[i] for i in test_indices]
        return SpeechCommandsDataset(test_data)
    
    def get_data_info(self) -> Dict[str, any]:
        """Get information about the dataset.
        
        Returns:
            Dictionary with dataset information
        """
        return {
            "num_classes": self.num_classes,
            "target_words": self.target_words,
            "input_dim": self.n_mfcc,
            "max_sequence_length": self.max_mfcc_len,
            "total_samples": len(self.processed_dataset),
            "num_clients": self.num_clients,
            "data_distribution": self.data_distribution
        }