"""Simple 1D CNN for audio classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SimpleAudioClassifier(nn.Module):
    """Simple 1D CNN for speech command classification.
    
    This model processes MFCC features using 1D convolutions followed by
    adaptive pooling and fully connected layers.
    """
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2):
        """Initialize the model.
        
        Args:
            input_dim: Number of input features (MFCC coefficients)
            output_dim: Number of output classes
            dropout: Dropout probability
        """
        super(SimpleAudioClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        
        # Pooling and regularization
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, output_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim, sequence_length)
            
        Returns:
            Output logits of shape (batch_size, output_dim)
        """
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Global average pooling
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_feature_dim(self) -> int:
        """Get the dimension of features before the final classifier."""
        return 128