"""Gated Keyword Spotting CNN for 2D spectrograms."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GKWS_CNN(nn.Module):
    """
    A Convolutional Neural Network for Gated Keyword Spotting (GKWS).
    Processes 2D Mel-spectrograms for speech command recognition.
    """
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2):
        """Initialize the GKWS CNN model.
        
        Args:
            input_dim: The number of Mel-spectrogram features (n_mels)
            output_dim: The number of output classes (keywords)
            dropout: Dropout probability for regularization
        """
        super(GKWS_CNN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # We assume input tensors of shape [batch_size, channels, freq, time]
        # Here, channels=1 (monoaural), freq=input_dim
        
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize model weights using appropriate initialization schemes."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input Mel-spectrogram tensor of shape [batch, 1, freq, time]
            
        Returns:
            Output logits tensor of shape [batch, output_dim]
        """
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global average pooling
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        
        # Fully connected layers with dropout
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def get_feature_dim(self) -> int:
        """Get the dimension of features before the final classifier."""
        return 64