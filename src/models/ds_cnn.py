"""Depthwise Separable CNN for efficient speech recognition.

Reference: Zhang et al., "Hello Edge: Keyword Spotting on Microcontrollers", 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0)
    ):
        """Initialize depthwise separable convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Stride for convolution
            padding: Padding for convolution
        """
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            bias=False
        )
        
        # Batch normalization
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)


class DS_CNN(nn.Module):
    """Depthwise Separable CNN for keyword spotting.
    
    This is an efficient architecture designed for edge devices and
    resource-constrained environments.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_filters: int = 64,
        dropout: float = 0.2
    ):
        """Initialize DS-CNN model.
        
        Args:
            input_dim: Number of input features (e.g., MFCC coefficients)
            output_dim: Number of output classes
            num_filters: Number of filters in convolutional layers
            dropout: Dropout probability
        """
        super(DS_CNN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # First standard convolution
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=(10, 4), stride=(2, 2), padding=(5, 1))
        self.bn1 = nn.BatchNorm2d(num_filters)
        
        # Depthwise separable convolutions
        self.ds_conv1 = DepthwiseSeparableConv(
            num_filters, num_filters, kernel_size=(3, 3), padding=(1, 1)
        )
        self.ds_conv2 = DepthwiseSeparableConv(
            num_filters, num_filters, kernel_size=(3, 3), padding=(1, 1)
        )
        self.ds_conv3 = DepthwiseSeparableConv(
            num_filters, num_filters, kernel_size=(3, 3), padding=(1, 1)
        )
        self.ds_conv4 = DepthwiseSeparableConv(
            num_filters, num_filters, kernel_size=(3, 3), padding=(1, 1)
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(num_filters, output_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize model weights."""
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
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 1, freq, time) or (batch, freq, time)
            
        Returns:
            Output logits of shape (batch, output_dim)
        """
        # Ensure 4D input (batch, channels, freq, time)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # First convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Depthwise separable convolutions
        x = self.ds_conv1(x)
        x = self.ds_conv2(x)
        x = self.ds_conv3(x)
        x = self.ds_conv4(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        # Dropout and classification
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def get_feature_dim(self) -> int:
        """Get the dimension of features before the final classifier."""
        return self.fc.in_features
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)