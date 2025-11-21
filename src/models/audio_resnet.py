"""ResNet architecture adapted for audio/speech recognition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class BasicBlock(nn.Module):
    """Basic ResNet block with two convolutional layers."""
    
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        """Initialize basic block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for first convolution
            downsample: Downsample layer for skip connection
        """
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class AudioResNet(nn.Module):
    """ResNet architecture for audio classification.
    
    Adapted from the original ResNet for image classification to work
    with audio spectrograms.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layers: List[int] = [2, 2, 2, 2],
        initial_filters: int = 64,
        dropout: float = 0.2
    ):
        """Initialize Audio ResNet.
        
        Args:
            input_dim: Number of input features (frequency bins)
            output_dim: Number of output classes
            layers: Number of blocks in each layer
            initial_filters: Number of filters in first layer
            dropout: Dropout probability
        """
        super(AudioResNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_channels = initial_filters
        
        # Initial convolution
        self.conv1 = nn.Conv2d(
            1, initial_filters, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(initial_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(initial_filters, layers[0], stride=1)
        self.layer2 = self._make_layer(initial_filters * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(initial_filters * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(initial_filters * 8, layers[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(initial_filters * 8 * BasicBlock.expansion, output_dim)
        
        self._initialize_weights()
    
    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """Create a ResNet layer with multiple blocks.
        
        Args:
            out_channels: Number of output channels
            num_blocks: Number of blocks in this layer
            stride: Stride for first block
            
        Returns:
            Sequential container of blocks
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion
        
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
            x = x.unsqueeze(1)
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling
        x = self.avgpool(x)
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


def audio_resnet18(input_dim: int, output_dim: int, **kwargs) -> AudioResNet:
    """Construct AudioResNet-18 model."""
    return AudioResNet(input_dim, output_dim, layers=[2, 2, 2, 2], **kwargs)


def audio_resnet34(input_dim: int, output_dim: int, **kwargs) -> AudioResNet:
    """Construct AudioResNet-34 model."""
    return AudioResNet(input_dim, output_dim, layers=[3, 4, 6, 3], **kwargs)