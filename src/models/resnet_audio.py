"""ResNet architecture adapted for 1D audio data."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class BasicBlock1D(nn.Module):
    """Basic ResNet block for 1D audio data."""
    
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
            stride: Convolution stride
            downsample: Downsample layer for residual connection
        """
        super(BasicBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
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


class Bottleneck1D(nn.Module):
    """Bottleneck ResNet block for 1D audio data."""
    
    expansion = 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        """Initialize bottleneck block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Convolution stride
            downsample: Downsample layer for residual connection
        """
        super(Bottleneck1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.conv3 = nn.Conv1d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        
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
        out = F.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class ResNetAudio(nn.Module):
    """ResNet architecture for 1D audio classification."""
    
    def __init__(
        self,
        block: nn.Module,
        layers: List[int],
        input_dim: int,
        output_dim: int,
        dropout: float = 0.2,
        zero_init_residual: bool = False
    ):
        """Initialize ResNet for audio.
        
        Args:
            block: ResNet block type (BasicBlock1D or Bottleneck1D)
            layers: Number of blocks in each layer
            input_dim: Number of input features
            output_dim: Number of output classes
            dropout: Dropout probability
            zero_init_residual: Whether to zero-initialize residual connections
        """
        super(ResNetAudio, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv1d(
            input_dim, 64, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * block.expansion, output_dim)
        
        self._initialize_weights(zero_init_residual)
    
    def _make_layer(
        self,
        block: nn.Module,
        channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """Create a ResNet layer.
        
        Args:
            block: Block type
            channels: Number of channels
            blocks: Number of blocks
            stride: Stride for first block
            
        Returns:
            Sequential layer
        """
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels, channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(channels * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self, zero_init_residual: bool) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck1D):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim, sequence_length)
            
        Returns:
            Output logits of shape (batch_size, output_dim)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def get_feature_dim(self) -> int:
        """Get the dimension of features before the final classifier."""
        return self.fc.in_features


def resnet18_audio(input_dim: int, output_dim: int, **kwargs) -> ResNetAudio:
    """ResNet-18 for audio classification."""
    return ResNetAudio(BasicBlock1D, [2, 2, 2, 2], input_dim, output_dim, **kwargs)


def resnet34_audio(input_dim: int, output_dim: int, **kwargs) -> ResNetAudio:
    """ResNet-34 for audio classification."""
    return ResNetAudio(BasicBlock1D, [3, 4, 6, 3], input_dim, output_dim, **kwargs)


def resnet50_audio(input_dim: int, output_dim: int, **kwargs) -> ResNetAudio:
    """ResNet-50 for audio classification."""
    return ResNetAudio(Bottleneck1D, [3, 4, 6, 3], input_dim, output_dim, **kwargs)