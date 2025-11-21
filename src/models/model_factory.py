"""Model factory for creating different neural network architectures."""

import torch.nn as nn
from typing import Dict, Any

from .simple_audio_classifier import SimpleAudioClassifier
from .gkws_cnn import GKWS_CNN
from .ds_cnn import DS_CNN
from .resnet_audio import resnet18_audio, resnet34_audio, resnet50_audio


class ModelFactory:
    """Factory class for creating different model architectures."""
    
    @staticmethod
    def create_model(
        architecture: str,
        input_dim: int,
        output_dim: int,
        **kwargs
    ) -> nn.Module:
        """Create a model based on the specified architecture.
        
        Args:
            architecture: Model architecture name
            input_dim: Input dimension (number of features)
            output_dim: Output dimension (number of classes)
            **kwargs: Additional model parameters
            
        Returns:
            Initialized model
            
        Raises:
            ValueError: If architecture is not supported
        """
        architecture = architecture.lower()
        
        if architecture == "simpleaudioclassifier":
            return SimpleAudioClassifier(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=kwargs.get('dropout', 0.2)
            )
        
        elif architecture == "gkws_cnn":
            return GKWS_CNN(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=kwargs.get('dropout', 0.2)
            )
        
        elif architecture == "ds_cnn":
            return DS_CNN(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=kwargs.get('dropout', 0.2),
                width_multiplier=kwargs.get('width_multiplier', 1.0)
            )
        
        elif architecture == "resnet18":
            return resnet18_audio(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=kwargs.get('dropout', 0.2)
            )
        
        elif architecture == "resnet34":
            return resnet34_audio(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=kwargs.get('dropout', 0.2)
            )
        
        elif architecture == "resnet50":
            return resnet50_audio(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=kwargs.get('dropout', 0.2)
            )
        
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    
    @staticmethod
    def get_available_architectures() -> list:
        """Get list of available model architectures.
        
        Returns:
            List of available architecture names
        """
        return [
            "simpleaudioclassifier",
            "gkws_cnn", 
            "ds_cnn",
            "resnet18",
            "resnet34", 
            "resnet50"
        ]
    
    @staticmethod
    def get_model_info(architecture: str) -> Dict[str, Any]:
        """Get information about a specific model architecture.
        
        Args:
            architecture: Model architecture name
            
        Returns:
            Dictionary with model information
        """
        architecture = architecture.lower()
        
        model_info = {
            "simpleaudioclassifier": {
                "description": "Simple 1D CNN with batch normalization",
                "parameters": "~100K",
                "complexity": "Low",
                "best_for": "Quick experiments and baseline"
            },
            "gkws_cnn": {
                "description": "2D CNN for mel-spectrograms with gating",
                "parameters": "~200K", 
                "complexity": "Medium",
                "best_for": "2D spectrogram inputs"
            },
            "ds_cnn": {
                "description": "Depthwise separable CNN for efficiency",
                "parameters": "~50K",
                "complexity": "Low-Medium",
                "best_for": "Mobile/edge deployment"
            },
            "resnet18": {
                "description": "ResNet-18 adapted for 1D audio",
                "parameters": "~500K",
                "complexity": "Medium-High", 
                "best_for": "High accuracy requirements"
            },
            "resnet34": {
                "description": "ResNet-34 adapted for 1D audio",
                "parameters": "~1M",
                "complexity": "High",
                "best_for": "Large datasets, high accuracy"
            },
            "resnet50": {
                "description": "ResNet-50 with bottleneck blocks for 1D audio",
                "parameters": "~2M",
                "complexity": "Very High",
                "best_for": "Very large datasets, maximum accuracy"
            }
        }
        
        return model_info.get(architecture, {"description": "Unknown architecture"})


def create_model_from_config(config: Dict[str, Any]) -> nn.Module:
    """Create a model from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    model_config = config.get('model', {})
    
    architecture = model_config.get('architecture', 'simpleaudioclassifier')
    input_dim = model_config.get('input_dim', 64)
    output_dim = model_config.get('output_dim', 10)
    
    # Extract additional parameters
    model_params = {
        'dropout': model_config.get('dropout', 0.2),
        'width_multiplier': model_config.get('width_multiplier', 1.0)
    }
    
    return ModelFactory.create_model(
        architecture=architecture,
        input_dim=input_dim,
        output_dim=output_dim,
        **model_params
    )