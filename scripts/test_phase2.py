"""Test script for Phase 2 enhancements."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torch
from src.utils.config_loader import load_config
from src.utils.reproducibility import set_seed
from src.data import SpeechCommandsDataLoader
from src.models import ModelFactory, create_model_from_config
from src.federated import FederatedClient, FederatedAggregator


def test_models():
    """Test all model architectures."""
    print("Testing model architectures...")
    
    input_dim = 64
    output_dim = 10
    batch_size = 4
    seq_len = 100
    
    architectures = ModelFactory.get_available_architectures()
    
    for arch in architectures:
        try:
            print(f"\nTesting {arch}...")
            model = ModelFactory.create_model(arch, input_dim, output_dim)
            
            # Prepare input based on model type
            if 'gkws' in arch:
                # GKWS_CNN expects 4D input [batch, 1, features, time]
                x = torch.randn(batch_size, 1, input_dim, seq_len)
            else:
                # Other models expect 3D input [batch, features, time]
                x = torch.randn(batch_size, input_dim, seq_len)
            
            # Forward pass
            output = model(x)
            assert output.shape == (batch_size, output_dim), f"Wrong output shape: {output.shape}"
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  ✓ Parameters: {total_params:,}")
            print(f"  ✓ Output shape: {output.shape}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")


def test_data_loader():
    """Test enhanced data loader."""
    print("\nTesting enhanced data loader...")
    
    try:
        # Test with small dataset for speed
        data_loader = SpeechCommandsDataLoader(
            num_clients=3,
            data_distribution="non_iid",
            alpha=0.5
        )
        
        print(f"  ✓ Loaded dataset with {len(data_loader.processed_dataset)} samples")
        
        # Test client data
        for i in range(3):
            client_data = data_loader.get_client_data(i)
            print(f"  ✓ Client {i}: {len(client_data)} samples")
        
        # Test data info
        data_info = data_loader.get_data_info()
        print(f"  ✓ Data info: {data_info}")
        
    except Exception as e:
        print(f"  ✗ Data loader failed: {e}")


def test_federated_components():
    """Test federated learning components."""
    print("\nTesting federated learning components...")
    
    try:
        # Load config
        config = load_config()
        device = torch.device("cpu")
        
        # Create small dataset
        data_loader = SpeechCommandsDataLoader(num_clients=2)
        client_data = data_loader.get_client_data(0)
        
        # Create model
        model = ModelFactory.create_model("simpleaudioclassifier", 64, 10)
        
        # Test client
        client = FederatedClient(
            client_id="test_client",
            model=model,
            train_dataset=client_data,
            device=device,
            config=config.to_dict()
        )
        
        print(f"  ✓ Created client with {len(client_data)} samples")
        
        # Test aggregator
        aggregator = FederatedAggregator("fedavg")
        print(f"  ✓ Created aggregator: {aggregator.aggregation_method}")
        
        # Test client training (1 epoch for speed)
        config.set('federated_learning.local_epochs', 1)
        client.config = config.to_dict()
        
        results = client.local_train()
        print(f"  ✓ Client training completed: {results['final_accuracy']:.2f}% accuracy")
        
    except Exception as e:
        print(f"  ✗ Federated components failed: {e}")


def test_config_system():
    """Test configuration system."""
    print("\nTesting configuration system...")
    
    try:
        config = load_config()
        
        # Test getting values
        num_clients = config.get('federated_learning.num_clients')
        print(f"  ✓ Config loaded: {num_clients} clients")
        
        # Test setting values
        config.set('test.value', 42)
        assert config.get('test.value') == 42
        print(f"  ✓ Config setting works")
        
        # Test model creation from config
        model = create_model_from_config(config.to_dict())
        print(f"  ✓ Model created from config: {type(model).__name__}")
        
    except Exception as e:
        print(f"  ✗ Config system failed: {e}")


def main():
    """Run all tests."""
    print("="*60)
    print("PHASE 2 ENHANCEMENT TESTS")
    print("="*60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Run tests
    test_config_system()
    test_models()
    test_data_loader()
    test_federated_components()
    
    print("\n" + "="*60)
    print("PHASE 2 TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()