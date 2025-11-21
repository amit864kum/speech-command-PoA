"""Test script to verify the federated learning system works correctly."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.utils.config_loader import load_config
from src.utils.reproducibility import set_seed, get_device
from src.utils.logger import get_logger
from src.data import SpeechCommandsDataLoader
from src.models import SimpleAudioClassifier
from src.federated import FederatedClient, FederatedTrainer, create_aggregator
from torch.utils.data import DataLoader


def test_system():
    """Test the federated learning system with minimal configuration."""
    
    print("="*70)
    print("TESTING FEDERATED LEARNING SYSTEM")
    print("="*70)
    print()
    
    # Test 1: Configuration Loading
    print("Test 1: Loading configuration...")
    try:
        config = load_config("Speech_command/configs/default_config.yaml")
        print("✓ Configuration loaded successfully")
        print(f"  - Experiment name: {config.get('experiment.name')}")
        print(f"  - Num clients: {config.get('federated_learning.num_clients')}")
        print(f"  - Data distribution: {config.get('federated_learning.data_distribution')}")
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False
    
    # Test 2: Reproducibility Setup
    print("\nTest 2: Setting up reproducibility...")
    try:
        seed = config.get("experiment.seed", 42)
        set_seed(seed)
        device = get_device("cpu")  # Use CPU for testing
        print(f"✓ Reproducibility setup complete")
        print(f"  - Seed: {seed}")
        print(f"  - Device: {device}")
    except Exception as e:
        print(f"✗ Reproducibility setup failed: {e}")
        return False
    
    # Test 3: Logger Setup
    print("\nTest 3: Setting up logger...")
    try:
        logger = get_logger(
            name="test_experiment",
            config={
                "log_dir": "Speech_command/logs",
                "log_level": "INFO",
                "log_to_file": True,
                "log_to_console": True
            }
        )
        logger.info("Logger test message")
        print("✓ Logger setup complete")
    except Exception as e:
        print(f"✗ Logger setup failed: {e}")
        return False
    
    # Test 4: Data Loading (with small subset)
    print("\nTest 4: Loading dataset (this may take a moment)...")
    try:
        # Use only 3 clients for quick testing
        data_loader = SpeechCommandsDataLoader(
            num_clients=3,
            download_dir="Speech_command/data",
            version="v0.02",
            data_distribution="iid",  # Use IID for faster testing
            min_samples_per_client=10
        )
        data_info = data_loader.get_data_info()
        print("✓ Dataset loaded successfully")
        print(f"  - Total samples: {data_info['total_samples']}")
        print(f"  - Num classes: {data_info['num_classes']}")
        print(f"  - Input dim: {data_info['input_dim']}")
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False
    
    # Test 5: Model Creation
    print("\nTest 5: Creating model...")
    try:
        model = SimpleAudioClassifier(
            input_dim=data_info["input_dim"],
            output_dim=data_info["num_classes"],
            dropout=0.2
        )
        num_params = sum(p.numel() for p in model.parameters())
        print("✓ Model created successfully")
        print(f"  - Architecture: SimpleAudioClassifier")
        print(f"  - Total parameters: {num_params:,}")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False
    
    # Test 6: Client Creation
    print("\nTest 6: Creating federated clients...")
    try:
        clients = []
        for client_id in range(3):
            client_data = data_loader.get_client_data(client_id)
            client = FederatedClient(
                client_id=f"Client_{client_id}",
                model=SimpleAudioClassifier(
                    data_info["input_dim"],
                    data_info["num_classes"],
                    0.2
                ),
                train_data=client_data,
                device=device,
                learning_rate=0.001,
                batch_size=16,  # Small batch for testing
                local_epochs=1  # Just 1 epoch for testing
            )
            clients.append(client)
        print("✓ Clients created successfully")
        print(f"  - Number of clients: {len(clients)}")
        for i, client in enumerate(clients):
            stats = client.get_statistics()
            print(f"  - Client {i}: {stats['total_samples']} samples")
    except Exception as e:
        print(f"✗ Client creation failed: {e}")
        return False
    
    # Test 7: Aggregator Creation
    print("\nTest 7: Creating aggregator...")
    try:
        aggregator = create_aggregator("fedavg")
        print("✓ Aggregator created successfully")
        print(f"  - Type: {aggregator.get_name()}")
    except Exception as e:
        print(f"✗ Aggregator creation failed: {e}")
        return False
    
    # Test 8: Test Data Creation
    print("\nTest 8: Creating test dataset...")
    try:
        test_data = data_loader.get_test_data(test_fraction=0.05)  # Small test set
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
        print("✓ Test dataset created successfully")
        print(f"  - Test samples: {len(test_data)}")
    except Exception as e:
        print(f"✗ Test dataset creation failed: {e}")
        return False
    
    # Test 9: Trainer Creation
    print("\nTest 9: Creating federated trainer...")
    try:
        trainer = FederatedTrainer(
            model=model,
            clients=clients,
            aggregator=aggregator,
            test_loader=test_loader,
            device=device,
            logger=logger,
            client_fraction=1.0
        )
        print("✓ Trainer created successfully")
    except Exception as e:
        print(f"✗ Trainer creation failed: {e}")
        return False
    
    # Test 10: Single Training Round
    print("\nTest 10: Running single training round...")
    try:
        metrics = trainer.train_round(round_num=1, verbose=False)
        print("✓ Training round completed successfully")
        print(f"  - Train Loss: {metrics['train_loss']:.4f}")
        print(f"  - Train Accuracy: {metrics['train_accuracy']:.2f}%")
        print(f"  - Test Loss: {metrics['test_loss']:.4f}")
        print(f"  - Test Accuracy: {metrics['test_accuracy']:.2f}%")
        print(f"  - Round Time: {metrics['round_time']:.2f}s")
    except Exception as e:
        print(f"✗ Training round failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 11: Model Saving
    print("\nTest 11: Saving model...")
    try:
        save_path = "Speech_command/results/test_model.pt"
        trainer.save_model(save_path)
        print("✓ Model saved successfully")
        print(f"  - Path: {save_path}")
    except Exception as e:
        print(f"✗ Model saving failed: {e}")
        return False
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✓")
    print("="*70)
    print("\nThe federated learning system is working correctly.")
    print("You can now run full experiments with:")
    print("  python Speech_command/scripts/run_experiment.py")
    print("="*70)
    
    return True


if __name__ == "__main__":
    success = test_system()
    sys.exit(0 if success else 1)