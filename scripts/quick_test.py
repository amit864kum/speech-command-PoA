"""Quick test with synthetic data to verify system components."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.models import SimpleAudioClassifier
from src.federated import FederatedClient, FederatedTrainer, create_aggregator
from src.utils.reproducibility import set_seed

def create_synthetic_data(num_samples=100, input_dim=64, seq_len=81, num_classes=10):
    """Create synthetic audio data for testing."""
    features = torch.randn(num_samples, input_dim, seq_len)
    labels = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(features, labels)

def quick_test():
    print("="*70)
    print("QUICK SYSTEM TEST WITH SYNTHETIC DATA")
    print("="*70)
    
    # Setup
    set_seed(42)
    device = torch.device("cpu")
    
    # Parameters
    input_dim = 64
    output_dim = 10
    num_clients = 3
    
    print("\n1. Creating synthetic datasets...")
    train_datasets = [create_synthetic_data(100) for _ in range(num_clients)]
    test_dataset = create_synthetic_data(50)
    test_loader = DataLoader(test_dataset, batch_size=16)
    print(f"✓ Created {num_clients} training datasets and 1 test dataset")
    
    print("\n2. Creating model...")
    model = SimpleAudioClassifier(input_dim, output_dim, dropout=0.2)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {num_params:,} parameters")
    
    print("\n3. Creating federated clients...")
    clients = []
    for i in range(num_clients):
        client = FederatedClient(
            client_id=f"Client_{i}",
            model=SimpleAudioClassifier(input_dim, output_dim, 0.2),
            train_data=train_datasets[i],
            device=device,
            learning_rate=0.01,
            batch_size=16,
            local_epochs=2
        )
        clients.append(client)
    print(f"✓ Created {len(clients)} clients")
    
    print("\n4. Testing aggregation strategies...")
    for agg_type in ["fedavg", "krum", "trimmed_mean"]:
        aggregator = create_aggregator(agg_type)
        print(f"  ✓ {aggregator.get_name()} created")
    
    print("\n5. Creating trainer with FedAvg...")
    aggregator = create_aggregator("fedavg")
    trainer = FederatedTrainer(
        model=model,
        clients=clients,
        aggregator=aggregator,
        test_loader=test_loader,
        device=device,
        client_fraction=1.0
    )
    print("✓ Trainer created")
    
    print("\n6. Running 3 training rounds...")
    for round_num in range(1, 4):
        metrics = trainer.train_round(round_num, verbose=False)
        print(f"  Round {round_num}: "
              f"Train Acc={metrics['train_accuracy']:.2f}%, "
              f"Test Acc={metrics['test_accuracy']:.2f}%, "
              f"Time={metrics['round_time']:.2f}s")
    
    print("\n7. Testing model save/load...")
    save_path = "Speech_command/results/quick_test_model.pt"
    trainer.save_model(save_path)
    print(f"✓ Model saved to {save_path}")
    
    # Test loading
    trainer.load_model(save_path)
    print("✓ Model loaded successfully")
    
    print("\n8. Final evaluation...")
    final_metrics = trainer.evaluate(test_loader)
    print(f"✓ Final Test Accuracy: {final_metrics['accuracy']:.2f}%")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✓")
    print("="*70)
    print("\nThe federated learning system is working correctly!")
    print("\nNext steps:")
    print("1. Run with real data: python Speech_command/src/main.py")
    print("2. Or use the quick experiment script")
    print("="*70)
    
    return True

if __name__ == "__main__":
    try:
        success = quick_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)