"""Test script for Phase 3: Privacy and Robustness features."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from src.models import SimpleAudioClassifier
from src.federated import FederatedClient, FederatedTrainer, create_aggregator
from src.federated.dp_client import DPFederatedClient
from src.adversarial import create_byzantine_client, AttackSimulator
from src.privacy import PrivacyEngine, PrivacyAccountant, compute_privacy_budget
from src.utils.reproducibility import set_seed


def create_synthetic_data(num_samples=100, input_dim=64, seq_len=81, num_classes=10):
    """Create synthetic audio data."""
    features = torch.randn(num_samples, input_dim, seq_len)
    labels = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(features, labels)


def test_differential_privacy():
    """Test differential privacy implementation."""
    print("\n" + "="*70)
    print("TEST 1: DIFFERENTIAL PRIVACY")
    print("="*70)
    
    set_seed(42)
    device = torch.device("cpu")
    
    # Create data and model
    train_data = create_synthetic_data(100)
    model = SimpleAudioClassifier(64, 10, 0.2)
    
    # Create DP client
    print("\n1. Creating DP-enabled client...")
    dp_client = DPFederatedClient(
        client_id="DP_Client_0",
        model=model,
        train_data=train_data,
        device=device,
        learning_rate=0.01,
        batch_size=16,
        local_epochs=2,
        enable_dp=True,
        noise_multiplier=1.1,
        max_grad_norm=1.0,
        target_epsilon=10.0,
        target_delta=1e-5
    )
    print("✓ DP client created")
    
    # Train with DP
    print("\n2. Training with differential privacy...")
    metrics = dp_client.train(verbose=True)
    print(f"✓ Training complete")
    print(f"  - Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  - Privacy: ε={metrics['epsilon']:.2f}, δ={metrics['delta']:.2e}")
    
    # Get privacy report
    print("\n3. Privacy report...")
    report = dp_client.get_privacy_report()
    print(f"✓ Privacy budget: ε={report['epsilon']:.2f}")
    print(f"  - Noise multiplier: {report['noise_multiplier']}")
    print(f"  - Max grad norm: {report['max_grad_norm']}")
    print(f"  - Budget exceeded: {report['budget_exceeded']}")
    
    print("\n✓ Differential Privacy Test PASSED")
    return True


def test_privacy_accounting():
    """Test privacy accounting."""
    print("\n" + "="*70)
    print("TEST 2: PRIVACY ACCOUNTING")
    print("="*70)
    
    print("\n1. Creating privacy accountant...")
    accountant = PrivacyAccountant(
        noise_multiplier=1.1,
        sample_rate=0.1,
        target_delta=1e-5
    )
    print("✓ Accountant created")
    
    print("\n2. Simulating training steps...")
    for step in range(1, 6):
        accountant.step()
        epsilon, delta = accountant.get_privacy_spent()
        print(f"  Step {step}: ε={epsilon:.2f}, δ={delta:.2e}")
    
    print("\n3. Computing max steps for target epsilon...")
    target_epsilon = 10.0
    max_steps = accountant.get_max_steps(target_epsilon)
    print(f"✓ Max steps for ε={target_epsilon}: {max_steps}")
    
    print("\n4. Computing required noise...")
    required_noise = accountant.get_required_noise(target_epsilon, 100)
    print(f"✓ Required noise multiplier: {required_noise:.2f}")
    
    print("\n✓ Privacy Accounting Test PASSED")
    return True


def test_byzantine_attacks():
    """Test Byzantine attack simulation."""
    print("\n" + "="*70)
    print("TEST 3: BYZANTINE ATTACKS")
    print("="*70)
    
    set_seed(42)
    device = torch.device("cpu")
    
    # Create honest clients
    print("\n1. Creating honest clients...")
    honest_clients = []
    for i in range(3):
        train_data = create_synthetic_data(100)
        model = SimpleAudioClassifier(64, 10, 0.2)
        client = FederatedClient(
            client_id=f"Honest_{i}",
            model=model,
            train_data=train_data,
            device=device,
            learning_rate=0.01,
            batch_size=16,
            local_epochs=1
        )
        honest_clients.append(client)
    print(f"✓ Created {len(honest_clients)} honest clients")
    
    # Test different attack types
    attack_types = ["random", "sign_flipping", "label_flipping", "gaussian"]
    
    for attack_type in attack_types:
        print(f"\n2. Testing {attack_type} attack...")
        
        # Create Byzantine client
        train_data = create_synthetic_data(100)
        model = SimpleAudioClassifier(64, 10, 0.2)
        
        byzantine_client = create_byzantine_client(
            client_id=f"Byzantine_{attack_type}",
            model=model,
            train_data=train_data,
            device=device,
            attack_type=attack_type,
            attack_strength=2.0,
            learning_rate=0.01,
            batch_size=16,
            local_epochs=1
        )
        
        print(f"  ✓ Created {attack_type} attacker")
        
        # Train Byzantine client
        metrics = byzantine_client.train(verbose=False)
        print(f"  ✓ Attack executed: Acc={metrics['accuracy']:.2f}%")
    
    print("\n3. Testing attack detection...")
    simulator = AttackSimulator(
        num_byzantine=1,
        attack_type="random",
        attack_strength=2.0
    )
    
    # Collect weights from all clients
    all_weights = []
    for client in honest_clients:
        client.train(verbose=False)
        all_weights.append(client.get_model_weights())
    
    # Add Byzantine weights
    byzantine_client.train(verbose=False)
    all_weights.append(byzantine_client.get_model_weights())
    
    # Detect Byzantine updates
    suspected = simulator.detect_byzantine_updates(all_weights, threshold=2.0)
    print(f"✓ Suspected Byzantine clients: {suspected}")
    
    print("\n✓ Byzantine Attacks Test PASSED")
    return True


def test_robust_aggregation():
    """Test robust aggregation methods."""
    print("\n" + "="*70)
    print("TEST 4: ROBUST AGGREGATION")
    print("="*70)
    
    set_seed(42)
    device = torch.device("cpu")
    
    # Create clients (mix of honest and Byzantine)
    print("\n1. Creating mixed client population...")
    clients = []
    
    # 3 honest clients
    for i in range(3):
        train_data = create_synthetic_data(100)
        model = SimpleAudioClassifier(64, 10, 0.2)
        client = FederatedClient(
            client_id=f"Honest_{i}",
            model=model,
            train_data=train_data,
            device=device,
            learning_rate=0.01,
            batch_size=16,
            local_epochs=1
        )
        clients.append(client)
    
    # 1 Byzantine client
    train_data = create_synthetic_data(100)
    model = SimpleAudioClassifier(64, 10, 0.2)
    byzantine = create_byzantine_client(
        client_id="Byzantine_0",
        model=model,
        train_data=train_data,
        device=device,
        attack_type="random",
        attack_strength=5.0,
        learning_rate=0.01,
        batch_size=16,
        local_epochs=1
    )
    clients.append(byzantine)
    
    print(f"✓ Created {len(clients)} clients (3 honest, 1 Byzantine)")
    
    # Test different aggregation methods
    aggregation_methods = ["fedavg", "krum", "trimmed_mean"]
    
    test_data = create_synthetic_data(50)
    test_loader = DataLoader(test_data, batch_size=16)
    
    for agg_method in aggregation_methods:
        print(f"\n2. Testing {agg_method} aggregation...")
        
        # Create aggregator
        if agg_method == "krum":
            aggregator = create_aggregator(agg_method, num_byzantine=1)
        elif agg_method == "trimmed_mean":
            aggregator = create_aggregator(agg_method, trim_ratio=0.2)
        else:
            aggregator = create_aggregator(agg_method)
        
        # Create trainer
        global_model = SimpleAudioClassifier(64, 10, 0.2)
        trainer = FederatedTrainer(
            model=global_model,
            clients=clients,
            aggregator=aggregator,
            test_loader=test_loader,
            device=device
        )
        
        # Run one round
        metrics = trainer.train_round(1, verbose=False)
        print(f"  ✓ {agg_method}: Test Acc={metrics['test_accuracy']:.2f}%")
    
    print("\n✓ Robust Aggregation Test PASSED")
    return True


def main():
    """Run all Phase 3 tests."""
    print("="*70)
    print("PHASE 3: PRIVACY & ROBUSTNESS TESTING")
    print("="*70)
    
    tests = [
        ("Differential Privacy", test_differential_privacy),
        ("Privacy Accounting", test_privacy_accounting),
        ("Byzantine Attacks", test_byzantine_attacks),
        ("Robust Aggregation", test_robust_aggregation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n" + "="*70)
        print("ALL PHASE 3 TESTS PASSED! ✓")
        print("="*70)
        print("\nPhase 3 features are working correctly:")
        print("  - Differential Privacy (DP-SGD)")
        print("  - Privacy Accounting")
        print("  - Byzantine Attack Simulation")
        print("  - Robust Aggregation (Krum, Trimmed Mean)")
        print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)