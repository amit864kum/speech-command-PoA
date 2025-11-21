"""
Demo script showcasing Phase 3 enhancements:
- Differential Privacy
- Byzantine Attack Simulation
- Robust Aggregation
- Privacy Accounting
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Import existing modules
from data_loader import SpeechCommandsDataLoader
from fl_node import SimpleAudioClassifier
from client import DecentralizedClient, aggregate_local_models
from ehr_chain import EHRChain
from miner import Miner

# Import new modules
try:
    from src.privacy.privacy_accountant import PrivacyAccountant, compute_privacy_budget
    from src.federated.aggregator import create_aggregator
    ENHANCED_FEATURES = True
except ImportError:
    ENHANCED_FEATURES = False
    print("[WARNING] Enhanced features not available. Run from Speech_command directory.")

def demo_differential_privacy():
    """Demonstrate differential privacy in action."""
    print("\n" + "="*70)
    print("DEMO 1: DIFFERENTIAL PRIVACY")
    print("="*70)
    
    if not ENHANCED_FEATURES:
        print("Enhanced features not available.")
        return
    
    print("\nDifferential Privacy protects individual client data by:")
    print("  1. Clipping gradients to bound sensitivity")
    print("  2. Adding calibrated Gaussian noise")
    print("  3. Tracking privacy budget (ε, δ)")
    
    # Load small dataset
    print("\nLoading dataset...")
    data_loader = SpeechCommandsDataLoader(num_clients=1)
    client_data = data_loader.get_client_data(0)
    
    # Create DP-enabled client
    print("\nCreating client with Differential Privacy...")
    dp_client = DecentralizedClient(
        client_id="DP_Demo_Client",
        miner_id="DP_Miner",
        client_data=client_data,
        input_dim=64,
        output_dim=10,
        target_words=data_loader.target_words,
        enable_dp=True,
        noise_multiplier=1.1,
        max_grad_norm=1.0,
        target_epsilon=10.0
    )
    
    print("\n✓ Client configured with:")
    print(f"  - Noise multiplier: 1.1")
    print(f"  - Max gradient norm: 1.0")
    print(f"  - Target privacy budget: ε=10.0")
    
    # Train with DP (5 epochs minimum)
    print("\nTraining with differential privacy (5 epochs)...")
    weights, acc, model_id, preds = dp_client.local_train(
        epochs=5,
        batch_size=32,
        lr=0.001
    )
    
    print(f"\n✓ Training complete:")
    print(f"  - Accuracy: {acc*100:.2f}%")
    print(f"  - Privacy preserved with formal guarantees")
    
    # Show privacy accounting
    if dp_client.privacy_engine:
        epsilon, delta = dp_client.privacy_engine.get_privacy_spent()
        print(f"\n📊 Privacy Budget Spent:")
        print(f"  - ε (epsilon): {epsilon:.2f}")
        print(f"  - δ (delta): {delta:.2e}")
        print(f"  - Budget remaining: {dp_client.privacy_engine.check_privacy_budget()}")


def demo_byzantine_attacks():
    """Demonstrate Byzantine attack simulation."""
    print("\n" + "="*70)
    print("DEMO 2: BYZANTINE ATTACK SIMULATION")
    print("="*70)
    
    print("\nByzantine attacks simulate malicious clients that:")
    print("  1. Send corrupted model updates")
    print("  2. Try to degrade global model performance")
    print("  3. Can be detected and mitigated")
    
    # Load dataset
    print("\nLoading dataset...")
    data_loader = SpeechCommandsDataLoader(num_clients=2)
    
    # Create honest client
    print("\nCreating honest client...")
    honest_client = DecentralizedClient(
        client_id="Honest_Client",
        miner_id="Honest_Miner",
        client_data=data_loader.get_client_data(0),
        input_dim=64,
        output_dim=10,
        target_words=data_loader.target_words,
        is_byzantine=False
    )
    
    # Create Byzantine client
    print("Creating Byzantine attacker...")
    byzantine_client = DecentralizedClient(
        client_id="Byzantine_Client",
        miner_id="Byzantine_Miner",
        client_data=data_loader.get_client_data(1),
        input_dim=64,
        output_dim=10,
        target_words=data_loader.target_words,
        is_byzantine=True,
        attack_type="random",
        attack_strength=2.0
    )
    
    print("\n✓ Clients configured:")
    print("  - 1 Honest client")
    print("  - 1 Byzantine attacker (random attack, strength=2.0)")
    
    # Train both clients
    print("\nTraining clients...")
    print("\n[Honest Client]")
    honest_weights, honest_acc, _, _ = honest_client.local_train(
        epochs=1, batch_size=32, lr=0.001
    )
    print(f"  Accuracy: {honest_acc*100:.2f}%")
    
    print("\n[Byzantine Client]")
    byzantine_weights, byzantine_acc, _, _ = byzantine_client.local_train(
        epochs=1, batch_size=32, lr=0.001
    )
    print(f"  Accuracy: {byzantine_acc*100:.2f}%")
    
    print("\n✓ Byzantine attack executed!")
    print("  The attacker's model updates are now corrupted.")
    print("  Robust aggregation methods (Krum, Trimmed Mean) can detect and mitigate this.")


def demo_robust_aggregation():
    """Demonstrate robust aggregation methods."""
    print("\n" + "="*70)
    print("DEMO 3: ROBUST AGGREGATION")
    print("="*70)
    
    if not ENHANCED_FEATURES:
        print("Enhanced features not available.")
        return
    
    print("\nRobust aggregation methods protect against Byzantine attacks:")
    print("  - FedAvg: Standard averaging (vulnerable)")
    print("  - Krum: Selects most representative model")
    print("  - Trimmed Mean: Removes outliers before averaging")
    
    print("\nAggregation methods available:")
    for method in ["fedavg", "krum", "trimmed_mean", "median"]:
        aggregator = create_aggregator(method)
        print(f"  ✓ {aggregator.get_name()}")


def demo_privacy_accounting():
    """Demonstrate privacy budget accounting."""
    print("\n" + "="*70)
    print("DEMO 4: PRIVACY ACCOUNTING")
    print("="*70)
    
    if not ENHANCED_FEATURES:
        print("Enhanced features not available.")
        return
    
    print("\nPrivacy accounting tracks cumulative privacy loss:")
    print("  - Computes privacy budget (ε, δ) over time")
    print("  - Helps determine when to stop training")
    print("  - Ensures privacy guarantees are maintained")
    
    # Create privacy accountant
    print("\nCreating privacy accountant...")
    accountant = PrivacyAccountant(
        noise_multiplier=1.1,
        sample_rate=0.1,
        target_delta=1e-5
    )
    
    print("\n✓ Accountant configured:")
    print(f"  - Noise multiplier: 1.1")
    print(f"  - Sample rate: 0.1")
    print(f"  - Target δ: 1e-5")
    
    # Simulate training steps
    print("\nSimulating training steps:")
    print("\n  Step | Privacy Budget (ε)")
    print("  " + "-"*30)
    
    for step in [10, 50, 100, 200, 500]:
        accountant.steps = step
        epsilon = accountant.compute_epsilon()
        print(f"  {step:4d} | ε = {epsilon:.2f}")
    
    # Compute max steps
    target_epsilon = 10.0
    max_steps = accountant.get_max_steps(target_epsilon)
    print(f"\n✓ For target ε={target_epsilon}:")
    print(f"  Maximum training steps: {max_steps}")


def main():
    """Run all demos."""
    print("="*70)
    print("ENHANCED FEDERATED LEARNING DEMO")
    print("Phase 3: Privacy & Robustness Features")
    print("="*70)
    
    if not ENHANCED_FEATURES:
        print("\n[WARNING] Enhanced features not fully available.")
        print("Make sure you're running from the Speech_command directory.")
        print("Some demos may not work.\n")
    
    demos = [
        ("Differential Privacy", demo_differential_privacy),
        ("Byzantine Attacks", demo_byzantine_attacks),
        ("Robust Aggregation", demo_robust_aggregation),
        ("Privacy Accounting", demo_privacy_accounting)
    ]
    
    for demo_name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n[ERROR] {demo_name} demo failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("  ✓ Differential Privacy protects individual client data")
    print("  ✓ Byzantine attacks can be simulated and detected")
    print("  ✓ Robust aggregation methods provide defense")
    print("  ✓ Privacy accounting ensures formal guarantees")
    print("\nFor full experiments, see:")
    print("  - scripts/test_phase3.py")
    print("  - src/main.py")
    print("="*70)


if __name__ == "__main__":
    main()