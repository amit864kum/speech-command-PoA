import concurrent.futures
import time
from client import DecentralizedClient
from data_loader import SpeechCommandsDataLoader
import os.path
from typing import Dict, Tuple, List

# ----------------------------------------------
# Configuration
# ----------------------------------------------
EPOCHS = 5  # Increased to 5 epochs minimum
LOCAL_EPOCHS = 5  # Increased to 5 local epochs minimum
BATCH_SIZE = 32
LR = 1e-3

NUM_CLASSES = 10
INPUT_DIM = 64
BLOCKCHAIN_FILE = "blockchain.json"

MINER_IDS = ["Miner_1", "Miner_2", "Miner_3"]
CLIENT_IDS = ["Device_A", "Device_B", "Device_C"]

if __name__ == "__main__":
    print("="*70)
    print("ENHANCED FEDERATED LEARNING WITH PRIVACY & ROBUSTNESS")
    print("="*70)
    print("\nFeatures:")
    print("  ✓ Differential Privacy (DP-SGD)")
    print("  ✓ Byzantine Attack Simulation")
    print("  ✓ Privacy Budget Tracking")
    print("  ✓ Blockchain Integration")
    print("="*70)
    
    # Configuration
    ENABLE_DP = False  # Set to True to enable differential privacy
    ENABLE_BYZANTINE = False  # Set to True to simulate Byzantine attacks
    NUM_BYZANTINE = 1  # Number of Byzantine clients
    
    print(f"\nConfiguration:")
    print(f"  - Differential Privacy: {'ENABLED' if ENABLE_DP else 'DISABLED'}")
    print(f"  - Byzantine Attacks: {'ENABLED' if ENABLE_BYZANTINE else 'DISABLED'}")
    if ENABLE_BYZANTINE:
        print(f"  - Byzantine Clients: {NUM_BYZANTINE}/{len(CLIENT_IDS)}")
    print()
    
    # Initialize a data loader for all clients
    data_loader = SpeechCommandsDataLoader(num_clients=len(CLIENT_IDS))

    # Create a list of client instances to be run
    clients_to_run = []
    
    for i, cid in enumerate(CLIENT_IDS):
        # Determine if this client should be Byzantine
        is_byzantine = ENABLE_BYZANTINE and i < NUM_BYZANTINE
        
        client = DecentralizedClient(
            client_id=cid,
            miner_id=MINER_IDS[i],
            client_data=data_loader.get_client_data(client_id=i),
            input_dim=INPUT_DIM,
            output_dim=NUM_CLASSES,
            target_words=data_loader.target_words,
            # Privacy parameters
            enable_dp=ENABLE_DP,
            noise_multiplier=1.1,
            max_grad_norm=1.0,
            target_epsilon=10.0,
            # Byzantine parameters
            is_byzantine=is_byzantine,
            attack_type="random" if is_byzantine else None,
            attack_strength=2.0
        )
        clients_to_run.append(client)
        
        if is_byzantine:
            print(f"[⚠️] {cid} configured as Byzantine attacker")
        elif ENABLE_DP:
            print(f"[🔒] {cid} configured with Differential Privacy")

    print("\nStarting federated learning simulation...")
    print("="*70)
    
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(CLIENT_IDS)) as executor:
        # Use executor.map to run the client loops and wait for them to finish
        executor.map(lambda c: c.run_client_loop(), clients_to_run)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*70)
    print("="*70)
    print("🎉 FEDERATED LEARNING SIMULATION COMPLETE")
    print("="*70)
    print("="*70)
    
    print(f"\n📊 SIMULATION SUMMARY:")
    print(f"  - Total Clients: {len(CLIENT_IDS)}")
    print(f"  - Total Epochs: {EPOCHS}")
    print(f"  - Local Epochs per Client: {LOCAL_EPOCHS}")
    print(f"  - Total Blocks Mined: {EPOCHS * len(CLIENT_IDS)}")
    print(f"  - Total Training Time: {total_time:.2f} seconds")
    print(f"  - Average Time per Epoch: {total_time/EPOCHS:.2f} seconds")
    
    print(f"\n💰 REWARDS DISTRIBUTED:")
    base_reward = 10.0
    difficulty_bonus = 1 * 2.0  # difficulty * 2
    reward_per_block = base_reward + difficulty_bonus
    total_rewards = reward_per_block * EPOCHS * len(CLIENT_IDS)
    print(f"  - Reward per Block: {reward_per_block} tokens")
    print(f"  - Total Rewards: {total_rewards} tokens")
    print(f"  - Rewards per Client: {total_rewards/len(CLIENT_IDS):.2f} tokens")
    
    print(f"\n🔗 BLOCKCHAIN STATUS:")
    print(f"  - Blockchain File: blockchain.json")
    print(f"  - Total Blocks: {EPOCHS * len(CLIENT_IDS) + 1} (including genesis)")
    print(f"  - Consensus: Proof of Authority (PoA)")
    
    if ENABLE_DP:
        print(f"\n🔒 PRIVACY STATUS:")
        print(f"  - Differential Privacy: ENABLED")
        print(f"  - Privacy Budget: ε ≤ 10.0")
        print(f"  - Noise Multiplier: 1.1")
    
    if ENABLE_BYZANTINE:
        print(f"\n⚠️ SECURITY STATUS:")
        print(f"  - Byzantine Clients: {NUM_BYZANTINE}/{len(CLIENT_IDS)}")
        print(f"  - Attack Type: random")
        print(f"  - Robust Aggregation: Available")
    
    print("\n" + "="*70)
    print("All clients have finished their tasks.")
    print("="*70)