"""
Complete Federated Learning Implementation with Blockchain Integration
Shows model aggregation, global model updates, and blockchain storage
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Dict, List
from collections import OrderedDict

from data_loader import SpeechCommandsDataLoader
from fl_node import SimpleAudioClassifier
from client import DecentralizedClient, aggregate_local_models, compute_global_model_hash
from ehr_chain import EHRChain
from miner import Miner

# Try to import enhanced modules
try:
    from src.storage.ipfs_manager import IPFSManager
    from src.blockchain.transaction import ModelUpdateTransaction
    from src.blockchain.incentives import IncentiveManager, StakingManager
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

# Configuration
EPOCHS = 5
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3
NUM_CLASSES = 10
INPUT_DIM = 64

CLIENT_IDS = ["Device_A", "Device_B", "Device_C"]
MINER_IDS = ["Miner_1", "Miner_2", "Miner_3"]

# Feature flags
ENABLE_DP = False
ENABLE_BYZANTINE = False
NUM_BYZANTINE = 1
ENABLE_IPFS = True if ENHANCED_AVAILABLE else False
ENABLE_INCENTIVES = True if ENHANCED_AVAILABLE else False


def federated_averaging(client_weights_list: List[Dict], client_sizes: List[int]) -> Dict:
    """
    Perform Federated Averaging (FedAvg) to aggregate client models.
    
    Args:
        client_weights_list: List of client model weights
        client_sizes: List of number of samples per client
        
    Returns:
        Aggregated global model weights
    """
    print(f"\n{'='*70}")
    print("🔄 PERFORMING FEDERATED AVERAGING")
    print(f"{'='*70}")
    print(f"Aggregating {len(client_weights_list)} client models...")
    
    total_samples = sum(client_sizes)
    print(f"Total samples: {total_samples}")
    
    # Initialize global weights
    global_weights = {}
    
    # Get first client's weights as template
    first_client = list(client_weights_list[0].keys())[0]
    for key in client_weights_list[0][first_client].keys():
        global_weights[key] = np.zeros_like(client_weights_list[0][first_client][key])
    
    # Weighted average
    for i, (client_weights, client_size) in enumerate(zip(client_weights_list, client_sizes)):
        weight = client_size / total_samples
        client_id = list(client_weights.keys())[0]
        
        print(f"  Client {i+1}: weight={weight:.3f} ({client_size} samples)")
        
        for key in global_weights.keys():
            global_weights[key] += client_weights[client_id][key] * weight
    
    print(f"✅ Aggregation complete!")
    print(f"{'='*70}")
    
    return global_weights


def main():
    """Main federated learning with blockchain integration."""
    
    print("="*70)
    print("="*70)
    print("🌐 FEDERATED LEARNING WITH BLOCKCHAIN")
    print("="*70)
    print("="*70)
    
    print("\n📋 SYSTEM CONFIGURATION:")
    print(f"  - Total Clients: {len(CLIENT_IDS)}")
    print(f"  - FL Rounds (Epochs): {EPOCHS}")
    print(f"  - Local Epochs: {LOCAL_EPOCHS}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Learning Rate: {LR}")
    print(f"  - Differential Privacy: {'ENABLED' if ENABLE_DP else 'DISABLED'}")
    print(f"  - Byzantine Attacks: {'ENABLED' if ENABLE_BYZANTINE else 'DISABLED'}")
    print(f"  - IPFS Storage: {'ENABLED' if ENABLE_IPFS else 'DISABLED'}")
    print(f"  - Incentives: {'ENABLED' if ENABLE_INCENTIVES else 'DISABLED'}")
    
    # Initialize components
    print("\n🔧 INITIALIZING COMPONENTS...")
    
    # Data loader
    print("  Loading dataset...")
    data_loader = SpeechCommandsDataLoader(num_clients=len(CLIENT_IDS))
    print(f"  ✓ Dataset loaded: {len(data_loader.processed_dataset)} samples")
    
    # Global model
    print("  Creating global model...")
    global_model = SimpleAudioClassifier(INPUT_DIM, NUM_CLASSES)
    print(f"  ✓ Global model created")
    
    # Blockchain
    print("  Initializing blockchain...")
    global_chain = EHRChain()
    if not global_chain.chain:
        global_chain.create_genesis_block()
    print(f"  ✓ Blockchain initialized")
    
    # IPFS (if available)
    ipfs_manager = None
    if ENABLE_IPFS:
        print("  Initializing IPFS...")
        ipfs_manager = IPFSManager()
        print(f"  ✓ IPFS initialized")
    
    # Incentives (if available)
    incentive_manager = None
    staking_manager = None
    if ENABLE_INCENTIVES:
        print("  Initializing incentive system...")
        incentive_manager = IncentiveManager()
        staking_manager = StakingManager()
        
        # Initial staking
        for client_id in CLIENT_IDS:
            staking_manager.stake(client_id, 500.0)
        print(f"  ✓ Incentive system initialized")
    
    # Create clients
    print("\n👥 CREATING CLIENTS...")
    clients = []
    for i, cid in enumerate(CLIENT_IDS):
        is_byzantine = ENABLE_BYZANTINE and i < NUM_BYZANTINE
        
        client = DecentralizedClient(
            client_id=cid,
            miner_id=MINER_IDS[i],
            client_data=data_loader.get_client_data(client_id=i),
            input_dim=INPUT_DIM,
            output_dim=NUM_CLASSES,
            target_words=data_loader.target_words,
            enable_dp=ENABLE_DP,
            noise_multiplier=1.1,
            max_grad_norm=1.0,
            target_epsilon=10.0,
            is_byzantine=is_byzantine,
            attack_type="random" if is_byzantine else None,
            attack_strength=2.0
        )
        clients.append(client)
        
        status = "BYZANTINE" if is_byzantine else ("DP-ENABLED" if ENABLE_DP else "HONEST")
        print(f"  ✓ {cid} created [{status}]")
    
    print("\n" + "="*70)
    print("🚀 STARTING FEDERATED LEARNING")
    print("="*70)
    
    # Federated Learning Loop
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'#'*70}")
        print(f"{'#'*70}")
        print(f"##  FEDERATED LEARNING ROUND {epoch}/{EPOCHS}")
        print(f"{'#'*70}")
        print(f"{'#'*70}")
        
        # Step 1: Broadcast global model to all clients
        print(f"\n📤 STEP 1: BROADCASTING GLOBAL MODEL TO CLIENTS")
        global_weights_dict = {k: v.cpu().numpy() for k, v in global_model.state_dict().items()}
        
        for client in clients:
            client.update_model(global_weights_dict)
        print(f"✅ Global model broadcasted to {len(clients)} clients")
        
        # Step 2: Local training on each client
        print(f"\n📚 STEP 2: LOCAL TRAINING ON CLIENTS")
        client_updates = []
        client_accuracies = []
        client_sizes = []
        
        for client in clients:
            print(f"\n  Training {client.client_id}...")
            weights, acc, model_id, preds = client.local_train(LOCAL_EPOCHS, BATCH_SIZE, LR)
            
            client_updates.append({client.client_id: weights})
            client_accuracies.append(acc)
            client_sizes.append(len(client.local_data))
            
            print(f"  ✓ {client.client_id}: Accuracy={acc*100:.2f}%")
            
            # Upload to IPFS if enabled
            if ENABLE_IPFS and ipfs_manager:
                cid = client.upload_model_to_ipfs(acc, len(client.local_data))
        
        # Step 3: Aggregate models (FedAvg)
        print(f"\n🔄 STEP 3: AGGREGATING CLIENT MODELS")
        aggregated_weights = federated_averaging(client_updates, client_sizes)
        
        # Update global model
        global_model_state = OrderedDict()
        for key, value in aggregated_weights.items():
            global_model_state[key] = torch.from_numpy(value)
        global_model.load_state_dict(global_model_state)
        
        global_model_hash = compute_global_model_hash({"global": aggregated_weights})
        print(f"✅ Global model updated")
        print(f"  - Global Model Hash: {global_model_hash[:32]}...")
        print(f"  - Average Accuracy: {np.mean(client_accuracies)*100:.2f}%")
        
        # Step 4: Mine blocks and add to blockchain
        print(f"\n⛏️  STEP 4: MINING BLOCKS")
        for i, client in enumerate(clients):
            txs = [{
                "client": client.client_id,
                "miner": client.miner_id,
                "accuracy": client_accuracies[i],
                "samples": client_sizes[i],
                "epoch": epoch
            }]
            
            winner_block = client.miner.mine_block(txs, global_model_hash)
            
            if winner_block:
                global_chain.add_block(winner_block)
                
                # Calculate and distribute rewards
                if ENABLE_INCENTIVES and incentive_manager:
                    reward = incentive_manager.calculate_reward(
                        participant=client.client_id,
                        contribution_quality=client_accuracies[i],
                        num_samples=client_sizes[i],
                        round_number=epoch
                    )
                    print(f"  💰 {client.client_id} earned {reward:.2f} tokens")
        
        # Save blockchain
        global_chain.save_to_file("blockchain.json")
        
        print(f"\n{'='*70}")
        print(f"ROUND {epoch} COMPLETE")
        print(f"  - Global Model Accuracy: {np.mean(client_accuracies)*100:.2f}%")
        print(f"  - Blocks Added: {len(clients)}")
        print(f"  - Total Chain Length: {len(global_chain.chain)}")
        print(f"{'='*70}")
    
    # Final Summary
    print("\n" + "="*70)
    print("="*70)
    print("🎉 FEDERATED LEARNING COMPLETE")
    print("="*70)
    print("="*70)
    
    print(f"\n📊 FINAL STATISTICS:")
    print(f"  - Total Rounds: {EPOCHS}")
    print(f"  - Total Clients: {len(CLIENT_IDS)}")
    print(f"  - Total Blocks: {len(global_chain.chain)}")
    print(f"  - Final Global Model Hash: {global_model_hash[:32]}...")
    
    if ENABLE_INCENTIVES and incentive_manager:
        print(f"\n💰 REWARDS SUMMARY:")
        stats = incentive_manager.get_statistics()
        print(f"  - Total Rewards: {stats['total_rewards_distributed']:.2f} tokens")
        print(f"  - Participants: {stats['num_participants']}")
        
        for client_id in CLIENT_IDS:
            balance = incentive_manager.get_balance(client_id)
            print(f"  - {client_id}: {balance:.2f} tokens")
    
    if ENABLE_INCENTIVES and staking_manager:
        print(f"\n🔒 STAKING SUMMARY:")
        stats = staking_manager.get_statistics()
        print(f"  - Total Staked: {stats['total_staked']:.2f} tokens")
        print(f"  - Active Stakers: {stats['num_stakers']}")
    
    print(f"\n📁 FILES SAVED:")
    print(f"  - blockchain.json (complete blockchain)")
    print(f"  - blockchain_detailed.log (human-readable)")
    print(f"  - blockchain_summary.txt (quick overview)")
    
    print("\n" + "="*70)
    print("All data saved successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
