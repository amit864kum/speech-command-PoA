"""
Complete Federated Learning + Blockchain Integration
This file demonstrates the full integration of federated learning with blockchain storage
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
from typing import Dict, List, Tuple
from collections import OrderedDict

# Core components
from data_loader import SpeechCommandsDataLoader
from fl_node import SimpleAudioClassifier
from client import DecentralizedClient
from ehr_chain import EHRChain
from block import Block
from miner import Miner

# Enhanced modules
try:
    from src.federated.aggregator import FederatedAggregator
    from src.blockchain.transaction import ModelUpdateTransaction
    from src.blockchain.incentives import IncentiveManager, StakingManager
    from src.storage.ipfs_manager import IPFSManager
    from src.privacy.differential_privacy import PrivacyEngine
    ENHANCED_MODULES = True
except ImportError:
    ENHANCED_MODULES = False
    print("[INFO] Some enhanced modules not available")


class FederatedBlockchainSystem:
    """
    Complete Federated Learning System with Blockchain Integration
    
    This class manages:
    1. Federated learning rounds
    2. Model aggregation
    3. Blockchain storage of model updates
    4. IPFS storage of model weights
    5. Incentive distribution
    6. Privacy preservation
    """
    
    def __init__(self, 
                 num_clients: int = 3,
                 num_classes: int = 10,
                 input_dim: int = 64,
                 enable_blockchain: bool = True,
                 enable_ipfs: bool = False,
                 enable_incentives: bool = False,
                 enable_privacy: bool = False):
        """
        Initialize the Federated Blockchain System
        
        Args:
            num_clients: Number of federated learning clients
            num_classes: Number of output classes
            input_dim: Input dimension for the model
            enable_blockchain: Enable blockchain storage
            enable_ipfs: Enable IPFS for model storage
            enable_incentives: Enable token-based incentives
            enable_privacy: Enable differential privacy
        """
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.input_dim = input_dim
        
        # Feature flags
        self.enable_blockchain = enable_blockchain
        self.enable_ipfs = enable_ipfs and ENHANCED_MODULES
        self.enable_incentives = enable_incentives and ENHANCED_MODULES
        self.enable_privacy = enable_privacy and ENHANCED_MODULES
        
        # Initialize components
        self.global_model = SimpleAudioClassifier(input_dim, num_classes)
        self.clients: List[DecentralizedClient] = []
        self.blockchain = None
        self.ipfs_manager = None
        self.incentive_manager = None
        self.staking_manager = None
        
        # Statistics
        self.round_history = []
        self.accuracy_history = []
        
        print(f"\n{'='*70}")
        print("🌐 FEDERATED BLOCKCHAIN SYSTEM INITIALIZED")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  - Clients: {num_clients}")
        print(f"  - Classes: {num_classes}")
        print(f"  - Input Dim: {input_dim}")
        print(f"  - Blockchain: {'✓' if enable_blockchain else '✗'}")
        print(f"  - IPFS: {'✓' if self.enable_ipfs else '✗'}")
        print(f"  - Incentives: {'✓' if self.enable_incentives else '✗'}")
        print(f"  - Privacy: {'✓' if self.enable_privacy else '✗'}")
        print(f"{'='*70}\n")
    
    def setup_blockchain(self):
        """Initialize blockchain for storing federated learning updates"""
        if not self.enable_blockchain:
            return
        
        print("🔗 Setting up blockchain...")
        self.blockchain = EHRChain()
        
        # Load existing blockchain or create genesis
        try:
            self.blockchain = EHRChain.load_from_file("blockchain.json")
            print(f"  ✓ Loaded existing blockchain with {len(self.blockchain.chain)} blocks")
        except:
            self.blockchain.create_genesis_block()
            print(f"  ✓ Created new blockchain with genesis block")
    
    def setup_ipfs(self):
        """Initialize IPFS for decentralized model storage"""
        if not self.enable_ipfs:
            return
        
        print("📦 Setting up IPFS...")
        try:
            self.ipfs_manager = IPFSManager()
            print(f"  ✓ IPFS manager initialized")
        except Exception as e:
            print(f"  ✗ IPFS setup failed: {e}")
            self.enable_ipfs = False
    
    def setup_incentives(self):
        """Initialize incentive and staking mechanisms"""
        if not self.enable_incentives:
            return
        
        print("💰 Setting up incentive system...")
        self.incentive_manager = IncentiveManager()
        self.staking_manager = StakingManager()
        print(f"  ✓ Incentive system initialized")
    
    def create_clients(self, data_loader: SpeechCommandsDataLoader):
        """
        Create federated learning clients with their local data
        
        Args:
            data_loader: Data loader with partitioned client data
        """
        print(f"\n👥 Creating {self.num_clients} clients...")
        
        for i in range(self.num_clients):
            client_id = f"Client_{i+1}"
            miner_id = f"Miner_{i+1}"
            
            # Get client's local data
            client_data = data_loader.get_client_data(client_id=i)
            
            # Create client
            client = DecentralizedClient(
                client_id=client_id,
                miner_id=miner_id,
                client_data=client_data,
                input_dim=self.input_dim,
                output_dim=self.num_classes,
                target_words=data_loader.target_words,
                enable_dp=self.enable_privacy,
                noise_multiplier=1.1,
                max_grad_norm=1.0,
                target_epsilon=10.0
            )
            
            self.clients.append(client)
            
            # Initial staking if incentives enabled
            if self.enable_incentives and self.staking_manager:
                self.staking_manager.stake(client_id, 500.0)
            
            print(f"  ✓ {client_id} created with {len(client_data)} samples")
        
        print(f"✅ All clients created successfully\n")
    
    def broadcast_global_model(self):
        """Broadcast current global model to all clients"""
        global_weights = {k: v.cpu().numpy() for k, v in self.global_model.state_dict().items()}
        
        for client in self.clients:
            client.update_model(global_weights)
    
    def aggregate_client_models(self, 
                                client_updates: List[Dict],
                                client_sizes: List[int]) -> Dict:
        """
        Perform Federated Averaging to aggregate client models
        
        Args:
            client_updates: List of client model weights
            client_sizes: Number of samples per client
            
        Returns:
            Aggregated global model weights
        """
        total_samples = sum(client_sizes)
        global_weights = {}
        
        # Initialize with zeros
        first_client_id = list(client_updates[0].keys())[0]
        for key in client_updates[0][first_client_id].keys():
            global_weights[key] = np.zeros_like(client_updates[0][first_client_id][key])
        
        # Weighted average based on number of samples
        for client_weights, client_size in zip(client_updates, client_sizes):
            weight = client_size / total_samples
            client_id = list(client_weights.keys())[0]
            
            for key in global_weights.keys():
                global_weights[key] += client_weights[client_id][key] * weight
        
        return global_weights
    
    def store_round_on_blockchain(self,
                                  round_num: int,
                                  global_model_hash: str,
                                  client_accuracies: List[float],
                                  client_sizes: List[int]):
        """
        Store federated learning round information on blockchain
        
        Args:
            round_num: Current round number
            global_model_hash: Hash of aggregated global model
            client_accuracies: Accuracy of each client
            client_sizes: Number of samples per client
        """
        if not self.enable_blockchain or not self.blockchain:
            return
        
        print(f"\n⛏️  Mining blocks for round {round_num}...")
        
        for i, client in enumerate(self.clients):
            # Create transaction data
            transaction = {
                "round": round_num,
                "client_id": client.client_id,
                "miner_id": client.miner_id,
                "accuracy": client_accuracies[i],
                "num_samples": client_sizes[i],
                "timestamp": time.time(),
                "model_hash": global_model_hash
            }
            
            # Mine block
            winner_block = client.miner.mine_block([transaction], global_model_hash)
            
            if winner_block:
                self.blockchain.add_block(winner_block)
                print(f"  ✓ Block #{winner_block.index} mined by {client.miner_id}")
                
                # Distribute rewards
                if self.enable_incentives and self.incentive_manager:
                    reward = self.incentive_manager.calculate_reward(
                        participant=client.client_id,
                        contribution_quality=client_accuracies[i],
                        num_samples=client_sizes[i],
                        round_number=round_num
                    )
                    print(f"    💰 Reward: {reward:.2f} tokens")
        
        # Save blockchain to file
        self.blockchain.save_to_file("blockchain.json")
        print(f"  ✓ Blockchain saved ({len(self.blockchain.chain)} blocks total)")
    
    def train_round(self, 
                   round_num: int,
                   local_epochs: int = 5,
                   batch_size: int = 32,
                   learning_rate: float = 1e-3) -> Tuple[float, str]:
        """
        Execute one round of federated learning
        
        Args:
            round_num: Current round number
            local_epochs: Number of local training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            
        Returns:
            Tuple of (average_accuracy, global_model_hash)
        """
        print(f"\n{'#'*70}")
        print(f"##  FEDERATED LEARNING ROUND {round_num}")
        print(f"{'#'*70}\n")
        
        # Step 1: Broadcast global model
        print("📤 Step 1: Broadcasting global model to clients...")
        self.broadcast_global_model()
        print("  ✓ Global model broadcasted\n")
        
        # Step 2: Local training
        print("📚 Step 2: Local training on clients...")
        client_updates = []
        client_accuracies = []
        client_sizes = []
        
        for client in self.clients:
            print(f"  Training {client.client_id}...")
            weights, acc, model_id, preds = client.local_train(
                local_epochs, batch_size, learning_rate
            )
            
            client_updates.append({client.client_id: weights})
            client_accuracies.append(acc)
            client_sizes.append(len(client.local_data))
            
            print(f"    ✓ Accuracy: {acc*100:.2f}%")
            
            # Upload to IPFS if enabled
            if self.enable_ipfs and client.ipfs_manager:
                cid = client.upload_model_to_ipfs(acc, len(client.local_data))
        
        print()
        
        # Step 3: Aggregate models
        print("🔄 Step 3: Aggregating client models...")
        aggregated_weights = self.aggregate_client_models(client_updates, client_sizes)
        
        # Update global model
        global_model_state = OrderedDict()
        for key, value in aggregated_weights.items():
            global_model_state[key] = torch.from_numpy(value)
        self.global_model.load_state_dict(global_model_state)
        
        # Compute global model hash
        import hashlib
        weights_str = json.dumps(
            {k: v.tolist() for k, v in aggregated_weights.items()},
            sort_keys=True
        )
        global_model_hash = hashlib.sha256(weights_str.encode()).hexdigest()
        
        avg_accuracy = np.mean(client_accuracies)
        
        print(f"  ✓ Aggregation complete")
        print(f"  - Global Model Hash: {global_model_hash[:32]}...")
        print(f"  - Average Accuracy: {avg_accuracy*100:.2f}%")
        
        # Step 4: Store on blockchain
        if self.enable_blockchain:
            self.store_round_on_blockchain(
                round_num, global_model_hash, client_accuracies, client_sizes
            )
        
        # Update history
        self.round_history.append({
            "round": round_num,
            "accuracy": avg_accuracy,
            "model_hash": global_model_hash,
            "client_accuracies": client_accuracies
        })
        self.accuracy_history.append(avg_accuracy)
        
        print(f"\n{'='*70}")
        print(f"ROUND {round_num} COMPLETE")
        print(f"{'='*70}\n")
        
        return avg_accuracy, global_model_hash
    
    def run_federated_learning(self,
                              num_rounds: int = 5,
                              local_epochs: int = 5,
                              batch_size: int = 32,
                              learning_rate: float = 1e-3):
        """
        Run complete federated learning process
        
        Args:
            num_rounds: Number of federated learning rounds
            local_epochs: Local training epochs per round
            batch_size: Training batch size
            learning_rate: Learning rate
        """
        print(f"\n{'='*70}")
        print("🚀 STARTING FEDERATED LEARNING")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  - Rounds: {num_rounds}")
        print(f"  - Local Epochs: {local_epochs}")
        print(f"  - Batch Size: {batch_size}")
        print(f"  - Learning Rate: {learning_rate}")
        print(f"{'='*70}\n")
        
        # Run training rounds
        for round_num in range(1, num_rounds + 1):
            avg_acc, model_hash = self.train_round(
                round_num, local_epochs, batch_size, learning_rate
            )
        
        # Final summary
        self.print_summary()
    
    def print_summary(self):
        """Print final summary of federated learning"""
        print(f"\n{'='*70}")
        print("🎉 FEDERATED LEARNING COMPLETE")
        print(f"{'='*70}\n")
        
        print("📊 Training Summary:")
        print(f"  - Total Rounds: {len(self.round_history)}")
        print(f"  - Total Clients: {len(self.clients)}")
        print(f"  - Final Accuracy: {self.accuracy_history[-1]*100:.2f}%")
        print(f"  - Best Accuracy: {max(self.accuracy_history)*100:.2f}%")
        
        if self.enable_blockchain and self.blockchain:
            print(f"\n🔗 Blockchain Summary:")
            print(f"  - Total Blocks: {len(self.blockchain.chain)}")
            print(f"  - Storage: blockchain.json")
        
        if self.enable_incentives and self.incentive_manager:
            print(f"\n💰 Incentives Summary:")
            stats = self.incentive_manager.get_statistics()
            print(f"  - Total Rewards: {stats['total_rewards_distributed']:.2f} tokens")
            print(f"  - Participants: {stats['num_participants']}")
            
            for client in self.clients:
                balance = self.incentive_manager.get_balance(client.client_id)
                print(f"  - {client.client_id}: {balance:.2f} tokens")
        
        print(f"\n📁 Output Files:")
        print(f"  - blockchain.json (complete blockchain)")
        print(f"  - blockchain_detailed.log (human-readable log)")
        print(f"  - blockchain_summary.txt (quick summary)")
        
        print(f"\n{'='*70}\n")
    
    def get_blockchain_details(self) -> Dict:
        """
        Get detailed information about blockchain storage
        
        Returns:
            Dictionary with blockchain details
        """
        if not self.enable_blockchain or not self.blockchain:
            return {"error": "Blockchain not enabled"}
        
        details = {
            "total_blocks": len(self.blockchain.chain),
            "blocks": []
        }
        
        for block in self.blockchain.chain:
            block_info = {
                "index": block.index,
                "timestamp": block.timestamp,
                "previous_hash": block.prev_hash,
                "current_hash": block.hash,
                "model_hash": block.model_hash,
                "miner": block.miner,
                "difficulty": block.difficulty,
                "nonce": block.nonce,
                "num_transactions": len(block.records),
                "transactions": block.records
            }
            details["blocks"].append(block_info)
        
        return details


def main():
    """Main execution function"""
    
    # Configuration
    NUM_CLIENTS = 3
    NUM_ROUNDS = 5
    LOCAL_EPOCHS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_CLASSES = 10
    INPUT_DIM = 64
    
    # Initialize system
    fl_system = FederatedBlockchainSystem(
        num_clients=NUM_CLIENTS,
        num_classes=NUM_CLASSES,
        input_dim=INPUT_DIM,
        enable_blockchain=True,
        enable_ipfs=False,  # Set to True if IPFS is available
        enable_incentives=False,  # Set to True if incentives desired
        enable_privacy=False  # Set to True for differential privacy
    )
    
    # Setup components
    fl_system.setup_blockchain()
    fl_system.setup_ipfs()
    fl_system.setup_incentives()
    
    # Load and partition data
    print("📊 Loading and partitioning data...")
    data_loader = SpeechCommandsDataLoader(num_clients=NUM_CLIENTS)
    print(f"  ✓ Dataset loaded: {len(data_loader.processed_dataset)} samples\n")
    
    # Create clients
    fl_system.create_clients(data_loader)
    
    # Run federated learning
    fl_system.run_federated_learning(
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    # Get blockchain details
    blockchain_details = fl_system.get_blockchain_details()
    
    # Save blockchain details to JSON
    with open("blockchain_details.json", "w") as f:
        json.dump(blockchain_details, f, indent=4)
    print("📄 Blockchain details saved to blockchain_details.json")


if __name__ == "__main__":
    main()
