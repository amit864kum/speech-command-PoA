"""
Complete System Demonstration
Shows how blockchain stores federated learning details
"""

import json
from federated_blockchain_integration import FederatedBlockchainSystem
from data_loader import SpeechCommandsDataLoader


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}\n")


def demonstrate_blockchain_storage():
    """Demonstrate where and how blockchain details are stored"""
    
    print_section("🌐 COMPLETE FEDERATED LEARNING + BLOCKCHAIN DEMO")
    
    # Configuration
    NUM_CLIENTS = 3
    NUM_ROUNDS = 3  # Reduced for demo
    LOCAL_EPOCHS = 2
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    
    print("📋 Configuration:")
    print(f"  - Clients: {NUM_CLIENTS}")
    print(f"  - Rounds: {NUM_ROUNDS}")
    print(f"  - Local Epochs: {LOCAL_EPOCHS}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    
    # Initialize system
    print_section("🔧 STEP 1: INITIALIZING SYSTEM")
    
    fl_system = FederatedBlockchainSystem(
        num_clients=NUM_CLIENTS,
        num_classes=10,
        input_dim=64,
        enable_blockchain=True,
        enable_ipfs=False,
        enable_incentives=False,
        enable_privacy=False
    )
    
    # Setup components
    fl_system.setup_blockchain()
    fl_system.setup_ipfs()
    fl_system.setup_incentives()
    
    # Load data
    print_section("📊 STEP 2: LOADING DATA")
    data_loader = SpeechCommandsDataLoader(num_clients=NUM_CLIENTS)
    print(f"✓ Dataset loaded: {len(data_loader.processed_dataset)} samples")
    
    # Create clients
    print_section("👥 STEP 3: CREATING CLIENTS")
    fl_system.create_clients(data_loader)
    
    # Run federated learning
    print_section("🚀 STEP 4: RUNNING FEDERATED LEARNING")
    fl_system.run_federated_learning(
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    # Show blockchain details
    print_section("🔍 STEP 5: EXAMINING BLOCKCHAIN STORAGE")
    
    print("📁 Files Created:")
    print("  1. blockchain.json - Complete blockchain data")
    print("  2. blockchain_detailed.log - Human-readable log")
    print("  3. blockchain_summary.txt - Quick summary")
    print("  4. blockchain_details.json - Programmatic export")
    
    # Get and display blockchain details
    blockchain_details = fl_system.get_blockchain_details()
    
    print(f"\n📊 Blockchain Statistics:")
    print(f"  - Total Blocks: {blockchain_details['total_blocks']}")
    
    # Show first few blocks
    print(f"\n🔗 Block Details (First 3 blocks):")
    for i, block in enumerate(blockchain_details['blocks'][:3]):
        print(f"\n  Block #{block['index']}:")
        print(f"    - Hash: {block['current_hash'][:32]}...")
        print(f"    - Previous Hash: {block['previous_hash'][:32]}...")
        print(f"    - Model Hash: {block['model_hash'][:32]}...")
        print(f"    - Miner: {block['miner']}")
        print(f"    - Difficulty: {block['difficulty']}")
        print(f"    - Nonce: {block['nonce']}")
        print(f"    - Transactions: {block['num_transactions']}")
        
        if block['transactions']:
            print(f"    - Transaction Data:")
            for tx in block['transactions']:
                if isinstance(tx, dict):
                    print(f"      • Client: {tx.get('client_id', 'N/A')}")
                    print(f"      • Accuracy: {tx.get('accuracy', 0)*100:.2f}%")
                    print(f"      • Samples: {tx.get('num_samples', 0)}")
    
    # Save detailed export
    with open("blockchain_details.json", "w") as f:
        json.dump(blockchain_details, f, indent=4)
    
    print(f"\n✅ Blockchain details exported to blockchain_details.json")
    
    # Show how to access blockchain programmatically
    print_section("💻 STEP 6: PROGRAMMATIC ACCESS EXAMPLE")
    
    print("Python code to access blockchain:")
    print("""
    from ehr_chain import EHRChain
    
    # Load blockchain
    blockchain = EHRChain.load_from_file("blockchain.json")
    
    # Access blocks
    for block in blockchain.chain:
        print(f"Block #{block.index}")
        print(f"  Hash: {block.hash}")
        print(f"  Model Hash: {block.model_hash}")
        print(f"  Miner: {block.miner}")
        print(f"  Transactions: {block.records}")
    
    # Get latest block
    latest_block = blockchain.chain[-1]
    print(f"Latest block: #{latest_block.index}")
    """)
    
    # Show blockchain verification
    print_section("🔒 STEP 7: BLOCKCHAIN VERIFICATION")
    
    print("Verifying blockchain integrity...")
    
    if fl_system.blockchain:
        is_valid = True
        for i in range(1, len(fl_system.blockchain.chain)):
            current = fl_system.blockchain.chain[i]
            previous = fl_system.blockchain.chain[i-1]
            
            # Check hash linkage
            if current.prev_hash != previous.hash:
                is_valid = False
                print(f"  ✗ Block #{i}: Hash linkage broken")
                break
            
            # Check proof of work
            if not current.hash.startswith("0" * current.difficulty):
                is_valid = False
                print(f"  ✗ Block #{i}: Invalid proof of work")
                break
        
        if is_valid:
            print("  ✓ Blockchain integrity verified")
            print("  ✓ All blocks properly linked")
            print("  ✓ All proofs of work valid")
    
    # Summary
    print_section("📋 SUMMARY")
    
    print("✅ Demonstration Complete!")
    print()
    print("What was demonstrated:")
    print("  1. ✓ Federated learning with multiple clients")
    print("  2. ✓ Local training on each client")
    print("  3. ✓ Model aggregation (FedAvg)")
    print("  4. ✓ Blockchain storage of FL rounds")
    print("  5. ✓ Block mining with PoA consensus")
    print("  6. ✓ Complete audit trail")
    print()
    print("Where blockchain details are stored:")
    print("  • blockchain.json - Complete data")
    print("  • blockchain_detailed.log - Human-readable")
    print("  • blockchain_summary.txt - Quick overview")
    print("  • blockchain_details.json - Programmatic export")
    print()
    print("What each block contains:")
    print("  • Block header (index, timestamp, nonce, difficulty)")
    print("  • Hashes (previous, current, model)")
    print("  • Transaction data (FL round, accuracy, samples)")
    print("  • Mining info (miner ID, consensus)")
    print("  • Digital signatures")
    print()
    print("How to access:")
    print("  • Python: Use EHRChain.load_from_file()")
    print("  • JSON: Read blockchain.json directly")
    print("  • API: Use FederatedBlockchainSystem.get_blockchain_details()")
    print()
    print(f"{'='*70}")


def show_blockchain_structure():
    """Show the structure of stored blockchain data"""
    
    print_section("📖 BLOCKCHAIN STORAGE STRUCTURE")
    
    print("blockchain.json structure:")
    print("""
{
  "blockchain_info": {
    "version": "2.0",
    "consensus": "Proof of Authority (PoA)",
    "total_blocks": <number>,
    "total_rewards": <number>,
    "last_updated": "<timestamp>"
  },
  "chain": [
    {
      "block_header": {
        "index": <block_number>,
        "timestamp": <unix_timestamp>,
        "timestamp_readable": "<readable_time>",
        "nonce": <nonce>,
        "difficulty": <difficulty>
      },
      "hashes": {
        "previous_hash": "<prev_hash>",
        "current_hash": "<curr_hash>",
        "model_hash": "<model_hash>"
      },
      "signature": {
        "public_key": "<public_key>",
        "signature": "<signature>",
        "miner": "<miner_id>"
      },
      "data": {
        "records": [<transactions>],
        "access_logs": [<logs>],
        "num_transactions": <count>
      },
      "mining_info": {
        "miner_id": "<miner>",
        "difficulty": <diff>,
        "nonce": <nonce>,
        "consensus": "PoA"
      },
      "rewards": {
        "base_reward": 10.0,
        "difficulty_bonus": <bonus>,
        "total_reward": <total>
      }
    }
  ]
}
    """)
    
    print("\nTransaction data in each block:")
    print("""
{
  "round": <fl_round_number>,
  "client_id": "<client_identifier>",
  "miner_id": "<miner_identifier>",
  "accuracy": <model_accuracy>,
  "num_samples": <training_samples>,
  "timestamp": <unix_timestamp>,
  "model_hash": "<global_model_hash>",
  "predictions": [
    {
      "predicted_word": "<word>",
      "confidence": <percentage>
    }
  ]
}
    """)


if __name__ == "__main__":
    # Show structure first
    show_blockchain_structure()
    
    # Run demonstration
    demonstrate_blockchain_storage()
