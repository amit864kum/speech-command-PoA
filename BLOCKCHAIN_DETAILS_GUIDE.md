# Blockchain Storage and Details Guide

## Overview
This document explains where and how blockchain details are stored in the federated learning system.

## Storage Locations

### 1. **blockchain.json** (Primary Storage)
**Location:** `Speech_command/blockchain.json`

**Purpose:** Main blockchain storage file with complete block details

**Structure:**
```json
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
        "timestamp_readable": "<human_readable_time>",
        "nonce": <proof_of_work_nonce>,
        "difficulty": <mining_difficulty>
      },
      "hashes": {
        "previous_hash": "<hash_of_previous_block>",
        "current_hash": "<hash_of_current_block>",
        "model_hash": "<hash_of_ml_model>"
      },
      "signature": {
        "public_key": "<miner_public_key>",
        "signature": "<digital_signature>",
        "miner": "<miner_id>"
      },
      "data": {
        "records": [<transactions>],
        "access_logs": [<access_records>],
        "num_transactions": <count>
      },
      "mining_info": {
        "miner_id": "<miner_identifier>",
        "difficulty": <difficulty_level>,
        "nonce": <nonce_value>,
        "consensus": "PoA"
      },
      "rewards": {
        "base_reward": 10.0,
        "difficulty_bonus": <bonus_amount>,
        "total_reward": <total_tokens>
      }
    }
  ]
}
```

### 2. **blockchain_detailed.log** (Human-Readable Log)
**Location:** `Speech_command/blockchain_detailed.log`

**Purpose:** Detailed human-readable log of all blockchain operations

**Contents:**
- Block-by-block breakdown
- Complete transaction details
- Mining information
- Reward calculations
- Digital signatures
- Timestamps in readable format

**Example:**
```
================================================================================
BLOCK #1
================================================================================

BLOCK HEADER:
  Index: 1
  Timestamp: 2025-11-18 10:30:45
  Nonce: 12345
  Difficulty: 2

HASHES:
  Previous Hash: 0000abc123...
  Current Hash:  0000def456...
  Model Hash:    789ghi012...

DIGITAL SIGNATURE:
  Miner: Miner_1
  Public Key: abc123...
  Signature: def456...

MINING INFO:
  Miner ID: Miner_1
  Consensus: Proof of Authority (PoA)
  Difficulty: 2

REWARDS:
  Base Reward: 10.0 tokens
  Difficulty Bonus: 4.0 tokens
  Total Reward: 14.0 tokens

TRANSACTIONS:
  Count: 1
  Transaction 1: {...}
```

### 3. **blockchain_summary.txt** (Quick Overview)
**Location:** `Speech_command/blockchain_summary.txt`

**Purpose:** Quick summary of blockchain state

**Contents:**
- Total blocks
- Total rewards distributed
- Block list with key information
- Miner contributions

**Example:**
```
================================================================================
BLOCKCHAIN SUMMARY
================================================================================

Version: 2.0
Consensus: Proof of Authority (PoA)
Total Blocks: 15
Total Rewards: 210.0 tokens
Last Updated: 2025-11-18 10:35:22

BLOCK LIST:
--------------------------------------------------------------------------------
Block    Miner           Timestamp            Hash                 Reward    
--------------------------------------------------------------------------------
#0       Genesis         2025-11-18 10:30:00  0000000000000000...  0.0       
#1       Miner_1         2025-11-18 10:30:45  0000abc123456789...  14.0      
#2       Miner_2         2025-11-18 10:31:12  0000def987654321...  14.0      
...
```

### 4. **blockchain_details.json** (Detailed Export)
**Location:** `Speech_command/blockchain_details.json`

**Purpose:** Programmatic access to blockchain details

**Structure:**
```json
{
  "total_blocks": <number>,
  "blocks": [
    {
      "index": <block_number>,
      "timestamp": <unix_timestamp>,
      "previous_hash": "<hash>",
      "current_hash": "<hash>",
      "model_hash": "<ml_model_hash>",
      "miner": "<miner_id>",
      "difficulty": <difficulty>,
      "nonce": <nonce>,
      "num_transactions": <count>,
      "transactions": [<transaction_list>]
    }
  ]
}
```

## Block Structure Details

### Block Class (block.py)
Each block contains the following attributes:

```python
class Block:
    def __init__(self, index, prev_hash, records, access_logs, miner, model_hash, difficulty, nonce=0):
        self.index = index                    # Block number in chain
        self.timestamp = time.time()          # Unix timestamp
        self.prev_hash = prev_hash            # Hash of previous block
        self.records = records                # Transaction data
        self.access_logs = access_logs        # Access control logs
        self.miner = miner                    # Miner identifier
        self.model_hash = model_hash          # ML model hash
        self.difficulty = difficulty          # Mining difficulty
        self.nonce = nonce                    # Proof of work nonce
        self.hash = self.compute_hash()       # Current block hash
```

### Transaction Data in Blocks

Each block stores federated learning transactions:

```json
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
```

## Blockchain Operations

### 1. **Creating Blocks**
Blocks are created during federated learning rounds:

```python
# In client.py or federated_blockchain_integration.py
transaction = {
    "round": round_num,
    "client_id": client.client_id,
    "accuracy": accuracy,
    "num_samples": num_samples,
    "model_hash": global_model_hash
}

winner_block = miner.mine_block([transaction], model_hash)
blockchain.add_block(winner_block)
```

### 2. **Saving Blockchain**
Blockchain is saved after each round:

```python
# In ehr_chain.py
blockchain.save_to_file("blockchain.json")
```

This automatically creates:
- `blockchain.json`
- `blockchain_detailed.log`
- `blockchain_summary.txt`

### 3. **Loading Blockchain**
Load existing blockchain:

```python
# In ehr_chain.py
blockchain = EHRChain.load_from_file("blockchain.json")
```

## Federated Learning Integration

### How FL Updates Are Stored

1. **Local Training:** Each client trains locally
2. **Model Aggregation:** Server aggregates client models
3. **Hash Computation:** Global model hash is computed
4. **Block Creation:** Transaction with FL round info is created
5. **Mining:** Block is mined with PoA consensus
6. **Storage:** Block is added to chain and saved to files

### Data Flow

```
Client Training → Model Weights → Aggregation → Global Model
                                                      ↓
                                                 Model Hash
                                                      ↓
                                              Transaction Data
                                                      ↓
                                                Block Mining
                                                      ↓
                                              Blockchain Storage
                                                      ↓
                                    ┌─────────────────┴─────────────────┐
                                    ↓                                   ↓
                            blockchain.json                  blockchain_detailed.log
                                    ↓
                          blockchain_summary.txt
```

## Accessing Blockchain Data

### Python API

```python
# Load blockchain
from ehr_chain import EHRChain
blockchain = EHRChain.load_from_file("blockchain.json")

# Access blocks
for block in blockchain.chain:
    print(f"Block #{block.index}")
    print(f"  Hash: {block.hash}")
    print(f"  Model Hash: {block.model_hash}")
    print(f"  Miner: {block.miner}")
    print(f"  Transactions: {block.records}")

# Get specific block
latest_block = blockchain.chain[-1]
genesis_block = blockchain.chain[0]
```

### Using FederatedBlockchainSystem

```python
from federated_blockchain_integration import FederatedBlockchainSystem

# Initialize system
fl_system = FederatedBlockchainSystem(
    num_clients=3,
    enable_blockchain=True
)

# Get blockchain details
details = fl_system.get_blockchain_details()

# Access block information
for block_info in details["blocks"]:
    print(f"Block {block_info['index']}: {block_info['current_hash']}")
```

## IPFS Integration (Optional)

If IPFS is enabled, model weights are also stored on IPFS:

### IPFS Storage
- **Location:** Distributed IPFS network
- **Content:** Model weights + metadata
- **Reference:** CID (Content Identifier) stored in blockchain

### Structure
```json
{
  "model_weights": {<layer_weights>},
  "metadata": {
    "client_id": "<client>",
    "accuracy": <accuracy>,
    "num_samples": <samples>,
    "timestamp": <time>
  }
}
```

### Accessing IPFS Data
```python
from src.storage.ipfs_manager import IPFSManager

ipfs = IPFSManager()
weights, metadata = ipfs.download_model_weights(cid)
```

## Security Features

### 1. **Cryptographic Hashing**
- SHA-256 for block hashes
- Model weights hashed for integrity
- Chain integrity through linked hashes

### 2. **Digital Signatures**
- Miners sign blocks with private keys
- Public key verification
- Non-repudiation of block creation

### 3. **Proof of Authority**
- Authorized miners only
- Difficulty-based mining
- Consensus mechanism

## Verification

### Verify Blockchain Integrity

```python
def verify_blockchain(blockchain):
    for i in range(1, len(blockchain.chain)):
        current = blockchain.chain[i]
        previous = blockchain.chain[i-1]
        
        # Check hash linkage
        if current.prev_hash != previous.hash:
            return False
        
        # Check proof of work
        if not current.hash.startswith("0" * current.difficulty):
            return False
    
    return True
```

### Verify Model Hash

```python
import hashlib
import json

def verify_model_hash(model_weights, stored_hash):
    weights_str = json.dumps(
        {k: v.tolist() for k, v in model_weights.items()},
        sort_keys=True
    )
    computed_hash = hashlib.sha256(weights_str.encode()).hexdigest()
    return computed_hash == stored_hash
```

## Summary

### Key Storage Files
1. **blockchain.json** - Complete blockchain data (machine-readable)
2. **blockchain_detailed.log** - Detailed human-readable log
3. **blockchain_summary.txt** - Quick overview
4. **blockchain_details.json** - Programmatic export

### What's Stored in Each Block
- Block metadata (index, timestamp, nonce, difficulty)
- Cryptographic hashes (previous, current, model)
- Transaction data (FL round info, accuracy, samples)
- Mining information (miner ID, consensus)
- Digital signatures (public key, signature)
- Rewards (base, bonus, total)

### How to Access
- **Python API:** Use `EHRChain` class
- **JSON Files:** Direct file access
- **System API:** Use `FederatedBlockchainSystem.get_blockchain_details()`

### Integration with Federated Learning
- Each FL round creates new blocks
- Model updates are hashed and stored
- Client contributions are tracked
- Incentives are calculated and distributed
- Complete audit trail maintained
