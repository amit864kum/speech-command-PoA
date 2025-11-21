# Quick Reference Guide

## Where Are Blockchain Details Stored?

### Primary Storage Files

| File | Location | Purpose | Format |
|------|----------|---------|--------|
| **blockchain.json** | `Speech_command/blockchain.json` | Complete blockchain data | JSON |
| **blockchain_detailed.log** | `Speech_command/blockchain_detailed.log` | Human-readable log | Text |
| **blockchain_summary.txt** | `Speech_command/blockchain_summary.txt` | Quick overview | Text |
| **blockchain_details.json** | `Speech_command/blockchain_details.json` | Programmatic export | JSON |

### What's Stored in Each Block?

```
Block Structure:
├── Block Header
│   ├── Index (block number)
│   ├── Timestamp
│   ├── Nonce (proof of work)
│   └── Difficulty
├── Hashes
│   ├── Previous Hash (links to previous block)
│   ├── Current Hash (this block's hash)
│   └── Model Hash (ML model hash)
├── Digital Signature
│   ├── Public Key
│   ├── Signature
│   └── Miner ID
├── Transaction Data
│   ├── FL Round Number
│   ├── Client ID
│   ├── Model Accuracy
│   ├── Number of Samples
│   └── Predictions
├── Mining Info
│   ├── Miner ID
│   ├── Consensus (PoA)
│   ├── Difficulty
│   └── Nonce
└── Rewards
    ├── Base Reward
    ├── Difficulty Bonus
    └── Total Reward
```

## How to Run Federated Learning

### Option 1: Using Existing Main File
```bash
python federated_learning_main.py
```

### Option 2: Using New Integration File
```bash
python federated_blockchain_integration.py
```

### Option 3: Using Demo
```bash
python demo_complete_system.py
```

## How to Access Blockchain Data

### Python API

```python
# Load blockchain
from ehr_chain import EHRChain
blockchain = EHRChain.load_from_file("blockchain.json")

# Access all blocks
for block in blockchain.chain:
    print(f"Block #{block.index}: {block.hash}")

# Get specific block
latest_block = blockchain.chain[-1]
genesis_block = blockchain.chain[0]

# Access block details
print(f"Miner: {latest_block.miner}")
print(f"Model Hash: {latest_block.model_hash}")
print(f"Transactions: {latest_block.records}")
```

### Using System API

```python
from federated_blockchain_integration import FederatedBlockchainSystem

# Initialize
fl_system = FederatedBlockchainSystem(
    num_clients=3,
    enable_blockchain=True
)

# Get details
details = fl_system.get_blockchain_details()

# Access blocks
for block in details["blocks"]:
    print(f"Block {block['index']}: {block['current_hash']}")
```

### Direct JSON Access

```python
import json

# Read blockchain.json
with open("blockchain.json", "r") as f:
    data = json.load(f)

# Access blockchain info
info = data["blockchain_info"]
print(f"Total Blocks: {info['total_blocks']}")

# Access blocks
for block in data["chain"]:
    print(f"Block #{block['index']}")
    print(f"  Hash: {block['hash']}")
    print(f"  Miner: {block['miner']}")
```

## Key Components

### 1. Block Class (`block.py`)
- Defines block structure
- Computes block hash
- Stores all block data

### 2. EHRChain Class (`ehr_chain.py`)
- Manages blockchain
- Validates blocks
- Saves/loads blockchain
- Creates detailed logs

### 3. DecentralizedClient (`client.py`)
- Performs local training
- Uploads to IPFS (optional)
- Mines blocks
- Stores updates on blockchain

### 4. FederatedBlockchainSystem (`federated_blockchain_integration.py`)
- Orchestrates FL rounds
- Aggregates models
- Manages blockchain
- Distributes incentives

## Federated Learning Flow

```
1. Initialize System
   ↓
2. Create Clients (with local data)
   ↓
3. For each FL Round:
   ├── Broadcast global model to clients
   ├── Clients train locally
   ├── Aggregate client models (FedAvg)
   ├── Update global model
   ├── Compute model hash
   ├── Create transactions
   ├── Mine blocks
   └── Store on blockchain
   ↓
4. Save blockchain to files
   ↓
5. Generate logs and summaries
```

## Configuration Options

### Basic Configuration
```python
NUM_CLIENTS = 3          # Number of FL clients
NUM_ROUNDS = 5           # FL rounds
LOCAL_EPOCHS = 5         # Local training epochs
BATCH_SIZE = 32          # Training batch size
LEARNING_RATE = 1e-3     # Learning rate
```

### Feature Flags
```python
enable_blockchain = True    # Store on blockchain
enable_ipfs = False        # Use IPFS storage
enable_incentives = False  # Token rewards
enable_privacy = False     # Differential privacy
```

### Byzantine Attacks (Testing)
```python
ENABLE_BYZANTINE = True    # Enable attacks
NUM_BYZANTINE = 1          # Number of malicious clients
attack_type = "random"     # Attack type
attack_strength = 2.0      # Attack strength
```

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

# Use it
from ehr_chain import EHRChain
blockchain = EHRChain.load_from_file("blockchain.json")
is_valid = verify_blockchain(blockchain)
print(f"Blockchain valid: {is_valid}")
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

## Common Tasks

### Task 1: Run FL and Store on Blockchain
```bash
python federated_blockchain_integration.py
```

### Task 2: View Blockchain Details
```bash
# View summary
cat blockchain_summary.txt

# View detailed log
cat blockchain_detailed.log

# View JSON
cat blockchain.json
```

### Task 3: Load and Inspect Blockchain
```python
from ehr_chain import EHRChain

blockchain = EHRChain.load_from_file("blockchain.json")
print(f"Total blocks: {len(blockchain.chain)}")

for block in blockchain.chain:
    print(f"\nBlock #{block.index}")
    print(f"  Miner: {block.miner}")
    print(f"  Hash: {block.hash[:32]}...")
    print(f"  Model Hash: {block.model_hash[:32]}...")
    print(f"  Transactions: {len(block.records)}")
```

### Task 4: Extract FL Metrics
```python
import json

with open("blockchain.json", "r") as f:
    data = json.load(f)

accuracies = []
for block in data["chain"]:
    for tx in block.get("records", []):
        if isinstance(tx, dict) and "accuracy" in tx:
            accuracies.append(tx["accuracy"])

print(f"Average accuracy: {sum(accuracies)/len(accuracies)*100:.2f}%")
```

## File Locations Summary

```
Speech_command/
├── block.py                              # Block class definition
├── ehr_chain.py                          # Blockchain management
├── client.py                             # FL client with blockchain
├── federated_learning_main.py            # Original FL main
├── federated_blockchain_integration.py   # New integrated system
├── demo_complete_system.py               # Demonstration script
├── blockchain.json                       # ← BLOCKCHAIN DATA HERE
├── blockchain_detailed.log               # ← DETAILED LOG HERE
├── blockchain_summary.txt                # ← SUMMARY HERE
├── blockchain_details.json               # ← PROGRAMMATIC EXPORT HERE
├── BLOCKCHAIN_DETAILS_GUIDE.md           # Complete guide
└── QUICK_REFERENCE.md                    # This file
```

## Need Help?

1. **Read the detailed guide:** `BLOCKCHAIN_DETAILS_GUIDE.md`
2. **Run the demo:** `python demo_complete_system.py`
3. **Check existing docs:**
   - `BLOCKCHAIN_STORAGE.md`
   - `BLOCKCHAIN_OUTPUT_GUIDE.md`
   - `STORAGE_LOCATIONS.md`

## Quick Commands

```bash
# Run federated learning
python federated_blockchain_integration.py

# Run demo
python demo_complete_system.py

# View blockchain summary
cat blockchain_summary.txt

# View detailed log
cat blockchain_detailed.log

# Check blockchain file
cat blockchain.json | python -m json.tool | head -50
```
