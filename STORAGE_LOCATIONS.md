# Blockchain Storage Locations

## 📁 Where All Block Details Are Stored

### 1. Main Blockchain File
**Location**: `blockchain.json`

**Contains:**
- Complete blockchain with all blocks
- Block headers (index, timestamp, nonce, difficulty)
- All hashes (previous, current, model)
- Digital signatures (public key, signature)
- Mining information (miner ID, consensus)
- Rewards (base, bonus, total)
- Transactions and records
- Blockchain metadata

**Structure:**
```json
{
  "blockchain_info": {
    "version": "2.0",
    "consensus": "Proof of Authority (PoA)",
    "total_blocks": 16,
    "total_rewards": 192.0,
    "last_updated": "2024-11-08 15:30:45"
  },
  "chain": [
    {
      "block_header": {
        "index": 0,
        "timestamp": 1699456845.123,
        "timestamp_readable": "2024-11-08 15:30:45",
        "nonce": 0,
        "difficulty": 2
      },
      "hashes": {
        "previous_hash": "0",
        "current_hash": "a3f5b2c8...",
        "model_hash": "b4c5d8e9..."
      },
      "signature": {
        "public_key": "pubkey-Miner_1",
        "signature": "c8d9e1f4...",
        "miner": "Miner_1"
      },
      "rewards": {
        "base_reward": 10.0,
        "difficulty_bonus": 4.0,
        "total_reward": 14.0
      }
    }
  ]
}
```

### 2. Detailed Log File
**Location**: `blockchain_detailed.log`

**Contains:**
- Human-readable format
- Complete block information
- All headers, hashes, signatures
- Mining details and rewards
- Transaction details

**Format:**
```
================================================================================
BLOCKCHAIN DETAILED LOG
================================================================================
Generated: 2024-11-08 15:30:45
Total Blocks: 16
================================================================================

================================================================================
BLOCK #0
================================================================================

BLOCK HEADER:
  Index: 0
  Timestamp: 2024-11-08 15:30:45
  Nonce: 0
  Difficulty: 2

HASHES:
  Previous Hash: 0
  Current Hash:  a3f5b2c8d9e1f4a7b6c5d8e9f2a3b4c5...
  Model Hash:    b4c5d8e9f2a3b4c5a3f5b2c8d9e1f4a7...

DIGITAL SIGNATURE:
  Miner: Miner_1
  Public Key: pubkey-Miner_1
  Signature: c8d9e1f4a7b6c5d8e9f2a3b4c5a3f5b2...

MINING INFO:
  Miner ID: Miner_1
  Consensus: PoA
  Difficulty: 2

REWARDS:
  Base Reward: 10.0 tokens
  Difficulty Bonus: 4.0 tokens
  Total Reward: 14.0 tokens

TRANSACTIONS:
  Count: 1
  Transaction 1: {...}
```

### 3. Summary File
**Location**: `blockchain_summary.txt`

**Contains:**
- Quick overview
- Block list with key info
- Total statistics

**Format:**
```
================================================================================
BLOCKCHAIN SUMMARY
================================================================================

Version: 2.0
Consensus: Proof of Authority (PoA)
Total Blocks: 16
Total Rewards: 192.0 tokens
Last Updated: 2024-11-08 15:30:45

BLOCK LIST:
--------------------------------------------------------------------------------
Block    Miner           Timestamp            Hash                 Reward    
--------------------------------------------------------------------------------
#0       Genesis         2024-11-08 15:30:45  a3f5b2c8d9e1f4a7...  14.0      
#1       Miner_1         2024-11-08 15:31:12  b4c5d8e9f2a3b4c5...  12.0      
#2       Miner_2         2024-11-08 15:31:45  c5d8e9f2a3b4c5a3...  12.0      
...
```

### 4. Individual Client Chains
**Location**: `ehr_chain.json` (per client)

**Contains:**
- Each client's local blockchain copy
- Same structure as main blockchain
- Client-specific view

### 5. Logs Directory
**Location**: `Speech_command/logs/`

**Contains:**
- Experiment logs with timestamps
- Training metrics
- Privacy budgets
- Error logs

### 6. Results Directory
**Location**: `Speech_command/results/`

**Contains:**
- Saved models (`.pt` files)
- Metrics (`.json` files)
- Experimental results

## 📊 What Information Is Stored

### Block Header
- ✅ Block Index (sequential number)
- ✅ Timestamp (Unix + readable)
- ✅ Nonce (proof-of-work value)
- ✅ Difficulty (mining difficulty)

### Hashes
- ✅ Previous Block Hash (chain linkage)
- ✅ Current Block Hash (unique ID)
- ✅ Global Model Hash (model identifier)

### Digital Signature
- ✅ Public Key (miner's public key)
- ✅ Signature (cryptographic signature)
- ✅ Miner ID (who signed)

### Mining Information
- ✅ Miner ID (who mined the block)
- ✅ Consensus Mechanism (PoA/PoW)
- ✅ Difficulty Level
- ✅ Nonce Value

### Rewards
- ✅ Base Reward (10 tokens)
- ✅ Difficulty Bonus (difficulty × 2)
- ✅ Total Reward (base + bonus)

### Transactions
- ✅ Transaction Count
- ✅ Transaction Data
- ✅ Client Information
- ✅ Model Predictions

### Metadata
- ✅ Blockchain Version
- ✅ Total Blocks
- ✅ Total Rewards
- ✅ Last Update Time

## 🔍 How to Access Stored Data

### View Main Blockchain
```bash
# View JSON file
cat blockchain.json

# Or use Python
python -c "import json; print(json.dumps(json.load(open('blockchain.json')), indent=2))"
```

### View Detailed Log
```bash
# View log file
cat blockchain_detailed.log

# Or
type blockchain_detailed.log  # Windows
```

### View Summary
```bash
cat blockchain_summary.txt
```

### Programmatic Access
```python
import json

# Load blockchain
with open('blockchain.json', 'r') as f:
    blockchain = json.load(f)

# Access specific block
block_5 = blockchain['chain'][5]
print(f"Block #5 Hash: {block_5['hashes']['current_hash']}")
print(f"Miner: {block_5['signature']['miner']}")
print(f"Reward: {block_5['rewards']['total_reward']} tokens")

# Get blockchain info
info = blockchain['blockchain_info']
print(f"Total Blocks: {info['total_blocks']}")
print(f"Total Rewards: {info['total_rewards']} tokens")
```

## 📈 Storage Format Comparison

| File | Format | Size | Purpose |
|------|--------|------|---------|
| blockchain.json | JSON | Large | Complete data, machine-readable |
| blockchain_detailed.log | Text | Medium | Human-readable, detailed |
| blockchain_summary.txt | Text | Small | Quick overview |
| ehr_chain.json | JSON | Large | Client-specific chain |

## 🎯 What Gets Saved When

### After Each Epoch
- ✅ blockchain.json (updated)
- ✅ blockchain_detailed.log (updated)
- ✅ blockchain_summary.txt (updated)
- ✅ ehr_chain.json (per client)

### After Complete Run
- ✅ All files contain complete history
- ✅ All blocks with full details
- ✅ All rewards calculated
- ✅ All signatures verified

## 🔐 Data Integrity

### Verification
- ✅ Hash chain ensures immutability
- ✅ Digital signatures prove authenticity
- ✅ Timestamps provide ordering
- ✅ Nonces prove computational work

### Audit Trail
- ✅ Complete history of all blocks
- ✅ All mining activities recorded
- ✅ All rewards tracked
- ✅ All transactions logged

## 📊 Example: Accessing Block Details

```python
import json

# Load blockchain
with open('blockchain.json', 'r') as f:
    data = json.load(f)

# Print all block hashes
for block in data['chain']:
    print(f"Block #{block['index']}: {block['hashes']['current_hash'][:16]}...")

# Calculate total rewards
total = sum(b['rewards']['total_reward'] for b in data['chain'])
print(f"Total Rewards: {total} tokens")

# Find blocks by miner
miner_blocks = [b for b in data['chain'] if b['miner'] == 'Miner_1']
print(f"Miner_1 mined {len(miner_blocks)} blocks")
```

## ✅ Summary

All block details are stored in **3 different formats**:

1. **blockchain.json** - Complete, structured, machine-readable
2. **blockchain_detailed.log** - Detailed, human-readable
3. **blockchain_summary.txt** - Quick overview

**Every block contains:**
- Block header (index, timestamp, nonce, difficulty)
- All hashes (previous, current, model)
- Digital signature (public key, signature, miner)
- Mining info (miner ID, consensus, difficulty)
- Rewards (base, bonus, total)
- Transactions (count, data)

**Files are updated after each epoch and saved in the Speech_command root directory!**