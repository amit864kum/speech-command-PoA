# Quick Start Guide - Phase 3 Features

## 🚀 5-Minute Quick Start

### 1. Verify Installation (30 seconds)

```bash
cd Speech_command
python scripts/test_phase3.py
```

**Expected Output:**
```
ALL PHASE 3 TESTS PASSED! ✓
  - Differential Privacy
  - Privacy Accounting
  - Byzantine Attacks
  - Robust Aggregation
```

### 2. Run Interactive Demo (2 minutes)

```bash
python demo_enhanced_features.py
```

**What You'll See:**
- Demo 1: Differential Privacy in action
- Demo 2: Byzantine attack simulation
- Demo 3: Robust aggregation methods
- Demo 4: Privacy budget accounting

### 3. Try Differential Privacy (1 minute)

**Edit `ehr_main.py`:**
```python
ENABLE_DP = True  # Change from False to True
```

**Run:**
```bash
python ehr_main.py
```

**Look for:**
```
[🔒] Privacy: ε=0.70, δ=1.00e-05
```

### 4. Try Byzantine Attacks (1 minute)

**Edit `ehr_main.py`:**
```python
ENABLE_BYZANTINE = True  # Change from False to True
NUM_BYZANTINE = 1
```

**Run:**
```bash
python ehr_main.py
```

**Look for:**
```
[⚠️] Device_A configured as Byzantine attacker
[⚠️] Byzantine: Applied random attack
```

### 5. Use Enhanced Features (30 seconds)

```bash
# Run with new modules
python src/main.py --config configs/default_config.yaml
```

## 📖 Common Use Cases

### Use Case 1: Privacy-Preserving Training

```python
from client import DecentralizedClient

client = DecentralizedClient(
    client_id="PrivateClient",
    miner_id="Miner1",
    client_data=data,
    input_dim=64,
    output_dim=10,
    target_words=words,
    enable_dp=True,           # Enable DP
    noise_multiplier=1.1,     # Privacy noise
    max_grad_norm=1.0,        # Gradient clipping
    target_epsilon=10.0       # Privacy budget
)

# Train with privacy
weights, acc, model_id, preds = client.local_train(
    epochs=5,
    batch_size=32,
    lr=0.001
)

# Check privacy
if client.privacy_engine:
    epsilon, delta = client.privacy_engine.get_privacy_spent()
    print(f"Privacy: ε={epsilon:.2f}, δ={delta:.2e}")
```

### Use Case 2: Simulate Byzantine Attack

```python
from client import DecentralizedClient

attacker = DecentralizedClient(
    client_id="Attacker",
    miner_id="Miner2",
    client_data=data,
    input_dim=64,
    output_dim=10,
    target_words=words,
    is_byzantine=True,        # Byzantine flag
    attack_type="random",     # Attack type
    attack_strength=2.0       # Attack strength
)

# Attack during training
weights, acc, model_id, preds = attacker.local_train(
    epochs=5,
    batch_size=32,
    lr=0.001
)
# Model weights are now corrupted!
```

### Use Case 3: Robust Aggregation

```python
from src.federated.aggregator import create_aggregator

# Create robust aggregator
aggregator = create_aggregator("krum", num_byzantine=2)

# Or trimmed mean
aggregator = create_aggregator("trimmed_mean", trim_ratio=0.2)

# Aggregate (automatically handles Byzantine clients)
global_weights = aggregator.aggregate(client_weights, client_sizes)
```

### Use Case 4: Privacy Accounting

```python
from src.privacy import PrivacyAccountant

accountant = PrivacyAccountant(
    noise_multiplier=1.1,
    sample_rate=0.1,
    target_delta=1e-5
)

# Track training
for step in range(100):
    accountant.step()
    epsilon, delta = accountant.get_privacy_spent()
    
    if epsilon > 10.0:
        print("Privacy budget exceeded!")
        break
```

## 🎯 Configuration Quick Reference

### `ehr_main.py` Settings

```python
# Privacy
ENABLE_DP = True/False
# If True: Adds differential privacy to all clients

# Byzantine Attacks
ENABLE_BYZANTINE = True/False
NUM_BYZANTINE = 1  # Number of attackers
# If True: First NUM_BYZANTINE clients become attackers
```

### `configs/default_config.yaml` Settings

```yaml
# Privacy
privacy:
  enable_differential_privacy: true
  epsilon: 10.0              # Privacy budget
  noise_multiplier: 1.1      # DP noise
  max_grad_norm: 1.0         # Gradient clipping

# Adversarial
adversarial:
  enable_attacks: true
  num_byzantine: 2           # Number of attackers
  attack_type: "random"      # Attack type
  attack_strength: 2.0       # Attack strength
```

## 🔧 Attack Types Reference

| Attack Type | Description | Strength Range |
|------------|-------------|----------------|
| `random` | Random noise | 0.5 - 5.0 |
| `sign_flipping` | Flip update signs | 1.0 - 3.0 |
| `label_flipping` | Corrupt labels | 0.1 - 1.0 |
| `gaussian` | Gaussian noise | 0.5 - 5.0 |
| `scaling` | Scale updates | 2.0 - 10.0 |

## 📊 Privacy Budget Guide

| Epsilon (ε) | Privacy Level | Use Case |
|------------|---------------|----------|
| 0.1 - 1.0 | Very Strong | Highly sensitive data |
| 1.0 - 5.0 | Strong | Medical/financial data |
| 5.0 - 10.0 | Moderate | General applications |
| > 10.0 | Weak | Less sensitive data |

## ⚡ Performance Tips

### For Faster Training:
```python
# Reduce local epochs
local_epochs = 2  # Instead of 5

# Increase batch size
batch_size = 64  # Instead of 32

# Disable DP for testing
enable_dp = False
```

### For Better Privacy:
```python
# Lower noise multiplier
noise_multiplier = 0.8  # Instead of 1.1

# Stricter gradient clipping
max_grad_norm = 0.5  # Instead of 1.0

# Lower target epsilon
target_epsilon = 5.0  # Instead of 10.0
```

### For Better Robustness:
```python
# Use Krum aggregation
aggregator = create_aggregator("krum", num_byzantine=2)

# Or trimmed mean
aggregator = create_aggregator("trimmed_mean", trim_ratio=0.3)

# Increase detection threshold
threshold = 2.0  # For Byzantine detection
```

## 🐛 Common Issues

### Issue 1: Import Errors
```
ImportError: cannot import name 'PrivacyEngine'
```
**Solution:** Run from Speech_command directory
```bash
cd Speech_command
python ehr_main.py
```

### Issue 2: Privacy Not Working
```
[INFO] Privacy modules not available
```
**Solution:** Check if src/privacy/ exists
```bash
ls src/privacy/
# Should show: __init__.py, differential_privacy.py, etc.
```

### Issue 3: Tests Failing
```
✗ Test FAILED
```
**Solution:** Run quick test first
```bash
python scripts/quick_test.py
# If this passes, Phase 3 should work
```

## 📚 Learn More

- **Full Documentation:** `PHASE3_COMPLETE.md`
- **All Changes:** `CHANGES_PHASE3.md`
- **Test Suite:** `scripts/test_phase3.py`
- **Demo Script:** `demo_enhanced_features.py`

## ✅ Checklist

Before publishing, verify:

- [ ] `python scripts/test_phase3.py` passes
- [ ] `python demo_enhanced_features.py` runs
- [ ] `python ehr_main.py` works with ENABLE_DP=True
- [ ] `python ehr_main.py` works with ENABLE_BYZANTINE=True
- [ ] All documentation files present
- [ ] Configuration files updated

## 🎉 You're Ready!

Your federated learning system now has:
- ✅ Differential Privacy
- ✅ Byzantine Attack Simulation
- ✅ Robust Aggregation
- ✅ Privacy Accounting
- ✅ Comprehensive Testing

**Start experimenting and good luck with your publication!** 🚀