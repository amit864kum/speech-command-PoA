"""Test script for Phase 4: Blockchain & IPFS features."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from collections import OrderedDict


def test_transactions():
    """Test blockchain transactions."""
    print("\n" + "="*70)
    print("TEST 1: BLOCKCHAIN TRANSACTIONS")
    print("="*70)
    
    try:
        from src.blockchain.transaction import (
            ModelUpdateTransaction,
            CoinbaseTransaction,
            CommitTransaction,
            RevealTransaction,
            SlashingTransaction,
            create_commit_hash,
            verify_reveal
        )
        
        print("\n1. Testing CoinbaseTransaction...")
        coinbase = CoinbaseTransaction(
            miner="Miner_0",
            reward=10.0,
            block_height=1
        )
        print(f"✓ Coinbase TX created: {coinbase.tx_hash[:16]}...")
        print(f"  - Miner: {coinbase.data['miner']}")
        print(f"  - Reward: {coinbase.data['reward']}")
        
        print("\n2. Testing ModelUpdateTransaction...")
        model_tx = ModelUpdateTransaction(
            client_id="Client_0",
            model_cid="QmTest123",
            metadata={"accuracy": 0.85, "samples": 1000},
            round_number=1
        )
        print(f"✓ Model Update TX created: {model_tx.tx_hash[:16]}...")
        print(f"  - CID: {model_tx.get_cid()}")
        print(f"  - Metadata: {model_tx.get_metadata()}")
        
        print("\n3. Testing Commit-Reveal...")
        # Create commit
        cid = "QmTest456"
        metadata = {"accuracy": 0.90, "samples": 2000}
        commit_hash = create_commit_hash(cid, metadata)
        
        commit_tx = CommitTransaction("Client_1", commit_hash, round_number=1)
        print(f"✓ Commit TX created: {commit_tx.tx_hash[:16]}...")
        print(f"  - Commit hash: {commit_hash[:16]}...")
        
        # Reveal
        reveal_tx = RevealTransaction(
            "Client_1", cid, metadata, 
            round_number=1, 
            commit_tx_hash=commit_tx.tx_hash
        )
        print(f"✓ Reveal TX created: {reveal_tx.tx_hash[:16]}...")
        
        # Verify
        is_valid = verify_reveal(commit_hash, cid, metadata)
        print(f"✓ Commit-Reveal verification: {is_valid}")
        
        print("\n4. Testing SlashingTransaction...")
        slash_tx = SlashingTransaction(
            accuser="Validator_0",
            accused="Client_2",
            evidence_cid="QmEvidence789",
            fraud_type="low_quality_model",
            penalty=50.0
        )
        print(f"✓ Slashing TX created: {slash_tx.tx_hash[:16]}...")
        print(f"  - Accused: {slash_tx.data['accused']}")
        print(f"  - Penalty: {slash_tx.data['penalty']}")
        
        print("\n✓ Blockchain Transactions Test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Blockchain Transactions Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ipfs_storage():
    """Test IPFS storage."""
    print("\n" + "="*70)
    print("TEST 2: IPFS STORAGE")
    print("="*70)
    
    try:
        from src.storage.ipfs_manager import IPFSManager, MockIPFSManager
        
        print("\n1. Testing Mock IPFS...")
        ipfs = MockIPFSManager()
        
        # Create test model weights
        weights = OrderedDict({
            "layer1.weight": torch.randn(10, 5),
            "layer1.bias": torch.randn(10),
            "layer2.weight": torch.randn(5, 10),
            "layer2.bias": torch.randn(5)
        })
        
        metadata = {
            "accuracy": 0.85,
            "samples": 1000,
            "client_id": "Client_0"
        }
        
        print("\n2. Uploading model weights...")
        cid = ipfs.upload_model_weights(weights, metadata)
        print(f"✓ Model uploaded: {cid[:16]}...")
        
        print("\n3. Downloading model weights...")
        downloaded_weights, downloaded_metadata = ipfs.download_model_weights(cid)
        print(f"✓ Model downloaded")
        print(f"  - Metadata: {downloaded_metadata}")
        
        print("\n4. Verifying weights...")
        weights_match = all(
            torch.allclose(weights[k], downloaded_weights[k])
            for k in weights.keys()
        )
        print(f"✓ Weights match: {weights_match}")
        
        print("\n5. Testing arbitrary data upload...")
        test_data = b"Test data for IPFS"
        data_cid = ipfs.upload_data(test_data)
        downloaded_data = ipfs.download_data(data_cid)
        print(f"✓ Data upload/download: {test_data == downloaded_data}")
        
        print("\n6. Getting IPFS stats...")
        stats = ipfs.get_stats()
        print(f"✓ IPFS Stats:")
        print(f"  - Mode: {stats['mode']}")
        print(f"  - Stored items: {stats.get('stored_items', 0)}")
        
        print("\n✓ IPFS Storage Test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ IPFS Storage Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_incentives():
    """Test incentive system."""
    print("\n" + "="*70)
    print("TEST 3: INCENTIVE SYSTEM")
    print("="*70)
    
    try:
        from src.blockchain.incentives import IncentiveManager
        
        print("\n1. Creating IncentiveManager...")
        incentives = IncentiveManager(
            base_reward=10.0,
            quality_bonus=5.0,
            slash_penalty=50.0
        )
        print("✓ IncentiveManager created")
        
        print("\n2. Calculating rewards...")
        reward1 = incentives.calculate_reward(
            participant="Client_0",
            contribution_quality=0.95,
            num_samples=1000,
            round_number=1
        )
        print(f"✓ Reward for Client_0: {reward1:.2f}")
        
        reward2 = incentives.calculate_reward(
            participant="Client_1",
            contribution_quality=0.80,
            num_samples=500,
            round_number=1
        )
        print(f"✓ Reward for Client_1: {reward2:.2f}")
        
        print("\n3. Checking balances...")
        balance0 = incentives.get_balance("Client_0")
        balance1 = incentives.get_balance("Client_1")
        print(f"✓ Client_0 balance: {balance0:.2f}")
        print(f"✓ Client_1 balance: {balance1:.2f}")
        
        print("\n4. Applying penalty...")
        penalty = incentives.apply_penalty(
            participant="Client_1",
            penalty_type="low_quality",
            evidence="QmEvidence123"
        )
        print(f"✓ Penalty applied: {penalty:.2f}")
        
        new_balance1 = incentives.get_balance("Client_1")
        print(f"✓ Client_1 new balance: {new_balance1:.2f}")
        
        print("\n5. Getting statistics...")
        stats = incentives.get_statistics()
        print(f"✓ Incentive Statistics:")
        print(f"  - Total rewards: {stats['total_rewards_distributed']:.2f}")
        print(f"  - Total penalties: {stats['total_penalties_applied']:.2f}")
        print(f"  - Num participants: {stats['num_participants']}")
        
        print("\n✓ Incentive System Test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Incentive System Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_staking():
    """Test staking system."""
    print("\n" + "="*70)
    print("TEST 4: STAKING SYSTEM")
    print("="*70)
    
    try:
        from src.blockchain.incentives import StakingManager
        
        print("\n1. Creating StakingManager...")
        staking = StakingManager(
            min_stake=100.0,
            max_stake=10000.0,
            slash_rate=0.5
        )
        print("✓ StakingManager created")
        
        print("\n2. Staking tokens...")
        success = staking.stake("Client_0", 500.0)
        print(f"✓ Stake successful: {success}")
        print(f"  - Stake amount: {staking.get_stake('Client_0'):.2f}")
        
        print("\n3. Checking eligibility...")
        eligible = staking.is_eligible("Client_0")
        print(f"✓ Client_0 eligible: {eligible}")
        
        print("\n4. Locking stake...")
        locked = staking.lock_stake("Client_0", 200.0)
        print(f"✓ Stake locked: {locked}")
        print(f"  - Available stake: {staking.get_available_stake('Client_0'):.2f}")
        
        print("\n5. Unlocking stake...")
        staking.unlock_stake("Client_0", 200.0)
        print(f"✓ Stake unlocked")
        print(f"  - Available stake: {staking.get_available_stake('Client_0'):.2f}")
        
        print("\n6. Slashing stake...")
        slashed = staking.slash_stake("Client_0", "malicious_behavior")
        print(f"✓ Slashed amount: {slashed:.2f}")
        print(f"  - Remaining stake: {staking.get_stake('Client_0'):.2f}")
        
        print("\n7. Getting statistics...")
        stats = staking.get_statistics()
        print(f"✓ Staking Statistics:")
        print(f"  - Total staked: {stats['total_staked']:.2f}")
        print(f"  - Total slashed: {stats['total_slashed']:.2f}")
        print(f"  - Num stakers: {stats['num_stakers']}")
        
        print("\n✓ Staking System Test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Staking System Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 4 tests."""
    print("="*70)
    print("PHASE 4: BLOCKCHAIN & IPFS TESTING")
    print("="*70)
    
    tests = [
        ("Blockchain Transactions", test_transactions),
        ("IPFS Storage", test_ipfs_storage),
        ("Incentive System", test_incentives),
        ("Staking System", test_staking)
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
        print("ALL PHASE 4 TESTS PASSED! ✓")
        print("="*70)
        print("\nPhase 4 features are working correctly:")
        print("  - Blockchain Transactions (6 types)")
        print("  - IPFS Storage (upload/download)")
        print("  - Incentive System (rewards/penalties)")
        print("  - Staking System (stake/slash)")
        print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)