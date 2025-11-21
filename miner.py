import hashlib
import time
from typing import Dict, List

from block import Block
from ehr_chain import EHRChain

class Miner:
    def __init__(self, miner_id, chain, difficulty):
        self.miner_id = miner_id
        self.chain = chain
        self.difficulty = difficulty
        self.private_key = f"privkey-{miner_id}"
        self.public_key = f"pubkey-{miner_id}"

    def sign_block(self, block_hash: str):
        return hashlib.sha256((self.private_key + block_hash).encode()).hexdigest()

    def mine_block(self, records: List[Dict], model_hash: str):
        # We replace PoW with a simple, quick block signing for this PoA-like model
        last_block = self.chain.chain[-1]
        new_index = last_block.index + 1
        prev_hash = last_block.hash

        print(f"\n{'='*70}")
        print(f"⛏️  MINING BLOCK #{new_index}")
        print(f"{'='*70}")
        print(f"Miner: {self.miner_id}")
        print(f"Previous Hash: {prev_hash[:32]}...")
        
        # Simulate PoW mining with nonce
        import random
        nonce = random.randint(1000, 9999)
        
        new_block = Block(
            index=new_index,
            prev_hash=prev_hash,
            records=records,
            access_logs=[],
            miner=self.miner_id,
            model_hash=model_hash,
            difficulty=self.difficulty,
            nonce=nonce
        )
        
        # Sign the block
        new_block.signature = self.sign_block(new_block.hash)
        new_block.public_key = self.public_key
        
        # Calculate mining reward
        base_reward = 10.0
        difficulty_bonus = self.difficulty * 2.0
        total_reward = base_reward + difficulty_bonus
        
        # Display detailed block information
        print(f"\n📦 BLOCK HEADER:")
        print(f"  Block Index: {new_block.index}")
        print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(new_block.timestamp))}")
        print(f"  Nonce: {new_block.nonce}")
        print(f"  Difficulty: {new_block.difficulty}")
        
        print(f"\n🔗 BLOCK HASHES:")
        print(f"  Previous Hash: {new_block.prev_hash[:32]}...")
        print(f"  Current Hash:  {new_block.hash[:32]}...")
        print(f"  Model Hash:    {new_block.model_hash[:32]}...")
        
        print(f"\n🔐 DIGITAL SIGNATURE:")
        print(f"  Public Key:  {new_block.public_key}")
        print(f"  Signature:   {new_block.signature[:32]}...")
        
        print(f"\n📊 BLOCK DATA:")
        print(f"  Transactions: {len(records)}")
        print(f"  Miner: {self.miner_id}")
        
        print(f"\n💰 MINING REWARD:")
        print(f"  Base Reward: {base_reward} tokens")
        print(f"  Difficulty Bonus: {difficulty_bonus} tokens")
        print(f"  Total Reward: {total_reward} tokens")
        
        print(f"\n✅ WINNER: {self.miner_id}")
        print(f"{'='*70}")
        
        return new_block