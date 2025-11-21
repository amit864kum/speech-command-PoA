# ehr_chain.py
import hashlib
import json
import time
from typing import List, Dict

from block import Block

class EHRChain:
    def __init__(self):
        self.chain = []
        self.difficulty = 2

    def create_genesis_block(self):
        genesis_block = Block(
            index=0,
            prev_hash="0",
            records=[],
            access_logs=[],
            miner="Genesis",
            model_hash="0",
            difficulty=self.difficulty
        )
        self.chain.append(genesis_block)
    
    @classmethod
    def load_from_file(cls, filepath="blockchain.json"):
        """Loads a blockchain from a JSON file."""
        ehr_chain = cls()
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                if 'chain' in data:
                    ehr_chain.chain.clear()
                    for block_data in data['chain']:
                        block = Block(
                            index=block_data['index'],
                            prev_hash=block_data['prev_hash'],
                            records=block_data['records'],
                            access_logs=block_data.get('access_logs', []),
                            miner=block_data['miner'],
                            model_hash=block_data['model_hash'],
                            difficulty=block_data.get('difficulty', ehr_chain.difficulty),
                            # Corrected: Read the 'proof' key from the file and pass it as the 'nonce'
                            nonce=block_data.get('proof', None) 
                        )
                        block.hash = block_data['hash']
                        ehr_chain.chain.append(block)
                print(f"[📂] Loaded blockchain with {len(ehr_chain.chain)} blocks from {filepath}.")
                return ehr_chain
        except FileNotFoundError:
            print(f"[❌] {filepath} not found. Creating a new blockchain.")
            ehr_chain.create_genesis_block()
            return ehr_chain

    def add_block(self, block):
        if not self.is_valid_block(block):
            return False
        
        self.chain.append(block)
        return True

    def is_valid_block(self, block):
        last_block = self.chain[-1]
        if block.prev_hash != last_block.hash:
            print("Block rejected: Previous hash is invalid.")
            return False
        
        if not block.hash.startswith("0" * self.difficulty):
            print("Block rejected: Proof of Work is invalid.")
            return False
        
        return True

    def save_to_file(self, filepath="blockchain.json"):
        """Saves the blockchain to a JSON file with complete details."""
        import time as time_module
        
        chain_list = []
        for block in self.chain:
            # Convert timestamp to readable format
            readable_timestamp = time_module.strftime(
                '%Y-%m-%d %H:%M:%S', 
                time_module.localtime(block.timestamp)
            )
            
            block_dict = {
                # Block Header
                "block_header": {
                    "index": block.index,
                    "timestamp": block.timestamp,
                    "timestamp_readable": readable_timestamp,
                    "nonce": block.nonce,
                    "difficulty": block.difficulty
                },
                # Block Hashes
                "hashes": {
                    "previous_hash": block.prev_hash,
                    "current_hash": block.hash,
                    "model_hash": block.model_hash
                },
                # Digital Signature
                "signature": {
                    "public_key": getattr(block, 'public_key', None),
                    "signature": getattr(block, 'signature', None),
                    "miner": block.miner
                },
                # Block Data
                "data": {
                    "records": block.records,
                    "access_logs": block.access_logs,
                    "num_transactions": len(block.records)
                },
                # Mining Info
                "mining_info": {
                    "miner_id": block.miner,
                    "difficulty": block.difficulty,
                    "nonce": block.nonce,
                    "consensus": "PoA"
                },
                # Rewards (calculated)
                "rewards": {
                    "base_reward": 10.0,
                    "difficulty_bonus": block.difficulty * 2.0,
                    "total_reward": 10.0 + (block.difficulty * 2.0)
                },
                # Legacy fields for compatibility
                "index": block.index,
                "timestamp": block.timestamp,
                "records": block.records,
                "proof": block.nonce,
                "prev_hash": block.prev_hash,
                "model_hash": block.model_hash,
                "hash": block.hash,
                "miner": block.miner,
                "signature": getattr(block, 'signature', None),
                "public_key": getattr(block, 'public_key', None),
                "difficulty": block.difficulty
            }
            chain_list.append(block_dict)

        # Calculate blockchain statistics
        total_blocks = len(chain_list)
        total_rewards = sum(b["rewards"]["total_reward"] for b in chain_list)
        
        data_to_save = {
            "blockchain_info": {
                "version": "2.0",
                "consensus": "Proof of Authority (PoA)",
                "total_blocks": total_blocks,
                "total_rewards": total_rewards,
                "last_updated": time_module.strftime('%Y-%m-%d %H:%M:%S')
            },
            "chain": chain_list
        }
        
        # Save main blockchain file
        with open(filepath, "w") as f:
            json.dump(data_to_save, f, indent=4)
        print(f"[💾] Blockchain saved to {filepath}")
        
        # Also save a detailed log file
        self._save_detailed_log(chain_list)
        
        # Save blockchain summary
        self._save_summary(data_to_save)
    
    def _save_detailed_log(self, chain_list):
        """Save a human-readable detailed log."""
        import time as time_module
        
        log_filepath = "blockchain_detailed.log"
        with open(log_filepath, "w") as f:
            f.write("="*80 + "\n")
            f.write("BLOCKCHAIN DETAILED LOG\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {time_module.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Blocks: {len(chain_list)}\n")
            f.write("="*80 + "\n\n")
            
            for block in chain_list:
                f.write(f"\n{'='*80}\n")
                f.write(f"BLOCK #{block['index']}\n")
                f.write(f"{'='*80}\n\n")
                
                f.write("BLOCK HEADER:\n")
                f.write(f"  Index: {block['block_header']['index']}\n")
                f.write(f"  Timestamp: {block['block_header']['timestamp_readable']}\n")
                f.write(f"  Nonce: {block['block_header']['nonce']}\n")
                f.write(f"  Difficulty: {block['block_header']['difficulty']}\n\n")
                
                f.write("HASHES:\n")
                f.write(f"  Previous Hash: {block['hashes']['previous_hash']}\n")
                f.write(f"  Current Hash:  {block['hashes']['current_hash']}\n")
                f.write(f"  Model Hash:    {block['hashes']['model_hash']}\n\n")
                
                f.write("DIGITAL SIGNATURE:\n")
                f.write(f"  Miner: {block['signature']['miner']}\n")
                f.write(f"  Public Key: {block['signature']['public_key']}\n")
                f.write(f"  Signature: {block['signature']['signature']}\n\n")
                
                f.write("MINING INFO:\n")
                f.write(f"  Miner ID: {block['mining_info']['miner_id']}\n")
                f.write(f"  Consensus: {block['mining_info']['consensus']}\n")
                f.write(f"  Difficulty: {block['mining_info']['difficulty']}\n\n")
                
                f.write("REWARDS:\n")
                f.write(f"  Base Reward: {block['rewards']['base_reward']} tokens\n")
                f.write(f"  Difficulty Bonus: {block['rewards']['difficulty_bonus']} tokens\n")
                f.write(f"  Total Reward: {block['rewards']['total_reward']} tokens\n\n")
                
                f.write("TRANSACTIONS:\n")
                f.write(f"  Count: {block['data']['num_transactions']}\n")
                if block['data']['records']:
                    for i, record in enumerate(block['data']['records'], 1):
                        f.write(f"  Transaction {i}: {record}\n")
                f.write("\n")
        
        print(f"[📝] Detailed log saved to {log_filepath}")
    
    def _save_summary(self, data):
        """Save a blockchain summary."""
        import time as time_module
        
        summary_filepath = "blockchain_summary.txt"
        with open(summary_filepath, "w") as f:
            f.write("="*80 + "\n")
            f.write("BLOCKCHAIN SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            info = data["blockchain_info"]
            f.write(f"Version: {info['version']}\n")
            f.write(f"Consensus: {info['consensus']}\n")
            f.write(f"Total Blocks: {info['total_blocks']}\n")
            f.write(f"Total Rewards: {info['total_rewards']} tokens\n")
            f.write(f"Last Updated: {info['last_updated']}\n\n")
            
            f.write("BLOCK LIST:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Block':<8} {'Miner':<15} {'Timestamp':<20} {'Hash':<20} {'Reward':<10}\n")
            f.write("-" * 80 + "\n")
            
            for block in data["chain"]:
                f.write(f"#{block['index']:<7} ")
                f.write(f"{block['miner']:<15} ")
                f.write(f"{block['block_header']['timestamp_readable']:<20} ")
                f.write(f"{block['hash'][:16]}... ")
                f.write(f"{block['rewards']['total_reward']:<10.1f}\n")
            
            f.write("-" * 80 + "\n")
        
        print(f"[📊] Summary saved to {summary_filepath}")