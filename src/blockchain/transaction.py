"""Transaction types for blockchain-based federated learning."""

import hashlib
import json
import time
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class Transaction:
    """Base transaction class."""
    
    tx_type: str
    sender: str
    timestamp: float
    data: Dict[str, Any]
    signature: Optional[str] = None
    tx_hash: Optional[str] = None
    
    def __post_init__(self):
        """Compute transaction hash after initialization."""
        if self.tx_hash is None:
            self.tx_hash = self.compute_hash()
    
    def compute_hash(self) -> str:
        """Compute SHA-256 hash of transaction."""
        tx_dict = {
            "tx_type": self.tx_type,
            "sender": self.sender,
            "timestamp": self.timestamp,
            "data": self.data
        }
        tx_string = json.dumps(tx_dict, sort_keys=True)
        return hashlib.sha256(tx_string.encode()).hexdigest()
    
    def sign(self, private_key: str) -> str:
        """Sign transaction with private key."""
        message = f"{self.tx_hash}{private_key}"
        self.signature = hashlib.sha256(message.encode()).hexdigest()
        return self.signature
    
    def verify_signature(self, public_key: str) -> bool:
        """Verify transaction signature."""
        if not self.signature:
            return False
        expected = hashlib.sha256(f"{self.tx_hash}{public_key}".encode()).hexdigest()
        return self.signature == expected
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CoinbaseTransaction(Transaction):
    """Coinbase transaction for miner rewards."""
    
    def __init__(self, miner: str, reward: float, block_height: int):
        """Initialize coinbase transaction.
        
        Args:
            miner: Miner address
            reward: Mining reward amount
            block_height: Block height
        """
        super().__init__(
            tx_type="coinbase",
            sender="system",
            timestamp=time.time(),
            data={
                "miner": miner,
                "reward": reward,
                "block_height": block_height
            }
        )


@dataclass
class ModelUpdateTransaction(Transaction):
    """Transaction for model update with IPFS CID."""
    
    def __init__(
        self,
        client_id: str,
        model_cid: str,
        commit_hash: Optional[str] = None,
        metadata: Optional[Dict] = None,
        round_number: int = 0
    ):
        """Initialize model update transaction.
        
        Args:
            client_id: Client identifier
            model_cid: IPFS CID of model update
            commit_hash: Commit hash for commit-reveal
            metadata: Additional metadata (accuracy, samples, etc.)
            round_number: FL round number
        """
        data = {
            "model_cid": model_cid,
            "round_number": round_number,
            "metadata": metadata or {}
        }
        
        if commit_hash:
            data["commit_hash"] = commit_hash
        
        super().__init__(
            tx_type="model_update",
            sender=client_id,
            timestamp=time.time(),
            data=data
        )
    
    def get_cid(self) -> str:
        """Get IPFS CID from transaction."""
        return self.data.get("model_cid", "")
    
    def get_metadata(self) -> Dict:
        """Get metadata from transaction."""
        return self.data.get("metadata", {})


@dataclass
class CommitTransaction(Transaction):
    """Commit transaction for commit-reveal scheme."""
    
    def __init__(self, client_id: str, commit_hash: str, round_number: int):
        """Initialize commit transaction.
        
        Args:
            client_id: Client identifier
            commit_hash: Hash of (CID || metadata)
            round_number: FL round number
        """
        super().__init__(
            tx_type="commit",
            sender=client_id,
            timestamp=time.time(),
            data={
                "commit_hash": commit_hash,
                "round_number": round_number
            }
        )


@dataclass
class RevealTransaction(Transaction):
    """Reveal transaction for commit-reveal scheme."""
    
    def __init__(
        self,
        client_id: str,
        model_cid: str,
        metadata: Dict,
        round_number: int,
        commit_tx_hash: str
    ):
        """Initialize reveal transaction.
        
        Args:
            client_id: Client identifier
            model_cid: IPFS CID of model update
            metadata: Model metadata
            round_number: FL round number
            commit_tx_hash: Hash of corresponding commit transaction
        """
        super().__init__(
            tx_type="reveal",
            sender=client_id,
            timestamp=time.time(),
            data={
                "model_cid": model_cid,
                "metadata": metadata,
                "round_number": round_number,
                "commit_tx_hash": commit_tx_hash
            }
        )
    
    def verify_commit(self, commit_hash: str) -> bool:
        """Verify that reveal matches commit.
        
        Args:
            commit_hash: Original commit hash
            
        Returns:
            True if reveal matches commit
        """
        # Recompute commit hash from revealed data
        reveal_data = f"{self.data['model_cid']}{json.dumps(self.data['metadata'], sort_keys=True)}"
        computed_hash = hashlib.sha256(reveal_data.encode()).hexdigest()
        return computed_hash == commit_hash


@dataclass
class SlashingTransaction(Transaction):
    """Slashing transaction for fraud proof."""
    
    def __init__(
        self,
        accuser: str,
        accused: str,
        evidence_cid: str,
        fraud_type: str,
        penalty: float
    ):
        """Initialize slashing transaction.
        
        Args:
            accuser: Address of accuser
            accused: Address of accused party
            evidence_cid: IPFS CID of fraud evidence
            fraud_type: Type of fraud detected
            penalty: Penalty amount
        """
        super().__init__(
            tx_type="slashing",
            sender=accuser,
            timestamp=time.time(),
            data={
                "accused": accused,
                "evidence_cid": evidence_cid,
                "fraud_type": fraud_type,
                "penalty": penalty
            }
        )


def create_commit_hash(model_cid: str, metadata: Dict) -> str:
    """Create commit hash for commit-reveal scheme.
    
    Args:
        model_cid: IPFS CID of model
        metadata: Model metadata
        
    Returns:
        Commit hash
    """
    commit_data = f"{model_cid}{json.dumps(metadata, sort_keys=True)}"
    return hashlib.sha256(commit_data.encode()).hexdigest()


def verify_reveal(commit_hash: str, model_cid: str, metadata: Dict) -> bool:
    """Verify that reveal matches commit.
    
    Args:
        commit_hash: Original commit hash
        model_cid: Revealed IPFS CID
        metadata: Revealed metadata
        
    Returns:
        True if reveal matches commit
    """
    computed_hash = create_commit_hash(model_cid, metadata)
    return computed_hash == commit_hash