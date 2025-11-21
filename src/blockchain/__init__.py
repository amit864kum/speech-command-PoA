"""Blockchain components for federated learning."""

from .transaction import Transaction, CoinbaseTransaction, ModelUpdateTransaction
from .incentives import IncentiveManager, StakingManager

# Import optional modules if they exist
try:
    from .chain import BlockchainFL
    CHAIN_AVAILABLE = True
except ImportError:
    CHAIN_AVAILABLE = False

try:
    from .miner import EnhancedMiner
    MINER_AVAILABLE = True
except ImportError:
    MINER_AVAILABLE = False

try:
    from .verification import (
        ModelVerifier,
        MerkleTree,
        CommitRevealProtocol,
        create_parameter_merkle_tree,
        verify_parameter_subset
    )
    VERIFICATION_AVAILABLE = True
except ImportError:
    VERIFICATION_AVAILABLE = False

__all__ = [
    "Transaction",
    "CoinbaseTransaction", 
    "ModelUpdateTransaction",
    "IncentiveManager",
    "StakingManager"
]

if CHAIN_AVAILABLE:
    __all__.append("BlockchainFL")
if MINER_AVAILABLE:
    __all__.append("EnhancedMiner")
if VERIFICATION_AVAILABLE:
    __all__.extend([
        "ModelVerifier",
        "MerkleTree",
        "CommitRevealProtocol",
        "create_parameter_merkle_tree",
        "verify_parameter_subset"
    ])