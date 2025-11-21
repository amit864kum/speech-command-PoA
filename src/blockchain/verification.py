"""Model verification and fraud detection for blockchain FL."""

import hashlib
import json
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import time


class MerkleTree:
    """Merkle tree for model parameter verification."""
    
    def __init__(self, data: List[bytes]):
        """Initialize Merkle tree from data.
        
        Args:
            data: List of data items (as bytes)
        """
        self.leaves = [self._hash(item) for item in data]
        self.tree = self._build_tree(self.leaves)
        self.root = self.tree[0] if self.tree else None
    
    def _hash(self, data: bytes) -> str:
        """Hash data using SHA-256."""
        return hashlib.sha256(data).hexdigest()
    
    def _build_tree(self, leaves: List[str]) -> List[str]:
        """Build Merkle tree from leaves."""
        if not leaves:
            return []
        
        tree = leaves.copy()
        level = leaves
        
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    combined = level[i] + level[i + 1]
                    next_level.append(self._hash(combined.encode()))
                else:
                    next_level.append(level[i])
            tree.extend(next_level)
            level = next_level
        
        return tree
    
    def get_root(self) -> Optional[str]:
        """Get Merkle root."""
        return self.root
    
    def get_proof(self, index: int) -> List[Tuple[str, str]]:
        """Get Merkle proof for leaf at index.
        
        Args:
            index: Index of leaf
            
        Returns:
            List of (hash, position) tuples for proof
        """
        if index >= len(self.leaves):
            return []
        
        proof = []
        level_size = len(self.leaves)
        level_start = 0
        current_index = index
        
        while level_size > 1:
            # Find sibling
            if current_index % 2 == 0:
                # Right sibling
                if current_index + 1 < level_size:
                    sibling_index = level_start + current_index + 1
                    proof.append((self.tree[sibling_index], "right"))
            else:
                # Left sibling
                sibling_index = level_start + current_index - 1
                proof.append((self.tree[sibling_index], "left"))
            
            # Move to next level
            level_start += level_size
            current_index = current_index // 2
            level_size = (level_size + 1) // 2
        
        return proof
    
    @staticmethod
    def verify_proof(leaf: str, proof: List[Tuple[str, str]], root: str) -> bool:
        """Verify Merkle proof.
        
        Args:
            leaf: Leaf hash
            proof: Merkle proof
            root: Expected root hash
            
        Returns:
            True if proof is valid
        """
        current = leaf
        
        for sibling, position in proof:
            if position == "right":
                combined = current + sibling
            else:
                combined = sibling + current
            current = hashlib.sha256(combined.encode()).hexdigest()
        
        return current == root


class CommitRevealProtocol:
    """Commit-reveal protocol for model updates."""
    
    def __init__(self):
        """Initialize commit-reveal protocol."""
        self.commits = {}  # client_id -> (commit_hash, timestamp)
        self.reveals = {}  # client_id -> (model_cid, metadata, timestamp)
        self.commit_timeout = 300  # 5 minutes
    
    def create_commit(
        self,
        client_id: str,
        model_cid: str,
        metadata: Dict,
        salt: Optional[str] = None
    ) -> Tuple[str, str]:
        """Create commitment for model update.
        
        Args:
            client_id: Client identifier
            model_cid: IPFS CID of model
            metadata: Model metadata
            salt: Random salt (generated if not provided)
            
        Returns:
            Tuple of (commit_hash, salt)
        """
        if salt is None:
            import secrets
            salt = secrets.token_hex(32)
        
        # Create commitment
        commit_data = json.dumps({
            "client_id": client_id,
            "model_cid": model_cid,
            "metadata": metadata,
            "salt": salt
        }, sort_keys=True)
        
        commit_hash = hashlib.sha256(commit_data.encode()).hexdigest()
        
        # Store commitment
        self.commits[client_id] = (commit_hash, time.time())
        
        return commit_hash, salt
    
    def verify_reveal(
        self,
        client_id: str,
        model_cid: str,
        metadata: Dict,
        salt: str
    ) -> bool:
        """Verify revealed data matches commitment.
        
        Args:
            client_id: Client identifier
            model_cid: IPFS CID of model
            metadata: Model metadata
            salt: Salt used in commitment
            
        Returns:
            True if reveal matches commitment
        """
        if client_id not in self.commits:
            print(f"[Verification] No commitment found for {client_id}")
            return False
        
        commit_hash, commit_time = self.commits[client_id]
        
        # Check timeout
        if time.time() - commit_time > self.commit_timeout:
            print(f"[Verification] Commitment expired for {client_id}")
            return False
        
        # Recreate commitment
        commit_data = json.dumps({
            "client_id": client_id,
            "model_cid": model_cid,
            "metadata": metadata,
            "salt": salt
        }, sort_keys=True)
        
        expected_hash = hashlib.sha256(commit_data.encode()).hexdigest()
        
        # Verify
        if commit_hash == expected_hash:
            self.reveals[client_id] = (model_cid, metadata, time.time())
            print(f"[Verification] ✓ Reveal verified for {client_id}")
            return True
        else:
            print(f"[Verification] ✗ Reveal mismatch for {client_id}")
            return False
    
    def get_commit(self, client_id: str) -> Optional[Tuple[str, float]]:
        """Get commitment for client."""
        return self.commits.get(client_id)
    
    def get_reveal(self, client_id: str) -> Optional[Tuple[str, Dict, float]]:
        """Get reveal for client."""
        return self.reveals.get(client_id)


class ModelVerifier:
    """Verify model quality and detect fraud."""
    
    def __init__(
        self,
        audit_dataset: Optional[torch.utils.data.Dataset] = None,
        quality_threshold: float = 0.05,
        slash_threshold: float = 0.15
    ):
        """Initialize model verifier.
        
        Args:
            audit_dataset: Dataset for spot-checking
            quality_threshold: Acceptable quality deviation
            slash_threshold: Threshold for slashing
        """
        self.audit_dataset = audit_dataset
        self.quality_threshold = quality_threshold
        self.slash_threshold = slash_threshold
        self.verification_history = []
    
    def verify_model_quality(
        self,
        model: nn.Module,
        claimed_accuracy: float,
        device: torch.device = torch.device("cpu")
    ) -> Tuple[bool, float, Optional[str]]:
        """Verify model quality on audit set.
        
        Args:
            model: Model to verify
            claimed_accuracy: Claimed accuracy by client
            device: Device for computation
            
        Returns:
            Tuple of (is_valid, actual_accuracy, fraud_evidence)
        """
        if self.audit_dataset is None:
            print("[Verification] No audit dataset available")
            return True, claimed_accuracy, None
        
        # Evaluate on audit set
        model.eval()
        model.to(device)
        
        correct = 0
        total = 0
        
        audit_loader = torch.utils.data.DataLoader(
            self.audit_dataset,
            batch_size=32,
            shuffle=False
        )
        
        with torch.no_grad():
            for features, labels in audit_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                if len(features.shape) == 2:
                    features = features.unsqueeze(1)
                
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        actual_accuracy = correct / total if total > 0 else 0.0
        deviation = abs(actual_accuracy - claimed_accuracy)
        
        # Check if fraud
        is_valid = deviation <= self.quality_threshold
        fraud_evidence = None
        
        if not is_valid:
            fraud_evidence = json.dumps({
                "claimed_accuracy": claimed_accuracy,
                "actual_accuracy": actual_accuracy,
                "deviation": deviation,
                "threshold": self.quality_threshold,
                "audit_samples": total,
                "timestamp": time.time()
            })
            
            print(f"[Verification] ✗ FRAUD DETECTED!")
            print(f"  Claimed: {claimed_accuracy*100:.2f}%")
            print(f"  Actual: {actual_accuracy*100:.2f}%")
            print(f"  Deviation: {deviation*100:.2f}%")
        else:
            print(f"[Verification] ✓ Model quality verified")
            print(f"  Claimed: {claimed_accuracy*100:.2f}%")
            print(f"  Actual: {actual_accuracy*100:.2f}%")
            print(f"  Deviation: {deviation*100:.2f}%")
        
        # Record verification
        self.verification_history.append({
            "claimed_accuracy": claimed_accuracy,
            "actual_accuracy": actual_accuracy,
            "deviation": deviation,
            "is_valid": is_valid,
            "timestamp": time.time()
        })
        
        return is_valid, actual_accuracy, fraud_evidence
    
    def should_slash(self, deviation: float) -> bool:
        """Determine if deviation warrants slashing.
        
        Args:
            deviation: Accuracy deviation
            
        Returns:
            True if should slash
        """
        return deviation >= self.slash_threshold
    
    def create_fraud_proof(
        self,
        client_id: str,
        model_cid: str,
        claimed_accuracy: float,
        actual_accuracy: float,
        audit_results: Dict
    ) -> Dict:
        """Create fraud proof for slashing transaction.
        
        Args:
            client_id: Client identifier
            model_cid: IPFS CID of fraudulent model
            claimed_accuracy: Claimed accuracy
            actual_accuracy: Actual accuracy
            audit_results: Detailed audit results
            
        Returns:
            Fraud proof dictionary
        """
        fraud_proof = {
            "type": "quality_fraud",
            "client_id": client_id,
            "model_cid": model_cid,
            "claimed_accuracy": claimed_accuracy,
            "actual_accuracy": actual_accuracy,
            "deviation": abs(actual_accuracy - claimed_accuracy),
            "audit_results": audit_results,
            "timestamp": time.time(),
            "verifier": "ModelVerifier"
        }
        
        # Sign fraud proof
        proof_str = json.dumps(fraud_proof, sort_keys=True)
        fraud_proof["proof_hash"] = hashlib.sha256(proof_str.encode()).hexdigest()
        
        return fraud_proof
    
    def get_statistics(self) -> Dict:
        """Get verification statistics."""
        if not self.verification_history:
            return {
                "total_verifications": 0,
                "fraud_detected": 0,
                "fraud_rate": 0.0
            }
        
        total = len(self.verification_history)
        fraud_count = sum(1 for v in self.verification_history if not v["is_valid"])
        
        return {
            "total_verifications": total,
            "fraud_detected": fraud_count,
            "fraud_rate": fraud_count / total,
            "avg_deviation": sum(v["deviation"] for v in self.verification_history) / total
        }


def create_parameter_merkle_tree(model_state_dict: OrderedDict) -> MerkleTree:
    """Create Merkle tree from model parameters.
    
    Args:
        model_state_dict: Model state dictionary
        
    Returns:
        Merkle tree
    """
    # Convert parameters to bytes
    param_bytes = []
    for name, param in model_state_dict.items():
        param_data = {
            "name": name,
            "shape": list(param.shape),
            "data": param.cpu().numpy().tobytes()
        }
        param_bytes.append(json.dumps(param_data, sort_keys=True).encode())
    
    return MerkleTree(param_bytes)


def verify_parameter_subset(
    model_state_dict: OrderedDict,
    merkle_root: str,
    parameter_indices: List[int]
) -> bool:
    """Verify subset of parameters using Merkle proofs.
    
    Args:
        model_state_dict: Model state dictionary
        merkle_root: Expected Merkle root
        parameter_indices: Indices of parameters to verify
        
    Returns:
        True if all parameters verify
    """
    # Create Merkle tree
    tree = create_parameter_merkle_tree(model_state_dict)
    
    # Verify root matches
    if tree.get_root() != merkle_root:
        print("[Verification] Merkle root mismatch")
        return False
    
    print(f"[Verification] ✓ Merkle root verified")
    print(f"[Verification] Spot-checking {len(parameter_indices)} parameters...")
    
    # Verify selected parameters
    param_list = list(model_state_dict.items())
    for idx in parameter_indices:
        if idx >= len(param_list):
            continue
        
        name, param = param_list[idx]
        proof = tree.get_proof(idx)
        leaf = tree.leaves[idx]
        
        if not MerkleTree.verify_proof(leaf, proof, merkle_root):
            print(f"[Verification] ✗ Parameter {name} failed verification")
            return False
    
    print(f"[Verification] ✓ All spot-checked parameters verified")
    return True
