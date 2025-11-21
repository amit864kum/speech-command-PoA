import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import hashlib
import random
import time
from typing import Dict, List, Tuple, Optional
from fl_node import SimpleAudioClassifier
from data_loader import SpeechCommandsDataLoader
from ehr_chain import EHRChain
from miner import Miner
import numpy as np
import torch.nn.functional as F

# Import new privacy and adversarial modules
try:
    from src.privacy.differential_privacy import PrivacyEngine, clip_gradients_per_sample, add_gaussian_noise
    from src.privacy.privacy_accountant import PrivacyAccountant, compute_privacy_budget
    PRIVACY_AVAILABLE = True
except ImportError:
    PRIVACY_AVAILABLE = False
    print("[INFO] Privacy modules not available. Install to enable differential privacy.")

# Import IPFS and blockchain modules
try:
    from src.storage.ipfs_manager import IPFSManager
    from src.blockchain.transaction import ModelUpdateTransaction, create_commit_hash
    from src.blockchain.incentives import IncentiveManager
    BLOCKCHAIN_AVAILABLE = True
except ImportError:
    BLOCKCHAIN_AVAILABLE = False
    print("[INFO] Blockchain modules not available. Advanced features disabled.")

# FL Utility functions (moved from fl_trainer.py)
def aggregate_local_models(local_weights: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    if not local_weights:
        return {}
    num_participating_clients = len(local_weights)
    clients = list(local_weights.keys())
    global_weights = {name: torch.from_numpy(param) for name, param in local_weights[clients[0]].items()}
    for client_id in clients[1:]:
        for layer_name in global_weights.keys():
            global_weights[layer_name] += torch.from_numpy(local_weights[client_id][layer_name])
    for layer_name in global_weights.keys():
        global_weights[layer_name] = torch.div(global_weights[layer_name], num_participating_clients)
    return global_weights

def compute_global_model_hash(global_weights: Dict[str, torch.Tensor]) -> str:
    weights_str = json.dumps({name: param.cpu().numpy().tolist() for name, param in global_weights.items()}, sort_keys=True)
    return hashlib.sha256(weights_str.encode()).hexdigest()

class DecentralizedClient:
    def __init__(self, client_id, miner_id, client_data, input_dim, output_dim, target_words,
                 enable_dp=False, noise_multiplier=1.1, max_grad_norm=1.0, 
                 target_epsilon=None, is_byzantine=False, attack_type=None, attack_strength=1.0):
        """
        Enhanced Decentralized Client with Privacy and Robustness features.
        
        Args:
            client_id: Unique client identifier
            miner_id: Associated miner identifier
            client_data: Local training data
            input_dim: Model input dimension
            output_dim: Model output dimension (number of classes)
            target_words: List of target words for classification
            enable_dp: Enable differential privacy (default: False)
            noise_multiplier: DP noise multiplier (default: 1.1)
            max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
            target_epsilon: Target privacy budget (default: None)
            is_byzantine: Whether this is a Byzantine (malicious) client (default: False)
            attack_type: Type of attack if Byzantine (default: None)
            attack_strength: Strength of attack (default: 1.0)
        """
        self.client_id = client_id
        self.miner_id = miner_id
        self.local_data = client_data
        self.target_words = target_words
        self.device = torch.device("cpu")
        self.model = SimpleAudioClassifier(input_dim, output_dim).to(self.device)
        self.private_key = hashlib.sha256(self.client_id.encode()).hexdigest()
        self.ehr_chain = EHRChain() # Each client has its own chain
        self.miner = Miner(self.miner_id, self.ehr_chain, difficulty=1)
        
        # Privacy features
        self.enable_dp = enable_dp and PRIVACY_AVAILABLE
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.target_epsilon = target_epsilon
        self.privacy_engine = None
        
        if self.enable_dp:
            self.privacy_engine = PrivacyEngine(
                model=self.model,
                batch_size=32,  # Default batch size
                sample_size=len(client_data),
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                target_epsilon=target_epsilon,
                target_delta=1e-5
            )
        
        # Adversarial features
        self.is_byzantine = is_byzantine
        self.attack_type = attack_type
        self.attack_strength = attack_strength
        self.initial_weights = None
        
        # IPFS and blockchain features
        self.ipfs_manager = None
        self.use_ipfs = False
        if BLOCKCHAIN_AVAILABLE:
            try:
                self.ipfs_manager = IPFSManager()
                self.use_ipfs = True
                print(f"[📦] {client_id}: IPFS storage enabled")
            except Exception as e:
                print(f"[INFO] {client_id}: IPFS not available, using direct storage")

    def get_model_weights(self):
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}

    def update_model(self, global_weights):
        weights_tensor = {k: torch.tensor(v) for k, v in global_weights.items()}
        self.model.load_state_dict(weights_tensor)
    
    def upload_model_to_ipfs(self, accuracy: float, num_samples: int) -> Optional[str]:
        """Upload model weights to IPFS and return CID.
        
        Args:
            accuracy: Model accuracy
            num_samples: Number of training samples
            
        Returns:
            IPFS CID or None if IPFS not available
        """
        if not self.use_ipfs or not self.ipfs_manager:
            return None
        
        try:
            # Get model weights as OrderedDict
            from collections import OrderedDict
            weights = OrderedDict()
            for k, v in self.model.state_dict().items():
                weights[k] = v.cpu()
            
            # Create metadata
            metadata = {
                "client_id": self.client_id,
                "accuracy": accuracy,
                "num_samples": num_samples,
                "timestamp": time.time()
            }
            
            # Upload to IPFS
            cid = self.ipfs_manager.upload_model_weights(weights, metadata)
            print(f"[📦] {self.client_id}: Model uploaded to IPFS: {cid[:16]}...")
            return cid
        except Exception as e:
            print(f"[ERROR] {self.client_id}: IPFS upload failed: {e}")
            return None

    def local_train(self, epochs, batch_size, lr) -> Tuple[Dict, float, str, List]:
        """
        Local training with optional differential privacy and Byzantine behavior.
        """
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Store initial weights for Byzantine attacks
        if self.is_byzantine and self.attack_type in ["sign_flipping", "scaling"]:
            self.initial_weights = self.get_model_weights()
        
        local_dataloader = DataLoader(self.local_data, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            for i, (features, labels) in enumerate(local_dataloader):
                features, labels = features.to(self.device), labels.to(self.device)
                features = features.permute(0, 2, 1)
                
                # Label flipping attack
                if self.is_byzantine and self.attack_type == "label_flipping":
                    if random.random() < self.attack_strength:
                        num_classes = len(self.target_words)
                        labels = torch.randint(0, num_classes, labels.shape, device=self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Apply differential privacy
                if self.enable_dp and self.privacy_engine:
                    # Clip gradients
                    clip_gradients_per_sample(self.model, self.max_grad_norm)
                    # Add noise
                    noise_scale = self.noise_multiplier * self.max_grad_norm
                    add_gaussian_noise(self.model, noise_scale)
                    self.privacy_engine.step()
                
                optimizer.step()
        
        # Apply Byzantine attacks to model weights
        if self.is_byzantine:
            self._apply_byzantine_attack()
        
        self.model.eval()
        correct, total, predictions = 0, 0, []
        with torch.no_grad():
            for features, labels in local_dataloader:
                features, labels = features.to(self.device), labels.to(self.device)
                features = features.permute(0, 2, 1)
                outputs = self.model(features)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                for i in range(len(predicted_class)):
                    predictions.append({"predicted_word": self.target_words[predicted_class[i]], "confidence": confidence[i].item() * 100})
                total += labels.size(0)
                correct += (predicted_class == labels).sum().item()
        
        accuracy = correct / total
        weights_serialized = self.get_model_weights()
        model_id = compute_global_model_hash({self.client_id: weights_serialized})
        
        sample_size = min(5, len(predictions))
        sample_predictions = random.sample(predictions, sample_size)
        
        # Add privacy metrics if DP is enabled
        privacy_info = {}
        if self.enable_dp and self.privacy_engine:
            epsilon, delta = self.privacy_engine.get_privacy_spent()
            privacy_info = {
                "epsilon": epsilon,
                "delta": delta,
                "privacy_budget_exceeded": not self.privacy_engine.check_privacy_budget()
            }
            print(f"[🔒] Privacy: ε={epsilon:.2f}, δ={delta:.2e}")
        
        return weights_serialized, accuracy, model_id, sample_predictions
    
    def _apply_byzantine_attack(self):
        """Apply Byzantine attack to model weights."""
        if not self.is_byzantine:
            return
        
        with torch.no_grad():
            if self.attack_type == "random":
                # Add random noise
                for param in self.model.parameters():
                    noise = torch.randn_like(param) * self.attack_strength
                    param.add_(noise)
                print(f"[⚠️] Byzantine: Applied random attack (strength={self.attack_strength})")
            
            elif self.attack_type == "sign_flipping" and self.initial_weights:
                # Flip sign of updates
                current_weights = self.get_model_weights()
                for key in current_weights.keys():
                    if key in self.initial_weights:
                        update = current_weights[key] - self.initial_weights[key]
                        flipped = self.initial_weights[key] - self.attack_strength * update
                        current_weights[key] = flipped
                self.update_model(current_weights)
                print(f"[⚠️] Byzantine: Applied sign flipping attack")
            
            elif self.attack_type == "gaussian":
                # Add Gaussian noise scaled by parameter std
                for param in self.model.parameters():
                    param_std = param.std().item() if param.numel() > 1 else 1.0
                    noise_std = param_std * self.attack_strength
                    noise = torch.normal(0.0, noise_std, size=param.shape, device=param.device)
                    param.add_(noise)
                print(f"[⚠️] Byzantine: Applied Gaussian noise attack")
            
            elif self.attack_type == "scaling" and self.initial_weights:
                # Scale updates
                current_weights = self.get_model_weights()
                for key in current_weights.keys():
                    if key in self.initial_weights:
                        update = current_weights[key] - self.initial_weights[key]
                        scaled = self.initial_weights[key] + self.attack_strength * update
                        current_weights[key] = scaled
                self.update_model(current_weights)
                print(f"[⚠️] Byzantine: Applied scaling attack")
    
    def run_client_loop(self):
        print(f"\n{'='*70}")
        print(f"🚀 CLIENT {self.client_id} STARTING FEDERATED LEARNING")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  - Miner ID: {self.miner_id}")
        print(f"  - Local Data: {len(self.local_data)} samples")
        print(f"  - Differential Privacy: {'ENABLED' if self.enable_dp else 'DISABLED'}")
        print(f"  - Byzantine Mode: {'ACTIVE' if self.is_byzantine else 'HONEST'}")
        if self.is_byzantine:
            print(f"  - Attack Type: {self.attack_type}")
            print(f"  - Attack Strength: {self.attack_strength}")
        print(f"{'='*70}")
        
        # Simplified decentralized loop
        for epoch in range(1, EPOCHS + 1):
            print(f"\n{'#'*70}")
            print(f"{'#'*70}")
            print(f"##  EPOCH {epoch}/{EPOCHS} - CLIENT {self.client_id}")
            print(f"{'#'*70}")
            print(f"{'#'*70}")
            
            # This client trains its local model
            print(f"\n📚 TRAINING LOCAL MODEL...")
            print(f"  - Local Epochs: {LOCAL_EPOCHS}")
            print(f"  - Batch Size: {BATCH_SIZE}")
            print(f"  - Learning Rate: {LR}")
            
            weights, acc, model_id, preds = self.local_train(LOCAL_EPOCHS, BATCH_SIZE, LR)
            
            print(f"\n📈 TRAINING RESULTS:")
            print(f"  - Accuracy: {acc*100:.2f}%")
            print(f"  - Model ID: {model_id[:32]}...")
            print(f"  - Predictions: {len(preds)} samples")
            
            # Upload to IPFS if available
            if self.use_ipfs:
                cid = self.upload_model_to_ipfs(acc, len(self.local_data))
                if cid:
                    print(f"  - IPFS CID: {cid[:32]}...")
            
            # In a real system, this client would broadcast its weights to peers.
            # For our simulation, we just mine a block.
            txs = [{
                "miner": self.miner_id,
                "client": self.client_id,
                "accuracy": acc,
                "predictions": preds,
                "epoch": epoch
            }]
            
            winner_block = self.miner.mine_block(txs, model_id)
            
            if winner_block:
                self.ehr_chain.add_block(winner_block)
                print(f"\n✅ BLOCK SUCCESSFULLY ADDED TO BLOCKCHAIN")
                print(f"  - Block #{winner_block.index} appended to chain")
                print(f"  - Chain Length: {len(self.ehr_chain.chain)} blocks")
                print(f"  - Blockchain File: blockchain.json")
            
            self.ehr_chain.save_to_file()
            
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch} COMPLETE")
            print(f"{'='*70}")