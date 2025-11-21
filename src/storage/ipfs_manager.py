"""IPFS manager for off-chain model storage."""

import hashlib
import json
import pickle
from typing import Dict, Optional, Any
from collections import OrderedDict
import torch

# Try to import IPFS client
try:
    import ipfshttpclient
    IPFS_AVAILABLE = True
except ImportError:
    IPFS_AVAILABLE = False


class IPFSManager:
    """Manager for IPFS storage operations."""
    
    def __init__(
        self,
        api_host: str = "127.0.0.1",
        api_port: int = 5001,
        use_mock: bool = False
    ):
        """Initialize IPFS manager.
        
        Args:
            api_host: IPFS API host
            api_port: IPFS API port
            use_mock: Use mock IPFS for testing
        """
        self.api_host = api_host
        self.api_port = api_port
        self.use_mock = use_mock or not IPFS_AVAILABLE
        
        if not self.use_mock:
            try:
                self.client = ipfshttpclient.connect(
                    f'/ip4/{api_host}/tcp/{api_port}/http'
                )
                print(f"[IPFS] Connected to IPFS daemon at {api_host}:{api_port}")
            except Exception as e:
                print(f"[IPFS] Failed to connect: {e}. Using mock IPFS.")
                self.use_mock = True
                self.client = None
        else:
            self.client = None
            print("[IPFS] Using mock IPFS storage")
        
        # Mock storage for testing
        self.mock_storage = {}
    
    def upload_model_weights(
        self,
        weights: OrderedDict,
        metadata: Optional[Dict] = None
    ) -> str:
        """Upload model weights to IPFS.
        
        Args:
            weights: Model weights (state dict)
            metadata: Optional metadata
            
        Returns:
            IPFS CID
        """
        # Serialize weights
        weights_bytes = pickle.dumps(weights)
        
        # Create package with metadata
        package = {
            "weights": weights_bytes,
            "metadata": metadata or {}
        }
        package_bytes = pickle.dumps(package)
        
        if self.use_mock:
            return self._mock_upload(package_bytes)
        else:
            return self._ipfs_upload(package_bytes)
    
    def download_model_weights(self, cid: str) -> tuple[OrderedDict, Dict]:
        """Download model weights from IPFS.
        
        Args:
            cid: IPFS CID
            
        Returns:
            Tuple of (weights, metadata)
        """
        if self.use_mock:
            package_bytes = self._mock_download(cid)
        else:
            package_bytes = self._ipfs_download(cid)
        
        # Deserialize package
        package = pickle.loads(package_bytes)
        weights = pickle.loads(package["weights"])
        metadata = package.get("metadata", {})
        
        return weights, metadata
    
    def upload_data(self, data: bytes) -> str:
        """Upload arbitrary data to IPFS.
        
        Args:
            data: Data bytes
            
        Returns:
            IPFS CID
        """
        if self.use_mock:
            return self._mock_upload(data)
        else:
            return self._ipfs_upload(data)
    
    def download_data(self, cid: str) -> bytes:
        """Download data from IPFS.
        
        Args:
            cid: IPFS CID
            
        Returns:
            Data bytes
        """
        if self.use_mock:
            return self._mock_download(cid)
        else:
            return self._ipfs_download(cid)
    
    def _ipfs_upload(self, data: bytes) -> str:
        """Upload to real IPFS.
        
        Args:
            data: Data bytes
            
        Returns:
            IPFS CID
        """
        try:
            result = self.client.add_bytes(data)
            cid = result
            print(f"[IPFS] Uploaded to IPFS: {cid}")
            return cid
        except Exception as e:
            print(f"[IPFS] Upload failed: {e}")
            # Fallback to mock
            return self._mock_upload(data)
    
    def _ipfs_download(self, cid: str) -> bytes:
        """Download from real IPFS.
        
        Args:
            cid: IPFS CID
            
        Returns:
            Data bytes
        """
        try:
            data = self.client.cat(cid)
            print(f"[IPFS] Downloaded from IPFS: {cid}")
            return data
        except Exception as e:
            print(f"[IPFS] Download failed: {e}")
            # Fallback to mock
            return self._mock_download(cid)
    
    def _mock_upload(self, data: bytes) -> str:
        """Mock IPFS upload for testing.
        
        Args:
            data: Data bytes
            
        Returns:
            Mock CID (hash of data)
        """
        # Generate CID as hash of data
        cid = "Qm" + hashlib.sha256(data).hexdigest()[:44]
        self.mock_storage[cid] = data
        print(f"[IPFS Mock] Stored data with CID: {cid}")
        return cid
    
    def _mock_download(self, cid: str) -> bytes:
        """Mock IPFS download for testing.
        
        Args:
            cid: Mock CID
            
        Returns:
            Data bytes
        """
        if cid not in self.mock_storage:
            raise ValueError(f"CID not found in mock storage: {cid}")
        
        print(f"[IPFS Mock] Retrieved data with CID: {cid}")
        return self.mock_storage[cid]
    
    def pin(self, cid: str) -> bool:
        """Pin content to keep it available.
        
        Args:
            cid: IPFS CID
            
        Returns:
            True if successful
        """
        if self.use_mock:
            return True
        
        try:
            self.client.pin.add(cid)
            print(f"[IPFS] Pinned: {cid}")
            return True
        except Exception as e:
            print(f"[IPFS] Pin failed: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get IPFS storage statistics.
        
        Returns:
            Dictionary with statistics
        """
        if self.use_mock:
            return {
                "mode": "mock",
                "stored_items": len(self.mock_storage),
                "total_size": sum(len(data) for data in self.mock_storage.values())
            }
        
        try:
            stats = self.client.stats.repo()
            return {
                "mode": "ipfs",
                "repo_size": stats.get("RepoSize", 0),
                "num_objects": stats.get("NumObjects", 0)
            }
        except:
            return {"mode": "ipfs", "error": "Failed to get stats"}


class MockIPFSManager(IPFSManager):
    """Mock IPFS manager for testing without IPFS daemon."""
    
    def __init__(self):
        """Initialize mock IPFS manager."""
        super().__init__(use_mock=True)


def compute_cid(data: bytes) -> str:
    """Compute CID-like hash for data.
    
    Args:
        data: Data bytes
        
    Returns:
        CID-like hash
    """
    return "Qm" + hashlib.sha256(data).hexdigest()[:44]