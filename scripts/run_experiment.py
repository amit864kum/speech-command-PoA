"""Quick start script for running federated learning experiments."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import main

if __name__ == "__main__":
    print("="*70)
    print("FEDERATED LEARNING FOR SPEECH COMMANDS")
    print("="*70)
    print()
    
    # Run with default configuration
    history, final_metrics = main("Speech_command/configs/default_config.yaml")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED")
    print("="*70)
    print(f"Final Test Accuracy: {final_metrics['accuracy']:.2f}%")
    print(f"Final Test Loss: {final_metrics['loss']:.4f}")
    print("="*70)