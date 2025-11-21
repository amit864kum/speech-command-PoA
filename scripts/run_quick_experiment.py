"""Quick experiment script to test the enhanced federated learning system."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torch
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.reproducibility import ReproducibilityManager
from src.data import SpeechCommandsDataLoader
from src.models import create_model_from_config
from src.federated import FederatedTrainer


def run_quick_experiment():
    """Run a quick federated learning experiment."""
    print("="*60)
    print("QUICK FEDERATED LEARNING EXPERIMENT")
    print("="*60)
    
    # Load config and modify for quick experiment
    config = load_config()
    
    # Quick experiment settings
    config.set('federated_learning.num_clients', 3)
    config.set('federated_learning.num_rounds', 3)
    config.set('federated_learning.local_epochs', 2)
    config.set('federated_learning.batch_size', 16)
    config.set('model.architecture', 'simpleaudioclassifier')
    config.set('experiment.seed', 42)
    
    # Set up reproducibility
    with ReproducibilityManager(seed=42, device='cpu') as repro_manager:
        device = repro_manager.get_device()
        
        # Initialize logger
        logger = get_logger("quick_experiment")
        
        logger.info("Starting quick experiment...")
        logger.info(f"Device: {device}")
        
        # Load data (small subset)
        logger.info("Loading dataset...")
        data_loader = SpeechCommandsDataLoader(
            num_clients=3,
            data_distribution="non_iid",
            alpha=0.5
        )
        
        # Update config with data info
        data_info = data_loader.get_data_info()
        config.set('model.input_dim', data_info['input_dim'])
        config.set('model.output_dim', data_info['num_classes'])
        
        logger.info(f"Dataset: {data_info['total_samples']} samples, {data_info['num_classes']} classes")
        
        # Create model
        logger.info("Creating model...")
        global_model = create_model_from_config(config.to_dict())
        
        total_params = sum(p.numel() for p in global_model.parameters())
        logger.info(f"Model: {total_params:,} parameters")
        
        # Prepare client data
        clients_data = []
        for i in range(3):
            client_dataset = data_loader.get_client_data(i)
            clients_data.append(client_dataset)
            logger.info(f"Client {i}: {len(client_dataset)} samples")
        
        # Create test dataset
        test_dataset = data_loader.get_test_data(test_fraction=0.2)
        logger.info(f"Test dataset: {len(test_dataset)} samples")
        
        # Initialize trainer
        logger.info("Initializing federated trainer...")
        trainer = FederatedTrainer(
            global_model=global_model,
            clients_data=clients_data,
            test_dataset=test_dataset,
            config=config.to_dict(),
            device=device,
            logger=logger
        )
        
        # Run training
        logger.info("Starting federated training...")
        results = trainer.train()
        
        # Print results
        logger.info("="*40)
        logger.info("EXPERIMENT RESULTS")
        logger.info("="*40)
        logger.info(f"Final accuracy: {results['final_metrics']['accuracy']:.2f}%")
        logger.info(f"Best accuracy: {results['best_accuracy']:.2f}%")
        logger.info(f"Total rounds: {results['total_rounds']}")
        
        # Print training history
        history = results['training_history']
        logger.info("\nTraining History:")
        for i, (acc, loss) in enumerate(zip(history['global_accuracy'], history['global_loss'])):
            logger.info(f"  Round {i+1}: Accuracy={acc:.2f}%, Loss={loss:.4f}")
        
        logger.info("\nQuick experiment completed successfully!")
        
        return results


if __name__ == "__main__":
    try:
        results = run_quick_experiment()
        print("\n✅ Experiment completed successfully!")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        raise