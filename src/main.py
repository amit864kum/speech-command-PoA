"""Main script for running federated learning experiments."""

import sys
import os
import argparse
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.utils.reproducibility import set_seed, get_device
from src.utils.logger import get_logger
from src.data import SpeechCommandsDataLoader
from src.models import SimpleAudioClassifier, GKWS_CNN, DS_CNN, audio_resnet18
from src.federated import FederatedClient, FederatedTrainer, create_aggregator
from torch.utils.data import DataLoader


def create_model(config: dict, data_info: dict) -> torch.nn.Module:
    """Create model based on configuration.
    
    Args:
        config: Configuration dictionary
        data_info: Dataset information
        
    Returns:
        Model instance
    """
    model_type = config.get("model.architecture", "SimpleAudioClassifier")
    input_dim = data_info["input_dim"]
    output_dim = data_info["num_classes"]
    dropout = config.get("model.dropout", 0.2)
    
    if model_type == "SimpleAudioClassifier":
        model = SimpleAudioClassifier(input_dim, output_dim, dropout)
    elif model_type == "GKWS_CNN":
        model = GKWS_CNN(input_dim, output_dim, dropout)
    elif model_type == "DS_CNN":
        model = DS_CNN(input_dim, output_dim, dropout=dropout)
    elif model_type == "AudioResNet18":
        model = audio_resnet18(input_dim, output_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def main(config_path: str = None):
    """Main function to run federated learning experiment.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set up reproducibility
    seed = config.get("experiment.seed", 42)
    set_seed(seed)
    
    # Get device
    device_pref = config.get("hardware.device", "auto")
    device = get_device(device_pref)
    
    # Set up logger
    logger = get_logger(
        name=config.get("experiment.name", "fl_experiment"),
        config={
            "log_dir": config.get("logging.log_dir", "Speech_command/logs"),
            "log_level": config.get("experiment.log_level", "INFO"),
            "log_to_file": config.get("logging.log_to_file", True),
            "log_to_console": True
        }
    )
    
    logger.log_experiment_start(config.to_dict())
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info("Loading Speech Commands dataset...")
    data_loader = SpeechCommandsDataLoader(
        num_clients=config.get("federated_learning.num_clients", 10),
        download_dir=config.get("dataset.data_dir", "Speech_command/data"),
        version=config.get("dataset.version", "v0.02"),
        target_words=config.get("dataset.target_words", None),
        sample_rate=config.get("dataset.sample_rate", 16000),
        n_mfcc=config.get("dataset.n_mfcc", 64),
        n_mels=config.get("dataset.n_mels", 64),
        data_distribution=config.get("federated_learning.data_distribution", "non_iid"),
        alpha=config.get("federated_learning.alpha", 0.5)
    )
    
    data_info = data_loader.get_data_info()
    logger.info(f"Dataset loaded: {data_info['total_samples']} samples")
    
    # Create global model
    logger.info("Creating model...")
    global_model = create_model(config, data_info)
    logger.info(f"Model: {config.get('model.architecture', 'SimpleAudioClassifier')}")
    logger.info(f"Total parameters: {sum(p.numel() for p in global_model.parameters())}")
    
    # Create federated clients
    logger.info("Creating federated clients...")
    clients = []
    for client_id in range(config.get("federated_learning.num_clients", 10)):
        client_data = data_loader.get_client_data(client_id)
        
        client = FederatedClient(
            client_id=f"Client_{client_id}",
            model=type(global_model)(
                data_info["input_dim"],
                data_info["num_classes"],
                config.get("model.dropout", 0.2)
            ),
            train_data=client_data,
            device=device,
            learning_rate=config.get("federated_learning.learning_rate", 0.001),
            batch_size=config.get("federated_learning.batch_size", 32),
            local_epochs=config.get("federated_learning.local_epochs", 5)
        )
        clients.append(client)
    
    logger.info(f"Created {len(clients)} clients")
    
    # Create test data loader
    test_data = data_loader.get_test_data(test_fraction=0.1)
    test_loader = DataLoader(
        test_data,
        batch_size=config.get("federated_learning.batch_size", 32),
        shuffle=False
    )
    logger.info(f"Test set: {len(test_data)} samples")
    
    # Create aggregator
    aggregator_type = config.get("federated_learning.aggregation_method", "fedavg")
    aggregator = create_aggregator(aggregator_type)
    logger.info(f"Aggregation method: {aggregator.get_name()}")
    
    # Create federated trainer
    trainer = FederatedTrainer(
        model=global_model,
        clients=clients,
        aggregator=aggregator,
        test_loader=test_loader,
        device=device,
        logger=logger,
        client_fraction=config.get("federated_learning.client_fraction", 1.0)
    )
    
    # Train
    logger.info("Starting federated training...")
    num_rounds = config.get("federated_learning.num_rounds", 50)
    history = trainer.train(num_rounds=num_rounds, verbose=True)
    
    # Save results
    output_dir = Path(config.get("experiment.output_dir", "Speech_command/results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / f"{config.get('experiment.name', 'model')}_final.pt"
    trainer.save_model(str(model_path))
    
    # Save metrics
    logger.save_metrics(str(output_dir / "metrics.json"))
    
    # Final evaluation
    final_metrics = trainer.evaluate(test_loader)
    logger.info(f"\nFinal Test Results:")
    logger.info(f"  Accuracy: {final_metrics['accuracy']:.2f}%")
    logger.info(f"  Loss: {final_metrics['loss']:.4f}")
    
    logger.log_experiment_end()
    
    return history, final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning for Speech Commands")
    parser.add_argument(
        "--config",
        type=str,
        default="Speech_command/configs/default_config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    main(args.config)