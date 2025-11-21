"""Logging utilities for the federated learning system."""

import os
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


class ExperimentLogger:
    """Enhanced logger for federated learning experiments."""
    
    def __init__(
        self,
        name: str = "fl_experiment",
        log_dir: str = "Speech_command/logs",
        log_level: str = "INFO",
        log_to_file: bool = True,
        log_to_console: bool = True
    ):
        """Initialize experiment logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files (relative to project root)
            log_level: Logging level
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add file handler
        if log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"{name}_{timestamp}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Add console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Metrics storage
        self.metrics = {}
        self.round_metrics = []
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def log_experiment_start(self, config: Dict[str, Any]) -> None:
        """Log experiment start with configuration."""
        self.info("=" * 50)
        self.info("EXPERIMENT STARTED")
        self.info("=" * 50)
        self.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    def log_experiment_end(self) -> None:
        """Log experiment end."""
        self.info("=" * 50)
        self.info("EXPERIMENT COMPLETED")
        self.info("=" * 50)
    
    def log_round_start(self, round_num: int) -> None:
        """Log federated learning round start."""
        self.info(f"Starting FL Round {round_num}")
    
    def log_round_end(self, round_num: int, metrics: Dict[str, float]) -> None:
        """Log federated learning round end with metrics."""
        self.info(f"Completed FL Round {round_num}")
        for metric_name, value in metrics.items():
            self.info(f"  {metric_name}: {value:.4f}")
        
        # Store round metrics
        round_data = {"round": round_num, **metrics}
        self.round_metrics.append(round_data)
    
    def log_client_update(self, client_id: str, metrics: Dict[str, float]) -> None:
        """Log client training update."""
        self.debug(f"Client {client_id} update: {metrics}")
    
    def log_blockchain_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log blockchain events."""
        self.info(f"Blockchain {event_type}: {details}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics for the current step."""
        step_info = f" (Step {step})" if step is not None else ""
        self.info(f"Metrics{step_info}: {metrics}")
        
        # Store metrics
        if step is not None:
            if step not in self.metrics:
                self.metrics[step] = {}
            self.metrics[step].update(metrics)
    
    def save_metrics(self, filepath: Optional[str] = None) -> None:
        """Save all collected metrics to file."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.log_dir / f"metrics_{timestamp}.json"
        
        all_metrics = {
            "round_metrics": self.round_metrics,
            "step_metrics": self.metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        self.info(f"Metrics saved to {filepath}")


def get_logger(
    name: str = "fl_experiment",
    config: Optional[Dict[str, Any]] = None
) -> ExperimentLogger:
    """Get configured logger instance.
    
    Args:
        name: Logger name
        config: Configuration dictionary
        
    Returns:
        ExperimentLogger instance
    """
    if config is None:
        config = {}
    
    return ExperimentLogger(
        name=name,
        log_dir=config.get("log_dir", "Speech_command/logs"),
        log_level=config.get("log_level", "INFO"),
        log_to_file=config.get("log_to_file", True),
        log_to_console=config.get("log_to_console", True)
    )