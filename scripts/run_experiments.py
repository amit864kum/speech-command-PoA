"""Comprehensive experiments for publication."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import json
import time
from typing import Dict, List
import numpy as np

from src.data import SpeechCommandsDataLoader
from src.models import SimpleAudioClassifier
from src.federated import FederatedClient, FederatedTrainer, create_aggregator
from src.utils import set_seed, get_device
from client import DecentralizedClient


class ExperimentRunner:
    """Run comprehensive experiments for publication."""
    
    def __init__(self, output_dir: str = "results/experiments"):
        """Initialize experiment runner."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}
    
    def run_all_experiments(self):
        """Run all experiments."""
        print("="*80)
        print("RUNNING COMPREHENSIVE EXPERIMENTS")
        print("="*80)
        
        # Experiment 1: Privacy-Utility Tradeoff
        print("\n" + "="*80)
        print("EXPERIMENT 1: Privacy-Utility Tradeoff")
        print("="*80)
        self.results["privacy_utility"] = self.experiment_privacy_utility()
        
        # Experiment 2: Byzantine Robustness
        print("\n" + "="*80)
        print("EXPERIMENT 2: Byzantine Robustness")
        print("="*80)
        self.results["byzantine_robustness"] = self.experiment_byzantine_robustness()
        
        # Experiment 3: Scalability
        print("\n" + "="*80)
        print("EXPERIMENT 3: Scalability Analysis")
        print("="*80)
        self.results["scalability"] = self.experiment_scalability()
        
        # Experiment 4: Aggregation Comparison
        print("\n" + "="*80)
        print("EXPERIMENT 4: Aggregation Method Comparison")
        print("="*80)
        self.results["aggregation_comparison"] = self.experiment_aggregation_comparison()
        
        # Save all results
        self.save_results()
        
        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETE!")
        print("="*80)
        print(f"Results saved to: {self.output_dir}")
    
    def experiment_privacy_utility(self) -> Dict:
        """Experiment 1: Privacy-utility tradeoff with different epsilon values."""
        print("\nTesting different privacy budgets (ε)...")
        
        epsilon_values = [0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]  # inf = no DP
        results = []
        
        for epsilon in epsilon_values:
            print(f"\n--- Testing ε = {epsilon} ---")
            set_seed(42)
            
            # Setup
            enable_dp = epsilon != float('inf')
            num_clients = 3
            num_rounds = 5
            local_epochs = 3
            
            # Load data
            data_loader = SpeechCommandsDataLoader(num_clients=num_clients)
            
            # Create clients with DP
            clients = []
            for i in range(num_clients):
                client_data = data_loader.get_client_data(i)
                client = DecentralizedClient(
                    client_id=f"Client_{i}",
                    miner_id=f"Miner_{i}",
                    client_data=client_data,
                    input_dim=64,
                    output_dim=10,
                    target_words=data_loader.target_words,
                    enable_dp=enable_dp,
                    noise_multiplier=1.1,
                    max_grad_norm=1.0,
                    target_epsilon=epsilon if enable_dp else None
                )
                clients.append(client)
            
            # Train
            accuracies = []
            for round_num in range(num_rounds):
                round_accs = []
                for client in clients:
                    weights, acc, model_id, preds = client.local_train(
                        local_epochs, 32, 0.001
                    )
                    round_accs.append(acc)
                accuracies.append(np.mean(round_accs))
            
            final_accuracy = np.mean(accuracies[-2:])  # Average last 2 rounds
            
            result = {
                "epsilon": epsilon if epsilon != float('inf') else "No DP",
                "final_accuracy": final_accuracy,
                "accuracy_history": accuracies
            }
            results.append(result)
            
            print(f"  Final Accuracy: {final_accuracy*100:.2f}%")
        
        return {
            "description": "Privacy-utility tradeoff with different epsilon values",
            "results": results
        }
    
    def experiment_byzantine_robustness(self) -> Dict:
        """Experiment 2: Byzantine robustness with different attack rates."""
        print("\nTesting Byzantine robustness...")
        
        byzantine_rates = [0.0, 0.1, 0.2, 0.3]  # 0%, 10%, 20%, 30%
        aggregation_methods = ["fedavg", "krum", "trimmed_mean"]
        results = []
        
        for byz_rate in byzantine_rates:
            for agg_method in aggregation_methods:
                print(f"\n--- Testing {agg_method} with {byz_rate*100:.0f}% Byzantine ---")
                set_seed(42)
                
                # Setup
                num_clients = 10
                num_byzantine = int(num_clients * byz_rate)
                num_rounds = 5
                local_epochs = 3
                
                # Load data
                data_loader = SpeechCommandsDataLoader(num_clients=num_clients)
                
                # Create clients (some Byzantine)
                clients = []
                for i in range(num_clients):
                    client_data = data_loader.get_client_data(i)
                    is_byzantine = i < num_byzantine
                    
                    client = DecentralizedClient(
                        client_id=f"Client_{i}",
                        miner_id=f"Miner_{i}",
                        client_data=client_data,
                        input_dim=64,
                        output_dim=10,
                        target_words=data_loader.target_words,
                        is_byzantine=is_byzantine,
                        attack_type="random" if is_byzantine else None,
                        attack_strength=2.0 if is_byzantine else 0.0
                    )
                    clients.append(client)
                
                # Train with aggregation
                accuracies = []
                for round_num in range(num_rounds):
                    # Local training
                    client_weights = []
                    client_sizes = []
                    round_accs = []
                    
                    for client in clients:
                        weights, acc, model_id, preds = client.local_train(
                            local_epochs, 32, 0.001
                        )
                        client_weights.append({client.client_id: weights})
                        client_sizes.append(len(client.local_data))
                        round_accs.append(acc)
                    
                    # Aggregate (this is where robustness matters)
                    aggregator = create_aggregator(
                        agg_method,
                        num_byzantine=num_byzantine if agg_method == "krum" else None,
                        trim_ratio=0.2 if agg_method == "trimmed_mean" else None
                    )
                    
                    # Convert to proper format for aggregator
                    from collections import OrderedDict
                    weights_list = []
                    for cw in client_weights:
                        client_id = list(cw.keys())[0]
                        weights_dict = OrderedDict()
                        for k, v in cw[client_id].items():
                            weights_dict[k] = torch.from_numpy(v)
                        weights_list.append(weights_dict)
                    
                    global_weights = aggregator.aggregate(weights_list, client_sizes)
                    
                    # Update clients with global model
                    global_weights_np = {k: v.cpu().numpy() for k, v in global_weights.items()}
                    for client in clients:
                        client.update_model(global_weights_np)
                    
                    accuracies.append(np.mean(round_accs))
                
                final_accuracy = np.mean(accuracies[-2:])
                
                result = {
                    "byzantine_rate": byz_rate,
                    "aggregation_method": agg_method,
                    "final_accuracy": final_accuracy,
                    "accuracy_history": accuracies
                }
                results.append(result)
                
                print(f"  Final Accuracy: {final_accuracy*100:.2f}%")
        
        return {
            "description": "Byzantine robustness with different attack rates and aggregation methods",
            "results": results
        }
    
    def experiment_scalability(self) -> Dict:
        """Experiment 3: Scalability with different numbers of clients."""
        print("\nTesting scalability...")
        
        client_counts = [5, 10, 20, 50]
        results = []
        
        for num_clients in client_counts:
            print(f"\n--- Testing with {num_clients} clients ---")
            set_seed(42)
            
            # Setup
            num_rounds = 3  # Fewer rounds for larger experiments
            local_epochs = 2
            
            # Load data
            data_loader = SpeechCommandsDataLoader(num_clients=num_clients)
            
            # Create clients
            clients = []
            for i in range(num_clients):
                client_data = data_loader.get_client_data(i)
                client = DecentralizedClient(
                    client_id=f"Client_{i}",
                    miner_id=f"Miner_{i}",
                    client_data=client_data,
                    input_dim=64,
                    output_dim=10,
                    target_words=data_loader.target_words
                )
                clients.append(client)
            
            # Train and measure time
            start_time = time.time()
            accuracies = []
            
            for round_num in range(num_rounds):
                round_accs = []
                for client in clients:
                    weights, acc, model_id, preds = client.local_train(
                        local_epochs, 32, 0.001
                    )
                    round_accs.append(acc)
                accuracies.append(np.mean(round_accs))
            
            total_time = time.time() - start_time
            time_per_round = total_time / num_rounds
            
            result = {
                "num_clients": num_clients,
                "final_accuracy": np.mean(accuracies[-2:]) if len(accuracies) >= 2 else accuracies[-1],
                "total_time": total_time,
                "time_per_round": time_per_round,
                "accuracy_history": accuracies
            }
            results.append(result)
            
            print(f"  Final Accuracy: {result['final_accuracy']*100:.2f}%")
            print(f"  Time per Round: {time_per_round:.2f}s")
        
        return {
            "description": "Scalability analysis with different numbers of clients",
            "results": results
        }
    
    def experiment_aggregation_comparison(self) -> Dict:
        """Experiment 4: Compare aggregation methods."""
        print("\nComparing aggregation methods...")
        
        aggregation_methods = ["fedavg", "krum", "trimmed_mean", "median"]
        results = []
        
        for agg_method in aggregation_methods:
            print(f"\n--- Testing {agg_method} ---")
            set_seed(42)
            
            # Setup
            num_clients = 10
            num_rounds = 5
            local_epochs = 3
            
            # Load data
            data_loader = SpeechCommandsDataLoader(num_clients=num_clients)
            
            # Create clients
            clients = []
            for i in range(num_clients):
                client_data = data_loader.get_client_data(i)
                client = DecentralizedClient(
                    client_id=f"Client_{i}",
                    miner_id=f"Miner_{i}",
                    client_data=client_data,
                    input_dim=64,
                    output_dim=10,
                    target_words=data_loader.target_words
                )
                clients.append(client)
            
            # Train
            accuracies = []
            for round_num in range(num_rounds):
                # Local training
                client_weights = []
                client_sizes = []
                round_accs = []
                
                for client in clients:
                    weights, acc, model_id, preds = client.local_train(
                        local_epochs, 32, 0.001
                    )
                    client_weights.append({client.client_id: weights})
                    client_sizes.append(len(client.local_data))
                    round_accs.append(acc)
                
                # Aggregate
                aggregator = create_aggregator(
                    agg_method,
                    num_byzantine=2 if agg_method == "krum" else None,
                    trim_ratio=0.2 if agg_method == "trimmed_mean" else None
                )
                
                # Convert format
                from collections import OrderedDict
                weights_list = []
                for cw in client_weights:
                    client_id = list(cw.keys())[0]
                    weights_dict = OrderedDict()
                    for k, v in cw[client_id].items():
                        weights_dict[k] = torch.from_numpy(v)
                    weights_list.append(weights_dict)
                
                global_weights = aggregator.aggregate(weights_list, client_sizes)
                
                # Update clients
                global_weights_np = {k: v.cpu().numpy() for k, v in global_weights.items()}
                for client in clients:
                    client.update_model(global_weights_np)
                
                accuracies.append(np.mean(round_accs))
            
            final_accuracy = np.mean(accuracies[-2:])
            
            result = {
                "aggregation_method": agg_method,
                "final_accuracy": final_accuracy,
                "accuracy_history": accuracies
            }
            results.append(result)
            
            print(f"  Final Accuracy: {final_accuracy*100:.2f}%")
        
        return {
            "description": "Comparison of different aggregation methods",
            "results": results
        }
    
    def save_results(self):
        """Save all experiment results."""
        output_file = os.path.join(self.output_dir, "experiment_results.json")
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"\n[Results] Saved to {output_file}")
        
        # Also save summary
        self.save_summary()
    
    def save_summary(self):
        """Save experiment summary."""
        summary_file = os.path.join(self.output_dir, "experiment_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EXPERIMENT RESULTS SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            # Privacy-Utility
            if "privacy_utility" in self.results:
                f.write("EXPERIMENT 1: Privacy-Utility Tradeoff\n")
                f.write("-"*80 + "\n")
                for result in self.results["privacy_utility"]["results"]:
                    f.write(f"ε = {result['epsilon']}: {result['final_accuracy']*100:.2f}%\n")
                f.write("\n")
            
            # Byzantine Robustness
            if "byzantine_robustness" in self.results:
                f.write("EXPERIMENT 2: Byzantine Robustness\n")
                f.write("-"*80 + "\n")
                for result in self.results["byzantine_robustness"]["results"]:
                    f.write(f"{result['aggregation_method']} @ {result['byzantine_rate']*100:.0f}% Byzantine: ")
                    f.write(f"{result['final_accuracy']*100:.2f}%\n")
                f.write("\n")
            
            # Scalability
            if "scalability" in self.results:
                f.write("EXPERIMENT 3: Scalability\n")
                f.write("-"*80 + "\n")
                for result in self.results["scalability"]["results"]:
                    f.write(f"{result['num_clients']} clients: {result['final_accuracy']*100:.2f}% ")
                    f.write(f"({result['time_per_round']:.2f}s/round)\n")
                f.write("\n")
            
            # Aggregation Comparison
            if "aggregation_comparison" in self.results:
                f.write("EXPERIMENT 4: Aggregation Comparison\n")
                f.write("-"*80 + "\n")
                for result in self.results["aggregation_comparison"]["results"]:
                    f.write(f"{result['aggregation_method']}: {result['final_accuracy']*100:.2f}%\n")
                f.write("\n")
        
        print(f"[Summary] Saved to {summary_file}")


if __name__ == "__main__":
    print("Starting comprehensive experiments...")
    print("This will take approximately 30-60 minutes.\n")
    
    runner = ExperimentRunner()
    runner.run_all_experiments()
    
    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE!")
    print("="*80)
    print("\nResults saved to: results/experiments/")
    print("  - experiment_results.json (detailed results)")
    print("  - experiment_summary.txt (summary)")
