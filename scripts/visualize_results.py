"""Visualize experiment results for publication."""

import json
import matplotlib.pyplot as plt
import numpy as np
import os


def load_results(results_file="results/experiments/experiment_results.json"):
    """Load experiment results."""
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_privacy_utility(results, output_dir="results/experiments/plots"):
    """Plot privacy-utility tradeoff."""
    os.makedirs(output_dir, exist_ok=True)
    
    data = results["privacy_utility"]["results"]
    
    epsilons = []
    accuracies = []
    
    for result in data:
        eps = result["epsilon"]
        if eps == "No DP":
            epsilons.append(float('inf'))
        else:
            epsilons.append(eps)
        accuracies.append(result["final_accuracy"] * 100)
    
    # Sort by epsilon
    sorted_pairs = sorted(zip(epsilons, accuracies))
    epsilons, accuracies = zip(*sorted_pairs)
    
    # Replace inf with label
    epsilon_labels = [f"{e}" if e != float('inf') else "No DP" for e in epsilons]
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(epsilons)), accuracies, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Privacy Budget (ε)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Privacy-Utility Tradeoff', fontsize=14, fontweight='bold')
    plt.xticks(range(len(epsilons)), epsilon_labels)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "privacy_utility.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved privacy-utility plot to {output_file}")
    plt.close()


def plot_byzantine_robustness(results, output_dir="results/experiments/plots"):
    """Plot Byzantine robustness."""
    os.makedirs(output_dir, exist_ok=True)
    
    data = results["byzantine_robustness"]["results"]
    
    # Organize by aggregation method
    methods = {}
    for result in data:
        method = result["aggregation_method"]
        if method not in methods:
            methods[method] = {"rates": [], "accuracies": []}
        methods[method]["rates"].append(result["byzantine_rate"] * 100)
        methods[method]["accuracies"].append(result["final_accuracy"] * 100)
    
    plt.figure(figsize=(10, 6))
    
    for method, data in methods.items():
        # Sort by rate
        sorted_pairs = sorted(zip(data["rates"], data["accuracies"]))
        rates, accs = zip(*sorted_pairs)
        plt.plot(rates, accs, 'o-', label=method.upper(), linewidth=2, markersize=8)
    
    plt.xlabel('Byzantine Client Rate (%)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Byzantine Robustness Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "byzantine_robustness.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved Byzantine robustness plot to {output_file}")
    plt.close()


def plot_scalability(results, output_dir="results/experiments/plots"):
    """Plot scalability analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    data = results["scalability"]["results"]
    
    num_clients = [r["num_clients"] for r in data]
    accuracies = [r["final_accuracy"] * 100 for r in data]
    times = [r["time_per_round"] for r in data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy vs clients
    ax1.plot(num_clients, accuracies, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Number of Clients', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy vs Number of Clients', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Time vs clients
    ax2.plot(num_clients, times, 'o-', linewidth=2, markersize=8, color='red')
    ax2.set_xlabel('Number of Clients', fontsize=12)
    ax2.set_ylabel('Time per Round (s)', fontsize=12)
    ax2.set_title('Training Time vs Number of Clients', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "scalability.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved scalability plot to {output_file}")
    plt.close()


def plot_aggregation_comparison(results, output_dir="results/experiments/plots"):
    """Plot aggregation method comparison."""
    os.makedirs(output_dir, exist_ok=True)
    
    data = results["aggregation_comparison"]["results"]
    
    methods = [r["aggregation_method"].upper() for r in data]
    accuracies = [r["final_accuracy"] * 100 for r in data]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.xlabel('Aggregation Method', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Aggregation Method Comparison', fontsize=14, fontweight='bold')
    plt.ylim([min(accuracies) - 5, max(accuracies) + 5])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "aggregation_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved aggregation comparison plot to {output_file}")
    plt.close()


def generate_all_plots(results_file="results/experiments/experiment_results.json"):
    """Generate all plots."""
    print("="*80)
    print("GENERATING PLOTS FOR PUBLICATION")
    print("="*80)
    
    results = load_results(results_file)
    
    print("\nGenerating plots...")
    
    if "privacy_utility" in results:
        plot_privacy_utility(results)
    
    if "byzantine_robustness" in results:
        plot_byzantine_robustness(results)
    
    if "scalability" in results:
        plot_scalability(results)
    
    if "aggregation_comparison" in results:
        plot_aggregation_comparison(results)
    
    print("\n" + "="*80)
    print("ALL PLOTS GENERATED!")
    print("="*80)
    print("\nPlots saved to: results/experiments/plots/")
    print("  - privacy_utility.png")
    print("  - byzantine_robustness.png")
    print("  - scalability.png")
    print("  - aggregation_comparison.png")


if __name__ == "__main__":
    generate_all_plots()
