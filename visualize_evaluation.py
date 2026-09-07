#!/usr/bin/env python3
"""
Visualize baseline evaluation results with plots and tables.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_results(json_file):
    """Load evaluation results"""
    with open(json_file, 'r') as f:
        return json.load(f)

def plot_results(results, output_dir=None):
    """Create visualization plots"""
    summary = results['summary']
    case_results = results['case_results']
    
    # Prepare data
    case_names = sorted(case_results.keys())
    metrics_names = ['DICE', 'Sensitivity', 'Specificity', 'IoU']
    
    data = {metric: [case_results[case][metric] for case in case_names] 
            for metric in metrics_names}
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('nnUNet Baseline Performance on Test Set (25 cases)', fontsize=16, fontweight='bold')
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    # Plot 1: Box plot of all metrics
    ax = axes[0, 0]
    box_data = [data[m] for m in metrics_names]
    bp = ax.boxplot(box_data, labels=metrics_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Score')
    ax.set_title('Metric Distributions')
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Bar chart with error bars
    ax = axes[0, 1]
    x_pos = np.arange(len(metrics_names))
    means = [summary[m]['mean'] for m in metrics_names]
    stds = [summary[m]['std'] for m in metrics_names]
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=colors)
    ax.set_ylabel('Score')
    ax.set_title('Mean ± Std Dev')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.3f}\n±{std:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 3: DICE scores per case
    ax = axes[1, 0]
    dice_scores = data['DICE']
    sorted_indices = np.argsort(dice_scores)
    sorted_names = [case_names[i] for i in sorted_indices]
    sorted_dice = [dice_scores[i] for i in sorted_indices]
    
    colors_dice = ['#e74c3c' if d < 0.75 else '#f39c12' if d < 0.85 else '#2ecc71' for d in sorted_dice]
    ax.barh(range(len(sorted_names)), sorted_dice, color=colors_dice, alpha=0.7)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel('DICE Score')
    ax.set_title('Per-Case DICE Scores')
    ax.set_xlim([0, 1])
    ax.axvline(x=summary['DICE']['mean'], color='red', linestyle='--', linewidth=2, label='Mean')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # Plot 4: Sensitivity vs Specificity scatter
    ax = axes[1, 1]
    sens = data['Sensitivity']
    spec = data['Specificity']
    scatter = ax.scatter(spec, sens, s=100, alpha=0.6, c=data['DICE'], cmap='RdYlGn', 
                        edgecolors='black', linewidth=0.5, vmin=0.7, vmax=1.0)
    
    # Add diagonal line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title('Sensitivity vs Specificity (colored by DICE)')
    ax.set_xlim([0.85, 1.01])
    ax.set_ylim([0.6, 1.01])
    ax.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('DICE')
    
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig_path = output_dir / "evaluation_plots.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plots saved to {fig_path}")
    
    plt.show()

def print_summary_table(results):
    """Print formatted summary table"""
    summary = results['summary']
    
    print("\n" + "="*90)
    print("BASELINE MODEL EVALUATION SUMMARY")
    print("="*90)
    print(f"\n{'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'N':>5}")
    print("-"*90)
    
    for metric in sorted(summary.keys()):
        stats = summary[metric]
        print(f"{metric:<20} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
              f"{stats['min']:>10.4f} {stats['max']:>10.4f} {stats['n']:>5}")
    
    print("="*90)

def print_case_details(results, top_n=5):
    """Print per-case details"""
    case_results = results['case_results']
    
    print("\n" + "="*90)
    print("PER-CASE RESULTS")
    print("="*90)
    print(f"\n{'Case':<25} {'DICE':>8} {'Sens':>8} {'Spec':>8} {'IoU':>8} {'Vol Ratio':>12}")
    print("-"*90)
    
    # Sort by DICE
    sorted_cases = sorted(case_results.items(), key=lambda x: x[1]['DICE'], reverse=True)
    
    print("\n--- Top Cases ---")
    for case, metrics in sorted_cases[:top_n]:
        print(f"{case:<25} {metrics['DICE']:>8.4f} {metrics['Sensitivity']:>8.4f} "
              f"{metrics['Specificity']:>8.4f} {metrics['IoU']:>8.4f} {metrics['Volume_Ratio']:>12.3f}")
    
    print("\n--- Bottom Cases ---")
    for case, metrics in sorted_cases[-top_n:]:
        print(f"{case:<25} {metrics['DICE']:>8.4f} {metrics['Sensitivity']:>8.4f} "
              f"{metrics['Specificity']:>8.4f} {metrics['IoU']:>8.4f} {metrics['Volume_Ratio']:>12.3f}")

def main():
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument("--json", type=str,
                       default="/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/evaluation/test_predictions_baseline/evaluation_results.json",
                       help="Path to evaluation_results.json")
    parser.add_argument("--output-dir", type=str,
                       help="Directory to save plots")
    parser.add_argument("--no-plot", action="store_true",
                       help="Skip plotting")
    
    args = parser.parse_args()
    
    if not Path(args.json).exists():
        print(f"Error: {args.json} not found")
        print("\nRun evaluation first:")
        print("  ./eval_baseline.sh sbatch    # or")
        print("  ./eval_baseline.sh run       # or")
        print("  ./eval_baseline.sh quick")
        return
    
    print(f"Loading results from {args.json}...")
    results = load_results(args.json)
    
    print_summary_table(results)
    print_case_details(results, top_n=3)
    
    if not args.no_plot:
        output_dir = args.output_dir or Path(args.json).parent
        plot_results(results, output_dir)

if __name__ == "__main__":
    main()
