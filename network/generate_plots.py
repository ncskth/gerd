"""Generate visualizations from saved training metrics."""

import argparse
from pathlib import Path
import sys
from datetime import datetime

from utils_visualization import (
    TrainingLogger, plot_losses, plot_l2_distances, plot_per_shape_losses,
    plot_learning_rate_schedule, plot_coordinate_error_distribution,
    plot_summary_dashboard, plot_heatmap_evolution, plot_gabor_filters
)


def find_latest_experiment():
    """Find the most recent experiment folder."""
    results_base = Path('./results')
    if not results_base.exists():
        return None
    
    exp_dirs = sorted([d for d in results_base.iterdir() if d.is_dir() and d.name.startswith('exp_')])
    return exp_dirs[-1] if exp_dirs else None


def main():
    parser = argparse.ArgumentParser(description='Generate visualizations from training results')
    parser.add_argument('--results-dir', type=Path, default=None,
                       help='Path to experiment directory containing training_metrics.pkl. '
                            'If not provided, uses the latest experiment.')
    parser.add_argument('--list-experiments', action='store_true',
                       help='List all available experiments')
    parser.add_argument('--plots', nargs='+', default='all',
                       choices=['losses', 'l2', 'shapes', 'lr', 'errors', 'dashboard', 'all'],
                       help='Which plots to generate')
    args = parser.parse_args()
    
    # Handle list experiments
    if args.list_experiments:
        results_base = Path('./results')
        if not results_base.exists():
            print("No experiments found (results directory doesn't exist)")
            return
        
        exp_dirs = sorted([d for d in results_base.iterdir() if d.is_dir() and d.name.startswith('exp_')])
        if not exp_dirs:
            print("No experiments found")
            return
        
        print(f"Available experiments ({len(exp_dirs)} total):")
        for exp_dir in exp_dirs[-10:]:  # Show last 10
            metrics_file = exp_dir / 'training_metrics.pkl'
            if metrics_file.exists():
                print(f"  ✓ {exp_dir.name}")
            else:
                print(f"  ✗ {exp_dir.name} (incomplete)")
        
        if len(exp_dirs) > 10:
            print(f"  ... and {len(exp_dirs) - 10} more")
        return
    
    # Determine results directory
    if args.results_dir is None:
        latest = find_latest_experiment()
        if latest is None:
            print("❌ No experiment found. Run 'python train.py' first or specify --results-dir")
            sys.exit(1)
        results_dir = latest
        print(f"Using latest experiment: {results_dir.name}")
    else:
        results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        sys.exit(1)
    
    viz_dir = results_dir / 'plots'
    viz_dir.mkdir(exist_ok=True)
    
    # Load metrics
    logger = TrainingLogger(results_dir)
    logger.load_metrics()
    
    print("="*60)
    print(f"Generating visualizations from: {results_dir}")
    print("="*60)
    
    plots_to_generate = args.plots if args.plots != 'all' else \
        ['losses', 'l2', 'shapes', 'lr', 'errors', 'dashboard']
    
    if 'losses' in plots_to_generate or 'all' in plots_to_generate:
        print("\n📊 Plotting losses...")
        plot_losses(logger, save_path=viz_dir / 'losses.png')
    
    if 'l2' in plots_to_generate or 'all' in plots_to_generate:
        print("📊 Plotting L2 distances...")
        plot_l2_distances(logger, save_path=viz_dir / 'l2_distances.png')
    
    if 'shapes' in plots_to_generate or 'all' in plots_to_generate:
        print("📊 Plotting per-shape losses...")
        plot_per_shape_losses(logger, save_path=viz_dir / 'per_shape_losses.png')
    
    if 'lr' in plots_to_generate or 'all' in plots_to_generate:
        print("📊 Plotting learning rate schedule...")
        plot_learning_rate_schedule(logger, save_path=viz_dir / 'lr_schedule.png')
    
    if 'errors' in plots_to_generate or 'all' in plots_to_generate:
        print("📊 Plotting coordinate error distribution...")
        # This requires coordinate history which may not be available in saved metrics
        print("⚠️  Coordinate error plot requires coordinate history (not saved by default)")
    
    if 'dashboard' in plots_to_generate or 'all' in plots_to_generate:
        print("📊 Plotting summary dashboard...")
        plot_summary_dashboard(logger, save_path=viz_dir / 'summary_dashboard.png')
    
    print("\n" + "="*60)
    print(f"✓ All visualizations saved to: {viz_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
