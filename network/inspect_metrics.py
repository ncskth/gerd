"""Quick analysis script for inspecting training metrics."""

import argparse
import numpy as np
from pathlib import Path
import pickle
import sys


def find_latest_experiment():
    """Find the most recent experiment folder."""
    results_base = Path('./results')
    if not results_base.exists():
        return None
    
    exp_dirs = sorted([d for d in results_base.iterdir() if d.is_dir() and d.name.startswith('exp_')])
    return exp_dirs[-1] if exp_dirs else None


def print_metrics_summary(results_dir: Path):
    """Print a human-readable summary of training metrics."""
    
    metrics_file = results_dir / 'training_metrics.pkl'
    
    if not metrics_file.exists():
        print(f"❌ Metrics file not found: {metrics_file}")
        return
    
    with open(metrics_file, 'rb') as f:
        metrics = pickle.load(f)
    
    print("\n" + "="*70)
    print("TRAINING METRICS SUMMARY")
    print("="*70)
    
    # Overall statistics
    train_losses = np.array(metrics['train_loss'])
    val_losses = np.array(metrics['val_loss'])
    
    print("\n📊 OVERALL LOSSES")
    print(f"  Training Loss:   START={train_losses[0]:.4f}  END={train_losses[-1]:.4f}  MIN={train_losses.min():.4f}")
    print(f"  Validation Loss: START={val_losses[0]:.4f}  END={val_losses[-1]:.4f}  MIN={val_losses.min():.4f}")
    print(f"  Improvement: {((val_losses[0] - val_losses[-1]) / val_losses[0] * 100):.1f}%")
    
    # L2 distances
    l2_distances = np.array(metrics['val_l2_distance_pixels'])
    print(f"\n📍 L2 DISTANCES (COORDINATE ERROR IN PIXELS)")
    print(f"  START: {l2_distances[0]:.2f}px")
    print(f"  END:   {l2_distances[-1]:.2f}px")
    print(f"  MIN:   {l2_distances.min():.2f}px (best)")
    print(f"  IMPROVEMENT: {((l2_distances[0] - l2_distances[-1]) / l2_distances[0] * 100):.1f}%")
    
    # Loss components
    train_co = np.array(metrics['train_loss_co'])
    train_reg = np.array(metrics['train_loss_reg'])
    val_co = np.array(metrics['val_loss_co'])
    val_reg = np.array(metrics['val_loss_reg'])
    
    print(f"\n🎯 LOSS COMPONENTS")
    print(f"  Coordinate Loss:")
    print(f"    Train: {train_co[0]:.4f} → {train_co[-1]:.4f}")
    print(f"    Val:   {val_co[0]:.4f} → {val_co[-1]:.4f}")
    print(f"  Regularization Loss:")
    print(f"    Train: {train_reg[0]:.4f} → {train_reg[-1]:.4f}")
    print(f"    Val:   {val_reg[0]:.4f} → {val_reg[-1]:.4f}")
    
    # Per-shape analysis
    print(f"\n🔷 PER-SHAPE ANALYSIS (Validation)")
    for shape in ['circle', 'triangle', 'square']:
        losses = np.array(metrics['per_shape_val_loss'][shape])
        l2s = np.array(metrics['per_shape_l2'][shape])
        
        print(f"\n  {shape.upper()}:")
        print(f"    Loss: {losses[0]:.4f} → {losses[-1]:.4f} (min: {losses.min():.4f})")
        print(f"    L2:   {l2s[0]:.2f}px → {l2s[-1]:.2f}px (min: {l2s.min():.2f}px)")
        
        # Determine best performing shape
        if losses.min() < val_losses.min() * 0.5:
            print(f"    ✓ Well-learned")
        elif losses.min() < val_losses.min() * 0.8:
            print(f"    ~ Moderate performance")
        else:
            print(f"    ✗ Struggling")
    
    # Learning rate info
    lrs = np.array(metrics['learning_rates'])
    print(f"\n⚡ LEARNING RATE")
    print(f"  Initial: {lrs[0]:.2e}")
    print(f"  Final:   {lrs[-1]:.2e}")
    print(f"  Min:     {lrs.min():.2e}")
    print(f"  Max:     {lrs.max():.2e}")
    
    # Convergence analysis
    print(f"\n📈 CONVERGENCE ANALYSIS")
    
    # Check if training is converged
    last_10_val = val_losses[-10:] if len(val_losses) >= 10 else val_losses
    val_std = np.std(last_10_val)
    
    if val_std < val_losses[-1] * 0.01:
        print(f"  Status: ✓ CONVERGED (low variance in last 10 epochs)")
    elif val_std < val_losses[-1] * 0.05:
        print(f"  Status: ~ NEAR CONVERGENCE (moderate variance)")
    else:
        print(f"  Status: ✗ NOT CONVERGED (still varying significantly)")
    
    # Overfitting check
    train_val_ratio = (train_losses[-1] / val_losses[-1]) if val_losses[-1] > 0 else 0
    print(f"\n🔍 OVERFITTING CHECK")
    print(f"  Final Train/Val Ratio: {train_val_ratio:.2f}")
    if train_val_ratio < 0.8:
        print(f"  Status: ✓ UNDERFITTING (train loss > val loss)")
    elif train_val_ratio < 1.2:
        print(f"  Status: ✓ GOOD (balanced training)")
    else:
        print(f"  Status: ✗ OVERFITTING (train loss significantly < val loss)")
    
    # Epochs and early stopping
    n_epochs = len(val_losses)
    print(f"\n⏱️  TRAINING DURATION")
    print(f"  Total Epochs: {n_epochs}")
    print(f"  Best Epoch: {val_losses.argmin()} (loss: {val_losses.min():.4f})")
    
    print("\n" + "="*70)


def print_failure_diagnosis(results_dir: Path):
    """Diagnose potential training issues."""
    
    metrics_file = results_dir / 'training_metrics.pkl'
    
    if not metrics_file.exists():
        return
    
    with open(metrics_file, 'rb') as f:
        metrics = pickle.load(f)
    
    issues = []
    
    # Check for exploding gradients
    train_losses = np.array(metrics['train_loss'])
    if np.any(train_losses > 10) or np.any(np.isnan(train_losses)):
        issues.append("⚠️  EXPLODING LOSS: Gradients likely exploding, LR too high")
    
    # Check for oscillations
    if len(train_losses) > 5:
        recent = train_losses[-5:]
        if np.std(recent) > np.mean(recent) * 0.1:
            issues.append("⚠️  OSCILLATING LOSS: Training unstable, consider lower LR")
    
    # Check for triangle performance
    if 'per_shape_l2' in metrics:
        tri_l2 = np.array(metrics['per_shape_l2']['triangle'])
        circle_l2 = np.array(metrics['per_shape_l2']['circle'])
        if tri_l2[-1] > circle_l2[-1] * 2:
            issues.append("⚠️  TRIANGLE STRUGGLING: L2 error 2x worse than circle")
    
    # Check for no convergence
    val_losses = np.array(metrics['val_loss'])
    if val_losses[-1] > val_losses[0] * 0.9:
        issues.append("⚠️  NOT IMPROVING: Validation loss hasn't improved significantly")
    
    if issues:
        print("\n🔴 POTENTIAL ISSUES DETECTED:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ No major issues detected!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect training metrics')
    parser.add_argument('--results-dir', type=Path, default=None,
                       help='Path to experiment directory. If not provided, uses the latest experiment.')
    parser.add_argument('--list-experiments', action='store_true',
                       help='List all available experiments')
    parser.add_argument('--diagnose', action='store_true',
                       help='Run failure diagnosis')
    args = parser.parse_args()
    
    # Handle list experiments
    if args.list_experiments:
        results_base = Path('./results')
        if not results_base.exists():
            print("No experiments found (results directory doesn't exist)")
            exit(0)
        
        exp_dirs = sorted([d for d in results_base.iterdir() if d.is_dir() and d.name.startswith('exp_')])
        if not exp_dirs:
            print("No experiments found")
            exit(0)
        
        print(f"Available experiments ({len(exp_dirs)} total):\n")
        for exp_dir in exp_dirs[-10:]:  # Show last 10
            metrics_file = exp_dir / 'training_metrics.pkl'
            if metrics_file.exists():
                with open(metrics_file, 'rb') as f:
                    metrics = pickle.load(f)
                    epochs = len(metrics.get('val_loss', []))
                    print(f"  ✓ {exp_dir.name:50s} ({epochs} epochs)")
            else:
                print(f"  ✗ {exp_dir.name:50s} (incomplete)")
        
        if len(exp_dirs) > 10:
            print(f"\n  ... and {len(exp_dirs) - 10} more")
        exit(0)
    
    # Determine results directory
    if args.results_dir is None:
        latest = find_latest_experiment()
        if latest is None:
            print("❌ No experiment found. Run 'python train.py' first or specify --results-dir")
            exit(1)
        results_dir = latest
        print(f"📊 Using latest experiment: {results_dir.name}\n")
    else:
        results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        exit(1)
    
    print_metrics_summary(results_dir)
    
    if args.diagnose:
        print_failure_diagnosis(args.results_dir)
