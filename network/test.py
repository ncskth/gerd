"""
Test script for running inference on the model and collecting results.

Usage:
    # Auto-detect latest experiment, use your test data
    python test.py --data-dir /path/to/test/data
    
    # Explicit experiment directory
    python test.py --data-dir /path/to/test/data --experiment-dir ./results/exp_covariant_20260327_123456_abcd1234
    
    # Custom test folder name
    python test.py --data-dir /path/to/test/data --test-name my_test_run
    
    # Override model type (auto-detected from experiment dir name otherwise)
    python test.py --data-dir /path/to/test/data --model non_covariant
"""

import argparse
import glob
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from covariant_net import SPT_Net, make_gaussian_targets
from net import SPT_Net_NonCovariant
from dataloader import SequenceChunkDataset
from loss import JensenShannonLoss, SmoothL1CoordinateLoss
from utils_visualization import (
    plot_losses, plot_l2_distances, plot_per_shape_losses,
    plot_coordinate_error_distribution, plot_summary_dashboard
)

# Device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)


def find_latest_experiment(results_dir='./results'):
    """Find the latest experiment directory by creation time."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return None
    
    exp_dirs = sorted([
        d for d in results_path.iterdir()
        if d.is_dir() and d.name.startswith('exp_')
    ])
    return exp_dirs[-1] if exp_dirs else None


def detect_network_type(exp_dir):
    """Infer network type from experiment directory name (exp_{type}_{timestamp}_{id})."""
    name = Path(exp_dir).name
    parts = name.split('_')
    # Try to match known types greedily from the parts after 'exp'
    if len(parts) >= 2:
        # non_covariant has two words
        if len(parts) >= 3 and parts[1] == 'non' and parts[2] == 'covariant':
            return 'non_covariant'
        if parts[1] == 'curriculum':
            return 'curriculum'
        if parts[1] == 'covariant':
            return 'covariant'
    return None


def build_model(network_type, warmup=20):
    """Instantiate the model matching the given network type."""
    if network_type == 'non_covariant':
        print(f"✓ Model architecture: SPT_Net_NonCovariant")
        return SPT_Net_NonCovariant(warmup=warmup)
    elif network_type == 'curriculum':
        print(f"✓ Model architecture: SPT_Net (curriculum)")
        return SPT_Net(warmup=warmup, curriculum=True)
    else:  # 'covariant' or unrecognised
        print(f"✓ Model architecture: SPT_Net (covariant)")
        return SPT_Net(warmup=warmup, curriculum=False)


def load_checkpoint(checkpoint_path, model, device):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Loaded checkpoint from {checkpoint_path}")
    return model


def create_test_data_loader(data_dir, batch_size=6, start_idx=8, datapoint_length=60):
    """Create data loader for test data."""
    data_path = Path(data_dir)
    file_paths = sorted(glob.glob(str(data_path / '*.dat')))
    
    if not file_paths:
        print(f"ERROR: No .dat files found in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(file_paths)} data files in {data_dir}")
    
    dataset = SequenceChunkDataset(
        file_paths,
        start_idx=start_idx,
        datapoint_length=datapoint_length
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"Created test loader with {len(dataset)} samples, batch_size={batch_size}")
    return dataloader, dataset


def run_inference(model, dataloader, device, warmup=20):
    """Run inference on test data and collect results."""
    
    model.eval()
    
    predictions = []
    ground_truth = []
    heatmaps = []
    losses = []
    l2_distances = []
    per_shape_l2 = {'square': []}
    per_shape_losses = {'square': []}
    
    # Loss functions
    coord_loss_fn = SmoothL1CoordinateLoss(beta=0.1)
    regularization = JensenShannonLoss()
    rectification = nn.Softmax(dim=-1)
    
    print("\n" + "="*70)
    print("Running inference...")
    print("="*70)
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass
            h, pred = model(x)
            
            # Process outputs
            h = h.permute(1, 0, 2, 3, 4)        # (B, T, 3, H, W)
            pred = pred.permute(1, 0, 2, 3)      # (B, T, 3, 2)
            h_used = h[:, warmup:]
            y_used = y[:, warmup:]
            y_used = y_used[..., [1, 0]]
            
            # Normalize coordinates
            res = torch.tensor([300, 300], device=device)
            y_norm = y_used / res
            
            # Store predictions and ground truth (denormalize back to pixels)
            predictions.append(pred.cpu() * 300)  # Convert back to pixel coordinates
            ground_truth.append(y_norm.cpu() * 300)
            heatmaps.append(h_used.cpu())
            
            # Compute losses
            batch_loss_co = coord_loss_fn(pred, y_norm)
            
            _, _, _, H, W = h_used.shape
            h_flat = h_used.flatten(3)
            h_rect = rectification(h_flat).reshape_as(h_used)
            
            gaussian_target = make_gaussian_targets(y_norm * H, H, W, sigma=1.0)
            batch_loss_reg = regularization(
                h_rect.reshape(-1, H, W),
                gaussian_target.reshape(-1, H, W)
            ) / x.shape[0]
            
            batch_loss = batch_loss_co + 0.1 * batch_loss_reg
            losses.append(batch_loss.item())
            
            # L2 distances
            l2_distance = torch.norm(pred - y_norm, dim=-1).mean().item() * 300
            l2_distances.append(l2_distance)
            
            # Per-shape metrics (only square)
            shape_pred = pred[..., 0, :]
            shape_gt = y_norm[..., 0, :]
            shape_l2 = torch.norm(shape_pred - shape_gt, dim=-1).mean().item() * 300
            per_shape_l2['square'].append(shape_l2)
            shape_loss = coord_loss_fn(shape_pred.unsqueeze(-2), shape_gt.unsqueeze(-2))
            per_shape_losses['square'].append(shape_loss.item())
            
            print(f"Batch {batch_idx+1}/{len(dataloader)}: "
                  f"Loss={batch_loss.item():.4f}, L2={l2_distance:.2f}px", end="\r")
    
    print("\n✓ Inference completed")
    
    # Aggregate results
    results = {
        'predictions': torch.cat(predictions, dim=0),      # (B*T, 3, 2)
        'ground_truth': torch.cat(ground_truth, dim=0),    # (B*T, 3, 2)
        'heatmaps': torch.cat(heatmaps, dim=0),            # (B*T, 3, H, W)
        'losses': losses,
        'l2_distances': l2_distances,
        'per_shape_l2': per_shape_l2,
        'per_shape_losses': per_shape_losses,
        'metadata': {
            'num_batches': len(dataloader),
            'warmup': warmup,
            'device': str(device),
        }
    }
    
    return results


def compute_test_metrics(results):
    """Compute aggregate metrics from test results."""
    metrics = {
        'total_loss': float(np.mean(results['losses'])),
        'total_l2_mean': float(np.mean(results['l2_distances'])),
        'total_l2_std': float(np.std(results['l2_distances'])),
        'total_l2_min': float(np.min(results['l2_distances'])),
        'total_l2_max': float(np.max(results['l2_distances'])),
    }
    
    # Per-shape metrics
    for shape_name in ['square']:
        shape_l2_values = results['per_shape_l2'][shape_name]
        shape_loss_values = results['per_shape_losses'][shape_name]
        metrics[f'{shape_name}_l2_mean'] = float(np.mean(shape_l2_values))
        metrics[f'{shape_name}_l2_std'] = float(np.std(shape_l2_values))
        metrics[f'{shape_name}_l2_min'] = float(np.min(shape_l2_values))
        metrics[f'{shape_name}_l2_max'] = float(np.max(shape_l2_values))
        metrics[f'{shape_name}_loss_mean'] = float(np.mean(shape_loss_values))
    
    return metrics


def save_test_results(test_dir, results, metrics):
    """Save test results to disk."""
    
    test_dir = Path(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Save pickled data
    with open(test_dir / 'predictions.pkl', 'wb') as f:
        pickle.dump(results['predictions'], f)
    
    with open(test_dir / 'ground_truth.pkl', 'wb') as f:
        pickle.dump(results['ground_truth'], f)
    
    with open(test_dir / 'heatmaps.pkl', 'wb') as f:
        pickle.dump(results['heatmaps'], f)
    
    # Save meta results
    meta_results = {
        'losses': results['losses'],
        'l2_distances': results['l2_distances'],
        'per_shape_l2': results['per_shape_l2'],
        'per_shape_losses': results['per_shape_losses'],
        'metrics': metrics,
        'metadata': results['metadata']
    }
    
    with open(test_dir / 'test_results.pkl', 'wb') as f:
        pickle.dump(meta_results, f)
    
    # Save metrics as text
    with open(test_dir / 'test_metrics.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("TEST RESULTS SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Metadata:\n")
        f.write(f"  Num batches: {results['metadata']['num_batches']}\n")
        f.write(f"  Warmup frames: {results['metadata']['warmup']}\n")
        f.write(f"  Device: {results['metadata']['device']}\n\n")
        f.write("Overall Metrics:\n")
        f.write(f"  Total Loss: {metrics['total_loss']:.4f}\n")
        f.write(f"  L2 Error (mean): {metrics['total_l2_mean']:.2f} px\n")
        f.write(f"  L2 Error (std):  {metrics['total_l2_std']:.2f} px\n")
        f.write(f"  L2 Error (min):  {metrics['total_l2_min']:.2f} px\n")
        f.write(f"  L2 Error (max):  {metrics['total_l2_max']:.2f} px\n\n")
        f.write("Per-Shape Metrics:\n")
        for shape_name in ['square']:
            f.write(f"\n  {shape_name.upper()}:\n")
            f.write(f"    L2 Error (mean): {metrics[f'{shape_name}_l2_mean']:.2f} px\n")
            f.write(f"    L2 Error (std):  {metrics[f'{shape_name}_l2_std']:.2f} px\n")
            f.write(f"    L2 Error (min):  {metrics[f'{shape_name}_l2_min']:.2f} px\n")
            f.write(f"    L2 Error (max):  {metrics[f'{shape_name}_l2_max']:.2f} px\n")
            f.write(f"    Loss (mean):     {metrics[f'{shape_name}_loss_mean']:.4f}\n")
        f.write("\n" + "="*70 + "\n")
    
    print(f"\n✓ Saved results to {test_dir}")
    print(f"  - predictions.pkl")
    print(f"  - ground_truth.pkl")
    print(f"  - heatmaps.pkl")
    print(f"  - test_results.pkl")
    print(f"  - test_metrics.txt")


def generate_test_visualizations(test_dir, results):
    """Generate visualizations for test results."""
    
    plots_dir = Path(test_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("Generating test visualizations...")
    print("="*70)
    
    # Plot L2 distances
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Overall L2 distance over samples
        axes[0, 0].plot(results['l2_distances'], 'b-', linewidth=1)
        axes[0, 0].set_title('L2 Distance Over Test Samples')
        axes[0, 0].set_xlabel('Sample Index')
        axes[0, 0].set_ylabel('L2 Error (pixels)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Per-shape L2 distances (only square)
        axes[0, 1].plot(results['per_shape_l2']['square'], label='square', linewidth=1)
        axes[0, 1].set_title('Per-Shape L2 Distances')
        axes[0, 1].set_xlabel('Sample Index')
        axes[0, 1].set_ylabel('L2 Error (pixels)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # L2 distance histogram
        axes[1, 0].hist(results['l2_distances'], bins=30, alpha=0.7, color='blue')
        axes[1, 0].set_title('L2 Distance Distribution')
        axes[1, 0].set_xlabel('L2 Error (pixels)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Per-shape L2 distribution (only square)
        axes[1, 1].hist(results['per_shape_l2']['square'], bins=20, alpha=0.5, label='square')
        axes[1, 1].set_title('Per-Shape L2 Distribution')
        axes[1, 1].set_xlabel('L2 Error (pixels)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'test_l2_distances.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Generated test_l2_distances.png")
        
        # Coordinate error analysis
        pred = results['predictions']  # (B*T, 3, 2)
        gt = results['ground_truth']    # (B*T, 3, 2)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Only square
        pred_shape = pred[:, 0, :]
        gt_shape = gt[:, 0, :]
        error_x = (pred_shape[:, 0] - gt_shape[:, 0]).numpy()
        error_y = (pred_shape[:, 1] - gt_shape[:, 1]).numpy()
        axes[0].scatter(error_x, error_y, alpha=0.5, s=20)
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        axes[0].set_title('Square Coordinate Errors')
        axes[0].set_xlabel('Error X (pixels)')
        axes[0].set_ylabel('Error Y (pixels)')
        axes[0].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'coordinate_errors.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Generated coordinate_errors.png")
        
    except Exception as e:
        print(f"✗ Error generating visualizations: {e}")
    
    print(f"✓ Visualizations saved to {plots_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Run test inference on trained model'
    )
    parser.add_argument(
        '--data-dir', '-d', required=True,
        help='Directory containing test .dat files'
    )
    parser.add_argument(
        '--experiment-dir', '-e', default=None,
        help='Experiment directory (auto-detect latest if not provided)'
    )
    parser.add_argument(
        '--batch-size', '-b', type=int, default=6,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--datapoint-length', type=int, default=60,
        help='Datapoint length'
    )
    parser.add_argument(
        '--start-idx', type=int, default=8,
        help='Start index for data chunks'
    )
    parser.add_argument(
        '--model', '-m',
        choices=['covariant', 'non_covariant', 'curriculum'],
        default=None,
        help='Model architecture to load. Auto-detected from experiment dir name if not provided.'
    )
    parser.add_argument(
        '--use-test-split', action='store_true',
        help='If set and using the original training data path, use only the test set split. Otherwise, use all files.'
    )
    
    args = parser.parse_args()
    
    # Find or validate experiment directory
    if args.experiment_dir is None:
        exp_dir = find_latest_experiment()
        if exp_dir is None:
            print("ERROR: No experiment directory found. "
                  "Please provide --experiment-dir or run training first.")
            sys.exit(1)
        print(f"\n✓ Using latest experiment: {exp_dir.name}")
    else:
        exp_dir = Path(args.experiment_dir)
        if not exp_dir.exists():
            print(f"ERROR: Experiment directory not found: {exp_dir}")
            sys.exit(1)
        print(f"\n✓ Using experiment: {exp_dir.name}")
    
    # Verify checkpoint exists
    checkpoint_path = exp_dir / 'best_model.pt'
    if not checkpoint_path.exists():
        print(f"ERROR: Best model checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Generate test folder name from data directory basename
    data_dir_basename = Path(args.data_dir).name
    if args.use_test_split:
        test_dir = exp_dir / 'test-s1.0'
    else:
        test_dir = exp_dir / f"test-{data_dir_basename}"
    
    # Determine if using original training data path and flag is set
    original_train_path = '../../../data/gerd/example-s1.0-v1.00-p0.25'
    use_test_split = (
        args.use_test_split and str(Path(args.data_dir).resolve()) == str(Path(original_train_path).resolve())
    )

    print("\n" + "="*70)
    print("TEST CONFIGURATION")
    print("="*70)
    print(f"Experiment:       {exp_dir.name}")
    print(f"Test folder:      test-s1.0")
    print(f"Data directory:   {args.data_dir}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Datapoint length: {args.datapoint_length}")
    print(f"Output directory: {test_dir}")
    print(f"Use test split:   {use_test_split}")
    print("="*70)
    
    # Determine network type
    if args.model is not None:
        network_type = args.model
        print(f"\n✓ Network type: {network_type} (from --model argument)")
    else:
        network_type = detect_network_type(exp_dir)
        if network_type is not None:
            print(f"\n✓ Network type: {network_type} (auto-detected from experiment directory name)")
        else:
            network_type = 'covariant'
            print(f"\n⚠ Could not detect network type from directory name; "
                  f"defaulting to 'covariant'. Use --model to override.")

    # Load model
    print("\n✓ Loading model...")
    model = build_model(network_type, warmup=20).to(device)
    model = load_checkpoint(checkpoint_path, model, device)
    
    # Create test dataloader
    print("\n✓ Creating test dataloader...")
    if use_test_split:
        # Use only the test set indices from the original split
        file_paths = sorted(glob.glob(str(Path(args.data_dir) / '*.dat')))
        n = len(file_paths)
        train_size = int(0.7 * n)
        val_size = int(0.15 * n)
        test_size = n - train_size - val_size
        test_files = file_paths[train_size + val_size:]
        dataset = SequenceChunkDataset(
            test_files,
            start_idx=args.start_idx,
            datapoint_length=args.datapoint_length
        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        print(f"Using only test set: {len(test_files)} files")
    else:
        dataloader, dataset = create_test_data_loader(
            args.data_dir,
            batch_size=args.batch_size,
            start_idx=args.start_idx,
            datapoint_length=args.datapoint_length
        )
    
    # Run inference
    results = run_inference(model, dataloader, device, warmup=20)
    
    # Compute metrics
    print("\n✓ Computing metrics...")
    metrics = compute_test_metrics(results)
    
    # Save results
    save_test_results(test_dir, results, metrics)
    
    # Generate visualizations
    generate_test_visualizations(test_dir, results)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"\nOverall Metrics:")
    print(f"  Total Loss:        {metrics['total_loss']:.4f}")
    print(f"  L2 Error (mean):   {metrics['total_l2_mean']:.2f} px")
    print(f"  L2 Error (std):    {metrics['total_l2_std']:.2f} px")
    print(f"  L2 Error (min):    {metrics['total_l2_min']:.2f} px")
    print(f"  L2 Error (max):    {metrics['total_l2_max']:.2f} px")
    
    print(f"\nPer-Shape Metrics:")
    for shape_name in ['square']:
        print(f"\n  {shape_name.upper()}:")
        print(f"    L2 Error (mean): {metrics[f'{shape_name}_l2_mean']:.2f} px")
        print(f"    L2 Error (std):  {metrics[f'{shape_name}_l2_std']:.2f} px")
        print(f"    Loss (mean):     {metrics[f'{shape_name}_loss_mean']:.4f}")
    
    print("\n" + "="*70)
    print(f"✓ Test completed!")
    print(f"✓ Results saved to: {test_dir.absolute()}")
    print(f"\nTo view results:")
    print(f"  $ cat {test_dir / 'test_metrics.txt'}")
    print(f"  $ open {test_dir / 'plots' / 'test_l2_distances.png'}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
