import glob
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import uuid

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from covariant_net import SPT_Net, make_gaussian_targets
from net import SPT_Net_NonCovariant
from dataloader import SequenceChunkDataset
from loss import JensenShannonLoss, SmoothL1CoordinateLoss
from utils import plot_predictions
from utils_visualization import (
    TrainingLogger, plot_losses, plot_l2_distances, plot_per_shape_losses,
    plot_learning_rate_schedule, plot_coordinate_error_distribution,
    plot_summary_dashboard
)


# ===== COMMAND LINE ARGUMENTS =====
parser = argparse.ArgumentParser(description='Train SPT_Net model')
parser.add_argument(
    '--model', '-m',
    choices=['original', 'non_covariant'],
    default='original',
    help='Model variant: original (SPT_Net) or non_covariant (SPT_Net_NonCovariant)'
)
parser.add_argument(
    '--curriculum', 
    action='store_true',
    help='Enable curriculum learning for time constants in SPT_Net'
)
parser.add_argument(
    '--no-save',
    action='store_true',
    help='Disable saving results, metrics, and plots (for quick debugging)'
)
parser.add_argument(
    '--resume-experiment', '-r',
    type=str,
    default=None,
    help='Resume training from an existing experiment folder (loads latest checkpoint).'
)
args = parser.parse_args()

# Device and reproducibility
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# Data loading
path_prefix = '../../../data/gerd/example-s1.0-v1.00-p0.25'
file_paths = glob.glob(f'{path_prefix}/*.dat')

dataset = SequenceChunkDataset(
    file_paths,
    start_idx=8,
    datapoint_length=60
)

n = len(dataset)
train_size = int(0.7 * n)
val_size = int(0.15 * n)
test_size = n - train_size - val_size

train_ds, val_ds, test_ds = random_split(
    dataset,
    [train_size, val_size, test_size]
)

batch_size = 10
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

# Model setup
warmup = 20
epochs = 50

if args.model == 'non_covariant':
    model = SPT_Net_NonCovariant(warmup=warmup).to(device)
    model_name = 'SPT_Net_NonCovariant (3x params, no time scales)'
    network_type = 'non_covariant'
else:
    # Original SPT_Net with optional curriculum learning
    model = SPT_Net(warmup=warmup, curriculum=args.curriculum, total_epochs=epochs).to(device)
    if args.curriculum:
        model_name = 'SPT_Net (Curriculum)'
        network_type = 'curriculum'
    else:
        model_name = 'SPT_Net (Original)'
        network_type = 'covariant'

# Training configuration
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-5)

# Learning rate scheduler with linear warmup + exponential decay
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda epoch: min(1.0, (epoch + 1) / 5) * (0.95 ** max(0, epoch - 4))
)

# Loss functions
regularization = JensenShannonLoss()
coord_loss_fn = SmoothL1CoordinateLoss(beta=0.1)
rectification = nn.Softmax(dim=-1)

# Regularization weight schedule
reg_scale_start = 0
reg_scale_end = 0

# Early stopping
patience = 100
best_val_loss = float('inf')  # Track for early stopping even in no-save mode
patience_count = 0


def find_latest_checkpoint(exp_dir: Path):
    """Find latest checkpoint file and return (path, epoch)."""
    checkpoint_paths = sorted(exp_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoint files found in: {exp_dir}")

    latest_path = checkpoint_paths[-1]
    epoch_token = latest_path.stem.split("_")[-1]
    try:
        latest_epoch = int(epoch_token)
    except ValueError as exc:
        raise ValueError(f"Could not parse epoch from checkpoint name: {latest_path.name}") from exc

    return latest_path, latest_epoch

# ===== LOGGING AND TRACKING =====
start_epoch = 0
resume_checkpoint_path = None

if args.resume_experiment and args.no_save:
    raise ValueError("Cannot use --resume-experiment together with --no-save.")

if not args.no_save:
    if args.resume_experiment:
        results_dir = Path(args.resume_experiment)
        if not results_dir.exists() or not results_dir.is_dir():
            raise FileNotFoundError(f"Resume experiment folder not found: {results_dir}")
        experiment_name = results_dir.name
        resume_checkpoint_path, latest_epoch = find_latest_checkpoint(results_dir)
    else:
        # Generate unique experiment folder with timestamp + random ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        experiment_name = f"exp_{network_type}_{timestamp}_{unique_id}"
        results_dir = Path('./results') / experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)
    
    logger = TrainingLogger(results_dir)

    if args.resume_experiment:
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            scheduler.last_epoch = checkpoint['epoch']

        if 'metrics' in checkpoint:
            logger.metrics = checkpoint['metrics']
        else:
            logger.load_metrics()

        val_history = logger.metrics.get('val_loss', [])
        inferred_best = min(val_history) if val_history else float('inf')
        best_val_loss = checkpoint.get('best_val_loss', inferred_best)
        logger.best_val_loss = best_val_loss

        if 'patience_count' in checkpoint:
            patience_count = int(checkpoint['patience_count'])
        else:
            if val_history:
                best_idx = int(np.argmin(val_history))
                patience_count = max(0, len(val_history) - best_idx - 1)
            else:
                patience_count = 0

        start_epoch = int(checkpoint['epoch']) + 1

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n{'='*70}")
        print("RESUMING TRAINING")
        print(f"{'='*70}")
        print(f"Experiment: {experiment_name}")
        print(f"Checkpoint: {resume_checkpoint_path.name}")
        print(f"Resuming at epoch: {start_epoch}")
        print(f"Current LR: {current_lr:.6g}")
        print(f"Best val loss so far: {best_val_loss:.6f}")
        print(f"Patience counter: {patience_count}/{patience}")
        print(f"{'='*70}\n")
    
    # Storage for heatmaps and coordinates (every N epochs for memory efficiency)
    checkpoint_interval = 5
    heatmaps_history = {}
    coords_history = {}
    gt_coords_history = {}
    print(f"\n{'='*70}")
    print(f"TRAINING SESSION")
    print(f"{'='*70}")
    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Epochs: {epochs}")
    print(f"Results directory: {results_dir}")
    print(f"{'='*70}\n")
else:
    # No saving mode - use dummy logger and skip all I/O
    logger = None
    results_dir = None
    checkpoint_interval = float('inf')  # Never save checkpoints
    heatmaps_history = {}
    coords_history = {}
    gt_coords_history = {}
    experiment_name = "debug_no_save"


# ===== TRAINING LOOP =====
try:
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        train_loss_co = 0
        train_loss_reg = 0
        train_l2_distance = 0
        batch_count = 0

        for idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            h, pred = model(x, epoch)

            h = h.permute(1, 0, 2, 3, 4)   # (B, T, N, H, W)
            pred = pred.permute(1, 0, 2, 3)
            h_used = h[:, warmup:]
            y_used = y[:, warmup:]
            y_used = y_used[..., [1, 0]]
            _, _, _, H, W = h_used.shape
            h_flat = h_used.flatten(3)
            h_rect = rectification(h_flat).reshape_as(h_used)

            res = torch.tensor([300, 300], device=device)
            y_norm = y_used / res

            # Coordinate loss
            loss_co = coord_loss_fn(pred, y_norm)
            
            # Gaussian targets with curriculum learning
            progress = epoch / epochs
            # sigma = 0.5 + 0.5 * progress
            sigma = 0.5
            gaussian_target = make_gaussian_targets(y_norm * H, H, W, sigma=sigma)
            
            # Regularization loss
            loss_reg = regularization(
                h_rect.reshape(-1, H, W),
                gaussian_target.reshape(-1, H, W)
            ) / batch_size
            
            # Combined loss
            reg_weight = reg_scale_start + (reg_scale_end - reg_scale_start) * progress
            loss = loss_co + reg_weight * loss_reg
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track losses
            train_loss += loss.item()
            train_loss_co += loss_co.item()
            train_loss_reg += loss_reg.item()
            
            # L2 distance in pixels
            l2_distance_pixels = torch.norm(pred - y_norm, dim=-1).mean().item() * 300
            train_l2_distance += l2_distance_pixels
            batch_count += 1

            # Log current LR
            current_lr = optimizer.param_groups[0]['lr']

            # Visualize periodically
            if idx % 10 == 0:
                plot_predictions(h, pred, y_used, x, network_type)

            print(
                f"Epoch {epoch}/{epochs} [{idx+1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} (co={loss_co.item():.4f}, reg={loss_reg.item():.4f}), "
                f"L2={l2_distance_pixels:.2f}px, σ={sigma:.2f}",
                end="\r"
            )

        # Average training metrics
        train_loss /= batch_count
        train_loss_co /= batch_count
        train_loss_reg /= batch_count
        train_l2_distance /= batch_count
        
        # Log training epoch metrics
        if not args.no_save:
            logger.log_train_epoch(train_loss, train_loss_co, train_loss_reg, 
                                 train_l2_distance, current_lr)
        
        scheduler.step()

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0
        val_loss_co = 0
        val_loss_reg = 0
        val_l2_distance = 0
        # per_shape_loss = {'circle': 0, 'triangle': 0, 'square': 0}
        # per_shape_l2 = {'circle': 0, 'triangle': 0, 'square': 0}
        # per_shape_count = {'circle': 0, 'triangle': 0, 'square': 0}

        per_shape_loss = {'square': 0}
        per_shape_l2 = {'square': 0}
        per_shape_count = {'square': 0}

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x = x.to(device)
                y = y.to(device)

                h, pred = model(x, epoch)

                h = h.permute(1, 0, 2, 3, 4)
                h_used = h[:, warmup:]
                y_used = y[:, warmup:]
                y_used = y_used[..., [1, 0]]
                pred = pred.permute(1, 0, 2, 3)

                res = torch.tensor([300, 300], device=device)
                y_norm = y_used / res
                
                # Total coordinate loss
                batch_loss_co = coord_loss_fn(pred, y_norm)
                val_loss_co += batch_loss_co.item()
                
                # Regularization loss
                _, _, _, H_val, W_val = h_used.shape
                h_flat_val = h_used.flatten(3)
                h_rect_val = rectification(h_flat_val).reshape_as(h_used)
                progress_val = epoch / epochs
                sigma_val = 0.5 + 1.5 * progress_val
                gaussian_target_val = make_gaussian_targets(y_norm * H_val, H_val, W_val, sigma=sigma_val)
                batch_loss_reg = regularization(
                    h_rect_val.reshape(-1, H_val, W_val),
                    gaussian_target_val.reshape(-1, H_val, W_val)
                ) / batch_size
                val_loss_reg += batch_loss_reg.item()
                
                reg_weight_val = reg_scale_start + (reg_scale_end - reg_scale_start) * progress_val
                batch_loss = batch_loss_co + reg_weight_val * batch_loss_reg
                val_loss += batch_loss.item()
                
                # L2 distance
                l2_distance = torch.norm(pred - y_norm, dim=-1).mean().item() * 300
                val_l2_distance += l2_distance

                # Per-shape metrics
                # Only square
                shape_pred = pred[..., 0, :]
                shape_gt = y_norm[..., 0, :]
                shape_l2 = torch.norm(shape_pred - shape_gt, dim=-1).mean().item() * 300
                per_shape_l2['square'] += shape_l2
                per_shape_count['square'] += 1
                shape_loss = coord_loss_fn(shape_pred.unsqueeze(-2), shape_gt.unsqueeze(-2))
                per_shape_loss['square'] += shape_loss.item()
                
                # Store heatmaps and coordinates for visualization (every checkpoint_interval)
                if epoch % checkpoint_interval == 0:
                    if epoch not in heatmaps_history:
                        heatmaps_history[epoch] = []
                        coords_history[epoch] = []
                        gt_coords_history[epoch] = []
                    
                    heatmaps_history[epoch].append(h_used)
                    coords_history[epoch].append(pred)
                    gt_coords_history[epoch].append(y_norm)

        # Average validation metrics
        val_loss /= len(val_loader)
        val_loss_co /= len(val_loader)
        val_loss_reg /= len(val_loader)
        val_l2_distance /= len(val_loader)
        
        for shape in per_shape_loss:
            per_shape_loss[shape] /= len(val_loader)
            per_shape_l2[shape] /= len(val_loader)
        
        # Log validation metrics
        if not args.no_save:
            logger.log_validation(val_loss, val_loss_co, val_loss_reg, val_l2_distance,
                                per_shape_loss, per_shape_l2)
        
        # Checkpoint management
        if not args.no_save:
            is_best = val_loss < logger.best_val_loss
            if is_best:
                patience_count = 0
                best_val_loss = val_loss
                logger.best_val_loss = val_loss
            else:
                patience_count += 1
            
            logger.save_checkpoint(
                epoch,
                model,
                optimizer,
                scheduler=scheduler,
                is_best=is_best,
                patience_count=patience_count,
                best_val_loss=best_val_loss,
            )
            logger.save_metrics()  # Save after each epoch so partial training is preserved
        else:
            # Still track patience for early stopping even in no-save mode
            if val_loss < best_val_loss:
                patience_count = 0
                best_val_loss = val_loss
            else:
                patience_count += 1

        # Print epoch summary
        print(f"\nEpoch {epoch:3d} | Train: {train_loss:.4f} (co={train_loss_co:.4f}, reg={train_loss_reg:.4f}) "
            f"| Val: {val_loss:.4f} (co={val_loss_co:.4f}, reg={val_loss_reg:.4f}) "
            f"| L2: {val_l2_distance:.2f}px | "
            f"Per-shape L2 - Square: {per_shape_l2['square']:.2f}px")
        
        # Early stopping
        if patience_count >= patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={patience}/{patience})")
            break

except Exception as e:
    print(f"\n⚠️ Training interrupted by exception: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Ensure metrics are saved even if the run is interrupted (only if saving enabled)
    if not args.no_save:
        logger.save_metrics()

print("\n✓ Training completed!")
if not args.no_save:
    print(f"✓ Best val loss: {logger.best_val_loss:.4f}")
    print(f"✓ Results saved to: {results_dir}")

# ===== GENERATE VISUALIZATIONS =====
if not args.no_save:
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)

    viz_dir = results_dir / 'plots'
    viz_dir.mkdir(exist_ok=True)

    # Plot losses
    print("Plotting losses...")
    plot_losses(logger, save_path=viz_dir / 'losses.png')

    # Plot L2 distances
    print("Plotting L2 distances...")
    plot_l2_distances(logger, save_path=viz_dir / 'l2_distances.png')

    # Plot per-shape losses
    print("Plotting per-shape losses...")
    plot_per_shape_losses(logger, save_path=viz_dir / 'per_shape_losses.png')

    # Plot learning rate schedule
    print("Plotting learning rate schedule...")
    plot_learning_rate_schedule(logger, save_path=viz_dir / 'lr_schedule.png')

    # Plot coordinate error distribution
    print("Plotting coordinate error distribution...")
    coords_list = [torch.cat(coords_history[e], dim=0) for e in sorted(coords_history.keys())]
    gt_coords_list = [torch.cat(gt_coords_history[e], dim=0) for e in sorted(gt_coords_history.keys())]
    if coords_list:
        plot_coordinate_error_distribution(coords_list, gt_coords_list, 
                                           save_path=viz_dir / 'coordinate_errors.png')

    # Plot summary dashboard
    print("Plotting summary dashboard...")
    plot_summary_dashboard(logger, save_path=viz_dir / 'summary_dashboard.png')

    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Experiment: {experiment_name}")
    print(f"Results saved to: {results_dir.absolute()}")
    print(f"\nTo view results:")
    print(f"  $ python inspect_metrics.py --results-dir {results_dir}")
    print(f"  $ open {results_dir / 'plots' / 'summary_dashboard.png'}")
    print("="*70)
else:
    print("\n" + "="*70)
    print("TRAINING COMPLETED (DEBUG MODE - NO SAVING)")
    print("="*70)
    print(f"Experiment: {experiment_name}")
    print("No results saved (--no-save flag enabled)")
    print("="*70)
