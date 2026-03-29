"""Visualization utilities for training monitoring and analysis."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle


class TrainingLogger:
    """Log and save training metrics."""
    
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'train_loss': [],
            'train_loss_co': [],
            'train_loss_reg': [],
            'val_loss': [],
            'val_loss_co': [],
            'val_loss_reg': [],
            'l2_distance_pixels': [],
            'val_l2_distance_pixels': [],
            'learning_rates': [],
            'per_shape_val_loss': {'square': []},
            'per_shape_l2': {'square': []},
        }
        
        self.best_model_path = None
        self.best_val_loss = np.inf
        self.checkpoints = {}  # epoch -> model state
        
    def log_train_batch(self, loss: float, loss_co: float, loss_reg: float, l2_distance: float, lr: float):
        """Log training batch metrics."""
        self.metrics['train_loss'].append(loss)
        self.metrics['train_loss_co'].append(loss_co)
        self.metrics['train_loss_reg'].append(loss_reg)
        self.metrics['l2_distance_pixels'].append(l2_distance)
        self.metrics['learning_rates'].append(lr)
    
    def log_train_epoch(self, train_loss: float, train_loss_co: float, train_loss_reg: float, 
                       train_l2: float, lr: float):
        """Log training epoch metrics (averaged over all batches)."""
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_loss_co'].append(train_loss_co)
        self.metrics['train_loss_reg'].append(train_loss_reg)
        self.metrics['l2_distance_pixels'].append(train_l2)
        self.metrics['learning_rates'].append(lr)
    
    def log_validation(self, val_loss: float, val_loss_co: float, val_loss_reg: float, 
                      val_l2: float, per_shape_loss: Dict[str, float], per_shape_l2: Dict[str, float]):
        """Log validation epoch metrics."""
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_loss_co'].append(val_loss_co)
        self.metrics['val_loss_reg'].append(val_loss_reg)
        self.metrics['val_l2_distance_pixels'].append(val_l2)
        
        for shape in ['square']:
            self.metrics['per_shape_val_loss'][shape].append(per_shape_loss[shape])
            self.metrics['per_shape_l2'][shape].append(per_shape_l2[shape])
    
    def save_checkpoint(
        self,
        epoch: int,
        model,
        optimizer,
        scheduler=None,
        is_best: bool = False,
        patience_count: Optional[int] = None,
        best_val_loss: Optional[float] = None,
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': self.metrics,
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if patience_count is not None:
            checkpoint['patience_count'] = int(patience_count)

        if best_val_loss is not None:
            checkpoint['best_val_loss'] = float(best_val_loss)
        
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch:03d}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            self.best_val_loss = self.metrics['val_loss'][-1]
    
    def save_metrics(self):
        """Save all metrics to file."""
        metrics_path = self.save_dir / 'training_metrics.pkl'
        with open(metrics_path, 'wb') as f:
            pickle.dump(self.metrics, f)
    
    def load_metrics(self) -> Dict:
        """Load metrics from file."""
        metrics_path = self.save_dir / 'training_metrics.pkl'
        if metrics_path.exists():
            with open(metrics_path, 'rb') as f:
                self.metrics = pickle.load(f)
        return self.metrics


def plot_losses(logger: TrainingLogger, save_path: Optional[Path] = None):
    """Plot training and validation losses."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(len(logger.metrics['val_loss']))
    
    # Total loss
    axes[0, 0].plot(logger.metrics['train_loss'], 'o-', linewidth=2, label='Train', alpha=0.7)
    axes[0, 0].plot(logger.metrics['val_loss'], 's-', linewidth=2, label='Val', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Coordinate loss
    axes[0, 1].plot(logger.metrics['train_loss_co'], 'o-', linewidth=2, label='Train', alpha=0.7)
    axes[0, 1].plot(logger.metrics['val_loss_co'], 's-', linewidth=2, label='Val', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Coordinate Loss')
    axes[0, 1].set_title('Coordinate Loss (L1)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Regularization loss
    axes[1, 0].plot(logger.metrics['train_loss_reg'], 'o-', linewidth=2, label='Train', alpha=0.7)
    axes[1, 0].plot(logger.metrics['val_loss_reg'], 's-', linewidth=2, label='Val', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Regularization Loss')
    axes[1, 0].set_title('Regularization Loss (Jensen-Shannon)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Log scale total loss
    train_loss_arr = np.array(logger.metrics['train_loss'])
    val_loss_arr = np.array(logger.metrics['val_loss'])
    axes[1, 1].semilogy(train_loss_arr[train_loss_arr > 0], 'o-', linewidth=2, label='Train', alpha=0.7)
    axes[1, 1].semilogy(val_loss_arr[val_loss_arr > 0], 's-', linewidth=2, label='Val', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Total Loss (log scale)')
    axes[1, 1].set_title('Total Loss (Log Scale)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_l2_distances(logger: TrainingLogger, save_path: Optional[Path] = None):
    """Plot L2 distances (coordinate errors in pixels)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw L2 distances
    axes[0].plot(logger.metrics['l2_distance_pixels'], 'o-', linewidth=2, alpha=0.7, label='Train')
    axes[0].plot(logger.metrics['val_l2_distance_pixels'], 's-', linewidth=2, alpha=0.7, label='Val')
    axes[0].set_xlabel('Epoch (validation only)')
    axes[0].set_ylabel('L2 Distance (pixels)')
    axes[0].set_title('Mean Coordinate Error')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Per-shape L2 distances
    for shape in ['square']:
        axes[1].plot(logger.metrics['per_shape_l2'][shape], 'o-', linewidth=2, label=shape, alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('L2 Distance (pixels)')
    axes[1].set_title('L2 Distance Per Shape')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_per_shape_losses(logger: TrainingLogger, save_path: Optional[Path] = None):
    """Plot per-shape validation losses."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for shape in ['square']:
        ax.plot(logger.metrics['per_shape_val_loss'][shape], 'o-', linewidth=2.5, 
               label=shape, markersize=6, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Per-Shape Validation Loss', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_learning_rate_schedule(logger: TrainingLogger, save_path: Optional[Path] = None):
    """Plot learning rate schedule during training."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Group by epochs (average per epoch)
    lrs_per_epoch = []
    n_lr = len(logger.metrics['learning_rates'])
    n_val = len(logger.metrics['val_loss'])
    if n_val == 0 or n_lr == 0:
        # Fall back to plotting raw LR history when no completed validation epochs are available
        lrs_per_epoch = logger.metrics['learning_rates']
    else:
        batches_per_epoch = n_lr // n_val
        if batches_per_epoch <= 0:
            batches_per_epoch = 1
        for i in range(n_val):
            start_idx = i * batches_per_epoch
            end_idx = min((i + 1) * batches_per_epoch, n_lr)
            if start_idx < end_idx:
                lrs_per_epoch.append(np.mean(logger.metrics['learning_rates'][start_idx:end_idx]))
    
    ax.semilogy(lrs_per_epoch, 'o-', linewidth=2.5, markersize=6, alpha=0.8, color='purple')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_heatmap_evolution(heatmaps_history: List[torch.Tensor], coords_history: List[torch.Tensor],
                          gt_coords_history: List[torch.Tensor], 
                          epochs_to_plot: List[int] = [0, 5, 10, 20, 50, 99],
                          save_path: Optional[Path] = None):
    """Plot evolution of predicted heatmaps and coordinates over training.
    
    Args:
        heatmaps_history: List of heatmaps at different epochs, each (T, B, 3, H, W)
        coords_history: List of predicted coords, each (T, B, 3, 2)
        gt_coords_history: List of ground truth coords, each (T, B, 3, 2)
        epochs_to_plot: Which epochs to visualize
        save_path: Path to save figure
    """
    n_epochs = min(len(epochs_to_plot), 6)
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, n_epochs, figure=fig)
    
    for idx, epoch in enumerate(epochs_to_plot[:n_epochs]):
        if epoch >= len(heatmaps_history):
            continue
        
        heatmaps = heatmaps_history[epoch]  # (T, B, 3, H, W)
        coords = coords_history[epoch]  # (T, B, 3, 2)
        gt_coords = gt_coords_history[epoch]  # (T, B, 3, 2)
        
        # Take first sample in batch, last timestep
        t_idx, b_idx = -1, 0
        h = heatmaps[t_idx, b_idx].detach().cpu().numpy()  # (3, H, W)
        c = coords[t_idx, b_idx].detach().cpu().numpy()  # (3, 2)
        gt_c = gt_coords[t_idx, b_idx].detach().cpu().numpy()  # (3, 2)
        
        shapes = ['Square']
        colors = ['blue']
        for shape_idx in range(1):
            ax = fig.add_subplot(gs[shape_idx, idx])
            h_norm = (h[shape_idx] - h[shape_idx].min()) / (h[shape_idx].max() - h[shape_idx].min() + 1e-6)
            ax.imshow(h_norm, cmap='hot', origin='upper')
            H_orig, W_orig = h_norm.shape
            pred_x, pred_y = c[shape_idx] * np.array([W_orig, H_orig])
            ax.plot(pred_x, pred_y, 'o', markersize=8, color=colors[shape_idx], label='Pred', alpha=0.8)
            gt_x, gt_y = gt_c[shape_idx] * np.array([W_orig, H_orig])
            ax.plot(gt_x, gt_y, 'x', markersize=10, color='white', markeredgewidth=2, label='GT', alpha=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            if shape_idx == 0:
                ax.set_title(f'Epoch {epoch}', fontsize=11, fontweight='bold')
            if idx == 0:
                ax.set_ylabel(f'{shapes[shape_idx]}', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_gabor_filters(model, layer_idx: int = 0, n_filters: int = 16, 
                       save_path: Optional[Path] = None):
    """Visualize learned Gabor-like filters from first convolution layer.
    
    Args:
        model: SPT_Net model
        layer_idx: Which conv layer to extract filters from
        n_filters: Number of filters to plot
        save_path: Path to save figure
    """
    # Extract weights from first conv layer (spt_conv1)
    weights = model.spt_conv1.weight.detach().cpu().numpy()  # (out_channels, in_channels, K, K)
    
    n_filters = min(n_filters, weights.shape[0])
    n_cols = 4
    n_rows = (n_filters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()
    
    for i in range(n_filters):
        ax = axes[i]
        
        # Average across input channels
        filter_vis = np.mean(np.abs(weights[i]), axis=0)
        
        ax.imshow(filter_vis, cmap='RdBu_r', origin='upper')
        ax.set_title(f'Filter {i}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for i in range(n_filters, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_coordinate_error_distribution(coords_history: List[torch.Tensor],
                                       gt_coords_history: List[torch.Tensor],
                                       save_path: Optional[Path] = None):
    """Plot distribution of coordinate errors across training.
    
    Args:
        coords_history: List of predicted coordinates at epochs
        gt_coords_history: List of GT coordinates at epochs
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Final epoch errors
    last_coords = coords_history[-1].detach().cpu().numpy()  # (T, B, 3, 2)
    last_gt = gt_coords_history[-1].detach().cpu().numpy()
    
    errors = np.linalg.norm(last_coords - last_gt, axis=-1).flatten() * 300  # Convert to pixels
    errors_per_shape = [
        np.linalg.norm(last_coords[..., 0, :] - last_gt[..., 0, :], axis=-1).flatten() * 300
    ]
    
    # Overall error distribution
    axes[0, 0].hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}px')
    axes[0, 0].set_xlabel('Coordinate Error (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Overall Coordinate Error Distribution (Final Epoch)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Per-shape errors
    shapes = ['Square']
    colors_per_shape = ['blue']
    for i, (shape, color) in enumerate(zip(shapes, colors_per_shape)):
        axes[0, 1].hist(errors_per_shape[i], bins=20, alpha=0.6, label=shape, color=color)
    axes[0, 1].set_xlabel('Coordinate Error (pixels)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Per-Shape Error Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error vs epoch
    mean_errors_per_epoch = []
    for coords, gt_coords in zip(coords_history, gt_coords_history):
        c = coords.detach().cpu().numpy()
        gt = gt_coords.detach().cpu().numpy()
        errs = np.linalg.norm(c - gt, axis=-1).flatten() * 300
        mean_errors_per_epoch.append(np.mean(errs))
    
    axes[1, 0].plot(mean_errors_per_epoch, 'o-', linewidth=2, markersize=6, alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Mean Coordinate Error (pixels)')
    axes[1, 0].set_title('Coordinate Error Over Training')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plot per shape (final epoch)
    axes[1, 1].boxplot(errors_per_shape, labels=shapes)
    axes[1, 1].set_ylabel('Coordinate Error (pixels)')
    axes[1, 1].set_title('Per-Shape Error (Final Epoch)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_summary_dashboard(logger: TrainingLogger, save_path: Optional[Path] = None):
    """Plot comprehensive training summary dashboard."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    
    # Loss curves
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(logger.metrics['train_loss'], 'o-', alpha=0.7, label='Train', linewidth=2)
    ax1.plot(logger.metrics['val_loss'], 's-', alpha=0.7, label='Val', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Learning rate
    ax2 = fig.add_subplot(gs[0, 2])
    batches_per_epoch = len(logger.metrics['learning_rates']) // len(logger.metrics['val_loss'])
    lrs_per_epoch = [np.mean(logger.metrics['learning_rates'][i*batches_per_epoch:(i+1)*batches_per_epoch])
                     for i in range(len(logger.metrics['val_loss']))]
    ax2.semilogy(lrs_per_epoch, 'o-', color='purple', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('LR Schedule')
    ax2.grid(True, alpha=0.3, which='both')
    
    # L2 distance
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(logger.metrics['val_l2_distance_pixels'], 'o-', color='orange', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('L2 Distance (px)')
    ax3.set_title('Coordinate Error')
    ax3.grid(True, alpha=0.3)
    
    # Coordinate loss
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(logger.metrics['train_loss_co'], 'o-', alpha=0.7, label='Train', linewidth=2)
    ax4.plot(logger.metrics['val_loss_co'], 's-', alpha=0.7, label='Val', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('Coordinate Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Regularization loss
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(logger.metrics['train_loss_reg'], 'o-', alpha=0.7, label='Train', linewidth=2)
    ax5.plot(logger.metrics['val_loss_reg'], 's-', alpha=0.7, label='Val', linewidth=2)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss')
    ax5.set_title('Regularization Loss')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Per-shape losses
    ax6 = fig.add_subplot(gs[2, :])
    for shape in ['square']:
        ax6.plot(logger.metrics['per_shape_val_loss'][shape], 'o-', label=shape, linewidth=2.5, markersize=5)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Validation Loss')
    ax6.set_title('Per-Shape Validation Loss')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig
