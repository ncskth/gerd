"""Utility functions for visualization and debugging."""

import torch
import matplotlib.pyplot as plt


def plot_predictions(h: torch.Tensor, pred: torch.Tensor, y_used: torch.Tensor, x: torch.Tensor, network_type: str) -> None:
    """Plot heatmap with predicted and ground-truth coordinates for three output channels.
    
    Args:
        h: Hidden activations of shape [T, B, C, H, W].
        pred: Predicted coordinates of shape [T, B, C, 2].
        y_used: Ground-truth coordinates of shape [T, B, C, 2].
        x: Input data of shape [B, T, C, H, W].
        network_type: Type of network (covariant, non_covariant, curriculum) for filename.
    """
    T, B, C, H, W = h.shape

    # Compute probability map from heatmaps
    heatmaps_flat = h.view(T, B, C, H * W)
    probs = torch.softmax(heatmaps_flat, dim=-1)
    # probs_map = probs[-1, -1, 1].detach().cpu().view(H, W)

    # Prepare input visualization
    x_sub = x[-1].cpu().detach()  # shape: [300, 300]
    x_sub = torch.sum(x_sub[-10:], dim=0)
    x_sub = x_sub[1] + x_sub[0]
    x_sub = x_sub.unsqueeze(0).unsqueeze(0).squeeze()

    # Create figure and plot heatmap
    plt.figure()
    im = plt.imshow(x_sub.numpy(), cmap='gray')

    # Plot predictions and ground truth for three channels (0=red, 1=green, 2=yellow)
    channel_colors = [
        ('green', 'green'),    # Channel 0
        # ('red', 'red'),        # Channel 1
        # ('yellow', 'yellow')   # Channel 2
    ]

    for channel, (pred_color, gt_color) in enumerate(channel_colors):
        x_pred, y_pred = pred[-1, -1, channel].detach().cpu() * 300
        x_gt, y_gt = y_used[-1, -1, channel].detach().cpu()

        plt.scatter(x_pred.item(), y_pred.item(), c=pred_color, marker='x', s=100, label=f'Pred Ch{channel}')
        plt.scatter(x_gt.item(), y_gt.item(), c=gt_color, marker='o', label=f'GT Ch{channel}')

    plt.colorbar(im)
    plt.title("Predictions vs Ground Truth")
    plt.savefig(f'prediction_{network_type}.png')
    plt.close()


def mean_position(h: torch.Tensor) -> None:
    """Print the mean position (center of mass) of a heatmap.
    
    Args:
        h: Heatmap of shape at least [..., H, W].
    """
    h_map = h[-1, -1, -1]  # Select last timestep, batch, and channel (H, W)

    H, W = h_map.shape
    ys = torch.arange(H, device=h_map.device, dtype=h_map.dtype)
    xs = torch.arange(W, device=h_map.device, dtype=h_map.dtype)

    # Normalize to probability distribution
    h_norm = h_map / (h_map.sum() + 1e-8)

    mean_y = (h_norm.sum(dim=1) * ys).sum()
    mean_x = (h_norm.sum(dim=0) * xs).sum()

    print(f"Heatmap mean position: (x={mean_x.item():.2f}, y={mean_y.item():.2f})")