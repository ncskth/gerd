"""Loss functions for neural network training."""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


def _kl_divergence(p: torch.Tensor, q: torch.Tensor, ndims: int) -> torch.Tensor:
    """Compute KL divergence between distributions p and q.
    
    Args:
        p: Distribution tensor.
        q: Distribution tensor.
        ndims: Number of dimensions to sum over.
        
    Returns:
        KL divergence value.
    """
    eps = 1e-24
    unsummed = p * ((p + eps).log() - (q + eps).log())
    for _ in range(ndims):
        unsummed = unsummed.sum(-1)
    return unsummed


def _jensen_shannon(p: torch.Tensor, q: torch.Tensor, ndims: int) -> torch.Tensor:
    """Compute Jensen-Shannon divergence between distributions p and q.
    
    Args:
        p: Distribution tensor.
        q: Distribution tensor.
        ndims: Number of dimensions to sum over.
        
    Returns:
        Jensen-Shannon divergence value.
    """
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m, ndims) + 0.5 * _kl_divergence(q, m, ndims)


class JensenShannonLoss(nn.Module):
    """Jensen-Shannon divergence loss between predicted and target distributions."""
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Jensen-Shannon loss.
        
        Args:
            x: Predicted distribution.
            y: Target distribution.
            
        Returns:
            Scalar loss value.
        """
        return sum(_jensen_shannon(tx, ty, 2) for tx, ty in zip(x, y))


class KLLoss(nn.Module):
    """Kullback-Leibler divergence loss between predicted and target distributions."""
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss.
        
        Args:
            x: Predicted distribution.
            y: Target distribution.
            
        Returns:
            Scalar loss value.
        """
        return sum(_kl_divergence(tx, ty, 2) for tx, ty in zip(x, y))


class VarianceLoss(nn.Module):
    """Loss based on variance difference between predicted and target distributions."""
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute variance loss.
        
        Args:
            x: Predicted distribution.
            y: Target distribution.
            
        Returns:
            Scalar loss value.
        """
        return sum((tx.var() - ty.var()).abs().mean() for tx, ty in zip(x, y))


# ============================================================================
# Better loss functions for coordinate prediction
# ============================================================================

class SmoothL1CoordinateLoss(nn.Module):
    """Smooth L1 loss for coordinate prediction with robustness to outliers."""
    
    def __init__(self, beta: float = 0.1):
        """Initialize Smooth L1 loss.
        
        Args:
            beta: Transition point between L1 and L2. Smaller β = sharper transition.
        """
        super().__init__()
        self.beta = beta
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute smooth L1 loss.
        
        Args:
            pred: Predicted coordinates of shape [..., 2].
            target: Target coordinates of shape [..., 2].
            
        Returns:
            Scalar loss value.
        """
        return torch.nn.functional.smooth_l1_loss(pred, target, beta=self.beta, reduction='mean')


class HuberCoordinateLoss(nn.Module):
    """Huber loss (smooth L1) for robust coordinate regression."""
    
    def __init__(self, delta: float = 0.5):
        """Initialize Huber loss.
        
        Args:
            delta: Transition point between L2 and L1.
        """
        super().__init__()
        self.delta = delta
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Huber loss.
        
        Args:
            pred: Predicted coordinates.
            target: Target coordinates.
            
        Returns:
            Scalar loss value.
        """
        return torch.nn.functional.huber_loss(pred, target, delta=self.delta, reduction='mean')


class GaussianNLLCoordinateLoss(nn.Module):
    """Gaussian Negative Log-Likelihood loss for uncertainty-aware coordinate prediction.
    
    Assumes coordinates follow a Gaussian distribution with learned variance.
    """
    
    def __init__(self, var_init: float = 0.1):
        """Initialize Gaussian NLL loss.
        
        Args:
            var_init: Initial variance estimate for each coordinate.
        """
        super().__init__()
        self.log_var = nn.Parameter(torch.full((2,), float(np.log(var_init))))
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian NLL loss.
        
        Args:
            pred: Predicted coordinates [..., 2].
            target: Target coordinates [..., 2].
            
        Returns:
            Scalar loss value.
        """
        var = torch.exp(self.log_var)
        loss = 0.5 * ((pred - target) ** 2 / var).mean() + 0.5 * self.log_var.mean()
        return loss


class DSNT(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        H, W = resolution
        self.probs_x = torch.linspace(-1, 1, W).repeat(H, 1).flatten()
        self.probs_y = torch.linspace(-1, 1, H).repeat(W, 1).T.flatten()

    def forward(self, x):
        device = x.device
        probs_x = self.probs_x.to(device)
        probs_y = self.probs_y.to(device)

        x_flat = x.flatten(-2)
        co_x = (x_flat * probs_x).sum(-1)
        co_y = (x_flat * probs_y).sum(-1)

        return torch.stack((co_y, co_x), dim=-1)




class DSNTLI(torch.nn.Module):
    def __init__(self, resolution: Tuple[int, int]):
        """A differentiable spatial-to-numerical transform (DSNT) with a linear
        integration (LI) step.

        Arguments:
            resolution: The resolution of the input image.
        """
        super().__init__()
        self.resolution = resolution
        self.probs_x = (
            torch.linspace(-1, 1, resolution[1]).repeat(resolution[0], 1).flatten()
        )
        self.probs_y = (
            torch.linspace(-1, 1, resolution[0]).repeat(resolution[1], 1).T.flatten()
        )
        self.li_tm = torch.nn.Parameter(torch.tensor([0.9, 0.9]))
        # self.lin = torch.nn.Linear(2, 2, bias=False)

    def forward(self, x: torch.Tensor, state=None):
        if not x.device == self.probs_x.device:
            self.probs_x = self.probs_x.to(x.device)
            self.probs_y = self.probs_y.to(x.device)
        co_1 = (x.flatten(-2) * self.probs_x).sum(-1)
        co_2 = (x.flatten(-2) * self.probs_y).sum(-1)

        cos = torch.stack((co_2, co_1), -1)
        if state is None:
            state = torch.zeros(2, device=x.device)

        out = []
        for t in cos:
            state = state - (state * self.li_tm) + t
            out.append(state.clone())

        return torch.stack(out), state
