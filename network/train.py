import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import glob
from covariant_net import SPT_Net, make_gaussian_targets
import torch.nn as nn
from dataloader import SequenceChunkDataset
from tqdm import tqdm





import matplotlib.pyplot as plt
import torch

def func(h, pred):
    plt.figure()

    T, B, C, H, W = h.shape

    # =========================
    # Select last timestep / batch / channel
    # =========================
    # shape: (H, W)
    heatmaps_flat = h.view(T, B, C, H * W)

    # Correct softmax over spatial dimension
    probs = torch.softmax(heatmaps_flat, dim=-1)
    # Select one example (same as your original indexing)
    probs_map = probs[-1, -1, 1].detach().cpu().view(H, W)

    # Predicted coordinates
    x_pred, y_pred = pred[-1, -1, 1].detach().cpu()

    # Ground truth coordinates
    y_out, x_out = y[-1, -1, 1].detach().cpu()

    # =========================
    # Center of mass (PyTorch-consistent)
    # =========================
    ys = torch.linspace(0, H - 1, H)
    xs = torch.linspace(0, W - 1, W)

    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

    grid_x = grid_x.reshape(H, W)
    grid_y = grid_y.reshape(H, W)

    total_mass = probs_map.sum()

    center_x = (probs_map * grid_x).sum() / total_mass
    center_y = (probs_map * grid_y).sum() / total_mass

    # =========================
    # Prints (same intent as original)
    # =========================
    # print(x_pred.item(), y_pred.item())
    # print(center_x.item(), center_y.item())
    # print(x_out.item(), y_out.item())

    # =========================
    # Plot heatmap
    # =========================
    im = plt.imshow(h.cpu().detach().numpy()[-1][-1][1], cmap='gray')
    # im = plt.imshow(probs_map.numpy(), cmap='gray')
    plt.scatter(x_pred.item(), y_pred.item(), c='red', marker='x', s=100, label='Prediction')

    # Ground truth (green)
    plt.scatter(x_out.item(), y_out.item(), c='green', marker='o', label='Ground Truth')

    # Center of mass (blue)
    plt.scatter(center_x.item(), center_y.item(), c='blue', marker='s', label='Center of Mass')
    # plt.colorbar(im)
    plt.colorbar(im)
    plt.title("Heatmap")
    plt.savefig('rand_test.png')
    plt.close()
    flat_idx = torch.argmax(probs_map)
    y_max = flat_idx // W
    x_max = flat_idx % W

    # print("Argmax:", x_max.item(), y_max.item())
    # print("Center of mass:", center_x.item(), center_y.item())






path_prefix = '../data/v1.00-p0.50-s2.0_1000'
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

batch_size = 2

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)


import torch
import torch.nn as nn
from tqdm import tqdm

class JensenShannonLoss(nn.Module):
    def forward(self, x, y):
        return sum(_js(tx, ty, 2) for tx, ty in zip(x, y))

class KLLoss(nn.Module):
    def forward(self, x, y):
        return sum(_kl(tx, ty, 2) for tx, ty in zip(x, y))

class VarianceLoss(nn.Module):
    def forward(self, x, y):
        return sum((tx.var() - ty.var()).abs().mean() for tx, ty in zip(x, y))

def _kl(p, q, ndims):
    eps = 1e-24
    unsummed = p * ((p + eps).log() - (q + eps).log())
    for _ in range(ndims):
        unsummed = unsummed.sum(-1)
    return unsummed

def _js(p, q, ndims):
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m, ndims) + 0.5 * _kl(q, m, ndims)


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


warmup = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SPT_Net(warmup=warmup).to(device)

optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

epochs = 200

regularization = JensenShannonLoss()
# regularization = KLLoss()
# regularization = VarianceLoss()

rectification = nn.Softmax(dim=-1)

reg_scale = 1e-4

min_val_loss = np.inf

for epoch in range(epochs):

    model.train()
    train_loss = 0

    for idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = (y.to(device) / 4)  # keep your scaling

        h, pred = model(x, epoch)

        h = h.permute(1, 0, 2, 3, 4)   # (B, T, N, H, W)
        pred = pred.permute(1, 0, 2, 3)

        h_used = h[:, warmup:]
        y_used = y[:, warmup:]

        B, T, N, H, W = h_used.shape

        h_flat = h_used.flatten(3)
        h_rect = rectification(h_flat).reshape_as(h_used)

        dsnt = DSNT((H, W)).to(device)
        pred_co = dsnt(h_rect)
        
        res = torch.tensor([H, W], device=device)
        y_norm = ((y_used * 2) / res) - 1

        y_norm = y_norm[:, warmup:]
        loss_co = torch.norm(pred_co - y_norm, dim=-1).mean()

        gaussian_target = make_gaussian_targets(y_used, H, W)

        loss_reg = regularization(
            h_rect.reshape(-1, H, W),
            gaussian_target.reshape(-1, H, W)
        ) * reg_scale

        loss = loss_co + loss_reg 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        func(h, pred)

        print(
            f"{idx+1}/{len(train_loader)} "
            f"Loss: {loss.item():.4f} "
            f"(coord={loss_co.item():.4f}, reg={loss_reg.item():.4f})",
            end="\r"
        )


    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = (y.to(device) / 4)

            h, _ = model(x, epoch)

            h = h.permute(1, 0, 2, 3, 4)
            h_used = h[:, warmup:]
            y_used = y[:, warmup:]

            B, T, N, H, W = h_used.shape

            h_flat = h_used.flatten(3)
            h_rect = rectification(h_flat).reshape_as(h_used)

            dsnt = DSNT((H, W)).to(device)
            pred_co = dsnt(h_rect)

            res = torch.tensor([H, W], device=device)
            y_norm = ((y_used * 2) / res) - 1
            y_norm = y_norm[:, warmup:]

            loss = torch.norm(pred_co - y_norm, dim=-1).mean()
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"\nEpoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}")
    
    if val_loss < min_val_loss:
        torch.save(model.state_dict(), f'best_model.pkl')
        min_val_loss = val_loss