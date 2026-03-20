import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import glob
from covariant_net import SPT_Net
import torch.nn as nn
from dataloader import SequenceChunkDataset
from tqdm import tqdm

path_prefix = '../data/v1.00-p0.50-s1.0_1100'
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


warmup = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SPT_Net(warmup=warmup).to(device)

loss_fn = nn.L1Loss()

epochs = 200

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


for epoch in range(epochs):

    model.train()
    train_loss = 0

    for idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)/4

        h, pred = model(x, epoch)
        pred = pred.permute(1, 0, 2, 3)

        loss = loss_fn(pred, y[:, warmup:])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        train_loss += loss.item()
        print(f'{idx+1}/{len(train_loader)} Loss: {loss.item():5f}', end ='\r')
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)/4

            _, pred = model(x, epoch)
            pred = pred.permute(1, 0, 2, 3)

            val_loss += loss_fn(pred, y[:, warmup:]).item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}")