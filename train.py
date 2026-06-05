"""Train a small CNN on the merged Quick Draw data (PyTorch)."""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from categories import CATEGORIES

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

EPOCHS = 12
BATCH_SIZE = 128
LR = 1e-3
VAL_SPLIT = 0.1
SEED = 42


class QuickDrawDataset(Dataset):
    """Per-sample augmentation via torchvision transforms."""

    def __init__(self, X_uint8: np.ndarray, y: np.ndarray, augment: bool):
        self.X = X_uint8  # (N, 28, 28) uint8
        self.y = y
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine(
                    degrees=15,
                    translate=(0.1, 0.1),
                    scale=(0.85, 1.15),
                ),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.transform(self.X[idx]), int(self.y[idx])

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


class QuickDrawCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


def load_data():
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    # Keep as uint8 (0-255); the transform pipeline handles ToTensor normalization.
    X = X.reshape(-1, 28, 28)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    n_val = int(len(X) * VAL_SPLIT)
    X_train, X_val = X[n_val:], X[:n_val]
    y_train, y_val = y[n_val:], y[:n_val]
    return X_train, y_train, X_val, y_val


def make_loader(X, y, batch_size, shuffle, augment):
    ds = QuickDrawDataset(X, y, augment=augment)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        persistent_workers=True,
    )


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb, reduction="sum")
        loss_sum += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def predictions(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_targets.append(yb.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    print(f"Device: {DEVICE}")
    print("Loading data...")
    X_train, y_train, X_val, y_val = load_data()
    print(f"  train: {X_train.shape}, val: {X_val.shape}")

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True, augment=True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False, augment=False)

    num_classes = len(CATEGORIES)
    model = QuickDrawCNN(num_classes).to(DEVICE)
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    print("\nTraining...")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        running_loss, n_seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * yb.size(0)
            n_seen += yb.size(0)
        train_loss = running_loss / n_seen
        val_loss, val_acc = evaluate(model, val_loader)
        scheduler.step(val_loss)
        dt = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  epoch {epoch:>2}/{EPOCHS}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"lr={current_lr:.1e}  ({dt:.1f}s)"
        )

    print("\nFinal evaluation:")
    val_loss, val_acc = evaluate(model, val_loader)
    print(f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    preds, targets = predictions(model, val_loader)
    cm = confusion_matrix(targets, preds, num_classes)
    name_width = max(len(c) for c in CATEGORIES)
    header = " " * (name_width + 2) + " ".join(f"{i:>4}" for i in range(num_classes))
    print(header)
    for i, name in enumerate(CATEGORIES):
        row = " ".join(f"{cm[i, j]:>4}" for j in range(num_classes))
        print(f"{name:>{name_width}}  {row}")

    print("\nSaving model and labels...")
    torch.save(
        {"state_dict": model.state_dict(), "num_classes": num_classes},
        os.path.join(MODEL_DIR, "quickdraw.pt"),
    )
    with open(os.path.join(MODEL_DIR, "labels.json"), "w") as f:
        json.dump(CATEGORIES, f)
    print("Done.")


if __name__ == "__main__":
    main()
