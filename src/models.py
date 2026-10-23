"""Classifiers: cosine library search and a small 1D CNN."""
import numpy as np
import torch
from torch import nn


class CosineLibrary:
    """Classic spectral library matching: pick the reference with highest cosine similarity."""

    def fit(self, X, y):
        self.X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        self.y = np.asarray(y)
        return self

    def scores(self, Q, allowed=None):
        Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
        S = Q @ self.X.T
        if allowed is not None:
            S[:, ~np.isin(self.y, list(allowed))] = -np.inf
        return S

    def predict(self, Q, allowed=None):
        return self.y[np.argmax(self.scores(Q, allowed), axis=1)]


class SpectrumCNN(nn.Module):
    def __init__(self, n_in, n_classes):
        super().__init__()
        def block(cin, cout, k):
            return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout),
                                 nn.ReLU(), nn.MaxPool1d(2))
        self.features = nn.Sequential(block(1, 16, 21), block(16, 32, 15), block(32, 64, 9), block(64, 64, 5))
        n_flat = 64 * (n_in // 16)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(n_flat, 256), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x.unsqueeze(1)))


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train_cnn(X, y_idx, n_classes, epochs=25, batch=256, lr=2e-3, seed=0, shift_aug=3, log=print):
    """Train on fixed arrays. shift_aug = max random roll (grid points) applied per batch."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device()
    model = SpectrumCNN(X.shape[1], n_classes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * int(np.ceil(len(X) / batch)))
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y_idx, dtype=torch.long)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    for ep in range(epochs):
        model.train()
        perm = torch.tensor(rng.permutation(len(X)))
        total = 0.0
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            xb = Xt[idx]
            if shift_aug:
                xb = torch.roll(xb, int(rng.integers(-shift_aug, shift_aug + 1)), dims=1)
            xb, yb = xb.to(dev), yt[idx].to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        if ep % 5 == 0 or ep == epochs - 1:
            log(f"  epoch {ep + 1}/{epochs} loss {total / len(X):.3f}")
    return model.eval()


@torch.no_grad()
def cnn_proba(model, X, batch=1024):
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(X), batch):
        xb = torch.tensor(X[i:i + batch], dtype=torch.float32, device=dev)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out)
