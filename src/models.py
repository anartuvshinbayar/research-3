"""1D CNN for Raman spectra."""
import numpy as np
import torch
from torch import nn


class SpectrumCNN(nn.Module):
    def __init__(self, n_in, n_classes):
        super().__init__()

        def block(cin, cout, k):
            return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout),
                                 nn.ReLU(), nn.MaxPool1d(2))

        self.features = nn.Sequential(block(1, 16, 21), block(16, 32, 15), block(32, 64, 9), block(64, 64, 5))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(64 * (n_in // 16), 256), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x.unsqueeze(1)))


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train_cnn(X, y, n_classes, epochs=25, batch=256, lr=2e-3, seed=0, max_shift=3, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device()
    model = SpectrumCNN(X.shape[1], n_classes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * int(np.ceil(len(X) / batch)))
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    for ep in range(epochs):
        model.train()
        perm = torch.tensor(rng.permutation(len(X)))
        total = 0.0
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            xb = Xt[idx]
            if max_shift:
                xb = torch.roll(xb, int(rng.integers(-max_shift, max_shift + 1)), dims=1)
            opt.zero_grad()
            loss = loss_fn(model(xb.to(dev)), yt[idx].to(dev))
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        if ep % 5 == 0 or ep == epochs - 1:
            log(f"epoch {ep + 1}/{epochs} loss {total / len(X):.3f}")
    return model.eval()


@torch.no_grad()
def cnn_proba(model, X, batch=1024):
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(X), batch):
        xb = torch.tensor(X[i:i + batch], dtype=torch.float32, device=dev)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out)
