"""Cosine-similarity library search."""
import numpy as np


class CosineLibrary:
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
