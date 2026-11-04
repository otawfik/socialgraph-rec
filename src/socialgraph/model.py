"""Two-tower recommender, trained from scratch in NumPy.

User tower:  profile features -> linear projection + free user embedding.
Item tower:  profile features -> linear projection + free item embedding.
Both towers are L2-normalised, so the score is cosine similarity.

Training loss is binary cross-entropy on a positive vs. a set of negatives
(in-batch negatives plus mined hard negatives), optimised with Adam.
Free embeddings let the model memorise graph structure; the profile
projection is what makes cold-start users recommendable at all.
"""
from __future__ import annotations

import numpy as np


class TwoTower:
    def __init__(self, n_users: int, n_feat: int, dim: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.dim = dim
        s = np.sqrt(2.0 / (n_feat + dim))
        self.Wu = rng.normal(0, s, size=(n_feat, dim)).astype(np.float64)
        self.Wi = rng.normal(0, s, size=(n_feat, dim)).astype(np.float64)
        self.Eu = rng.normal(0, 0.05, size=(n_users, dim))
        self.Ei = rng.normal(0, 0.05, size=(n_users, dim))
        self.params = ["Wu", "Wi", "Eu", "Ei"]
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.t = 0

    def user_emb(self, users, X):
        e = X[users] @ self.Wu + self.Eu[users]
        return e / np.linalg.norm(e, axis=1, keepdims=True).clip(min=1e-9)

    def item_emb(self, items, X):
        e = X[items] @ self.Wi + self.Ei[items]
        return e / np.linalg.norm(e, axis=1, keepdims=True).clip(min=1e-9)

    def _adam(self, grads, lr):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for p in self.params:
            g = grads[p]
            self.m[p] = b1 * self.m[p] + (1 - b1) * g
            self.v[p] = b2 * self.v[p] + (1 - b2) * g * g
            mh = self.m[p] / (1 - b1 ** self.t)
            vh = self.v[p] / (1 - b2 ** self.t)
            setattr(self, p, getattr(self, p) - lr * mh / (np.sqrt(vh) + eps))

    def train_step(self, users, pos, negs, X, lr=0.02):
        """users: (B,), pos: (B,), negs: (B, N). Returns mean loss."""
        B, N = negs.shape
        ue = self.user_emb(users, X)                       # (B, d)
        pe = self.item_emb(pos, X)                         # (B, d)
        ne = np.stack([self.item_emb(negs[i], X) for i in range(B)])  # (B, N, d)
        # Scores, with a temperature that keeps gradients alive early on.
        sp = np.sum(ue * pe, axis=1) / 0.1                 # (B,)
        sn = np.einsum("bd,bnd->bn", ue, ne) / 0.1         # (B, N)
        # Binary cross-entropy: positive label 1, negatives 0.
        def sig(z):
            return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        pp, pn = sig(sp), sig(sn)
        loss = float(-np.log(pp.clip(min=1e-9)).mean() - np.log((1 - pn).clip(min=1e-9)).mean())
        # Gradients through the normalised embeddings (approx: treat the
        # normalisation as constant for the backward pass, standard for
        # two-tower training at this scale; the projection still learns).
        dp = (pp - 1.0) / 0.1 / B                          # (B,)
        dn = pn / 0.1 / B                                  # (B, N)
        g_ue = dp[:, None] * pe + np.einsum("bn,bnd->bd", dn, ne)
        g_pe = dp[:, None] * ue
        g_ne = dn[:, :, None] * ue[:, None, :]
        gWu = X[users].T @ g_ue
        gWi = X[pos].T @ g_pe + sum(X[negs[i]].T @ g_ne[i] for i in range(B))
        gEu = np.zeros_like(self.Eu)
        gEi = np.zeros_like(self.Ei)
        np.add.at(gEu, users, g_ue)
        np.add.at(gEi, pos, g_pe)
        for i in range(B):
            np.add.at(gEi, negs[i], g_ne[i])
        self._adam({"Wu": gWu, "Wi": gWi, "Eu": gEu, "Ei": gEi}, lr)
        return loss
