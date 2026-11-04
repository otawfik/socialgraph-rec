"""Baselines: popularity and Adamic-Adar common neighbours.

Both are computed from the train graph only, so they are fair comparisons
for the two-tower model on the held-out test connections.
"""
from __future__ import annotations

import numpy as np


def popularity_scores(n_users: int, train: dict) -> np.ndarray:
    counts = np.zeros(n_users)
    for positives in train.values():
        for v in positives:
            counts[v] += 1
    return counts


def adamic_adar_scores(user: int, friends: dict, n_users: int) -> np.ndarray:
    """Adamic-Adar over the friendship graph, restricted to non-friends."""
    scores = np.zeros(n_users)
    deg = {u: max(1, len(f)) for u, f in friends.items()}
    for f in friends[user]:
        for v in friends[f]:
            if v != user and v not in friends[user]:
                scores[v] += 1.0 / np.log(deg[f] + 1)
    return scores


def rank(scores: np.ndarray, user: int, friends: dict, train: dict, k: int = 10):
    s = scores.astype(float).copy()
    s[user] = -np.inf
    for v in friends[user]:
        s[v] = -np.inf
    for v in train.get(user, []):
        s[v] = -np.inf
    return np.argsort(-s)[:k].tolist()
