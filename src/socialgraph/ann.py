"""Approximate nearest neighbour retrieval: IVF index.

Item embeddings are clustered with k-means; a query probes only the
nearest clusters and reranks those candidates exactly. This is the
standard IVF (inverted file) scheme, and it is what makes serving
thousands of candidates cheap: probe 3 of 16 clusters instead of all.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


class IVFIndex:
    def __init__(self, item_emb: np.ndarray, n_clusters: int = 16, seed: int = 0):
        self.emb = item_emb
        self.km = KMeans(n_clusters=n_clusters, n_init=4, random_state=seed).fit(item_emb)
        self.labels = self.km.labels_
        self.centroids = self.km.cluster_centers_
        self.centroids /= np.linalg.norm(self.centroids, axis=1, keepdims=True).clip(min=1e-9)

    def search(self, query: np.ndarray, k: int = 10, n_probe: int = 3, exclude=()):
        sims = self.centroids @ query
        clusters = np.argsort(-sims)[:n_probe]
        cand = np.where(np.isin(self.labels, clusters))[0]
        cand = np.array([c for c in cand if c not in set(exclude)], dtype=int)
        if len(cand) == 0:
            return []
        scores = self.emb[cand] @ query
        top = cand[np.argsort(-scores)[:k]]
        return top.tolist()

    def brute_force(self, query: np.ndarray, k: int = 10, exclude=()):
        scores = self.emb @ query
        for e in exclude:
            scores[e] = -np.inf
        return np.argsort(-scores)[:k].tolist()
