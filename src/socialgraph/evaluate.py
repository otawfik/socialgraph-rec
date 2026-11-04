"""Evaluation: Recall@K, NDCG@K, ANN fidelity, cold start."""
from __future__ import annotations

import numpy as np


def recall_ndcg(ranked, relevant, k=10):
    ranked = ranked[:k]
    hits = [1.0 if v in relevant else 0.0 for v in ranked]
    recall = sum(hits) / max(1, len(relevant))
    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return recall, (dcg / ideal if ideal else 0.0)


def evaluate_all(rec_fn, test_users, test):
    recalls, ndcgs = [], []
    for u in test_users:
        if u not in test or not test[u]:
            continue
        ranked = rec_fn(u)
        r, n = recall_ndcg(ranked, set(test[u]))
        recalls.append(r)
        ndcgs.append(n)
    return float(np.mean(recalls)), float(np.mean(ndcgs))
