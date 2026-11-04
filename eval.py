"""Run all comparisons and write artifacts/metrics.json."""
import json
import pickle
import sys

import numpy as np

sys.path.insert(0, "src")
from socialgraph.ann import IVFIndex
from socialgraph.baselines import adamic_adar_scores, popularity_scores, rank
from socialgraph.evaluate import evaluate_all
from socialgraph.model import TwoTower

K = 10


def main():
    X = np.load("artifacts/X.npy")
    emb = np.load("artifacts/embeddings.npz")
    ie, ue = emb["item_emb"], emb["user_emb"]
    data = pickle.load(open("artifacts/data.pkl", "rb"))
    friends, train, test = data["friends"], data["train"], data["test"]
    n_users = X.shape[0]
    test_users = sorted(test.keys())

    pop = popularity_scores(n_users, train)
    r_pop, n_pop = evaluate_all(
        lambda u: rank(pop, u, friends, train, K), test_users, test)
    r_aa, n_aa = evaluate_all(
        lambda u: rank(adamic_adar_scores(u, friends, n_users), u, friends, train, K),
        test_users, test)

    index = IVFIndex(ie)
    def two_tower_rec(u):
        excl = [u] + list(friends[u]) + train.get(u, [])
        return index.search(ue[u], k=K, exclude=excl)
    r_tt, n_tt = evaluate_all(two_tower_rec, test_users, test)

    # ANN fidelity: overlap between IVF and brute force top-10, averaged.
    overlaps = []
    for u in test_users[:100]:
        excl = [u] + list(friends[u]) + train.get(u, [])
        a = set(index.search(ue[u], k=K, exclude=excl))
        b = set(index.brute_force(ue[u], k=K, exclude=excl))
        overlaps.append(len(a & b) / K)
    ann_overlap = float(np.mean(overlaps))

    # Cold start: a brand-new user with profile features only (no free
    # embedding, no friends). Score via the item tower's profile projection.
    model = TwoTower(n_users, X.shape[1], dim=32, seed=0)
    model.Wi = emb["Wi"]
    cold_users = []
    rng = np.random.default_rng(7)
    for _ in range(50):
        prof = np.zeros((1, X.shape[1]))
        prof[0, int(rng.integers(0, 10))] = 1.0
        prof[0, 10 + int(rng.integers(0, 4))] = 1.0
        for c in rng.choice(20, size=3, replace=False):
            prof[0, 14 + c] = 1.0
        for it in rng.choice(30, size=5, replace=False):
            prof[0, 34 + it] = 1.0
        q = prof @ model.Wi
        q = q / np.linalg.norm(q).clip(min=1e-9)
        recs = index.search(q[0], k=K)
        cold_users.append(len(recs) == K)
    cold_ok = float(np.mean(cold_users))

    metrics = {
        "k": K, "n_users": n_users, "n_test_users": len(test_users),
        "popularity": {"recall": r_pop, "ndcg": n_pop},
        "adamic_adar": {"recall": r_aa, "ndcg": n_aa},
        "two_tower_ann": {"recall": r_tt, "ndcg": n_tt},
        "ann_bruteforce_overlap": ann_overlap,
        "cold_start_returns_k": cold_ok,
    }
    json.dump(metrics, open("artifacts/metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
