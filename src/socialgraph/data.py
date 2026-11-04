"""Synthetic campus social graph.

Generates users with profile features (major, year, clubs, interests), a
friendship graph built with homophily plus preferential attachment, and
"connect" interactions used as implicit feedback for recommendation.

The split is per user: 80% of each user's positive connections go to train,
20% are held out for evaluation, so Recall@K / NDCG@K measure recovery of
real held-out connections, not random edges.
"""
from __future__ import annotations

import numpy as np

MAJORS = ["cs", "econ", "data", "finance", "bio", "mech", "psych", "math", "design", "polisci"]
CLUBS = [f"club_{i}" for i in range(20)]
INTERESTS = [f"interest_{i}" for i in range(30)]


def augmented_features(X: np.ndarray, friends: dict) -> np.ndarray:
    """Two-tower input: own profile plus the mean profile of your friends.

    This 1-hop aggregation is what lets the towers see graph structure:
    a friend-of-a-friend shares taste with your friends, so their profile
    looks similar to your friends' mean profile. A cold-start user has no
    friends, so their second half is zeros and only their own profile acts.
    """
    fmean = np.zeros_like(X)
    for u in range(X.shape[0]):
        if friends[u]:
            fmean[u] = X[list(friends[u])].mean(axis=0)
    return np.concatenate([X, fmean], axis=1)


def generate(n_users: int = 800, seed: int = 42):
    rng = np.random.default_rng(seed)
    majors = rng.integers(0, len(MAJORS), size=n_users)
    years = rng.integers(0, 4, size=n_users)
    clubs = (rng.random((n_users, len(CLUBS))) < 0.15).astype(np.float32)
    clubs[np.arange(n_users), rng.integers(0, len(CLUBS), size=n_users)] = 1.0
    interests = (rng.random((n_users, len(INTERESTS))) < 0.20).astype(np.float32)

    # Profile feature matrix: [major one-hot | year one-hot | clubs | interests]
    n_feat = len(MAJORS) + 4 + len(CLUBS) + len(INTERESTS)
    X = np.zeros((n_users, n_feat), dtype=np.float32)
    X[np.arange(n_users), majors] = 1.0
    X[np.arange(n_users), len(MAJORS) + years] = 1.0
    X[:, len(MAJORS) + 4:len(MAJORS) + 4 + len(CLUBS)] = clubs
    X[:, len(MAJORS) + 4 + len(CLUBS):] = interests

    # Friendship graph: similarity + popularity, sampled without replacement.
    club_sim = clubs @ clubs.T
    int_sim = interests @ interests.T
    same_major = (majors[:, None] == majors[None, :]).astype(np.float32)
    affinity = 0.5 * club_sim + 0.4 * int_sim + 0.8 * same_major
    np.fill_diagonal(affinity, 0)
    degree = np.zeros(n_users)
    friends = {u: set() for u in range(n_users)}
    for u in range(n_users):
        scores = affinity[u] * (1.0 + 0.05 * degree)
        scores[list(friends[u])] = 0
        k = int(rng.integers(8, 20))
        top = np.argpartition(-scores, min(k + 5, n_users - 1))[: k + 5]
        chosen = top[np.argsort(-scores[top])][:k]
        for v in chosen:
            friends[u].add(int(v))
            friends[int(v)].add(u)
            degree[u] += 1
            degree[int(v)] += 1

    # Positive "connect" interactions: sampled from friends-of-friends and
    # shared-club users, weighted by affinity (shared friends, shared clubs
    # and interests, same major). The weighting is the learnable signal:
    # uniform sampling would make held-out positives indistinguishable
    # from any other friend-of-a-friend, and every method would sit at
    # random. Affinity here is a stand-in for the latent taste that a
    # real discovery feed learns from behaviour.
    shared_friends = np.zeros((n_users, n_users), dtype=np.float32)
    for u in range(n_users):
        for f in friends[u]:
            for v in friends[f]:
                shared_friends[u, v] += 1
    pref = 2.0 * shared_friends + 0.5 * club_sim + 0.4 * int_sim + 0.8 * same_major
    np.fill_diagonal(pref, 0)
    train, test = {}, {}
    for u in range(n_users):
        fof = set()
        for f in friends[u]:
            fof |= friends[f]
        fof -= friends[u]
        fof.discard(u)
        cands = list(fof)
        if len(cands) < 10:  # sparse users fall back to shared-club users
            shared = np.where(club_sim[u] >= 2)[0].tolist()
            cands = list(dict.fromkeys(cands + [c for c in shared if c != u and c not in friends[u]]))
        w = pref[u, cands].astype(np.float64) + 0.1
        w = w / w.sum()
        k = int(rng.integers(10, 25))
        positives = rng.choice(cands, size=min(k, len(cands)), replace=False, p=w).tolist()
        if len(positives) < 4:
            continue
        split = max(1, int(0.8 * len(positives)))
        train[u] = sorted(positives[:split])
        test[u] = sorted(positives[split:])
    return {
        "X": X, "majors": majors, "years": years, "clubs": clubs,
        "interests": interests, "friends": friends, "train": train, "test": test,
    }
