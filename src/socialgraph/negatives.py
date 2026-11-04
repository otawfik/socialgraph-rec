"""Hard negative mining.

A hard negative is a user who looks connectable but is not connected:
a friend-of-a-friend, or someone sharing 2+ clubs, who is neither a friend
nor a train positive. Random negatives are too easy (the model learns
"different major = no"), so each training example mixes 4 hard negatives
with 12 random in-batch-style negatives.
"""
from __future__ import annotations

import numpy as np


def hard_negatives(user, friends, train, clubs, n_users, rng, k=4):
    fof = set()
    for f in friends[user]:
        fof |= friends[f]
    fof -= friends[user]
    fof.discard(user)
    banned = set(train.get(user, [])) | friends[user] | {user}
    hard = [v for v in fof if v not in banned]
    shared = np.where((clubs @ clubs[user]) >= 2)[0].tolist()
    hard += [v for v in shared if v not in banned and v not in hard]
    rng.shuffle(hard)
    return hard[:k]


def sample_negatives(users, friends, train, clubs, n_users, rng, n_neg=16, n_hard=4):
    negs = np.zeros((len(users), n_neg), dtype=int)
    for i, u in enumerate(users):
        banned = set(train.get(int(u), [])) | friends[int(u)] | {int(u)}
        hard = hard_negatives(int(u), friends, train, clubs, n_users, rng, k=n_hard)
        row = list(hard)
        # Sample without rejection looping: a very popular user can be
        # connected to almost everyone, so rejection sampling could spin
        # forever when fewer than n_neg candidates exist. Draw from the
        # explicit allowed list instead, cycling if it is shorter than
        # n_neg (duplicated negatives are harmless for BPR).
        allowed = [v for v in range(n_users) if v not in banned and v not in row]
        rng.shuffle(allowed)
        row += allowed
        # Degenerate hub case: if almost everyone is banned, cycle what
        # exists (hard negatives, else anyone but self) so the row is
        # always exactly n_neg long and training never crashes or spins.
        pool = allowed if allowed else (list(row) if row else
                                        [v for v in range(n_users) if v != int(u)])
        j = 0
        while len(row) < n_neg:
            row.append(pool[j % len(pool)])
            j += 1
        negs[i] = row[:n_neg]
    return negs
