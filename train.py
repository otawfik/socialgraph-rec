"""Train the two-tower model and save embeddings + metrics."""
import json
import sys

import numpy as np

sys.path.insert(0, "src")
from socialgraph.data import augmented_features, generate
from socialgraph.model import TwoTower
from socialgraph.negatives import sample_negatives


def main(epochs: int = 15, seed: int = 42):
    data = generate(seed=seed)
    friends, train = data["friends"], data["train"]
    # Towers train on augmented features (own profile + friends' mean).
    X = augmented_features(data["X"], friends)
    n_users = X.shape[0]
    model = TwoTower(n_users, X.shape[1], dim=32, seed=seed)
    pairs = [(u, v) for u, vs in train.items() for v in vs]
    rng = np.random.default_rng(seed + 1)
    history = []
    for ep in range(epochs):
        rng.shuffle(pairs)
        losses = []
        for i in range(0, len(pairs), 256):
            batch = pairs[i:i + 256]
            users = np.array([u for u, _ in batch])
            pos = np.array([v for _, v in batch])
            negs = sample_negatives(users, friends, train, data["clubs"], n_users, rng)
            losses.append(model.train_step(users, pos, negs, X, lr=0.01))
        history.append(float(np.mean(losses)))
        print(f"epoch {ep + 1}/{epochs} loss {history[-1]:.4f}", flush=True)
    ue = np.stack([model.user_emb(np.array([u]), X)[0] for u in range(n_users)])
    ie = np.stack([model.item_emb(np.array([u]), X)[0] for u in range(n_users)])
    np.savez("artifacts/embeddings.npz", user_emb=ue, item_emb=ie,
             Wu=model.Wu, Wi=model.Wi)
    json.dump({"loss_history": history}, open("artifacts/train.json", "w"), indent=2)
    # Save the graph split so evaluation / serving reuse the exact split.
    import pickle
    with open("artifacts/data.pkl", "wb") as f:
        pickle.dump({k: data[k] for k in ("friends", "train", "test", "majors", "years")}, f)
    np.save("artifacts/X.npy", X)
    print("saved artifacts/embeddings.npz")


if __name__ == "__main__":
    main()
