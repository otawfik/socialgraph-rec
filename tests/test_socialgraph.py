import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from socialgraph.ann import IVFIndex
from socialgraph.data import augmented_features, generate
from socialgraph.evaluate import recall_ndcg
from socialgraph.model import TwoTower
from socialgraph.negatives import sample_negatives


@pytest.fixture(scope="module")
def data():
    return generate(n_users=200, seed=1)


def test_data_split_is_disjoint(data):
    for u, test in data["test"].items():
        assert not set(test) & set(data["train"].get(u, []))
        assert u not in test


def test_augmented_features_shape(data):
    xa = augmented_features(data["X"], data["friends"])
    assert xa.shape == (200, 2 * data["X"].shape[1])
    # A user with friends gets a nonzero second half.
    u = next(u for u in range(200) if data["friends"][u])
    assert xa[u, data["X"].shape[1]:].any()


def test_bpr_loss_decreases(data):
    X = augmented_features(data["X"], data["friends"])
    model = TwoTower(200, X.shape[1], dim=16, seed=0)
    pairs = [(u, v) for u, vs in data["train"].items() for v in vs]
    rng = np.random.default_rng(0)
    losses = []
    for ep in range(3):
        rng.shuffle(pairs)
        ep_losses = []
        for i in range(0, len(pairs), 128):
            b = pairs[i:i + 128]
            users = np.array([u for u, _ in b])
            pos = np.array([v for _, v in b])
            negs = sample_negatives(users, data["friends"], data["train"],
                                    data["clubs"], 200, rng)
            ep_losses.append(model.train_step(users, pos, negs, X, lr=0.01))
        losses.append(float(np.mean(ep_losses)))
    assert losses[-1] < losses[0], losses


def test_ann_close_to_brute_force():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(300, 16))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    index = IVFIndex(emb, n_clusters=8)
    q = emb[0]
    approx = set(index.search(q, k=10, n_probe=4))
    exact = set(index.brute_force(q, k=10))
    assert len(approx & exact) >= 6


def test_metrics_in_range():
    r, n = recall_ndcg([1, 2, 3], {2, 5}, k=3)
    assert 0.0 <= r <= 1.0 and 0.0 <= n <= 1.0
    assert r == 1 / 2


def test_api_endpoints():
    artifacts = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    if not os.path.exists(os.path.join(artifacts, "embeddings.npz")):
        pytest.skip("train first: artifacts not present")
    from fastapi.testclient import TestClient
    import app as app_module
    client = TestClient(app_module.app)
    assert client.get("/health").json()["status"] == "ok"
    rec = client.get("/recommend/0?k=5").json()
    assert len(rec["recommendations"]) == 5
    cold = client.post("/cold-start?k=5", json={"major": "cs", "year": 1,
                                                "clubs": [1], "interests": [2]}).json()
    assert len(cold["recommendations"]) == 5
    assert client.get("/recommend/99999").status_code == 404
