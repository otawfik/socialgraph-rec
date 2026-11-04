"""FastAPI serving for SocialGraph Rec.

Endpoints:
  GET  /health                 service + artifact status
  GET  /recommend/{user_id}    top-K people to connect with (IVF retrieval)
  GET  /similar/{user_id}      nearest neighbours in item-embedding space
  POST /cold-start             recommendations for a brand-new user profile
                               (major, year, clubs, interests; no history)
"""
import os
import pickle
import sys

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from socialgraph.ann import IVFIndex

BASE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="SocialGraph Rec")

_X = np.load(os.path.join(BASE, "artifacts", "X.npy"))
_emb = np.load(os.path.join(BASE, "artifacts", "embeddings.npz"))
_ue, _ie = _emb["user_emb"], _emb["item_emb"]
_data = pickle.load(open(os.path.join(BASE, "artifacts", "data.pkl"), "rb"))
_friends, _train = _data["friends"], _data["train"]
_index = IVFIndex(_ie)
MAJORS = ["cs", "econ", "data", "finance", "bio", "mech", "psych", "math", "design", "polisci"]


def _excluded(user_id):
    return [user_id] + list(_friends[user_id]) + _train.get(user_id, [])


def _profile_vector(major, year, clubs, interests):
    # _X is the augmented matrix (profile + friend-mean), so a raw
    # profile is half its width; cold-start users get zeros for the
    # friend-mean half.
    x = np.zeros((1, _X.shape[1] // 2))
    if major in MAJORS:
        x[0, MAJORS.index(major)] = 1.0
    if 0 <= year <= 3:
        x[0, 10 + year] = 1.0
    for c in clubs:
        if 0 <= c < 20:
            x[0, 14 + c] = 1.0
    for it in interests:
        if 0 <= it < 30:
            x[0, 34 + it] = 1.0
    return x


@app.get("/health")
def health():
    return {"status": "ok", "n_users": int(_X.shape[0]),
            "embedding_dim": int(_ie.shape[1])}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 10):
    if not 0 <= user_id < _X.shape[0]:
        raise HTTPException(404, "unknown user")
    recs = _index.search(_ue[user_id], k=k, exclude=_excluded(user_id))
    return {"user_id": user_id, "recommendations": [
        {"user_id": int(v), "major": MAJORS[int(_data["majors"][v])],
         "year": int(_data["years"][v])} for v in recs]}


@app.get("/similar/{user_id}")
def similar(user_id: int, k: int = 10):
    if not 0 <= user_id < _X.shape[0]:
        raise HTTPException(404, "unknown user")
    recs = _index.search(_ie[user_id], k=k, exclude=[user_id])
    return {"user_id": user_id, "similar": [int(v) for v in recs]}


class ColdStartProfile(BaseModel):
    major: str = "cs"
    year: int = 1
    clubs: list[int] = []
    interests: list[int] = []


@app.post("/cold-start")
def cold_start(profile: ColdStartProfile, k: int = 10):
    # New users have no free embedding and no friends, so only the item
    # tower's profile projection is available: reuse it as the query.
    # Friend-mean features are zero for a user with no friends yet.
    x = _profile_vector(profile.major, profile.year, profile.clubs, profile.interests)
    xa = np.concatenate([x, np.zeros_like(x)], axis=1)
    # The saved Wi maps the augmented feature space; the profile half is
    # what a cold-start user can actually fill in.
    Wi = _emb["Wi"]
    q = xa @ Wi
    q = q / np.linalg.norm(q).clip(min=1e-9)
    recs = _index.search(q[0], k=k)
    return {"recommendations": [int(v) for v in recs]}


app.mount("/demo", StaticFiles(directory=os.path.join(BASE, "demo"), html=True), name="demo")


@app.get("/")
def root():
    return FileResponse(os.path.join(BASE, "demo", "index.html"))
