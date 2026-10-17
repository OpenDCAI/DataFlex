"""Deterministic numerical building blocks shared by multimodal selectors.

No model loading or optional clustering imports happen at import time.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def allocate_budget(capacities, budget, weights=None):
    """Capped weighted allocation with deterministic rounding and exact count.

    Uniform weights implement Adapt-inf's equal skill budget. Capacity weights
    implement PreSel's proportional allocation within a task.
    """
    capacities = np.asarray(capacities, dtype=np.int64)
    if capacities.ndim != 1 or (capacities < 0).any():
        raise ValueError("capacities must be a nonnegative vector")
    if int(budget) != budget or budget < 0 or budget > capacities.sum():
        raise ValueError(f"budget {budget} exceeds available capacity {capacities.sum()}")
    weights = np.ones(len(capacities)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != capacities.shape or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite, nonnegative and match capacities")
    allocation = np.zeros_like(capacities)
    remaining = int(budget)
    while remaining:
        available = np.flatnonzero(allocation < capacities)
        active_weights = weights[available]
        if not active_weights.sum():
            active_weights = np.ones(len(available))
        quotas = remaining * active_weights / active_weights.sum()
        additions = np.minimum(np.floor(quotas).astype(int), capacities[available] - allocation[available])
        if additions.sum():
            allocation[available] += additions
            remaining -= int(additions.sum())
        else:
            order = np.argsort(-(quotas - np.floor(quotas)), kind="stable")
            chosen = available[order[:remaining]]
            allocation[chosen] += 1
            remaining -= len(chosen)
    return allocation.tolist()




def cluster_features(features, clusters, seed, minibatch=True):
    from sklearn.cluster import KMeans, MiniBatchKMeans

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or not len(features) or not np.isfinite(features).all():
        raise ValueError("features must be a nonempty finite matrix")
    if clusters < 1:
        raise ValueError("clusters must be positive")
    clusters = min(int(clusters), len(features))
    options = dict(n_clusters=clusters, random_state=int(seed), n_init=3)
    estimator = MiniBatchKMeans(batch_size=max(1024, clusters), **options) if minibatch else KMeans(**options)
    return estimator.fit_predict(features)


def neighbor_centrality(features, neighbors=5, chunk_size=1024):
    """Exact mean cosine similarity to k nearest *other* cluster members.

    Chunking bounds memory; this remains quadratic in time within a cluster.
    Explicit diagonal masking handles duplicate vectors correctly.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("features must be a finite matrix")
    if neighbors < 1 or chunk_size < 1:
        raise ValueError("neighbors and chunk_size must be positive")
    if len(features) <= 1:
        return np.ones(len(features), dtype=np.float32)
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    k = min(int(neighbors), len(features) - 1)
    result = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), chunk_size):
        stop = min(start + chunk_size, len(features))
        similarities = features[start:stop] @ features.T
        similarities[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        result[start:stop] = np.partition(similarities, -k, axis=1)[:, -k:].mean(axis=1)
    return result
