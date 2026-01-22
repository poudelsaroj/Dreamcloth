from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass(frozen=True)
class SelfProximityStats:
    self_proximity_rate: float
    self_proximity_depth_mean: float


def self_proximity_proxy(
    cloth_mesh: trimesh.Trimesh,
    distance_threshold: float,
    num_samples: int = 20000,
    seed: int = 0,
) -> SelfProximityStats:
    """
    Proxy for self-intersection / self-contact without requiring FCL.

    It flags frames where sampled points on the cloth surface come unusually close
    to other sampled points on different faces.
    """
    if distance_threshold <= 0:
        raise ValueError("distance_threshold must be > 0")

    rng = np.random.default_rng(seed)
    points, face_index = trimesh.sample.sample_surface(cloth_mesh, num_samples, seed=rng)
    points = np.asarray(points, dtype=np.float64)
    face_index = np.asarray(face_index, dtype=np.int64)

    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    dists, neighbors = tree.query(points, k=2, workers=-1)
    nearest_dist = dists[:, 1]
    nearest_neighbor = neighbors[:, 1]
    different_face = face_index != face_index[nearest_neighbor]
    proximity = (nearest_dist < distance_threshold) & different_face
    depth = np.where(proximity, distance_threshold - nearest_dist, 0.0)
    return SelfProximityStats(
        self_proximity_rate=float(np.mean(proximity)),
        self_proximity_depth_mean=float(np.mean(depth)),
    )

