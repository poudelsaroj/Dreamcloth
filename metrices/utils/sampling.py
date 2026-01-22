from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass(frozen=True)
class SurfaceSamples:
    points: np.ndarray  # (N, 3)
    face_index: np.ndarray  # (N,)


def sample_surface_points(mesh: trimesh.Trimesh, num_samples: int, seed: int = 0) -> SurfaceSamples:
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    rng = np.random.default_rng(seed)
    points, face_index = trimesh.sample.sample_surface(mesh, num_samples, seed=rng)
    return SurfaceSamples(
        points=np.asarray(points, dtype=np.float32),
        face_index=np.asarray(face_index, dtype=np.int64),
    )


def unique_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be (F, 3)")
    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return edges


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]  # (F, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return n / norm

