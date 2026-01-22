from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import trimesh

from ...utils.sampling import face_normals, unique_edges_from_faces


@dataclass(frozen=True)
class StretchStats:
    stretch_violation_mean: float
    stretch_violation_p95: float
    stretch_over_threshold_rate: float


def stretch_violation(
    cloth_vertices_t0: np.ndarray,
    cloth_vertices_t: np.ndarray,
    cloth_faces: np.ndarray,
    stretch_threshold: float,
) -> StretchStats:
    edges = unique_edges_from_faces(cloth_faces)
    e0 = np.linalg.norm(cloth_vertices_t0[edges[:, 0]] - cloth_vertices_t0[edges[:, 1]], axis=1) + 1e-12
    et = np.linalg.norm(cloth_vertices_t[edges[:, 0]] - cloth_vertices_t[edges[:, 1]], axis=1)
    ratio = et / e0
    violation = np.abs(ratio - 1.0)
    return StretchStats(
        stretch_violation_mean=float(np.mean(violation)),
        stretch_violation_p95=float(np.percentile(violation, 95)),
        stretch_over_threshold_rate=float(np.mean(ratio > stretch_threshold)),
    )


def _edge_to_two_faces(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges_sorted = np.sort(edges, axis=1)
    face_ids = np.repeat(np.arange(faces.shape[0]), 3)

    order = np.lexsort((edges_sorted[:, 1], edges_sorted[:, 0]))
    edges_sorted = edges_sorted[order]
    face_ids = face_ids[order]

    shared_edges = []
    face_pairs = []
    start = 0
    while start < edges_sorted.shape[0]:
        end = start + 1
        while end < edges_sorted.shape[0] and np.all(edges_sorted[end] == edges_sorted[start]):
            end += 1
        if end - start == 2:
            shared_edges.append(edges_sorted[start])
            face_pairs.append([face_ids[start], face_ids[start + 1]])
        start = end

    if not shared_edges:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 2), dtype=np.int64)
    return np.asarray(shared_edges, dtype=np.int64), np.asarray(face_pairs, dtype=np.int64)


@dataclass(frozen=True)
class BendingStats:
    bend_violation_mean: float
    bend_violation_p95: float


def bending_violation(
    cloth_vertices_t0: np.ndarray,
    cloth_vertices_t: np.ndarray,
    cloth_faces: np.ndarray,
) -> Optional[BendingStats]:
    _, face_pairs = _edge_to_two_faces(cloth_faces)
    if face_pairs.shape[0] == 0:
        return None

    normals0 = face_normals(cloth_vertices_t0, cloth_faces)
    normalst = face_normals(cloth_vertices_t, cloth_faces)

    n0_a = normals0[face_pairs[:, 0]]
    n0_b = normals0[face_pairs[:, 1]]
    nt_a = normalst[face_pairs[:, 0]]
    nt_b = normalst[face_pairs[:, 1]]

    theta0 = np.arccos(np.clip(np.sum(n0_a * n0_b, axis=1), -1.0, 1.0))
    thetat = np.arccos(np.clip(np.sum(nt_a * nt_b, axis=1), -1.0, 1.0))
    diff2 = (thetat - theta0) ** 2
    return BendingStats(
        bend_violation_mean=float(np.mean(diff2)),
        bend_violation_p95=float(np.percentile(diff2, 95)),
    )


@dataclass(frozen=True)
class TriangleQualityStats:
    flipped_face_rate: float
    degenerate_face_rate: float


def triangle_quality(
    cloth_vertices_t0: np.ndarray,
    cloth_vertices_t: np.ndarray,
    cloth_faces: np.ndarray,
    area_epsilon: float = 1e-12,
) -> TriangleQualityStats:
    normals0 = face_normals(cloth_vertices_t0, cloth_faces)
    normalst = face_normals(cloth_vertices_t, cloth_faces)
    alignment = np.sum(normals0 * normalst, axis=1)
    flipped = alignment < 0.0

    tri = cloth_vertices_t[cloth_faces]
    area_vec = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(area_vec, axis=1)
    degenerate = area < area_epsilon

    return TriangleQualityStats(
        flipped_face_rate=float(np.mean(flipped)),
        degenerate_face_rate=float(np.mean(degenerate)),
    )


@dataclass(frozen=True)
class AreaStats:
    area_m2: float
    area_ratio: float


def cloth_area_stats(cloth_mesh: trimesh.Trimesh, area_t0_m2: float) -> AreaStats:
    area = float(cloth_mesh.area)
    ratio = float(area / (area_t0_m2 + 1e-12))
    return AreaStats(area_m2=area, area_ratio=ratio)

