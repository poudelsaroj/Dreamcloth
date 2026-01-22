from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh


def _infer_outside_sign(body_mesh: trimesh.Trimesh) -> float:
    bounds = body_mesh.bounds
    center = 0.5 * (bounds[0] + bounds[1])
    extent = np.maximum(bounds[1] - bounds[0], 1e-6)
    far_point = center + 10.0 * extent
    from trimesh.proximity import signed_distance

    signed = float(signed_distance(body_mesh, far_point.reshape(1, 3))[0])
    return 1.0 if signed >= 0.0 else -1.0


@dataclass(frozen=True)
class PenetrationStats:
    penetration_rate: float
    penetration_depth_mean: float
    penetration_depth_p95: float


def cloth_body_penetration(cloth_vertices: np.ndarray, body_mesh: trimesh.Trimesh) -> Optional[PenetrationStats]:
    """
    Cloth-body penetration based on signed distance to the body surface.

    Returns None if signed distance is unavailable (e.g., non-watertight body or missing deps).
    """
    try:
        from trimesh.proximity import signed_distance

        outside_sign = _infer_outside_sign(body_mesh)
        signed = np.asarray(signed_distance(body_mesh, cloth_vertices), dtype=np.float64)
        inside = (signed * outside_sign) < 0.0
        depths = np.where(inside, np.abs(signed), 0.0)
        return PenetrationStats(
            penetration_rate=float(np.mean(inside)),
            penetration_depth_mean=float(np.mean(depths)),
            penetration_depth_p95=float(np.percentile(depths, 95)),
        )
    except Exception:
        return None


@dataclass(frozen=True)
class AnchorSet:
    indices: np.ndarray


def define_anchor_set(
    cloth_vertices_t0: np.ndarray,
    body_mesh_t0: trimesh.Trimesh,
    anchor_distance: float,
) -> AnchorSet:
    from trimesh.proximity import closest_point

    _, distances, _ = closest_point(body_mesh_t0, cloth_vertices_t0)
    indices = np.flatnonzero(np.asarray(distances) < anchor_distance).astype(np.int64)
    return AnchorSet(indices=indices)


@dataclass(frozen=True)
class GapStats:
    gap_mean: float
    gap_p95: float
    contact_rate: float


def anchored_gap(
    cloth_vertices: np.ndarray,
    body_mesh: trimesh.Trimesh,
    anchors: AnchorSet,
    contact_distance: float,
) -> Optional[GapStats]:
    if anchors.indices.size == 0:
        return None
    from trimesh.proximity import closest_point

    points = cloth_vertices[anchors.indices]
    _, distances, _ = closest_point(body_mesh, points)
    distances = np.asarray(distances, dtype=np.float64)
    gap = np.maximum(0.0, distances - contact_distance)
    contact_rate = float(np.mean(distances <= contact_distance))
    return GapStats(gap_mean=float(np.mean(gap)), gap_p95=float(np.percentile(gap, 95)), contact_rate=contact_rate)


@dataclass(frozen=True)
class SlipStats:
    slip_mean: float
    slip_p95: float


def anchored_slip(
    cloth_vertices_t: np.ndarray,
    cloth_vertices_t1: np.ndarray,
    body_mesh_t: trimesh.Trimesh,
    body_mesh_t1: trimesh.Trimesh,
    anchors: AnchorSet,
) -> Optional[SlipStats]:
    if anchors.indices.size == 0:
        return None

    from trimesh.proximity import closest_point

    cloth_t = cloth_vertices_t[anchors.indices]
    cloth_t1 = cloth_vertices_t1[anchors.indices]

    closest_t, _, tri_id_t = closest_point(body_mesh_t, cloth_t)
    closest_t1, _, _ = closest_point(body_mesh_t1, cloth_t1)

    normals_t = body_mesh_t.face_normals[np.asarray(tri_id_t, dtype=np.int64)]
    normals_t = np.asarray(normals_t, dtype=np.float64)

    delta_cloth = cloth_t1 - cloth_t
    delta_body = closest_t1 - closest_t
    delta_rel = delta_cloth - delta_body

    normal_component = np.sum(delta_rel * normals_t, axis=1, keepdims=True) * normals_t
    tangential = delta_rel - normal_component
    slip = np.linalg.norm(tangential, axis=1)
    return SlipStats(slip_mean=float(np.mean(slip)), slip_p95=float(np.percentile(slip, 95)))

