from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class JitterStats:
    jitter_mean: float
    jitter_p95: float


def vertex_acceleration_jitter(
    vertices_t_minus_1: np.ndarray,
    vertices_t: np.ndarray,
    vertices_t_plus_1: np.ndarray,
) -> Optional[JitterStats]:
    if vertices_t_minus_1.shape != vertices_t.shape or vertices_t.shape != vertices_t_plus_1.shape:
        return None
    if vertices_t.ndim != 2 or vertices_t.shape[1] != 3:
        return None

    accel = vertices_t_plus_1 - 2.0 * vertices_t + vertices_t_minus_1
    magnitude = np.linalg.norm(accel, axis=1)
    return JitterStats(jitter_mean=float(np.mean(magnitude)), jitter_p95=float(np.percentile(magnitude, 95)))

