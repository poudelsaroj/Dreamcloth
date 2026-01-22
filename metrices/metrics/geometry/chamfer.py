from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ChamferResult:
    chamfer_l2: float
    chamfer_l2_squared: float
    forward_mean: float
    backward_mean: float


def chamfer_distance(x: np.ndarray, y: np.ndarray) -> ChamferResult:
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != 3 or y.shape[1] != 3:
        raise ValueError("x and y must be (N, 3) and (M, 3)")

    tree_y = cKDTree(y)
    tree_x = cKDTree(x)

    d_xy, _ = tree_y.query(x, k=1, workers=-1)
    d_yx, _ = tree_x.query(y, k=1, workers=-1)

    forward_mean = float(np.mean(d_xy))
    backward_mean = float(np.mean(d_yx))
    chamfer_l2 = forward_mean + backward_mean
    chamfer_l2_squared = float(np.mean(d_xy**2) + np.mean(d_yx**2))
    return ChamferResult(
        chamfer_l2=chamfer_l2,
        chamfer_l2_squared=chamfer_l2_squared,
        forward_mean=forward_mean,
        backward_mean=backward_mean,
    )


@dataclass(frozen=True)
class FScoreResult:
    fscore: float
    precision: float
    recall: float


def fscore_at_threshold(x: np.ndarray, y: np.ndarray, threshold: float) -> FScoreResult:
    if threshold <= 0:
        raise ValueError("threshold must be > 0")
    tree_y = cKDTree(y)
    tree_x = cKDTree(x)
    d_xy, _ = tree_y.query(x, k=1, workers=-1)
    d_yx, _ = tree_x.query(y, k=1, workers=-1)

    precision = float(np.mean(d_xy < threshold))
    recall = float(np.mean(d_yx < threshold))
    denom = precision + recall
    fscore = float(0.0 if denom == 0.0 else (2.0 * precision * recall / denom))
    return FScoreResult(fscore=fscore, precision=precision, recall=recall)

