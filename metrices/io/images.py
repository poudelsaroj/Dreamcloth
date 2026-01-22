from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def load_image_rgb(path: Path) -> np.ndarray:
    from skimage.io import imread

    img = imread(str(path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return np.asarray(img)


def load_mask01(path: Path, threshold: float = 0.5) -> np.ndarray:
    mask = load_image_rgb(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0
    return (mask >= threshold).astype(np.float32)


def find_first_image(directory: Path) -> Optional[Path]:
    if not directory.exists():
        return None
    for suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        candidates = sorted(directory.glob(f"*{suffix}"))
        if candidates:
            return candidates[0]
    return None

