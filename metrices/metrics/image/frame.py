from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@dataclass(frozen=True)
class ImageMetricResult:
    psnr: Optional[float]
    ssim: Optional[float]
    lpips: Optional[float]


def _to_float01(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def _apply_mask(image: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return image
    if mask.ndim == 2:
        mask_3 = mask[..., None]
    else:
        mask_3 = mask
    return image * mask_3


def compute_psnr_ssim(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    pred01 = _apply_mask(_to_float01(pred), mask)
    target01 = _apply_mask(_to_float01(target), mask)
    psnr_value = float(peak_signal_noise_ratio(target01, pred01, data_range=1.0))
    ssim_value = float(structural_similarity(target01, pred01, channel_axis=-1, data_range=1.0))
    return psnr_value, ssim_value


def compute_lpips_optional(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> Optional[float]:
    try:
        import torch
        import lpips  # type: ignore
    except Exception:
        return None

    pred01 = _apply_mask(_to_float01(pred), mask)
    target01 = _apply_mask(_to_float01(target), mask)

    pred_t = torch.from_numpy(pred01).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    target_t = torch.from_numpy(target01).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    pred_t = pred_t.to(device)
    target_t = target_t.to(device)

    model = lpips.LPIPS(net="vgg").to(device)
    with torch.no_grad():
        value = model(pred_t, target_t).item()
    return float(value)

