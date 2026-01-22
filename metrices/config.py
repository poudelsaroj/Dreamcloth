from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


Unit = Literal["m", "cm", "mm"]


def unit_scale_to_meters(unit: Unit) -> float:
    if unit == "m":
        return 1.0
    if unit == "cm":
        return 0.01
    if unit == "mm":
        return 0.001
    raise ValueError(f"Unsupported unit: {unit}")


@dataclass(frozen=True)
class EvalPaths:
    pred_video_dir: Path
    gt_video_dir: Optional[Path] = None

    pred_render_dir: Optional[Path] = None
    gt_render_dir: Optional[Path] = None

    input_image: Optional[Path] = None
    input_mask: Optional[Path] = None

    raw_render_dir: Optional[Path] = None
    final_render_dir: Optional[Path] = None

    params_json: Optional[Path] = None


@dataclass(frozen=True)
class GeometryConfig:
    num_surface_samples: int = 20000
    chamfer_squared: bool = False
    fscore_threshold_m: float = 0.001  # 1mm if meshes are in meters


@dataclass(frozen=True)
class PhysicsConfig:
    anchor_distance_m: float = 0.005   # 5mm: define anchored vertices from frame 0
    contact_distance_m: float = 0.002  # 2mm: acceptable contact band for "gap" metric
    stretch_threshold: float = 1.10
    self_proximity_threshold_m: float = 0.002

    dc_physerr_tau_pen_m: float = 0.002
    dc_physerr_tau_gap_m: float = 0.005
    dc_physerr_tau_slip_m: float = 0.005


@dataclass(frozen=True)
class TemporalConfig:
    compute_jitter: bool = True


@dataclass(frozen=True)
class ImageConfig:
    compute_psnr: bool = True
    compute_ssim: bool = True
    compute_lpips: bool = True  # optional dependency; will auto-disable if unavailable

    mask_background: bool = True


@dataclass(frozen=True)
class EvalConfig:
    unit: Unit = "m"

    geometry: GeometryConfig = GeometryConfig()
    physics: PhysicsConfig = PhysicsConfig()
    temporal: TemporalConfig = TemporalConfig()
    image: ImageConfig = ImageConfig()

    max_frames: Optional[int] = None
    random_seed: int = 0

    def scale_to_meters(self) -> float:
        return unit_scale_to_meters(self.unit)
