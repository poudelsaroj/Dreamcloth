from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import EvalConfig, EvalPaths
from .io import find_first_image, load_image_rgb, load_mask01
from .io.mesh_sequence import MeshSequence, load_trimesh
from .metrics.geometry import chamfer_distance, fscore_at_threshold
from .metrics.image import compute_lpips_optional, compute_psnr_ssim
from .metrics.physics import (
    anchored_gap,
    anchored_slip,
    bending_violation,
    cloth_area_stats,
    cloth_body_penetration,
    define_anchor_set,
    self_proximity_proxy,
    stretch_violation,
    triangle_quality,
)
from .report import summarize_series
from .utils import sample_surface_points
from .metrics.temporal import vertex_acceleration_jitter


@dataclass(frozen=True)
class FrameResult:
    frame_index: int
    metrics: Dict[str, Any]


class Evaluator:
    """
    Evaluator for DreamCloth outputs.

    Primary use-case (single-image input):
    - Compute physics plausibility and temporal stability from the predicted rollout.
    Optional:
    - If GT cloth meshes exist: compute CD + F-score.
    - If render(s) exist: compute PSNR/SSIM/LPIPS against input image at t=0.
    - If raw/final renders exist: compute Phase3 correction budget.
    """

    def __init__(self, config: EvalConfig):
        self.config = config

    def evaluate(self, paths: EvalPaths) -> Dict[str, Any]:
        scale = self.config.scale_to_meters()
        pred_seq = MeshSequence(paths.pred_video_dir, max_frames=self.config.max_frames)

        has_gt = paths.gt_video_dir is not None

        pred_cloth_frames = pred_seq.load_frame_vertices("cloth")
        pred_body_frames = pred_seq.load_frame_vertices("body")

        frames_out: List[FrameResult] = []
        summary_series: Dict[str, List[float]] = {}

        t0_vertices = pred_cloth_frames[0][1] * scale
        t0_faces = pred_cloth_frames[0][2]
        t0_body_frame_index = pred_body_frames[0][0]
        t0_body_mesh = load_trimesh(paths.pred_video_dir / f"body_{t0_body_frame_index:04d}.obj")
        t0_body_mesh.apply_scale(scale)
        t0_cloth_mesh = load_trimesh(paths.pred_video_dir / f"cloth_{pred_cloth_frames[0][0]:04d}.obj")
        t0_cloth_mesh.apply_scale(scale)
        t0_area_m2 = float(t0_cloth_mesh.area)
        anchors = define_anchor_set(
            cloth_vertices_t0=t0_vertices,
            body_mesh_t0=t0_body_mesh,
            anchor_distance=self.config.physics.anchor_distance_m,
        )

        for idx, (frame_index, cloth_vertices, cloth_faces) in enumerate(pred_cloth_frames):
            cloth_vertices_m = cloth_vertices * scale
            body_mesh = load_trimesh(paths.pred_video_dir / f"body_{frame_index:04d}.obj")
            body_mesh.apply_scale(scale)

            per_frame: Dict[str, Any] = {}

            penetration = cloth_body_penetration(cloth_vertices_m, body_mesh)
            if penetration is not None:
                per_frame.update(asdict(penetration))

            gap = anchored_gap(
                cloth_vertices=cloth_vertices_m,
                body_mesh=body_mesh,
                anchors=anchors,
                contact_distance=self.config.physics.contact_distance_m,
            )
            if gap is not None:
                per_frame.update(asdict(gap))

            if idx + 1 < len(pred_cloth_frames):
                next_vertices_m = pred_cloth_frames[idx + 1][1] * scale
                next_frame_index = pred_cloth_frames[idx + 1][0]
                next_body_mesh = load_trimesh(paths.pred_video_dir / f"body_{next_frame_index:04d}.obj")
                next_body_mesh.apply_scale(scale)
                slip = anchored_slip(
                    cloth_vertices_t=cloth_vertices_m,
                    cloth_vertices_t1=next_vertices_m,
                    body_mesh_t=body_mesh,
                    body_mesh_t1=next_body_mesh,
                    anchors=anchors,
                )
                if slip is not None:
                    per_frame.update(asdict(slip))

            if cloth_vertices_m.shape[0] == t0_vertices.shape[0] and np.array_equal(cloth_faces, t0_faces):
                stretch = stretch_violation(
                    cloth_vertices_t0=t0_vertices,
                    cloth_vertices_t=cloth_vertices_m,
                    cloth_faces=cloth_faces,
                    stretch_threshold=self.config.physics.stretch_threshold,
                )
                per_frame.update(asdict(stretch))

                bending = bending_violation(
                    cloth_vertices_t0=t0_vertices,
                    cloth_vertices_t=cloth_vertices_m,
                    cloth_faces=cloth_faces,
                )
                if bending is not None:
                    per_frame.update(asdict(bending))

                tri_quality = triangle_quality(
                    cloth_vertices_t0=t0_vertices,
                    cloth_vertices_t=cloth_vertices_m,
                    cloth_faces=cloth_faces,
                )
                per_frame.update(asdict(tri_quality))

            cloth_mesh = load_trimesh(paths.pred_video_dir / f"cloth_{frame_index:04d}.obj")
            cloth_mesh.apply_scale(scale)
            per_frame.update(asdict(cloth_area_stats(cloth_mesh, area_t0_m2=t0_area_m2)))
            self_prox = self_proximity_proxy(
                cloth_mesh=cloth_mesh,
                distance_threshold=self.config.physics.self_proximity_threshold_m,
                num_samples=min(self.config.geometry.num_surface_samples, 20000),
                seed=self.config.random_seed + frame_index,
            )
            per_frame.update(asdict(self_prox))

            if self.config.temporal.compute_jitter and 0 < idx < len(pred_cloth_frames) - 1:
                prev_vertices_m = pred_cloth_frames[idx - 1][1] * scale
                next_vertices_m = pred_cloth_frames[idx + 1][1] * scale
                jitter = vertex_acceleration_jitter(prev_vertices_m, cloth_vertices_m, next_vertices_m)
                if jitter is not None:
                    per_frame.update(asdict(jitter))

            if has_gt:
                gt_cloth_path = paths.gt_video_dir / f"cloth_{frame_index:04d}.obj"  # type: ignore[operator]
                if gt_cloth_path.exists():
                    gt_mesh = load_trimesh(gt_cloth_path)
                    gt_mesh.apply_scale(scale)
                    pred_samples = sample_surface_points(
                        cloth_mesh, self.config.geometry.num_surface_samples, seed=self.config.random_seed + frame_index
                    ).points
                    gt_samples = sample_surface_points(
                        gt_mesh, self.config.geometry.num_surface_samples, seed=self.config.random_seed + 10_000 + frame_index
                    ).points
                    cd = chamfer_distance(pred_samples, gt_samples)
                    fscore = fscore_at_threshold(pred_samples, gt_samples, self.config.geometry.fscore_threshold_m)
                    per_frame.update(
                        {
                            "chamfer_l2": cd.chamfer_l2,
                            "chamfer_l2_squared": cd.chamfer_l2_squared,
                            "fscore": fscore.fscore,
                            "precision": fscore.precision,
                            "recall": fscore.recall,
                        }
                    )

            if (
                "penetration_depth_mean" in per_frame
                and "gap_mean" in per_frame
                and "slip_mean" in per_frame
                and self.config.physics.dc_physerr_tau_pen_m > 0
                and self.config.physics.dc_physerr_tau_gap_m > 0
                and self.config.physics.dc_physerr_tau_slip_m > 0
            ):
                per_frame["dc_physerr"] = (
                    float(per_frame["penetration_depth_mean"]) / self.config.physics.dc_physerr_tau_pen_m
                    + float(per_frame["gap_mean"]) / self.config.physics.dc_physerr_tau_gap_m
                    + float(per_frame["slip_mean"]) / self.config.physics.dc_physerr_tau_slip_m
                )

            frames_out.append(FrameResult(frame_index=frame_index, metrics=per_frame))
            for key, value in per_frame.items():
                if isinstance(value, (int, float)) and value is not None:
                    summary_series.setdefault(key, []).append(float(value))

        image_metrics = self._evaluate_single_image(paths)
        correction_budget = self._evaluate_correction_budget(paths)

        summary = {key: summarize_series(values) for key, values in summary_series.items()}
        if image_metrics:
            summary["t0_image"] = image_metrics
        if correction_budget:
            summary["correction_budget"] = correction_budget

        return {
            "meta": {
                "pred_video_dir": str(paths.pred_video_dir),
                "gt_video_dir": str(paths.gt_video_dir) if paths.gt_video_dir else None,
                "unit": self.config.unit,
                "scale_to_meters": scale,
                "num_frames": len(pred_seq),
                "anchor_vertex_count": int(anchors.indices.size),
            },
            "frames": [{"frame_index": fr.frame_index, **fr.metrics} for fr in frames_out],
            "summary": summary,
        }

    def _evaluate_single_image(self, paths: EvalPaths) -> Optional[Dict[str, Any]]:
        if paths.input_image is None:
            return None

        target = load_image_rgb(paths.input_image)
        mask = load_mask01(paths.input_mask) if (paths.input_mask and self.config.image.mask_background) else None

        pred_image_path = None
        if paths.pred_render_dir is not None:
            pred_image_path = find_first_image(paths.pred_render_dir)
        if pred_image_path is None:
            return None

        pred = load_image_rgb(pred_image_path)
        psnr_value, ssim_value = compute_psnr_ssim(pred, target, mask=mask)

        lpips_value = None
        if self.config.image.compute_lpips:
            lpips_value = compute_lpips_optional(pred, target, mask=mask, device="cpu")

        return {
            "pred_image": str(pred_image_path),
            "target_image": str(paths.input_image),
            "psnr": psnr_value if self.config.image.compute_psnr else None,
            "ssim": ssim_value if self.config.image.compute_ssim else None,
            "lpips": lpips_value,
            "mask": str(paths.input_mask) if paths.input_mask else None,
        }

    def _evaluate_correction_budget(self, paths: EvalPaths) -> Optional[Dict[str, Any]]:
        if paths.raw_render_dir is None or paths.final_render_dir is None:
            return None
        raw_image_path = find_first_image(paths.raw_render_dir)
        final_image_path = find_first_image(paths.final_render_dir)
        if raw_image_path is None or final_image_path is None:
            return None

        raw = load_image_rgb(raw_image_path)
        final = load_image_rgb(final_image_path)
        mask = load_mask01(paths.input_mask) if (paths.input_mask and self.config.image.mask_background) else None

        psnr_value, ssim_value = compute_psnr_ssim(final, raw, mask=mask)
        lpips_value = None
        if self.config.image.compute_lpips:
            lpips_value = compute_lpips_optional(final, raw, mask=mask, device="cpu")

        l1 = float(np.mean(np.abs(final.astype(np.float32) - raw.astype(np.float32))) / 255.0)
        return {
            "raw_image": str(raw_image_path),
            "final_image": str(final_image_path),
            "l1": l1,
            "psnr": psnr_value,
            "ssim": ssim_value,
            "lpips": lpips_value,
        }
