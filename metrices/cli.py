from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .config import EvalConfig, EvalPaths, GeometryConfig, ImageConfig, PhysicsConfig, TemporalConfig
from .evaluator import Evaluator
from .report import save_json


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DreamCloth evaluation metrics (Phase2 mesh rollouts + optional renders/GT)"
    )

    parser.add_argument(
        "--pred-video-dir",
        type=Path,
        required=True,
        help="Path to video_#### folder with cloth_*.obj and body_*.obj",
    )
    parser.add_argument(
        "--gt-video-dir",
        type=Path,
        default=None,
        help="Optional GT video_#### folder with cloth_*.obj (enables CD/F-score)",
    )

    parser.add_argument(
        "--unit",
        type=str,
        default="m",
        choices=["m", "cm", "mm"],
        help="Units of the OBJ meshes",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames for quick evaluation")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for surface sampling")

    parser.add_argument("--input-image", type=Path, default=None, help="Optional single input image (t=0 target)")
    parser.add_argument("--input-mask", type=Path,
     default=None, help="Optional foreground mask for image metrics")
    parser.add_argument(
        "--pred-render-dir",
        type=Path,
        default=None,
        help="Optional folder containing a rendered prediction image (frame 0)",
    )

    parser.add_argument(
        "--raw-render-dir",
        type=Path,
        default=None,
        help="Optional folder containing raw MPM render (for correction budget)",
    )
    parser.add_argument(
        "--final-render-dir",
        type=Path,
        default=None,
        help="Optional folder containing final Phase3 render (for correction budget)",
    )

    parser.add_argument(
        "--num-surface-samples",
        type=_positive_int,
        default=20000,
        help="Surface points for CD/F-score (when GT is available)",
    )
    parser.add_argument(
        "--fscore-threshold-mm",
        type=float,
        default=1.0,
        help="F-score threshold in millimeters (converted internally to meters)",
    )

    parser.add_argument("--anchor-distance-mm", type=float, default=5.0, help="Anchor set distance at t=0 in mm")
    parser.add_argument("--contact-distance-mm", type=float, default=2.0, help="Contact band in mm for anchored gap")
    parser.add_argument("--stretch-threshold", type=float, default=1.10, help="Stretch ratio threshold")
    parser.add_argument("--self-proximity-threshold-mm", type=float, default=2.0, help="Self-proximity proxy threshold in mm")
    parser.add_argument("--dc-tau-pen-mm", type=float, default=2.0, help="DC-PhysErr penetration depth scale in mm")
    parser.add_argument("--dc-tau-gap-mm", type=float, default=5.0, help="DC-PhysErr gap scale in mm")
    parser.add_argument("--dc-tau-slip-mm", type=float, default=5.0, help="DC-PhysErr slip scale in mm/frame")

    parser.add_argument("--out-json", type=Path, default=Path("metrics.json"), help="Output JSON path")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    geometry = GeometryConfig(
        num_surface_samples=args.num_surface_samples,
        fscore_threshold_m=(args.fscore_threshold_mm / 1000.0),
    )
    physics = PhysicsConfig(
        anchor_distance_m=(args.anchor_distance_mm / 1000.0),
        contact_distance_m=(args.contact_distance_mm / 1000.0),
        stretch_threshold=args.stretch_threshold,
        self_proximity_threshold_m=(args.self_proximity_threshold_mm / 1000.0),
        dc_physerr_tau_pen_m=(args.dc_tau_pen_mm / 1000.0),
        dc_physerr_tau_gap_m=(args.dc_tau_gap_mm / 1000.0),
        dc_physerr_tau_slip_m=(args.dc_tau_slip_mm / 1000.0),
    )
    config = EvalConfig(
        unit=args.unit,
        geometry=geometry,
        physics=physics,
        temporal=TemporalConfig(),
        image=ImageConfig(),
        max_frames=args.max_frames,
        random_seed=args.seed,
    )

    paths = EvalPaths(
        pred_video_dir=args.pred_video_dir,
        gt_video_dir=args.gt_video_dir,
        pred_render_dir=args.pred_render_dir,
        input_image=args.input_image,
        input_mask=args.input_mask,
        raw_render_dir=args.raw_render_dir,
        final_render_dir=args.final_render_dir,
    )

    evaluator = Evaluator(config)
    results = evaluator.evaluate(paths)
    save_json(args.out_json, results)
    print(f"Wrote: {args.out_json}")
