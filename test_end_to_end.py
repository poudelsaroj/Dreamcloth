"""
End-to-end test for PhaseI -> MPM -> Phase3 pipeline.

This is a lightweight smoke test that:
- Resolves PhaseI meshes
- Verifies MPM outputs are present
- Runs a single Phase3 loss/gradient step on rendered MPM video
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.amp import autocast

sys.path.insert(0, os.path.dirname(__file__))

from phase3.dataloader import create_dataloader
from phase3.model import create_video_diffusion_model
from phase3.train import MPMParameters
from phase3.utils import DiffusionScheduler, compute_diffusion_loss, normalize_video


def _find_first_obj(root: Path, patterns, label: str) -> Optional[Path]:
    if not root.exists():
        return None
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            if len(matches) > 1:
                print(f"[warn] Multiple {label} candidates found. Using: {matches[0]}")
            return matches[0]
    return None


def resolve_phase1_meshes(
    phase1_output_root: Path,
    body_mesh: Optional[Path],
    cloth_mesh: Optional[Path],
) -> Tuple[Path, Path]:
    if body_mesh and cloth_mesh:
        return body_mesh, cloth_mesh

    econ_root = phase1_output_root / "econ"
    garment_root = phase1_output_root / "garment3d"

    if body_mesh is None:
        body_mesh = _find_first_obj(
            econ_root,
            ["*_smpl.obj", "*smpl*.obj", "*body*.obj", "*.obj"],
            "body mesh",
        )

    if cloth_mesh is None:
        cloth_mesh = _find_first_obj(
            garment_root,
            ["*cloth*.obj", "*garment*.obj", "*.obj"],
            "cloth mesh",
        )

    if body_mesh is None or cloth_mesh is None:
        raise FileNotFoundError(
            "Could not resolve PhaseI meshes. Provide --body-mesh and --cloth-mesh."
        )

    return body_mesh, cloth_mesh


def adapt_video_with_params(
    base_videos: torch.Tensor,
    mpm_params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    young = mpm_params["young_modulus"]
    poisson = mpm_params["poisson_ratio"]
    shear = mpm_params["shear_stiffness"]
    normal = mpm_params["normal_stiffness"]
    friction = mpm_params["friction"]
    density = mpm_params["density"]
    damping = mpm_params["damping"]

    scale_young = (0.6 + 0.4 * torch.sigmoid((young - 100.0) / 100.0)).view(1, 1, 1, 1, 1)
    scale_poisson = (0.9 + 0.1 * poisson).view(1, 1, 1, 1, 1)
    scale_shear = (0.8 + 0.2 * torch.sigmoid((shear - 500.0) / 250.0)).view(1, 1, 1, 1, 1)
    scale_normal = (0.8 + 0.2 * torch.sigmoid((normal - 500.0) / 250.0)).view(1, 1, 1, 1, 1)
    scale_friction = (0.85 + 0.15 * friction).view(1, 1, 1, 1, 1)
    density_factor = torch.sigmoid(density - 1.0).view(1, 1, 1, 1, 1)
    damping_factor = (0.95 + 0.05 * damping).view(1, 1, 1, 1, 1)

    time = torch.linspace(0.0, 1.0, base_videos.shape[2], device=base_videos.device)
    time = time.view(1, 1, -1, 1, 1)

    video = base_videos * scale_young
    video = video * scale_poisson
    video = video * scale_shear
    video = video * scale_normal
    video = video * scale_friction
    video = video * (0.9 + 0.1 * density_factor * time)
    video = video * (damping_factor - 0.02 * time)
    return video.clamp(0.0, 1.0)


def validate_mpm_outputs(mpm_output_dir: Path, video_length: int) -> Path:
    video_dir = mpm_output_dir / "video_0000"
    if not video_dir.exists():
        raise FileNotFoundError(f"Missing MPM output directory: {video_dir}")

    cloth_files = sorted(video_dir.glob("cloth_*.obj"))
    if len(cloth_files) < video_length:
        raise FileNotFoundError(
            f"Not enough cloth frames in {video_dir} "
            f"(found {len(cloth_files)}, expected >= {video_length})."
        )

    return video_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end pipeline smoke test")
    parser.add_argument("--phase1-output-root", type=Path, default=Path("PhaseI/full_pipeline_outputs/collected_econ_and_garment3d"))
    parser.add_argument("--body-mesh", type=Path, default=None)
    parser.add_argument("--cloth-mesh", type=Path, default=None)
    parser.add_argument("--mpm-output-dir", type=Path, default=Path("output/phase2_sim"))
    parser.add_argument("--video-length", type=int, default=8)
    parser.add_argument("--spatial-size", type=int, nargs=2, default=(64, 64))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device == "cuda"

    body_mesh, cloth_mesh = resolve_phase1_meshes(
        args.phase1_output_root, args.body_mesh, args.cloth_mesh
    )
    print(f"[phase1] Body mesh:  {body_mesh}")
    print(f"[phase1] Cloth mesh: {cloth_mesh}")

    validate_mpm_outputs(args.mpm_output_dir, args.video_length)
    print(f"[phase2] MPM outputs OK: {args.mpm_output_dir}")

    dataloader = create_dataloader(
        data_root=str(args.mpm_output_dir),
        batch_size=args.batch_size,
        video_length=args.video_length,
        spatial_size=tuple(args.spatial_size),
        shuffle=False,
        use_dummy=False,
    )
    if len(dataloader.dataset) == 0:
        raise FileNotFoundError(
            f"No rendered videos found in {args.mpm_output_dir}. "
            "Check the MPM output path."
        )

    base_videos = next(iter(dataloader)).to(device)
    diffusion_model = create_video_diffusion_model(device=device, fp16=use_amp)
    scheduler = DiffusionScheduler(timesteps=1000, schedule="cosine", device=device)
    mpm_params = MPMParameters(device=device)

    params = mpm_params()
    simulated_videos = adapt_video_with_params(base_videos, params)
    simulated_videos_norm = normalize_video(simulated_videos)
    if use_amp:
        simulated_videos_norm = simulated_videos_norm.half()

    timesteps = torch.randint(0, scheduler.timesteps, (simulated_videos_norm.shape[0],), device=device).long()
    noise = torch.randn_like(simulated_videos_norm)
    noisy_videos = scheduler.q_sample(simulated_videos_norm, timesteps, noise)

    with autocast("cuda", enabled=use_amp):
        predicted_noise = diffusion_model(noisy_videos, timesteps)
        loss = compute_diffusion_loss(predicted_noise, noise)

    loss.backward()

    has_grad = all(p.grad is not None and p.grad.norm().item() > 0 for p in mpm_params.parameters())
    print(f"[phase3] Loss: {loss.item():.6f}")
    print(f"[phase3] Gradients OK: {has_grad}")

    if not has_grad:
        raise RuntimeError("Gradient check failed for MPM parameters.")


if __name__ == "__main__":
    main()
