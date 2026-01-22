"""
End-to-end training pipeline: PhaseI -> MPM -> Phase3.

This script orchestrates:
1) PhaseI outputs (body + cloth meshes)
2) MPM simulation (Phase2)
3) Phase3 optimization using diffusion guidance

Notes:
- MPM is not differentiable; we apply a small, differentiable modulation on
  top of the rendered MPM video so Phase3 parameters receive gradients.
- Provide explicit mesh paths if auto-detection picks the wrong file.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.amp import autocast, GradScaler
import trimesh

sys.path.insert(0, os.path.dirname(__file__))

from phase3.dataloader import create_dataloader
from phase3.model import create_video_diffusion_model
from phase3.train import MPMParameters
from phase3.utils import (
    AverageMeter,
    DiffusionScheduler,
    compute_diffusion_loss,
    normalize_video,
    save_checkpoint,
    set_seed,
)
from phase3.wan22_i2v_guidance import Wan22I2VConfig, Wan22I2VGuidance


def _maybe_init_wandb(args: argparse.Namespace, extra_config: Dict[str, Any] | None = None):
    if not getattr(args, "wandb", False):
        return None

    try:
        import wandb  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "W&B is enabled but 'wandb' is not installed. Install with: pip install wandb"
        ) from e

    config: Dict[str, Any] = {
        "seed": args.seed,
        "device": args.device,
        "use_amp": not args.no_amp,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "video_length": args.video_length,
        "spatial_size": list(args.spatial_size),
        "run_phase1": args.run_phase1,
        "skip_mpm": args.skip_mpm,
        "mpm_frames": args.mpm_frames,
        "mpm_save_every": args.mpm_save_every,
        "mpm_y_offset": args.mpm_y_offset,
        "phase1_output_root": str(args.phase1_output_root),
        "mpm_output_dir": str(args.mpm_output_dir),
        "checkpoint_dir": str(args.checkpoint_dir),
    }
    if extra_config:
        config.update(extra_config)

    tags = None
    if getattr(args, "wandb_tags", None):
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]

    run = wandb.init(
        entity=getattr(args, "wandb_entity", None) or None,
        project=getattr(args, "wandb_project", None) or None,
        name=getattr(args, "wandb_name", None) or None,
        group=getattr(args, "wandb_group", None) or None,
        notes=getattr(args, "wandb_notes", None) or None,
        tags=tags,
        dir=str(getattr(args, "wandb_dir", "wandb")),
        mode=getattr(args, "wandb_mode", "online"),
        config=config,
    )

    define_metric = getattr(run, "define_metric", None) or getattr(wandb, "define_metric", None)
    if callable(define_metric):
        define_metric("train/step")
        define_metric("train/*", step_metric="train/step")
        define_metric("mpm/*", step_metric="train/step")
        define_metric("time/*", step_metric="train/step")
    return run


def _find_first_obj(root: Path, patterns: List[str], label: str) -> Optional[Path]:
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


def load_mesh_vertices_faces(mesh_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise ValueError(f"Invalid mesh: {mesh_path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    return vertices, faces


def build_body_sequence(
    body_mesh: Path,
    body_sequence_dir: Optional[Path],
    n_frames: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if body_sequence_dir and body_sequence_dir.exists():
        frame_paths = sorted(body_sequence_dir.glob("*.obj"))
        if not frame_paths:
            raise FileNotFoundError(f"No .obj frames found in {body_sequence_dir}")
        frames = [load_mesh_vertices_faces(p) for p in frame_paths]
    else:
        frames = [load_mesh_vertices_faces(body_mesh)]

    if n_frames <= 0:
        return frames

    if len(frames) < n_frames:
        last = frames[-1]
        frames = frames + [last] * (n_frames - len(frames))
    else:
        frames = frames[:n_frames]

    return frames


def run_phase1(
    phase1_script: Path,
    input_image: Path,
    output_root: Path,
    skip_env: bool,
    skip_clone: bool,
) -> None:
    import subprocess

    cmd = ["bash", str(phase1_script), str(input_image), str(output_root)]
    env = os.environ.copy()
    if skip_env:
        env["SKIP_ENV"] = "1"
    if skip_clone:
        env["SKIP_CLONE"] = "1"
    subprocess.run(cmd, check=True, env=env)


def run_mpm_simulation(
    cloth_mesh_path: Path,
    body_sequence: List[Tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    save_every: int,
    y_offset: float,
) -> Path:
    import mpm.mpm_cloth_v28 as mpm

    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "video_0000"
    video_dir.mkdir(parents=True, exist_ok=True)

    cloth_mesh = trimesh.load(str(cloth_mesh_path), force="mesh")
    positions = mpm.init_particles_from_mesh(cloth_mesh, y_offset=y_offset)

    verts0, faces0 = body_sequence[0]
    mpm.setup_smart_attachment(positions, mpm.cloth_faces_np, verts0)
    mpm.update_body_collision(verts0, faces0)
    mpm.update_body_velocities(verts0, None)
    mpm.update_attachment_schedule(0)

    prev_verts = verts0.copy()
    save_index = 0

    for frame, (verts, faces) in enumerate(body_sequence):
        mpm.update_body_velocities(verts, prev_verts)
        mpm.update_body_collision(verts, faces, prev_verts)
        mpm.update_attachment_schedule(frame)

        for _ in range(mpm.SUBSTEPS):
            mpm.clear_collision_grid()
            mpm.mesh_to_grid_collision()
            mpm.p2g()
            mpm.grid_op()
            mpm.g2p()
            mpm.apply_kinematic_attachment()
            mpm.apply_distance_failsafe()
            mpm.apply_body_pushout(mpm.BODY_PUSHOUT)

        prev_verts = verts.copy()

        if frame % save_every == 0:
            cloth_path = video_dir / f"cloth_{save_index:04d}.obj"
            body_path = video_dir / f"body_{save_index:04d}.obj"
            mpm.save_cloth_mesh(str(cloth_path), mpm.cloth_faces_np)
            mpm.save_body_mesh(str(body_path), verts, faces)
            save_index += 1

    return video_dir


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


def train_one_epoch(
    diffusion_model: torch.nn.Module,
    mpm_params: MPMParameters,
    scheduler: DiffusionScheduler,
    optimizer: torch.optim.Optimizer,
    dataloader,
    device: str,
    epoch: int,
    scaler: Optional[GradScaler],
    use_amp: bool,
    wandb_run: Any = None,
    global_step: int = 0,
    log_every: int = 10,
    wan_guidance: Any = None,
    wan_cond_image_01: Any = None,
) -> Tuple[float, Dict[str, float], int]:
    mpm_params.train()
    diffusion_model.eval()

    loss_meter = AverageMeter()

    for batch_idx, base_videos in enumerate(dataloader):
        base_videos = base_videos.to(device)
        optimizer.zero_grad()

        params = mpm_params()
        simulated_videos = adapt_video_with_params(base_videos, params)

        if wan_guidance is not None:
            if wan_cond_image_01 is None:
                raise ValueError("WAN backend requires a conditioning image tensor.")
            batch_size = simulated_videos.shape[0]
            timesteps = torch.randint(0, wan_guidance.num_train_timesteps, (batch_size,), device=device).long()
            loss = wan_guidance.compute_loss(simulated_videos, wan_cond_image_01, timesteps=timesteps)
        else:
            simulated_videos_norm = normalize_video(simulated_videos)

            batch_size = simulated_videos_norm.shape[0]
            timesteps = torch.randint(0, scheduler.timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(simulated_videos_norm)

            if use_amp:
                noise = noise.half()
                simulated_videos_norm = simulated_videos_norm.half()

            noisy_videos = scheduler.q_sample(simulated_videos_norm, timesteps, noise)

            with autocast("cuda", enabled=use_amp):
                predicted_noise = diffusion_model(noisy_videos, timesteps)
                loss = compute_diffusion_loss(predicted_noise, noise)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        mpm_params.clamp_parameters()
        loss_meter.update(loss.item(), batch_size)

        global_step += 1
        if wandb_run is not None and log_every > 0 and (global_step % log_every == 0):
            wandb_run.log(
                {
                    "train/step": global_step,
                    "train/loss": float(loss.item()),
                    "train/loss_avg": float(loss_meter.avg),
                    "train/epoch": int(epoch),
                }
            )

        if batch_idx % 10 == 0:
            print(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.6f} (avg: {loss_meter.avg:.6f})"
            )

    return loss_meter.avg, mpm_params.get_params_dict(), global_step


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end training pipeline")
    parser.add_argument("--phase1-output-root", type=Path, default=Path("PhaseI/full_pipeline_outputs/collected_econ_and_garment3d"))
    parser.add_argument("--body-mesh", type=Path, default=None)
    parser.add_argument("--cloth-mesh", type=Path, default=None)
    parser.add_argument("--body-sequence-dir", type=Path, default=None)
    parser.add_argument("--run-phase1", action="store_true")
    parser.add_argument("--phase1-script", type=Path, default=Path("PhaseI/run_full_pipeline.sh"))
    parser.add_argument("--phase1-input-image", type=Path, default=None)
    parser.add_argument("--phase1-skip-env", action="store_true")
    parser.add_argument("--phase1-skip-clone", action="store_true")
    parser.add_argument("--skip-mpm", action="store_true")
    parser.add_argument("--mpm-output-dir", type=Path, default=Path("output/phase2_sim"))
    parser.add_argument("--mpm-frames", type=int, default=100)
    parser.add_argument("--mpm-save-every", type=int, default=2)
    parser.add_argument("--mpm-y-offset", type=float, default=-0.05)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--video-length", type=int, default=8)
    parser.add_argument("--spatial-size", type=int, nargs=2, default=(64, 64))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/end_to_end"))

    # Diffusion prior backend
    parser.add_argument("--diffusion-backend", type=str, default="local", choices=["local", "wan22-i2v"])

    # Wan2.2 I2V backend options
    parser.add_argument("--wan-repo-root", type=Path, default=None, help="Path to cloned Wan2.2 repo (so we can import `wan`)")
    parser.add_argument("--wan-ckpt-dir", type=Path, default=None, help="Path to Wan2.2-I2V checkpoint directory (contains low_noise_model/high_noise_model/VAE/T5)")
    parser.add_argument("--wan-prompt", type=str, default="", help="Prompt for Wan2.2 I2V conditioning")
    parser.add_argument("--wan-negative-prompt", type=str, default="", help="Negative prompt for Wan2.2 I2V conditioning")
    parser.add_argument("--wan-use-cfg", action="store_true", help="Enable classifier-free guidance inside Wan2.2 forward (2x compute)")
    parser.add_argument("--wan-cfg-scale", type=float, default=3.5, help="CFG scale for Wan2.2 when --wan-use-cfg is set")
    parser.add_argument("--wan-dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--cond-image", type=Path, default=None, help="Conditioning image for Wan2.2 I2V (defaults to --phase1-input-image if provided)")

    # Experiment tracking (Weights & Biases)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (user or team)")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb-group", type=str, default=None, help="W&B group name for sweeps/ablations")
    parser.add_argument("--wandb-notes", type=str, default=None, help="W&B notes")
    parser.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated tags")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-dir", type=Path, default=Path("wandb"), help="Local W&B directory")
    parser.add_argument("--wandb-log-every", type=int, default=10, help="Log every N steps")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device == "cuda"

    set_seed(args.seed)

    phase1_time_s = None
    phase2_time_s = None
    phase3_time_s = None

    if args.wandb_mode == "disabled":
        args.wandb = False

    if args.run_phase1:
        if args.phase1_input_image is None:
            raise ValueError("--phase1-input-image is required when --run-phase1 is set.")
        print("[phase1] Running PhaseI pipeline...")
        phase1_start = time.time()
        run_phase1(
            phase1_script=args.phase1_script,
            input_image=args.phase1_input_image,
            output_root=args.phase1_output_root.parent,
            skip_env=args.phase1_skip_env,
            skip_clone=args.phase1_skip_clone,
        )
        phase1_time_s = time.time() - phase1_start

    if args.body_mesh is None and args.body_sequence_dir is not None:
        seq_candidates = sorted(args.body_sequence_dir.glob("*.obj"))
        if seq_candidates:
            args.body_mesh = seq_candidates[0]

    body_mesh, cloth_mesh = resolve_phase1_meshes(
        args.phase1_output_root, args.body_mesh, args.cloth_mesh
    )

    print(f"[phase1] Body mesh:  {body_mesh}")
    print(f"[phase1] Cloth mesh: {cloth_mesh}")

    if not args.skip_mpm:
        print("[phase2] Running MPM simulation...")
        phase2_start = time.time()
        body_sequence = build_body_sequence(
            body_mesh=body_mesh,
            body_sequence_dir=args.body_sequence_dir,
            n_frames=args.mpm_frames,
        )
        run_mpm_simulation(
            cloth_mesh_path=cloth_mesh,
            body_sequence=body_sequence,
            output_dir=args.mpm_output_dir,
            save_every=args.mpm_save_every,
            y_offset=args.mpm_y_offset,
        )
        phase2_time_s = time.time() - phase2_start

    print("[phase3] Starting optimization...")
    diffusion_model = create_video_diffusion_model(device=device, fp16=use_amp)
    mpm_params = MPMParameters(device=device)
    scheduler = DiffusionScheduler(timesteps=1000, schedule="cosine", device=device)

    wan_guidance = None
    wan_cond_image_01 = None
    if args.diffusion_backend == "wan22-i2v":
        if args.wan_ckpt_dir is None:
            raise ValueError("--wan-ckpt-dir is required when --diffusion-backend wan22-i2v")
        cond_image_path = args.cond_image or args.phase1_input_image
        if cond_image_path is None:
            raise ValueError("--cond-image (or --phase1-input-image) is required when using --diffusion-backend wan22-i2v")

        from PIL import Image

        img = Image.open(cond_image_path).convert("RGB")
        img_np = np.asarray(img).astype(np.float32) / 255.0
        wan_cond_image_01 = torch.from_numpy(img_np).permute(2, 0, 1).contiguous().to(device)

        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        wan_dtype = dtype_map[args.wan_dtype]
        wan_cfg = Wan22I2VConfig(
            wan_repo_root=args.wan_repo_root,
            ckpt_dir=args.wan_ckpt_dir,
            device=device,
            dtype=wan_dtype,
            prompt=args.wan_prompt,
            negative_prompt=args.wan_negative_prompt,
            use_cfg=args.wan_use_cfg,
            cfg_scale=args.wan_cfg_scale,
        )
        wan_guidance = Wan22I2VGuidance(wan_cfg).to(device)

    dataloader = create_dataloader(
        data_root=str(args.mpm_output_dir),
        batch_size=args.batch_size,
        video_length=args.video_length,
        spatial_size=tuple(args.spatial_size),
        shuffle=True,
        use_dummy=False,
    )
    if len(dataloader.dataset) == 0:
        raise FileNotFoundError(
            f"No rendered videos found in {args.mpm_output_dir}. "
            "Check the MPM output path or run with --skip-mpm disabled."
        )

    optimizer = torch.optim.Adam(mpm_params.parameters(), lr=args.learning_rate)
    scaler = GradScaler("cuda") if use_amp else None

    best_loss = float("inf")
    total_start = time.time()
    phase3_start = time.time()

    wandb_run = _maybe_init_wandb(
        args,
        extra_config={
            "resolved_body_mesh": str(body_mesh),
            "resolved_cloth_mesh": str(cloth_mesh),
            "resolved_device": device,
            "use_amp_resolved": use_amp,
        },
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "train/step": 0,
                "time/phase1_sec": float(phase1_time_s) if phase1_time_s is not None else None,
                "time/phase2_sec": float(phase2_time_s) if phase2_time_s is not None else None,
            }
        )

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        avg_loss, param_dict, global_step = train_one_epoch(
            diffusion_model=diffusion_model,
            mpm_params=mpm_params,
            scheduler=scheduler,
            optimizer=optimizer,
            dataloader=dataloader,
            device=device,
            epoch=epoch,
            scaler=scaler,
            use_amp=use_amp,
            wandb_run=wandb_run,
            global_step=global_step,
            log_every=max(1, int(args.wandb_log_every)),
            wan_guidance=wan_guidance,
            wan_cond_image_01=wan_cond_image_01,
        )
        epoch_time_s = time.time() - epoch_start

        print(f"[phase3] Epoch {epoch} loss: {avg_loss:.6f}")
        for name, value in param_dict.items():
            print(f"  {name}: {value:.4f}")

        if wandb_run is not None:
            log_payload = {
                "train/step": global_step,
                "train/epoch": int(epoch),
                "train/epoch_loss": float(avg_loss),
                "time/epoch_sec": float(epoch_time_s),
            }
            for name, value in param_dict.items():
                log_payload[f"mpm/{name}"] = float(value)
            wandb_run.log(log_payload)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = args.checkpoint_dir / "best_checkpoint.pth"
            save_checkpoint(
                path=str(best_path),
                mpm_params=param_dict,
                optimizer_state=optimizer.state_dict(),
                epoch=epoch,
                loss=avg_loss,
            )
            if wandb_run is not None:
                try:
                    wandb_run.log({"train/step": global_step, "train/best_loss": float(best_loss)})
                    wandb_run.save(str(best_path))
                except Exception:
                    pass

    phase3_time_s = time.time() - phase3_start
    total = time.time() - total_start
    print(f"[done] Total time: {total:.2f}s, best loss: {best_loss:.6f}")

    if wandb_run is not None:
        try:
            wandb_run.summary["best_loss"] = float(best_loss)
            if phase2_time_s is not None:
                wandb_run.summary["phase2_sec"] = float(phase2_time_s)
            wandb_run.summary["phase3_sec"] = float(phase3_time_s)
            wandb_run.summary["total_sec"] = float(total)
        finally:
            wandb_run.finish()


if __name__ == "__main__":
    main()
