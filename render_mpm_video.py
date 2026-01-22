"""
Render MPM mesh sequence (cloth/body) into a video using PyTorch3D.

Example (WSL):
  python render_mpm_video.py \
    --input-dir /mnt/f/mpm/github/Dreamcloth/output/phase2_sim/video_0000 \
    --output /mnt/f/mpm/github/Dreamcloth/output/phase2_sim/video_0000.mp4 \
    --fps 24 --image-size 512 --include-body
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    PointLights,
    BlendParams,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import join_meshes_as_scene


def list_frames(input_dir: Path, prefix: str) -> List[Path]:
    return sorted(p for p in input_dir.iterdir() if p.name.startswith(prefix) and p.suffix == ".obj")


def make_renderer(
    device: torch.device,
    image_size: int,
    background: Tuple[float, float, float],
    distance: float,
    elevation: float,
    azimuth: float,
):
    R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth, device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    blend_params = BlendParams(background_color=background)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights, blend_params=blend_params),
    )
    return renderer, cameras


def view_angles(view: str) -> Tuple[float, float]:
    if view == "front":
        return 0.0, 0.0
    if view == "side":
        return 0.0, 90.0
    if view == "top":
        return 90.0, 0.0
    raise ValueError(f"Unknown camera view: {view}")


def color_mesh(mesh, color: Tuple[float, float, float]) -> None:
    verts = mesh.verts_packed()
    color_tensor = torch.tensor(color, device=verts.device).view(1, 3).repeat(verts.shape[0], 1)
    mesh.textures = TexturesVertex(verts_features=color_tensor.unsqueeze(0))


def compute_center_extent(mesh) -> Tuple[torch.Tensor, torch.Tensor]:
    verts = mesh.verts_packed()
    vmin = verts.min(dim=0).values
    vmax = verts.max(dim=0).values
    center = (vmin + vmax) * 0.5
    extent = (vmax - vmin).max()
    return center, extent


def apply_center_scale(mesh, center: torch.Tensor, extent: torch.Tensor, scale: float) -> None:
    if extent.item() > 0:
        verts = mesh.verts_packed()
        verts = (verts - center) / extent * scale
        mesh._verts_list = [verts]


def load_mesh_sequence(paths: List[Path], device: torch.device):
    return [load_objs_as_meshes([str(p)], device=device) for p in paths]


def render_sequence(
    cloth_paths: List[Path],
    body_paths: List[Path],
    device: torch.device,
    image_size: int,
    camera_view: str,
    include_body: bool,
    background: Tuple[float, float, float],
    distance: float,
    scale: float,
) -> List[torch.Tensor]:
    elev, azim = view_angles(camera_view)
    renderer, cameras = make_renderer(device, image_size, background, distance, elev, azim)

    cloth_meshes = load_mesh_sequence(cloth_paths, device)
    body_meshes = load_mesh_sequence(body_paths, device) if include_body else []

    frames = []
    for i, cloth_mesh in enumerate(cloth_meshes):
        color_mesh(cloth_mesh, (0.7, 0.2, 0.2))
        center, extent = compute_center_extent(cloth_mesh)
        apply_center_scale(cloth_mesh, center, extent, scale)
        render_mesh = cloth_mesh

        if include_body and i < len(body_meshes):
            body_mesh = body_meshes[i]
            color_mesh(body_mesh, (0.2, 0.2, 0.7))
            apply_center_scale(body_mesh, center, extent, scale)
            render_mesh = join_meshes_as_scene([render_mesh, body_mesh])

        images = renderer(render_mesh, cameras=cameras)
        frames.append(images[0, ..., :3].detach().cpu())

    return frames


def save_video(frames: List[torch.Tensor], output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if imageio is None:
        raise RuntimeError("imageio is not available. Install imageio to save videos.")
    with imageio.get_writer(str(output_path), fps=fps) as writer:
        for frame in frames:
            frame_np = (frame.numpy() * 255.0).clip(0, 255).astype("uint8")
            writer.append_data(frame_np)


def save_frames(frames: List[torch.Tensor], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if imageio is None:
        raise RuntimeError("imageio is not available. Install imageio to save frames.")
    for i, frame in enumerate(frames):
        frame_np = (frame.numpy() * 255.0).clip(0, 255).astype("uint8")
        imageio.imwrite(str(output_dir / f"frame_{i:04d}.png"), frame_np)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render MPM mesh sequence with PyTorch3D")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with cloth_*.obj and body_*.obj")
    parser.add_argument("--output", type=Path, required=True, help="Output .mp4 path or output frames directory")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--camera", type=str, default="front", choices=["front", "side", "top"])
    parser.add_argument("--include-body", action="store_true")
    parser.add_argument("--background", type=str, default="white", choices=["white", "black"])
    parser.add_argument("--distance", type=float, default=2.5, help="Camera distance")
    parser.add_argument("--scale", type=float, default=1.0, help="Normalize mesh extent to this scale")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    background = (1.0, 1.0, 1.0) if args.background == "white" else (0.0, 0.0, 0.0)

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {args.input_dir}")

    cloth_paths = list_frames(args.input_dir, "cloth_")
    body_paths = list_frames(args.input_dir, "body_")

    if not cloth_paths:
        raise FileNotFoundError(f"No cloth_*.obj found in {args.input_dir}")

    if args.include_body and not body_paths:
        raise FileNotFoundError(f"--include-body set but no body_*.obj found in {args.input_dir}")

    if args.include_body:
        frame_count = min(len(cloth_paths), len(body_paths))
        cloth_paths = cloth_paths[:frame_count]
        body_paths = body_paths[:frame_count]

    frames = render_sequence(
        cloth_paths=cloth_paths,
        body_paths=body_paths,
        device=device,
        image_size=args.image_size,
        camera_view=args.camera,
        include_body=args.include_body,
        background=background,
        distance=args.distance,
        scale=args.scale,
    )

    if args.output.suffix.lower() in [".mp4", ".gif"]:
        save_video(frames, args.output, fps=args.fps)
    else:
        save_frames(frames, args.output)


if __name__ == "__main__":
    main()
