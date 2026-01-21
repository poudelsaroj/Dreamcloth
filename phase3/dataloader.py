"""
Dataloader for Phase 3 training
Handles loading rendered cloth simulation videos
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from typing import Tuple, List, Optional
import trimesh


class DummyVideoDataset(Dataset):
    """
    Dummy dataset for testing training loop
    Generates synthetic video data on-the-fly
    """
    def __init__(
        self,
        num_samples: int = 100,
        video_length: int = 8,
        spatial_size: Tuple[int, int] = (64, 64),
        num_channels: int = 3
    ):
        self.num_samples = num_samples
        self.video_length = video_length
        self.spatial_size = spatial_size
        self.num_channels = num_channels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Generate a random video

        Returns:
            video: (C, T, H, W) in [0, 1]
        """
        video = torch.rand(self.num_channels, self.video_length, *self.spatial_size)
        return video


class RenderedVideoDataset(Dataset):
    """
    Dataset for loading pre-rendered MPM simulation videos
    Expected structure:
        data_root/
            video_0000/
                cloth_0000.obj
                cloth_0001.obj
                ...
            video_0001/
                ...
    """
    def __init__(
        self,
        data_root: str,
        video_length: int = 8,
        spatial_size: Tuple[int, int] = (64, 64),
        transform=None
    ):
        self.data_root = data_root
        self.video_length = video_length
        self.spatial_size = spatial_size
        self.transform = transform

        # Find all video directories
        self.video_dirs = self._find_video_dirs()

        if len(self.video_dirs) == 0:
            print(f"Warning: No videos found in {data_root}")

    def _find_video_dirs(self) -> List[str]:
        """Find all video directories"""
        video_dirs = []

        if not os.path.exists(self.data_root):
            return video_dirs

        for dirname in sorted(os.listdir(self.data_root)):
            dirpath = os.path.join(self.data_root, dirname)
            if os.path.isdir(dirpath) and dirname.startswith("video_"):
                # Check if it has mesh files
                mesh_files = [f for f in os.listdir(dirpath) if f.endswith(".obj")]
                if len(mesh_files) >= self.video_length:
                    video_dirs.append(dirpath)

        return video_dirs

    def __len__(self) -> int:
        return len(self.video_dirs)

    def _render_mesh_to_image(self, mesh_path: str) -> np.ndarray:
        """
        Simple mesh rendering to image (placeholder)
        In practice, would use proper rendering (Blender, PyTorch3D, etc.)

        Args:
            mesh_path: Path to .obj file

        Returns:
            (H, W, 3) in [0, 255] uint8
        """
        # Load mesh
        try:
            mesh = trimesh.load(mesh_path, force='mesh')
        except:
            # Return blank image if mesh loading fails
            return np.zeros((*self.spatial_size, 3), dtype=np.uint8)

        # Simple placeholder rendering: project vertices to 2D
        vertices = mesh.vertices

        # Create empty image
        img = np.zeros((*self.spatial_size, 3), dtype=np.uint8)

        # Normalize vertices to image space
        if len(vertices) > 0:
            vmin = vertices.min(axis=0)
            vmax = vertices.max(axis=0)
            vrange = vmax - vmin + 1e-8

            # Project to XY plane and scale to image size
            verts_2d = vertices[:, :2]  # Take X, Y
            verts_2d = (verts_2d - vmin[:2]) / vrange[:2]
            verts_2d[:, 0] *= self.spatial_size[1]  # X -> width
            verts_2d[:, 1] *= self.spatial_size[0]  # Y -> height

            # Color based on Z coordinate (height)
            z_norm = (vertices[:, 2] - vmin[2]) / vrange[2]

            # Draw points
            for i, (x, y) in enumerate(verts_2d):
                ix, iy = int(x), int(y)
                if 0 <= ix < self.spatial_size[1] and 0 <= iy < self.spatial_size[0]:
                    color_val = int(z_norm[i] * 255)
                    img[iy, ix] = [color_val, color_val // 2, 255 - color_val]

        return img

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load and return a video

        Returns:
            video: (C, T, H, W) in [0, 1]
        """
        video_dir = self.video_dirs[idx]

        # Get mesh files
        mesh_files = sorted([f for f in os.listdir(video_dir) if f.startswith("cloth_") and f.endswith(".obj")])

        # Load frames
        frames = []
        for i in range(self.video_length):
            if i < len(mesh_files):
                mesh_path = os.path.join(video_dir, mesh_files[i])
                img = self._render_mesh_to_image(mesh_path)
            else:
                # Repeat last frame if not enough frames
                img = frames[-1] if frames else np.zeros((*self.spatial_size, 3), dtype=np.uint8)

            frames.append(img)

        # Stack to (T, H, W, 3)
        video = np.stack(frames, axis=0)

        # Convert to torch tensor (T, H, W, 3) -> (C, T, H, W)
        video = torch.from_numpy(video).float() / 255.0
        video = video.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

        if self.transform:
            video = self.transform(video)

        return video


def create_dataloader(
    data_root: Optional[str] = None,
    batch_size: int = 1,
    video_length: int = 8,
    spatial_size: Tuple[int, int] = (64, 64),
    num_workers: int = 0,
    shuffle: bool = True,
    use_dummy: bool = False,
    num_dummy_samples: int = 100
) -> DataLoader:
    """
    Create dataloader for training

    Args:
        data_root: Root directory containing rendered videos
        batch_size: Batch size
        video_length: Number of frames per video
        spatial_size: Spatial resolution (H, W)
        num_workers: Number of dataloader workers
        shuffle: Whether to shuffle data
        use_dummy: Whether to use dummy synthetic data
        num_dummy_samples: Number of dummy samples if use_dummy=True

    Returns:
        DataLoader instance
    """
    if use_dummy or data_root is None:
        print(f"Using dummy dataset with {num_dummy_samples} synthetic videos")
        dataset = DummyVideoDataset(
            num_samples=num_dummy_samples,
            video_length=video_length,
            spatial_size=spatial_size
        )
    else:
        print(f"Loading rendered videos from {data_root}")
        dataset = RenderedVideoDataset(
            data_root=data_root,
            video_length=video_length,
            spatial_size=spatial_size
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Dataloader created: {len(dataset)} videos, batch_size={batch_size}")

    return dataloader


if __name__ == "__main__":
    # Test dataloader
    print("Testing dataloader...")

    # Test dummy dataset
    print("\n1. Testing dummy dataset:")
    dummy_loader = create_dataloader(
        use_dummy=True,
        batch_size=2,
        video_length=8,
        spatial_size=(64, 64),
        num_dummy_samples=10
    )

    for i, batch in enumerate(dummy_loader):
        print(f"  Batch {i}: shape={batch.shape}, dtype={batch.dtype}, range=[{batch.min():.3f}, {batch.max():.3f}]")
        if i >= 2:
            break

    # Test rendered dataset (will be empty unless data exists)
    print("\n2. Testing rendered dataset:")
    rendered_loader = create_dataloader(
        data_root="./test_data",
        batch_size=1,
        video_length=8,
        spatial_size=(64, 64),
        use_dummy=False  # Will fall back to dummy if no data
    )

    if len(rendered_loader.dataset) > 0:
        batch = next(iter(rendered_loader))
        print(f"  Loaded batch: shape={batch.shape}")
    else:
        print("  No data found (expected for testing)")

    print("\n✓ Dataloader tests passed!")
