"""
Utility functions for Phase 3 training
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import os


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """
    Linear beta schedule for diffusion

    Args:
        timesteps: Number of diffusion timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value

    Returns:
        Beta schedule tensor of shape (timesteps,)
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine beta schedule (Improved DDPM)

    Args:
        timesteps: Number of diffusion timesteps
        s: Small offset parameter

    Returns:
        Beta schedule tensor
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionScheduler:
    """
    Handles noise scheduling for diffusion process
    """
    def __init__(self, timesteps: int = 1000, schedule: str = "linear", device: str = "cuda"):
        self.timesteps = timesteps
        self.device = device

        # Beta schedule
        if schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # Pre-compute useful quantities
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Precompute for q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Precompute for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Register buffer and move to device"""
        setattr(self, name, tensor.to(self.device))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0)

        Args:
            x_start: Clean video (B, C, T, H, W)
            t: Timesteps (B,)
            noise: Optional noise tensor

        Returns:
            Noisy video at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting: (B, 1, 1, 1, 1)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None, None]

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise

        Args:
            x_t: Noisy video
            t: Timesteps
            noise: Predicted noise

        Returns:
            Predicted x_0
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None, None]

        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t


def normalize_video(video: torch.Tensor) -> torch.Tensor:
    """
    Normalize video to [-1, 1]

    Args:
        video: (B, C, T, H, W) in [0, 1]

    Returns:
        Normalized video in [-1, 1]
    """
    return video * 2.0 - 1.0


def denormalize_video(video: torch.Tensor) -> torch.Tensor:
    """
    Denormalize video from [-1, 1] to [0, 1]

    Args:
        video: (B, C, T, H, W) in [-1, 1]

    Returns:
        Denormalized video in [0, 1]
    """
    return (video + 1.0) / 2.0


def render_to_video_tensor(render_frames: np.ndarray, target_size: Tuple[int, int] = (64, 64)) -> torch.Tensor:
    """
    Convert rendered frames to video tensor

    Args:
        render_frames: (T, H, W, 3) in [0, 255] uint8
        target_size: Target spatial size (H, W)

    Returns:
        Video tensor (1, 3, T, H, W) in [0, 1] float32
    """
    # Convert to torch tensor
    video = torch.from_numpy(render_frames).float() / 255.0  # (T, H, W, 3) -> [0, 1]

    # Permute to (T, 3, H, W)
    video = video.permute(0, 3, 1, 2)

    # Resize if needed
    if video.shape[2:] != target_size:
        video = F.interpolate(video, size=target_size, mode='bilinear', align_corners=False)

    # Add batch dimension and permute to (1, 3, T, H, W)
    video = video.permute(1, 0, 2, 3).unsqueeze(0)

    return video


def video_tensor_to_numpy(video: torch.Tensor) -> np.ndarray:
    """
    Convert video tensor back to numpy

    Args:
        video: (B, 3, T, H, W) or (3, T, H, W) in [0, 1]

    Returns:
        (T, H, W, 3) in [0, 255] uint8
    """
    if video.dim() == 5:
        video = video[0]  # Remove batch dim

    # Permute to (T, H, W, 3)
    video = video.permute(1, 2, 3, 0)

    # Convert to numpy and scale to [0, 255]
    video = (video.cpu().numpy() * 255).astype(np.uint8)

    return video


def compute_diffusion_loss(
    predicted_noise: torch.Tensor,
    target_noise: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute simple MSE loss between predicted and target noise

    Args:
        predicted_noise: Model output
        target_noise: Ground truth noise
        reduction: 'mean' or 'sum'

    Returns:
        Loss scalar
    """
    loss = F.mse_loss(predicted_noise, target_noise, reduction=reduction)
    return loss


def save_checkpoint(
    path: str,
    mpm_params: dict,
    optimizer_state: dict,
    epoch: int,
    loss: float
):
    """
    Save training checkpoint

    Args:
        path: Checkpoint file path
        mpm_params: Dictionary of MPM parameters
        optimizer_state: Optimizer state dict
        epoch: Current epoch
        loss: Current loss value
    """
    checkpoint = {
        "mpm_params": mpm_params,
        "optimizer_state": optimizer_state,
        "epoch": epoch,
        "loss": loss
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: str) -> dict:
    """
    Load training checkpoint

    Args:
        path: Checkpoint file path

    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(path)
    print(f"Checkpoint loaded from {path} (epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.6f})")
    return checkpoint


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test scheduler
    scheduler = DiffusionScheduler(timesteps=1000, schedule="cosine", device=device)
    print(f"✓ Scheduler created with {scheduler.timesteps} timesteps")

    # Test q_sample
    x_start = torch.randn(2, 3, 8, 64, 64, device=device)
    t = torch.randint(0, 1000, (2,), device=device)
    x_noisy = scheduler.q_sample(x_start, t)
    print(f"✓ q_sample: {x_start.shape} -> {x_noisy.shape}")

    # Test normalization
    video = torch.rand(1, 3, 8, 64, 64)
    video_norm = normalize_video(video)
    video_denorm = denormalize_video(video_norm)
    assert torch.allclose(video, video_denorm, atol=1e-6)
    print("✓ Normalization/denormalization")

    # Test render conversion
    frames = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)
    video_tensor = render_to_video_tensor(frames, target_size=(64, 64))
    print(f"✓ Render to tensor: {frames.shape} -> {video_tensor.shape}")

    # Test loss computation
    pred = torch.randn(2, 3, 8, 64, 64)
    target = torch.randn(2, 3, 8, 64, 64)
    loss = compute_diffusion_loss(pred, target)
    print(f"✓ Diffusion loss: {loss.item():.6f}")

    print("\n✓ All utility tests passed!")
