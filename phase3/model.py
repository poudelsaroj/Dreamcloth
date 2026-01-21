"""
Lightweight Video Diffusion Model for Phase 3 Testing
~100M parameters, FP16 optimized for 4GB VRAM

Architecture: 3D U-Net with temporal attention for video denoising
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Timestep embeddings for diffusion"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TemporalAttention(nn.Module):
    """Temporal attention across video frames"""
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape

        # Reshape to (B*H*W, C, T)
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, T)

        # Normalize
        h = self.norm(x_flat)

        # QKV projection
        qkv = self.qkv(h)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # Multi-head attention
        head_dim = C // self.num_heads
        q = q.reshape(B * H * W, self.num_heads, head_dim, T).permute(0, 1, 3, 2)
        k = k.reshape(B * H * W, self.num_heads, head_dim, T).permute(0, 1, 3, 2)
        v = v.reshape(B * H * W, self.num_heads, head_dim, T).permute(0, 1, 3, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(B * H * W, C, T)

        # Output projection
        out = self.proj(out)

        # Residual connection and reshape back
        out = (out + x_flat).reshape(B, H, W, C, T).permute(0, 3, 4, 1, 2)

        return out


class ResBlock3D(nn.Module):
    """3D Residual block with temporal convolution"""
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class DownBlock3D(nn.Module):
    """Downsampling block with temporal attention"""
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, use_attention: bool = False):
        super().__init__()

        self.resblock1 = ResBlock3D(in_channels, out_channels, time_emb_dim)
        self.resblock2 = ResBlock3D(out_channels, out_channels, time_emb_dim)

        self.attention = TemporalAttention(out_channels) if use_attention else None

        # Downsample spatially but keep temporal resolution
        self.downsample = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.resblock1(x, time_emb)
        h = self.resblock2(h, time_emb)

        if self.attention is not None:
            h = self.attention(h)

        skip = h
        h = self.downsample(h)

        return h, skip


class UpBlock3D(nn.Module):
    """Upsampling block with temporal attention"""
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, use_attention: bool = False):
        super().__init__()

        # Upsample spatially but keep temporal resolution
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))

        self.resblock1 = ResBlock3D(in_channels + out_channels, out_channels, time_emb_dim)
        self.resblock2 = ResBlock3D(out_channels, out_channels, time_emb_dim)

        self.attention = TemporalAttention(out_channels) if use_attention else None

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.upsample(x)
        h = torch.cat([h, skip], dim=1)

        h = self.resblock1(h, time_emb)
        h = self.resblock2(h, time_emb)

        if self.attention is not None:
            h = self.attention(h)

        return h


class VideoUNet3D(nn.Module):
    """
    3D U-Net for video diffusion
    Input: (B, C, T, H, W) - noisy video
    Output: (B, C, T, H, W) - predicted noise

    Architecture designed for ~100M parameters at 64x64 resolution
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 96,  # Tuned for ~100M params
        channel_mult: Tuple[int, ...] = (1, 2, 3, 4),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim // 4),
            nn.Linear(time_emb_dim // 4, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Initial convolution
        self.conv_in = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            use_attn = i >= 2  # Use attention in deeper layers

            self.down_blocks.append(
                DownBlock3D(now_channels, out_ch, time_emb_dim, use_attention=use_attn)
            )

            now_channels = out_ch
            channels.append(now_channels)

        # Middle block
        self.mid_block1 = ResBlock3D(now_channels, now_channels, time_emb_dim)
        self.mid_attn = TemporalAttention(now_channels)
        self.mid_block2 = ResBlock3D(now_channels, now_channels, time_emb_dim)

        # Upsampling path
        self.up_blocks = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_mult)):
            out_ch = base_channels * mult
            skip_ch = channels[-(i+2)]
            use_attn = (len(channel_mult) - i - 1) >= 2

            self.up_blocks.append(
                UpBlock3D(now_channels, out_ch, time_emb_dim, use_attention=use_attn)
            )

            now_channels = out_ch

        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) noisy video
            timesteps: (B,) diffusion timesteps

        Returns:
            (B, C, T, H, W) predicted noise
        """
        # Time embedding
        t_emb = self.time_embed(timesteps)

        # Initial conv
        h = self.conv_in(x)

        # Downsampling with skip connections
        skips = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, t_emb)
            skips.append(skip)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Upsampling with skip connections
        for up_block in self.up_blocks:
            skip = skips.pop()
            h = up_block(h, skip, t_emb)

        # Output
        h = self.conv_out(h)

        return h

    def count_parameters(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_video_diffusion_model(device: str = "cuda", fp16: bool = True) -> VideoUNet3D:
    """
    Create video diffusion model with ~100M parameters

    Args:
        device: Device to place model on
        fp16: Whether to use FP16 precision

    Returns:
        Initialized model
    """
    model = VideoUNet3D(
        in_channels=3,
        out_channels=3,
        base_channels=96,
        channel_mult=(1, 2, 3, 4),
        time_emb_dim=256,
        dropout=0.0  # No dropout for "pretrained" model
    )

    model = model.to(device)

    if fp16:
        model = model.half()

    # Initialize with Xavier uniform (simulates pretrained weights)
    for module in model.modules():
        if isinstance(module, (nn.Conv3d, nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # Freeze all parameters (pretrained model)
    for param in model.parameters():
        param.requires_grad = False

    num_params = model.count_parameters()
    print(f"Video Diffusion Model: {num_params / 1e6:.1f}M parameters")

    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing Video Diffusion Model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_video_diffusion_model(device=device, fp16=True)

    print(f"Model created on {device}")
    print(f"Parameters: {model.count_parameters() / 1e6:.1f}M")

    # Test forward pass
    B, C, T, H, W = 1, 3, 8, 64, 64
    x = torch.randn(B, C, T, H, W, device=device, dtype=torch.float16)
    timesteps = torch.randint(0, 1000, (B,), device=device)

    print(f"\nInput shape: {x.shape}")

    with torch.no_grad():
        output = model(x, timesteps)

    print(f"Output shape: {output.shape}")
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\n✓ Model test passed!")
