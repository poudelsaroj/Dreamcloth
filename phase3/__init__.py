"""
Phase 3: Video Diffusion-Guided MPM Parameter Optimization

This module implements gradient propagation from a pretrained video diffusion model
to MPM simulation parameters, enabling learning of physical parameters from video data.
"""

from .model import create_video_diffusion_model, VideoUNet3D
from .utils import (
    DiffusionScheduler,
    normalize_video,
    denormalize_video,
    compute_diffusion_loss,
    save_checkpoint,
    load_checkpoint,
    set_seed
)
from .dataloader import create_dataloader
from .train import MPMParameters, DummyMPMSimulator

__all__ = [
    'create_video_diffusion_model',
    'VideoUNet3D',
    'DiffusionScheduler',
    'normalize_video',
    'denormalize_video',
    'compute_diffusion_loss',
    'save_checkpoint',
    'load_checkpoint',
    'set_seed',
    'create_dataloader',
    'MPMParameters',
    'DummyMPMSimulator'
]
