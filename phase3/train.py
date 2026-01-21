"""
Phase 3 Training: Gradient Propagation from Video Diffusion to MPM Parameters

Architecture:
    MPM Simulation -> Rendered Video -> Add Noise -> Diffusion Model -> Score Loss -> Backprop to MPM Params
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import numpy as np
import os
import sys
from typing import Dict, Tuple, Optional
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from phase3.model import create_video_diffusion_model
from phase3.utils import (
    DiffusionScheduler,
    normalize_video,
    denormalize_video,
    compute_diffusion_loss,
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
    set_seed
)
from phase3.dataloader import create_dataloader


class MPMParameters(nn.Module):
    """
    Wrapper for MPM physical parameters that need to be optimized
    These parameters will receive gradients from the diffusion loss
    """
    def __init__(self, device: str = "cuda"):
        super().__init__()

        # Physical parameters (initialized with default values)
        # These are the parameters that will be optimized

        # Material properties
        self.young_modulus = nn.Parameter(torch.tensor(100.0, device=device))  # E
        self.poisson_ratio = nn.Parameter(torch.tensor(0.3, device=device))    # nu
        self.shear_stiffness = nn.Parameter(torch.tensor(500.0, device=device)) # gamma
        self.normal_stiffness = nn.Parameter(torch.tensor(500.0, device=device)) # kappa
        self.density = nn.Parameter(torch.tensor(1.0, device=device))           # rho

        # Simulation parameters
        self.friction = nn.Parameter(torch.tensor(0.5, device=device))
        self.damping = nn.Parameter(torch.tensor(0.999, device=device))

        # Constrain parameters to valid ranges
        self.param_bounds = {
            'young_modulus': (10.0, 500.0),
            'poisson_ratio': (0.1, 0.49),
            'shear_stiffness': (100.0, 1000.0),
            'normal_stiffness': (100.0, 1000.0),
            'density': (0.1, 5.0),
            'friction': (0.0, 1.0),
            'damping': (0.95, 0.9999)
        }

    def get_params_dict(self) -> Dict[str, float]:
        """Get current parameter values as dictionary"""
        return {
            'young_modulus': self.young_modulus.item(),
            'poisson_ratio': self.poisson_ratio.item(),
            'shear_stiffness': self.shear_stiffness.item(),
            'normal_stiffness': self.normal_stiffness.item(),
            'density': self.density.item(),
            'friction': self.friction.item(),
            'damping': self.damping.item()
        }

    def clamp_parameters(self):
        """Clamp parameters to valid physical ranges"""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in self.param_bounds:
                    bounds = self.param_bounds[name]
                    param.clamp_(bounds[0], bounds[1])

    def forward(self) -> Dict[str, torch.Tensor]:
        """Return parameters as dictionary of tensors"""
        return {
            'young_modulus': self.young_modulus,
            'poisson_ratio': self.poisson_ratio,
            'shear_stiffness': self.shear_stiffness,
            'normal_stiffness': self.normal_stiffness,
            'density': self.density,
            'friction': self.friction,
            'damping': self.damping
        }


class DummyMPMSimulator:
    """
    Dummy MPM simulator for testing gradient flow
    In practice, this would call the actual Taichi MPM simulation
    """
    def __init__(self, device: str = "cuda"):
        self.device = device

    def simulate_and_render(
        self,
        mpm_params: Dict[str, torch.Tensor],
        num_frames: int = 8,
        spatial_size: Tuple[int, int] = (64, 64)
    ) -> torch.Tensor:
        """
        Simulate cloth and render to video

        Args:
            mpm_params: Dictionary of MPM parameters (with gradients)
            num_frames: Number of frames to simulate
            spatial_size: Spatial resolution

        Returns:
            video: (1, 3, T, H, W) in [0, 1] with gradient connection
        """
        # Dummy simulation: generate video as function of parameters
        # In reality, this would run Taichi MPM simulation and render

        # Create base video (random)
        video = torch.rand(1, 3, num_frames, *spatial_size, device=self.device)

        # Modulate video based on parameters to create gradient connection
        # This simulates how changing parameters affects the rendered output

        # Example: Young's modulus affects overall brightness
        young_factor = torch.sigmoid((mpm_params['young_modulus'] - 100.0) / 100.0)
        video = video * (0.5 + 0.5 * young_factor)

        # Friction affects spatial smoothness (dummy correlation)
        friction_factor = mpm_params['friction']
        video = video * (0.8 + 0.2 * friction_factor)

        # Density affects temporal variation (dummy correlation)
        density_factor = torch.sigmoid((mpm_params['density'] - 1.0))
        for t in range(num_frames):
            video[:, :, t] *= (0.9 + 0.1 * density_factor * (t / num_frames))

        return video


def train_one_epoch(
    diffusion_model: nn.Module,
    mpm_params: MPMParameters,
    simulator: DummyMPMSimulator,
    scheduler: DiffusionScheduler,
    optimizer: optim.Optimizer,
    dataloader,
    device: str,
    epoch: int,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = True
) -> Tuple[float, Dict[str, float]]:
    """
    Train for one epoch

    Args:
        diffusion_model: Pretrained video diffusion model (frozen)
        mpm_params: MPM parameters to optimize
        simulator: MPM simulator
        scheduler: Diffusion noise scheduler
        optimizer: Optimizer for MPM parameters
        dataloader: Data loader
        device: Device
        epoch: Current epoch
        scaler: GradScaler for mixed precision
        use_amp: Whether to use automatic mixed precision

    Returns:
        avg_loss: Average loss for epoch
        param_dict: Current parameter values
    """
    mpm_params.train()
    diffusion_model.eval()  # Frozen

    loss_meter = AverageMeter()

    for batch_idx, target_videos in enumerate(dataloader):
        # Target videos: (B, C, T, H, W) in [0, 1]
        target_videos = target_videos.to(device)

        optimizer.zero_grad()

        # === Step 1: MPM Simulation with current parameters ===
        current_params = mpm_params()  # Dict of tensors with gradients

        # Simulate and render video
        # This is where gradients will flow back from diffusion loss to parameters
        simulated_videos = simulator.simulate_and_render(
            current_params,
            num_frames=target_videos.shape[2],
            spatial_size=target_videos.shape[3:]
        )

        # Expand batch if needed
        if simulated_videos.shape[0] < target_videos.shape[0]:
            simulated_videos = simulated_videos.repeat(target_videos.shape[0], 1, 1, 1, 1)

        # === Step 2: Normalize videos to [-1, 1] ===
        simulated_videos_norm = normalize_video(simulated_videos)
        target_videos_norm = normalize_video(target_videos)

        # === Step 3: Add noise according to diffusion schedule ===
        # Sample random timesteps
        batch_size = target_videos.shape[0]
        timesteps = torch.randint(
            0,
            scheduler.timesteps,
            (batch_size,),
            device=device
        ).long()

        # Add noise to simulated video
        noise = torch.randn_like(simulated_videos_norm)

        if use_amp:
            noise = noise.half()
            simulated_videos_norm = simulated_videos_norm.half()

        noisy_videos = scheduler.q_sample(simulated_videos_norm, timesteps, noise)

        # === Step 4: Predict noise with diffusion model ===
        with autocast('cuda', enabled=use_amp):
            predicted_noise = diffusion_model(noisy_videos, timesteps)

            # === Step 5: Compute loss (MSE between predicted and true noise) ===
            loss = compute_diffusion_loss(predicted_noise, noise)

        # === Step 6: Backpropagate to MPM parameters ===
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Clamp parameters to valid ranges
        mpm_params.clamp_parameters()

        # Update metrics
        loss_meter.update(loss.item(), batch_size)

        # Print progress
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss.item():.6f} (avg: {loss_meter.avg:.6f})")

    return loss_meter.avg, mpm_params.get_params_dict()


def main():
    """Main training loop"""
    # === Configuration ===
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
        'epochs': 5,  # Small number for testing
        'batch_size': 1,  # Small batch for 4GB VRAM
        'learning_rate': 1e-3,
        'video_length': 8,
        'spatial_size': (64, 64),
        'num_dummy_samples': 20,  # Small dataset for testing
        'use_amp': True,
        'checkpoint_dir': './checkpoints/phase3',
        'log_interval': 1
    }

    print("="*60)
    print("Phase 3 Training: Diffusion Loss -> MPM Parameters")
    print("="*60)
    print(f"Device: {config['device']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Video size: {config['video_length']} x {config['spatial_size']}")
    print(f"AMP: {config['use_amp']}")
    print("="*60)

    # Set seed
    set_seed(config['seed'])

    # === Create models ===
    print("\n1. Creating video diffusion model...")
    diffusion_model = create_video_diffusion_model(
        device=config['device'],
        fp16=config['use_amp']
    )
    print(f"   Diffusion model: {diffusion_model.count_parameters() / 1e6:.1f}M params (frozen)")

    print("\n2. Creating MPM parameters...")
    mpm_params = MPMParameters(device=config['device'])
    num_mpm_params = sum(p.numel() for p in mpm_params.parameters())
    print(f"   MPM parameters: {num_mpm_params} trainable params")
    print(f"   Initial values: {mpm_params.get_params_dict()}")

    # === Create simulator ===
    print("\n3. Creating MPM simulator...")
    simulator = DummyMPMSimulator(device=config['device'])
    print("   Using dummy simulator for testing")

    # === Create scheduler ===
    print("\n4. Creating diffusion scheduler...")
    scheduler = DiffusionScheduler(
        timesteps=1000,
        schedule='cosine',
        device=config['device']
    )
    print(f"   Scheduler: {scheduler.timesteps} timesteps, cosine schedule")

    # === Create dataloader ===
    print("\n5. Creating dataloader...")
    dataloader = create_dataloader(
        use_dummy=True,
        num_dummy_samples=config['num_dummy_samples'],
        batch_size=config['batch_size'],
        video_length=config['video_length'],
        spatial_size=config['spatial_size'],
        shuffle=True
    )

    # === Create optimizer ===
    print("\n6. Creating optimizer...")
    optimizer = optim.Adam(mpm_params.parameters(), lr=config['learning_rate'])
    print(f"   Optimizer: Adam, lr={config['learning_rate']}")

    # === Create gradient scaler for AMP ===
    scaler = GradScaler('cuda') if config['use_amp'] else None

    # === Training loop ===
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    best_loss = float('inf')
    train_start = time.time()

    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.time()

        # Train one epoch
        avg_loss, param_dict = train_one_epoch(
            diffusion_model=diffusion_model,
            mpm_params=mpm_params,
            simulator=simulator,
            scheduler=scheduler,
            optimizer=optimizer,
            dataloader=dataloader,
            device=config['device'],
            epoch=epoch,
            scaler=scaler,
            use_amp=config['use_amp']
        )

        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Loss: {avg_loss:.6f}")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Parameters:")
        for name, value in param_dict.items():
            print(f"    {name}: {value:.4f}")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_checkpoint.pth')
            save_checkpoint(
                path=checkpoint_path,
                mpm_params=param_dict,
                optimizer_state=optimizer.state_dict(),
                epoch=epoch,
                loss=avg_loss
            )

        # Save regular checkpoint
        if epoch % config['log_interval'] == 0:
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(
                path=checkpoint_path,
                mpm_params=param_dict,
                optimizer_state=optimizer.state_dict(),
                epoch=epoch,
                loss=avg_loss
            )

    total_time = time.time() - train_start

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Final parameters: {mpm_params.get_params_dict()}")
    print("="*60)

    # Test gradient flow
    print("\nTesting gradient flow...")
    test_gradient_flow(diffusion_model, mpm_params, simulator, scheduler, config)


def test_gradient_flow(
    diffusion_model,
    mpm_params,
    simulator,
    scheduler,
    config
):
    """Test that gradients flow properly to MPM parameters"""
    print("  Creating synthetic video...")

    # Get current parameters
    params = mpm_params()

    # Simulate
    video = simulator.simulate_and_render(
        params,
        num_frames=config['video_length'],
        spatial_size=config['spatial_size']
    )

    # Add noise
    video_norm = normalize_video(video)
    if config['use_amp']:
        video_norm = video_norm.half()

    timesteps = torch.tensor([500], device=config['device'])
    noise = torch.randn_like(video_norm)

    noisy_video = scheduler.q_sample(video_norm, timesteps, noise)

    # Forward through diffusion model
    with autocast('cuda', enabled=config['use_amp']):
        predicted_noise = diffusion_model(noisy_video, timesteps)
        loss = compute_diffusion_loss(predicted_noise, noise)

    # Backward
    loss.backward()

    # Check gradients
    print("  Checking gradients on MPM parameters:")
    has_grad = True
    for name, param in mpm_params.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"    {name}: grad_norm = {grad_norm:.6e}")
            if grad_norm == 0:
                has_grad = False
        else:
            print(f"    {name}: NO GRADIENT")
            has_grad = False

    if has_grad:
        print("  ✓ Gradient flow verified!")
    else:
        print("  ✗ Warning: Some parameters have no gradient")


if __name__ == "__main__":
    main()
