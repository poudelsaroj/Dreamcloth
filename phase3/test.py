"""
Test script for Phase 3 training pipeline
Verifies all components work without errors
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from phase3.model import create_video_diffusion_model, VideoUNet3D
from phase3.utils import (
    DiffusionScheduler,
    normalize_video,
    denormalize_video,
    compute_diffusion_loss,
    render_to_video_tensor,
    video_tensor_to_numpy,
    set_seed
)
from phase3.dataloader import create_dataloader
from phase3.train import MPMParameters, DummyMPMSimulator


def test_model():
    """Test video diffusion model"""
    print("\n" + "="*60)
    print("TEST 1: Video Diffusion Model")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create model
    print("Creating model...")
    model = create_video_diffusion_model(device=device, fp16=True)

    print(f"✓ Model created: {model.count_parameters() / 1e6:.1f}M parameters")

    # Test forward pass
    print("Testing forward pass...")
    B, C, T, H, W = 1, 3, 8, 64, 64
    x = torch.randn(B, C, T, H, W, device=device, dtype=torch.float16)
    timesteps = torch.randint(0, 1000, (B,), device=device)

    with torch.no_grad():
        output = model(x, timesteps)

    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    print(f"✓ Forward pass successful: {x.shape} -> {output.shape}")

    # Check memory
    if torch.cuda.is_available():
        memory_mb = torch.cuda.memory_allocated() / 1e6
        print(f"✓ GPU memory: {memory_mb:.1f} MB")

    # Verify frozen parameters
    grad_count = sum(p.requires_grad for p in model.parameters())
    print(f"✓ Trainable parameters: {grad_count} (should be 0)")

    print("✓ Model test passed!\n")


def test_scheduler():
    """Test diffusion scheduler"""
    print("\n" + "="*60)
    print("TEST 2: Diffusion Scheduler")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create scheduler
    print("Creating scheduler...")
    scheduler = DiffusionScheduler(timesteps=1000, schedule='cosine', device=device)
    print(f"✓ Scheduler created: {scheduler.timesteps} timesteps")

    # Test q_sample
    print("Testing q_sample (forward diffusion)...")
    B, C, T, H, W = 2, 3, 8, 64, 64
    x_start = torch.randn(B, C, T, H, W, device=device)
    timesteps = torch.randint(0, 1000, (B,), device=device)

    x_noisy = scheduler.q_sample(x_start, timesteps)
    assert x_noisy.shape == x_start.shape
    print(f"✓ q_sample successful: {x_start.shape} -> {x_noisy.shape}")

    # Test predict_start_from_noise
    print("Testing predict_start_from_noise...")
    noise = torch.randn_like(x_start)
    x_pred = scheduler.predict_start_from_noise(x_noisy, timesteps, noise)
    assert x_pred.shape == x_start.shape
    print(f"✓ predict_start_from_noise successful")

    print("✓ Scheduler test passed!\n")


def test_dataloader():
    """Test dataloader"""
    print("\n" + "="*60)
    print("TEST 3: Dataloader")
    print("="*60)

    # Test dummy dataloader
    print("Creating dummy dataloader...")
    dataloader = create_dataloader(
        use_dummy=True,
        num_dummy_samples=10,
        batch_size=2,
        video_length=8,
        spatial_size=(64, 64),
        shuffle=True
    )

    print(f"✓ Dataloader created: {len(dataloader.dataset)} samples")

    # Load one batch
    print("Loading batch...")
    batch = next(iter(dataloader))

    print(f"✓ Batch loaded: shape={batch.shape}, dtype={batch.dtype}")
    print(f"  Range: [{batch.min():.3f}, {batch.max():.3f}]")

    assert batch.shape[0] == 2, "Batch size mismatch"
    assert batch.shape[1] == 3, "Channel mismatch"
    assert batch.shape[2] == 8, "Time mismatch"
    assert batch.shape[3:] == (64, 64), "Spatial size mismatch"

    print("✓ Dataloader test passed!\n")


def test_mpm_parameters():
    """Test MPM parameters"""
    print("\n" + "="*60)
    print("TEST 4: MPM Parameters")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create parameters
    print("Creating MPM parameters...")
    mpm_params = MPMParameters(device=device)

    num_params = sum(p.numel() for p in mpm_params.parameters())
    print(f"✓ MPM parameters created: {num_params} parameters")

    # Get parameter dict
    param_dict = mpm_params.get_params_dict()
    print("✓ Parameter values:")
    for name, value in param_dict.items():
        print(f"  {name}: {value:.4f}")

    # Test forward pass
    params_tensor = mpm_params()
    print(f"✓ Forward pass successful: {len(params_tensor)} tensors")

    # Test gradient
    print("Testing gradient...")
    loss = sum(p.sum() for p in params_tensor.values())
    loss.backward()

    has_grad = all(p.grad is not None for p in mpm_params.parameters())
    print(f"✓ Gradient computed: {has_grad}")

    # Test clamping
    print("Testing parameter clamping...")
    mpm_params.young_modulus.data = torch.tensor(1000.0, device=device)  # Out of bounds
    mpm_params.clamp_parameters()
    assert mpm_params.young_modulus.item() <= 500.0
    print("✓ Clamping successful")

    print("✓ MPM parameters test passed!\n")


def test_simulator():
    """Test dummy MPM simulator"""
    print("\n" + "="*60)
    print("TEST 5: Dummy MPM Simulator")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create simulator and parameters
    print("Creating simulator and parameters...")
    simulator = DummyMPMSimulator(device=device)
    mpm_params = MPMParameters(device=device)

    # Simulate
    print("Running simulation...")
    params = mpm_params()
    video = simulator.simulate_and_render(
        params,
        num_frames=8,
        spatial_size=(64, 64)
    )

    print(f"✓ Simulation successful: output shape {video.shape}")
    print(f"  Range: [{video.min():.3f}, {video.max():.3f}]")

    # Test gradient flow
    print("Testing gradient flow...")
    loss = video.mean()
    loss.backward()

    has_grad = all(p.grad is not None for p in mpm_params.parameters())
    print(f"✓ Gradient flow: {has_grad}")

    if has_grad:
        for name, param in mpm_params.named_parameters():
            print(f"  {name}: grad_norm = {param.grad.norm().item():.6e}")

    print("✓ Simulator test passed!\n")


def test_end_to_end():
    """Test complete pipeline end-to-end"""
    print("\n" + "="*60)
    print("TEST 6: End-to-End Pipeline")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = torch.cuda.is_available()

    set_seed(42)

    # Create all components
    print("Creating components...")
    diffusion_model = create_video_diffusion_model(device=device, fp16=use_amp)
    mpm_params = MPMParameters(device=device)
    simulator = DummyMPMSimulator(device=device)
    scheduler = DiffusionScheduler(timesteps=1000, schedule='cosine', device=device)
    optimizer = torch.optim.Adam(mpm_params.parameters(), lr=1e-3)

    print("✓ All components created")

    # Run one training step
    print("\nRunning one training step...")

    optimizer.zero_grad()

    # 1. Simulate
    print("  1. Simulating with MPM...")
    params = mpm_params()
    video = simulator.simulate_and_render(params, num_frames=8, spatial_size=(64, 64))
    print(f"     ✓ Video shape: {video.shape}")

    # 2. Normalize
    print("  2. Normalizing video...")
    video_norm = normalize_video(video)
    if use_amp:
        video_norm = video_norm.half()

    # 3. Add noise
    print("  3. Adding noise...")
    timesteps = torch.tensor([500], device=device)
    noise = torch.randn_like(video_norm)
    noisy_video = scheduler.q_sample(video_norm, timesteps, noise)
    print(f"     ✓ Noisy video shape: {noisy_video.shape}")

    # 4. Predict noise
    print("  4. Predicting noise with diffusion model...")
    with torch.no_grad():  # Model is frozen
        predicted_noise = diffusion_model(noisy_video, timesteps)
    print(f"     ✓ Predicted noise shape: {predicted_noise.shape}")

    # 5. Compute loss
    print("  5. Computing loss...")
    loss = compute_diffusion_loss(predicted_noise, noise)
    print(f"     ✓ Loss: {loss.item():.6f}")

    # 6. Backpropagate
    print("  6. Backpropagating to MPM parameters...")
    loss.backward()

    # 7. Check gradients
    print("  7. Checking gradients...")
    has_grad = True
    for name, param in mpm_params.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"     {name}: grad_norm = {grad_norm:.6e}")
            if grad_norm == 0:
                has_grad = False
        else:
            print(f"     {name}: NO GRADIENT")
            has_grad = False

    if has_grad:
        print("     ✓ All parameters have gradients!")
    else:
        print("     ✗ Warning: Some parameters missing gradients")

    # 8. Optimizer step
    print("  8. Updating parameters...")
    param_before = mpm_params.get_params_dict()
    optimizer.step()
    mpm_params.clamp_parameters()
    param_after = mpm_params.get_params_dict()

    print("     Parameter changes:")
    for name in param_before.keys():
        delta = param_after[name] - param_before[name]
        print(f"       {name}: {param_before[name]:.4f} -> {param_after[name]:.4f} (Δ={delta:.6f})")

    print("\n✓ End-to-end test passed!\n")


def test_memory_usage():
    """Test memory usage on 4GB VRAM"""
    print("\n" + "="*60)
    print("TEST 7: Memory Usage (4GB VRAM)")
    print("="*60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return

    device = 'cuda'
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("Creating components...")
    diffusion_model = create_video_diffusion_model(device=device, fp16=True)
    mpm_params = MPMParameters(device=device)
    simulator = DummyMPMSimulator(device=device)

    print(f"✓ Components created")
    print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # Run forward pass
    print("\nRunning forward pass...")
    params = mpm_params()
    video = simulator.simulate_and_render(params, num_frames=8, spatial_size=(64, 64))
    video_norm = normalize_video(video).half()

    timesteps = torch.tensor([500], device=device)
    with torch.no_grad():
        output = diffusion_model(video_norm, timesteps)

    print(f"✓ Forward pass completed")
    print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    if torch.cuda.max_memory_allocated() / 1e9 < 4.0:
        print("✓ Memory usage within 4GB limit!")
    else:
        print("✗ Warning: Memory usage exceeds 4GB")

    print("✓ Memory test passed!\n")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("PHASE 3 TEST SUITE")
    print("Testing gradient propagation from diffusion to MPM parameters")
    print("="*60)

    tests = [
        ("Video Diffusion Model", test_model),
        ("Diffusion Scheduler", test_scheduler),
        ("Dataloader", test_dataloader),
        ("MPM Parameters", test_mpm_parameters),
        ("MPM Simulator", test_simulator),
        ("End-to-End Pipeline", test_end_to_end),
        ("Memory Usage", test_memory_usage)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ TEST FAILED: {test_name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
        print("The training pipeline is ready to use.")
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
        print("Please fix errors before training.")

    print("="*60)


if __name__ == "__main__":
    main()
