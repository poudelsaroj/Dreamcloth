"""
SMPLX Running Motion Generator
Generates running animation using SMPLX body model with head stabilization.
"""

import torch
import smplx
import numpy as np
import os
import trimesh


class SMPLXRunning:
    """Generate running motion using SMPLX model with stable head."""

    def __init__(self, model_path: str, gender: str = 'neutral'):
        """
        Initialize SMPLX model.

        Args:
            model_path: Path to folder containing SMPLX_NEUTRAL.npz
            gender: 'neutral', 'male', or 'female'
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        print(f"Loading SMPLX model from {model_path}...")
        self.model = smplx.create(
            model_path,
            model_type='smplx',
            gender=gender,
            use_face_contour=False,
            num_betas=10,
            num_expression_coeffs=10,
            ext='npz'
        ).to(self.device)

        print(f"SMPLX model loaded: {self.model.num_betas} shape params")
        self.faces = self.model.faces

    def generate_running_pose(self, phase: float) -> dict:
        """
        Generate body pose parameters for running at given phase.
        Based on biomechanics of running gait cycle.

        Args:
            phase: Running cycle phase in radians [0, 2*pi]

        Returns:
            Dictionary with pose parameters
        """
        # Initialize pose (63 params = 21 joints × 3 rotation axes)
        body_pose = np.zeros(63, dtype=np.float32)

        # Global orientation (root tilt) - separate from body_pose
        global_orient = np.zeros(3, dtype=np.float32)
        global_orient[0] = 0.1  # Slight forward lean

        # Running motion - alternating legs
        left_phase = phase
        right_phase = phase + np.pi  # Opposite phase

        # ===== LEFT LEG =====
        # Left hip (indices 0, 1) - forward/backward swing
        body_pose[0] = 0.8 * np.sin(left_phase)   # hip flexion/extension
        body_pose[1] = 0.1 * np.sin(left_phase)   # hip adduction/abduction

        # Left knee (index 12) - NEGATIVE for flexion
        body_pose[12] = -0.3 - 0.6 * max(0, np.sin(left_phase))

        # Left ankle (index 21)
        body_pose[21] = 0.2 * np.sin(left_phase)

        # ===== RIGHT LEG =====
        # Right hip (indices 3, 4)
        body_pose[3] = 0.8 * np.sin(right_phase)
        body_pose[4] = 0.1 * np.sin(right_phase)

        # Right knee (index 15) - NEGATIVE for flexion
        body_pose[15] = -0.3 - 0.6 * max(0, np.sin(right_phase))

        # Right ankle (index 24)
        body_pose[24] = 0.2 * np.sin(right_phase)

        # ===== ARMS (opposite to legs for balance) =====
        # Left shoulder (indices 48, 49) - opposite to RIGHT leg
        body_pose[48] = -0.5 * np.sin(right_phase)  # shoulder flexion
        body_pose[49] = 0.2  # slight abduction

        # Left elbow (index 54) - NEGATIVE for flexion
        body_pose[54] = -0.3 - 0.3 * abs(np.sin(right_phase))

        # Right shoulder (indices 51, 52) - opposite to LEFT leg
        body_pose[51] = -0.5 * np.sin(left_phase)
        body_pose[52] = 0.2

        # Right elbow (index 57) - NEGATIVE for flexion
        body_pose[57] = -0.3 - 0.3 * abs(np.sin(left_phase))

        # ===== SPINE (indices 6, 7, 8) =====
        body_pose[6] = 0.0
        body_pose[7] = 0.1 * np.sin(phase)       # torso twist
        body_pose[8] = 0.05 * np.sin(2 * phase)  # torso bend

        # Vertical displacement (running bounce) - double frequency
        y_offset = 0.05 * abs(np.sin(2 * phase))

        return {
            'body_pose': torch.tensor(body_pose, device=self.device).unsqueeze(0),
            'global_orient': torch.tensor(global_orient, device=self.device).unsqueeze(0),
            'transl': torch.tensor([[0.0, 1.302 + y_offset, 0.0]], device=self.device),
            'betas': torch.zeros(1, 10, device=self.device),
        }

    def get_mesh(self, pose_params: dict) -> tuple:
        """
        Get mesh vertices and faces for given pose.

        Returns:
            vertices: (N, 3) numpy array
            faces: (F, 3) numpy array
        """
        with torch.no_grad():
            output = self.model(
                body_pose=pose_params['body_pose'],
                global_orient=pose_params['global_orient'],
                transl=pose_params['transl'],
                betas=pose_params['betas'],
                return_verts=True
            )

        vertices = output.vertices.cpu().numpy()[0]
        return vertices, self.faces

    def generate_running_sequence(self, n_frames: int, cycles: float = 3.0) -> list:
        """
        Generate a sequence of running poses.

        Args:
            n_frames: Number of frames to generate
            cycles: Number of complete running cycles (faster than walking)

        Returns:
            List of (vertices, faces) tuples
        """
        sequence = []

        for i in range(n_frames):
            phase = (i / n_frames) * cycles * 2 * np.pi
            pose = self.generate_running_pose(phase)
            verts, faces = self.get_mesh(pose)
            sequence.append((verts, faces))

            if i % 10 == 0:
                print(f"  Generated frame {i}/{n_frames}")

        return sequence

    def export_sequence(self, sequence: list, output_dir: str, prefix: str = "body"):
        """Export sequence as OBJ files."""
        os.makedirs(output_dir, exist_ok=True)

        for i, (verts, faces) in enumerate(sequence):
            filepath = os.path.join(output_dir, f"{prefix}_{i:04d}.obj")
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            mesh.export(filepath)

        print(f"Exported {len(sequence)} frames to {output_dir}")


def test_smplx_running():
    """Test SMPLX running generation."""
    model_path = r"F:\mpm\github\Dreamcloth\data"
    output_dir = r"F:\mpm\output\running_test"

    print("=" * 50)
    print("Testing SMPLX Running Generation")
    print("=" * 50)

    runner = SMPLXRunning(model_path, gender='neutral')

    print("\nGenerating 50 frames of running...")
    sequence = runner.generate_running_sequence(n_frames=50, cycles=3.0)

    print("\nExporting to OBJ files...")
    runner.export_sequence(sequence, output_dir)

    print("\nDone! Check", output_dir)


if __name__ == "__main__":
    test_smplx_running()
