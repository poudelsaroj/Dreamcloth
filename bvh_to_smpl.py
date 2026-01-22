import numpy as np
import pickle
import trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.sparse import issparse
from bvh import Bvh

class SMPLModel:
    def __init__(self, model_path):
        """Load SMPL model"""
        with open(model_path, 'rb') as f:
            smpl_data = pickle.load(f, encoding='latin1')
        
        self.faces = smpl_data['f']
        self.v_template = smpl_data['v_template']
        self.shapedirs = smpl_data['shapedirs']
        
        # Convert sparse matrices to dense
        self.J_regressor = smpl_data['J_regressor']
        if issparse(self.J_regressor):
            self.J_regressor = self.J_regressor.toarray()
        
        self.weights = smpl_data['weights']
        self.posedirs = smpl_data['posedirs']
        self.kintree_table = smpl_data['kintree_table']
        
    def rodrigues(self, r):
        """Convert axis-angle to rotation matrix"""
        theta = np.linalg.norm(r)
        if theta < 1e-8:
            return np.eye(3)
        
        r_normalized = r / theta
        K = np.array([
            [0, -r_normalized[2], r_normalized[1]],
            [r_normalized[2], 0, -r_normalized[0]],
            [-r_normalized[1], r_normalized[0], 0]
        ])
        
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    
    def forward(self, pose, betas=None, trans=None):
        """SMPL forward pass"""
        if betas is None:
            betas = np.zeros(10)
        if trans is None:
            trans = np.zeros(3)
        
        # Ensure everything is numpy arrays
        pose = np.array(pose)
        betas = np.array(betas)
        trans = np.array(trans)
        
        # Add shape blend shapes
        v_shaped = self.v_template + np.einsum('ijk,k->ij', self.shapedirs, betas)
        
        # Get joint locations
        J = np.dot(self.J_regressor, v_shaped)
        
        # Reshape pose to (24, 3)
        pose = pose.reshape(-1, 3)
        
        # Compute rotation matrices
        R_mats = np.array([self.rodrigues(p) for p in pose])
        
        # Pose blend shapes
        pose_feature = (R_mats[1:] - np.eye(3)).reshape(-1)
        v_posed = v_shaped + np.einsum('ij,j->i', self.posedirs.reshape(-1, self.posedirs.shape[-1]), pose_feature).reshape(-1, 3)
        
        # Linear Blend Skinning
        num_joints = 24
        T = np.zeros((num_joints, 4, 4))
        
        # Root transformation
        T[0] = np.eye(4)
        T[0, :3, :3] = R_mats[0]
        T[0, :3, 3] = J[0]
        
        # Compute global transformations
        for i in range(1, num_joints):
            parent = int(self.kintree_table[0, i])
            
            # Local transformation
            T_local = np.eye(4)
            T_local[:3, :3] = R_mats[i]
            T_local[:3, 3] = J[i] - J[parent]
            
            # Global transformation
            T[i] = np.dot(T[parent], T_local)
        
        # Remove rest pose
        T_rest = np.zeros((num_joints, 4, 4))
        for i in range(num_joints):
            T_rest[i] = np.eye(4)
            T_rest[i, :3, 3] = -J[i]
        
        # Apply skinning
        vertices_h = np.hstack([v_posed, np.ones((v_posed.shape[0], 1))])  # Homogeneous
        vertices = np.zeros_like(v_posed)
        
        for i in range(num_joints):
            T_final = np.dot(T[i], T_rest[i])
            weighted_transform = self.weights[:, i:i+1] * np.dot(T_final, vertices_h.T).T[:, :3]
            vertices += weighted_transform
        
        # Add translation
        vertices += trans
        
        return vertices


class BVHtoSMPL:
    def __init__(self, smpl_model_path):
        self.smpl = SMPLModel(smpl_model_path)
        
    def load_bvh(self, bvh_path):
        """Load BVH file"""
        with open(bvh_path) as f:
            self.mocap = Bvh(f.read())
        
        self.fps = 1.0 / self.mocap.frame_time
        self.num_frames = self.mocap.nframes
        
        print(f"Loaded BVH: {self.num_frames} frames @ {self.fps:.2f} fps")
        print(f"BVH joints: {self.mocap.get_joints_names()}")
    
    def create_joint_mapping(self):
        """Create mapping for your BVH structure"""
        bvh_joints = self.mocap.get_joints_names()
        
        mapping = {
            'Hips': 0,
            'LeftUpLeg': 1,
            'RightUpLeg': 2,
            'Spine': 3,
            'LeftLeg': 4,
            'RightLeg': 5,
            'Spine1': 6,
            'LeftFoot': 7,
            'RightFoot': 8,
            'Spine2': 9,
            'LeftToe': 10,
            'RightToe': 11,
            'Neck': 12,
            'LeftShoulder': 13,
            'RightShoulder': 14,
            'Head': 15,
            'LeftArm': 16,
            'RightArm': 17,
            'LeftForeArm': 18,
            'RightForeArm': 19,
            'LeftHand': 20,
            'RightHand': 21,
        }
        
        self.joint_mapping = {k: v for k, v in mapping.items() if k in bvh_joints}
        print(f"Mapped {len(self.joint_mapping)}/24 SMPL joints")
    
    def get_joint_channel_value(self, frame_idx, joint_name, channel_name):
        """Safely get channel value"""
        try:
            channels = self.mocap.joint_channels(joint_name)
            if channel_name in channels:
                channel_idx = channels.index(channel_name)
                
                # Access frame data directly
                frame_data = self.mocap.frames[frame_idx]
                
                # Calculate the offset for this joint's channels
                offset = 0
                for j in self.mocap.get_joints():
                    if j.name == joint_name:
                        break
                    offset += len(self.mocap.joint_channels(j.name))
                
                return float(frame_data[offset + channel_idx])
            return 0.0
        except:
            return 0.0
    
    def extract_pose_frame(self, frame_idx):
        """Extract pose"""
        pose = np.zeros(72)
        
        for bvh_joint, smpl_idx in self.joint_mapping.items():
            # Get rotation values
            z_rot = self.get_joint_channel_value(frame_idx, bvh_joint, 'Zrotation')
            y_rot = self.get_joint_channel_value(frame_idx, bvh_joint, 'Yrotation')
            x_rot = self.get_joint_channel_value(frame_idx, bvh_joint, 'Xrotation')
            
            # Convert to radians
            rotation = [
                np.radians(x_rot),
                np.radians(y_rot),
                np.radians(z_rot)
            ]
            
            # Convert Euler to axis-angle
            rot_mat = R.from_euler('xyz', rotation).as_matrix()
            axis_angle = R.from_matrix(rot_mat).as_rotvec()
            pose[smpl_idx*3:(smpl_idx+1)*3] = axis_angle
        
        return pose
    
    def extract_translation_frame(self, frame_idx):
        """Extract root translation"""
        root_joint = 'Hips'
        
        # Get position values
        x_pos = self.get_joint_channel_value(frame_idx, root_joint, 'Xposition')
        y_pos = self.get_joint_channel_value(frame_idx, root_joint, 'Yposition')
        z_pos = self.get_joint_channel_value(frame_idx, root_joint, 'Zposition')
        
        trans = np.array([x_pos, y_pos, z_pos])
        
        # Scale: BVH in cm, SMPL in meters
        trans = trans * 0.01
        
        return trans
    
    def convert_to_smpl_meshes(self, output_dir, betas=None):
        """Convert BVH to SMPL mesh sequence"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        if betas is None:
            betas = np.zeros(10)
        
        print(f"\nConverting {self.num_frames} frames to SMPL meshes...")
        
        for frame_idx in range(self.num_frames):
            # Extract pose and translation
            pose = self.extract_pose_frame(frame_idx)
            trans = self.extract_translation_frame(frame_idx)
            
            # Generate SMPL mesh
            vertices = self.smpl.forward(pose, betas, trans)
            
            # Create and save mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=self.smpl.faces)
            output_path = output_dir / f"frame_{frame_idx:05d}.obj"
            mesh.export(output_path)
            
            if (frame_idx + 1) % 10 == 0:
                print(f"  Processed {frame_idx + 1}/{self.num_frames} frames")
        
        print(f"\n✅ Conversion complete! Meshes saved to {output_dir}")


# ==================== USAGE ====================

if __name__ == "__main__":
    # Configuration
    SMPL_MODEL = "models/SMPL_NEUTRAL.pkl"
    OUTPUT_DIR = "./smpl_meshes_from_bvh"
    from pathlib import Path
    import re

    bvh_dir = Path("generation/text2motion/animations/0/")
    bvh_files = sorted([f for f in bvh_dir.glob("*.bvh") if f.is_file()], key=lambda f: f.stat().st_mtime)
    if not bvh_files:
        raise FileNotFoundError(f"No BVH files found in {bvh_dir}")
    BVH_FILE = str(bvh_files[-1])
    print(f"Automatically selected BVH file: {BVH_FILE}")
    
    # Convert
    converter = BVHtoSMPL(SMPL_MODEL)
    converter.load_bvh(BVH_FILE)
    converter.create_joint_mapping()
    converter.convert_to_smpl_meshes(OUTPUT_DIR)
    
    print("\n🎉 Done! Meshes saved to ./smpl_meshes_from_bvh/")
    print("View them in Blender or render with Open3D!")