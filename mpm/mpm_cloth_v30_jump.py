"""
MPM Cloth Simulation v30 - Jump Sequence
Uses precomputed body frames from F:\mpm\data\jump\frame_*.obj
and simulates cloth (tshirt-sim.obj) against those frames.
"""

import os
import time
import glob
import numpy as np
import trimesh

import mpm_cloth_v30 as mpm


def load_body_frame(path):
    mesh = trimesh.load(path, force='mesh')
    return mesh.vertices, mesh.faces


def _pca_axes(vertices):
    v = vertices - vertices.mean(axis=0)
    cov = v.T @ v / max(len(v), 1)
    _, vecs = np.linalg.eigh(cov)
    # Columns are eigenvectors; sort by descending eigenvalue
    return vecs[:, ::-1]


def align_cloth_to_body(cloth_mesh, body_vertices, y_offset=-0.02):
    cloth_v = cloth_mesh.vertices.copy()
    body_v = body_vertices

    # Rotate cloth to match body orientation via PCA axes
    cloth_axes = _pca_axes(cloth_v)
    body_axes = _pca_axes(body_v)
    R = body_axes @ cloth_axes.T
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    cloth_center = cloth_v.mean(axis=0)
    cloth_v = (cloth_v - cloth_center) @ R.T + cloth_center

    cloth_min, cloth_max = cloth_v.min(axis=0), cloth_v.max(axis=0)
    body_min, body_max = body_v.min(axis=0), body_v.max(axis=0)

    cloth_center = (cloth_min + cloth_max) * 0.5
    body_center = (body_min + body_max) * 0.5

    cloth_height = max(cloth_max[1] - cloth_min[1], 1e-6)
    body_height = max(body_max[1] - body_min[1], 1e-6)

    scale = body_height / cloth_height
    cloth_v = (cloth_v - cloth_center) * scale + cloth_center

    # Translate to body center, then apply small vertical offset
    translation = body_center - cloth_center
    translation[1] += y_offset
    cloth_v = cloth_v + translation

    cloth_mesh.vertices = cloth_v
    print(f"Align cloth: scale={scale:.3f}, translate={translation}")


def main():
    print("=" * 60)
    print("MPM Cloth v30 - Jump Sequence")
    print("Precomputed body frames + kinematic attachment")
    print("=" * 60)

    data_dir = r"F:\mpm\github\Dreamcloth\data"
    jump_dir = r"F:\mpm\github\Dreamcloth\data\jump"
    output_dir = r"F:\mpm\github\Dreamcloth\output\jump_v30"
    os.makedirs(output_dir, exist_ok=True)

    frame_paths = sorted(glob.glob(os.path.join(jump_dir, "frame_*.obj")))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {jump_dir}")
    n_frames = len(frame_paths)
    print(f"Frames: {n_frames} from {jump_dir}")

    print("\nLoading quality cloth mesh...")
    cloth_path = os.path.join(data_dir, "tshirt-sim.obj")
    if not os.path.exists(cloth_path):
        print(f"Warning: {cloth_path} not found, trying tshirt-decimated.obj")
        cloth_path = os.path.join(data_dir, "tshirt-decimated.obj")
    cloth_mesh = trimesh.load(cloth_path, force='mesh')

    verts0, faces0 = load_body_frame(frame_paths[0])
    print(f"Body: {len(verts0)} vertices, {len(faces0)} faces")

    print("\nAuto-aligning cloth to body...")
    align_cloth_to_body(cloth_mesh, verts0, y_offset=-0.02)

    print("\nInitializing simulation...")
    positions = mpm.init_particles_from_mesh(cloth_mesh, y_offset=0.0)

    n_attached, boundary_verts = mpm.setup_smart_attachment(positions, mpm.cloth_faces_np, verts0)
    if mpm.ENABLE_LEG_SEPARATION:
        mpm.setup_leg_separation(positions, verts0)

    print("\nSetting up collision...")
    mpm.update_body_collision(verts0, faces0)
    mpm.update_body_velocities(verts0, None)
    prev_verts = None

    mpm.update_attachment_schedule(0)

    print(f"\n{'='*60}")
    print(f"Parameters: dt={mpm.DT:.6f}, substeps={mpm.SUBSTEPS}")
    print(f"Material: E={mpm.YOUNG_MODULUS}, nu={mpm.POISSON_RATIO}")
    print(f"Anisotropic: gamma={mpm.SHEAR_STIFFNESS}, kappa={mpm.NORMAL_STIFFNESS}")
    print(f"Attachment: warmup={mpm.WARMUP_FRAMES} frames, decay={mpm.DECAY_FRAMES} frames")
    print(f"Attachment scale min={mpm.MIN_ATTACHMENT_SCALE}")
    print(f"Motion: JUMP ({n_frames} frames)")
    print(f"{'='*60}")

    total_start = time.time()

    for frame, frame_path in enumerate(frame_paths):
        t0 = time.time()

        verts, faces = load_body_frame(frame_path)
        mpm.update_body_velocities(verts, prev_verts)
        mpm.update_body_collision(verts, faces, prev_verts)

        mpm.update_attachment_schedule(frame)

        if mpm.ENABLE_LEG_SEPARATION:
            bmin, bmax, bcenter, bsize = mpm.compute_bbox(verts)
            plane_x = bcenter[0]
            cutoff_y = bmin[1] + mpm.LEG_CUTOFF_FRAC * bsize[1]
            mpm.leg_cutoff_y[None] = cutoff_y

        for _ in range(mpm.SUBSTEPS):
            mpm.clear_collision_grid()
            mpm.mesh_to_grid_collision()
            mpm.p2g()
            mpm.grid_op()
            mpm.g2p()
            mpm.apply_kinematic_attachment()
            mpm.apply_distance_failsafe()
            if mpm.ENABLE_LEG_SEPARATION:
                mpm.apply_midplane_separation(plane_x, cutoff_y, mpm.LEG_SEPARATION_MARGIN)
                mpm.apply_leg_restraints(plane_x, cutoff_y, mpm.LEG_MAX_LATERAL_DRIFT)
            mpm.apply_body_pushout(mpm.BODY_PUSHOUT)

        prev_verts = verts.copy()

        stats = mpm.get_stats()
        dt = time.time() - t0

        if frame % 2 == 0:
            mpm.save_cloth_mesh(os.path.join(output_dir, f"cloth_{frame:04d}.obj"), mpm.cloth_faces_np)
            mpm.save_body_mesh(os.path.join(output_dir, f"body_{frame:04d}.obj"), verts, faces)

        if frame % 10 == 0:
            print(f"Frame {frame:3d}: v={stats[0]:.2f} m/s, y=[{stats[1]:.3f},{stats[2]:.3f}], "
                  f"F_max={stats[4]:.2f}, time={dt:.2f}s, attach_scale={mpm.attachment_scale[None]:.2f}")

        if stats[0] > 100 or np.isnan(stats[0]):
            print(f"UNSTABLE at frame {frame}! v={stats[0]}")
            break

    total_time = time.time() - total_start
    print(f"\nTotal: {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"Avg: {total_time/n_frames:.2f}s per frame")
    print(f"\nOutput: {output_dir}/")


if __name__ == "__main__":
    main()
