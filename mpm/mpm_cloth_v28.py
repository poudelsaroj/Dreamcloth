"""
MPM Cloth Simulation v28 - Kinematic Warmup + Attachment Decay
Based on v27 with:
1. Initial kinematic warmup to lock cloth to body pose
2. Smooth decay of attachment strength after warmup
3. Same MPM + collision pipeline as v27

Goal:
- Prevent initial gravity-driven settling before collisions stabilize
- Keep cloth aligned to SMPL-X pose at the start
"""

import taichi as ti
import numpy as np
import trimesh
import os
import sys
import time
from scipy.spatial import cKDTree
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from smplx_running import SMPLXRunning


ti.init(arch=ti.cuda, default_fp=ti.f32)

# ============== Parameters from MPMAvatar Paper ==============
GRID_RES = 200  # Paper uses 200
MAX_PARTICLES = 140000
DT = 0.04 / 400  # dt=0.04 with 400 substeps = 1e-4 per substep
SUBSTEPS = 400

# Material parameters from paper
YOUNG_MODULUS = 100.0  # E (kappa_s - stretching stiffness)
POISSON_RATIO = 0.3    # nu
SHEAR_STIFFNESS = 500.0  # gamma (shear resistance)
NORMAL_STIFFNESS = 500.0  # kappa (normal compression resistance)
DENSITY = 1.0  # rho

GRAVITY = ti.Vector([0.0, -9.8, 0.0])
FRICTION = 0.5

# Attachment schedule (new in v28)
WARMUP_FRAMES = 60
DECAY_FRAMES = 80
MIN_ATTACHMENT_SCALE = 0.9

# Collision pushout (new in v28)
BODY_PUSHOUT = 0.01  # meters

# ============== Particle Data ==============
x = ti.Vector.field(3, dtype=ti.f32, shape=MAX_PARTICLES)  # position
v = ti.Vector.field(3, dtype=ti.f32, shape=MAX_PARTICLES)  # velocity
C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=MAX_PARTICLES)  # APIC affine velocity
F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=MAX_PARTICLES)  # deformation gradient
p_mass = ti.field(dtype=ti.f32, shape=MAX_PARTICLES)
p_vol = ti.field(dtype=ti.f32, shape=MAX_PARTICLES)
n_particles = ti.field(dtype=ti.i32, shape=())

# Material directions (D = original, updated from mesh)
D1 = ti.Vector.field(3, dtype=ti.f32, shape=MAX_PARTICLES)  # in-plane direction 1
D2 = ti.Vector.field(3, dtype=ti.f32, shape=MAX_PARTICLES)  # in-plane direction 2
D3 = ti.Vector.field(3, dtype=ti.f32, shape=MAX_PARTICLES)  # normal direction

# Per-particle face info for material directions
particle_face = ti.field(dtype=ti.i32, shape=MAX_PARTICLES)

# Grid
grid_v = ti.Vector.field(3, dtype=ti.f32, shape=(GRID_RES, GRID_RES, GRID_RES))
grid_m = ti.field(dtype=ti.f32, shape=(GRID_RES, GRID_RES, GRID_RES))

# Mesh-based collision - collider velocity and normal on grid
grid_collider_v = ti.Vector.field(3, dtype=ti.f32, shape=(GRID_RES, GRID_RES, GRID_RES))
grid_collider_n = ti.Vector.field(3, dtype=ti.f32, shape=(GRID_RES, GRID_RES, GRID_RES))
grid_collider_w = ti.field(dtype=ti.f32, shape=(GRID_RES, GRID_RES, GRID_RES))

# Body mesh faces for collision
MAX_BODY_FACES = 25000
body_face_centers = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODY_FACES)
body_face_normals = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODY_FACES)
body_face_velocities = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODY_FACES)
n_body_faces = ti.field(dtype=ti.i32, shape=())

# Attachment data
attachment_body_idx = ti.field(dtype=ti.i32, shape=MAX_PARTICLES)
attachment_offset = ti.Vector.field(3, dtype=ti.f32, shape=MAX_PARTICLES)
attachment_weight = ti.field(dtype=ti.f32, shape=MAX_PARTICLES)

# Attachment schedule state (new in v28)
attachment_scale = ti.field(dtype=ti.f32, shape=())
warmup_kinematic = ti.field(dtype=ti.i32, shape=())

# Body vertices for attachment
MAX_BODY_VERTS = 10500
body_verts = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODY_VERTS)
body_vertex_velocities = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODY_VERTS)  # NEW: for kinematic attachment
body_vertex_normals = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODY_VERTS)

# Grid params
dx = ti.field(dtype=ti.f32, shape=())
inv_dx = ti.field(dtype=ti.f32, shape=())
origin = ti.Vector.field(3, dtype=ti.f32, shape=())

# Cloth mesh storage
cloth_faces_np = None


@ti.func
def qr_decomposition_3x3(A: ti.types.matrix(3, 3, ti.f32)):
    """
    QR decomposition for 3x3 matrix using Gram-Schmidt
    Returns Q (orthogonal) and R (upper triangular)
    """
    # Column vectors of A
    a1 = ti.Vector([A[0, 0], A[1, 0], A[2, 0]])
    a2 = ti.Vector([A[0, 1], A[1, 1], A[2, 1]])
    a3 = ti.Vector([A[0, 2], A[1, 2], A[2, 2]])

    # Gram-Schmidt
    u1 = a1
    e1_norm = u1.norm()
    e1 = u1 / ti.max(e1_norm, 1e-10)

    u2 = a2 - a2.dot(e1) * e1
    e2_norm = u2.norm()
    e2 = u2 / ti.max(e2_norm, 1e-10)

    u3 = a3 - a3.dot(e1) * e1 - a3.dot(e2) * e2
    e3_norm = u3.norm()
    e3 = u3 / ti.max(e3_norm, 1e-10)

    # Q matrix
    Q = ti.Matrix([
        [e1[0], e2[0], e3[0]],
        [e1[1], e2[1], e3[1]],
        [e1[2], e2[2], e3[2]]
    ])

    # R = Q^T * A
    R = Q.transpose() @ A

    return Q, R


@ti.func
def compute_anisotropic_stress(F_local: ti.types.matrix(3, 3, ti.f32)):
    """
    Compute Cauchy stress using anisotropic constitutive model from MPMAvatar.
    The strain energy is decomposed into:
    - psi_normal: penalizes normal compression (R_33)
    - psi_shear: penalizes shear (R_13, R_23)
    - psi_in_plane: in-plane stretching (R_11, R_12, R_22)

    This implements: Psi(F) = (kappa_s/2)||F-R||^2 + (kappa_b/2)||nabla^2 u||^2
    """
    # QR decomposition: F = Q * R
    Q, R = qr_decomposition_3x3(F_local)

    # Extract R components
    R11, R12, R22 = R[0, 0], R[0, 1], R[1, 1]
    R13, R23, R33 = R[0, 2], R[1, 2], R[2, 2]

    # Initialize stress derivative
    dPsi_dR = ti.Matrix.zero(ti.f32, 3, 3)

    # 1. Normal term: psi_normal = kappa/3 * (1 - R33)^3 if R33 <= 1, else 0
    if R33 <= 1.0:
        dPsi_dR33 = -NORMAL_STIFFNESS * (1.0 - R33) ** 2
        dPsi_dR[2, 2] = dPsi_dR33

    # 2. Shear term: psi_shear = gamma/2 * (R13^2 + R23^2)
    dPsi_dR[0, 2] = SHEAR_STIFFNESS * R13
    dPsi_dR[1, 2] = SHEAR_STIFFNESS * R23

    # 3. In-plane term using fixed corotated model
    # psi_in_plane = mu * ((sigma_1 - 1)^2 + (sigma_2 - 1)^2) + lambda/2 * (J - 1)^2
    mu = YOUNG_MODULUS / (2.0 * (1.0 + POISSON_RATIO))
    lam = YOUNG_MODULUS * POISSON_RATIO / ((1.0 + POISSON_RATIO) * (1.0 - 2.0 * POISSON_RATIO))

    # For 2x2 in-plane part of R
    J_2d = R11 * R22  # determinant of 2x2

    # Derivative of fixed corotated for 2D
    dPsi_dR[0, 0] = 2.0 * mu * (R11 - 1.0) + lam * (J_2d - 1.0) * R22
    dPsi_dR[1, 1] = 2.0 * mu * (R22 - 1.0) + lam * (J_2d - 1.0) * R11
    dPsi_dR[0, 1] = mu * R12  # shear in-plane

    # First Piola-Kirchhoff stress: P = Q * dPsi/dR
    P = Q @ dPsi_dR

    # Cauchy stress: sigma = (1/J) * P * F^T
    J = F_local.determinant()
    J = ti.max(ti.abs(J), 1e-6)
    sigma = (1.0 / J) * P @ F_local.transpose()

    return sigma


@ti.kernel
def p2g():
    """Particle to Grid transfer with anisotropic stress (NO attachment force)"""
    # Clear grid
    for i, j, k in grid_m:
        grid_v[i, j, k] = ti.Vector.zero(ti.f32, 3)
        grid_m[i, j, k] = 0.0

    for p in range(n_particles[None]):
        Xp = (x[p] - origin[None]) * inv_dx[None]
        base = ti.cast(Xp - 0.5, ti.i32)
        fx = Xp - ti.cast(base, ti.f32)

        # Quadratic B-spline weights
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

        # Compute stress from anisotropic constitutive model
        stress = compute_anisotropic_stress(F[p])

        # Stress contribution: -vol * stress (this is f = -V_p * sigma_p * grad(w_ip))
        stress_term = -p_vol[p] * stress

        # NOTE: No attachment force here - handled by kinematic constraint after G2P

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            idx = base + offset
            if 0 <= idx[0] < GRID_RES and 0 <= idx[1] < GRID_RES and 0 <= idx[2] < GRID_RES:
                weight = w[i][0] * w[j][1] * w[k][2]
                dpos = (ti.cast(offset, ti.f32) - fx) * dx[None]

                # APIC momentum transfer
                momentum = weight * p_mass[p] * (v[p] + C[p] @ dpos)

                # Add stress contribution only (no attachment force)
                affine_force = stress_term @ dpos

                grid_v[idx] += momentum + DT * weight * affine_force
                grid_m[idx] += weight * p_mass[p]


@ti.kernel
def grid_op():
    """Grid operations: normalize, gravity, boundaries, collision handling"""
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 1e-10:
            grid_v[i, j, k] /= grid_m[i, j, k]

            # Add gravity
            grid_v[i, j, k] += DT * GRAVITY

            # Damping
            grid_v[i, j, k] *= 0.999

            # Grid boundaries
            bound = 3
            if j < bound and grid_v[i, j, k][1] < 0:
                grid_v[i, j, k][1] = 0
            if j > GRID_RES - bound and grid_v[i, j, k][1] > 0:
                grid_v[i, j, k][1] = 0
            if i < bound and grid_v[i, j, k][0] < 0:
                grid_v[i, j, k][0] = 0
            if i > GRID_RES - bound and grid_v[i, j, k][0] > 0:
                grid_v[i, j, k][0] = 0
            if k < bound and grid_v[i, j, k][2] < 0:
                grid_v[i, j, k][2] = 0
            if k > GRID_RES - bound and grid_v[i, j, k][2] > 0:
                grid_v[i, j, k][2] = 0

            # Mesh-based collision from MPMAvatar (Algorithm 1)
            # If collider weight > 0, we have collision info at this grid node
            if grid_collider_w[i, j, k] > 0:
                v_c = grid_collider_v[i, j, k] / grid_collider_w[i, j, k]
                n_c = grid_collider_n[i, j, k]
                n_len = n_c.norm()
                if n_len > 1e-6:
                    n_c = n_c / n_len

                    # Relative velocity
                    v_rel = grid_v[i, j, k] - v_c

                    # If relative velocity points inward (toward collider)
                    v_n = v_rel.dot(n_c)
                    if v_n < 0:
                        # Project out normal component (keep tangential)
                        v_t = v_rel - v_n * n_c

                        # Apply friction to tangential
                        v_t_norm = v_t.norm()
                        if v_t_norm > 1e-6:
                            friction_mag = ti.min(FRICTION * ti.abs(v_n), v_t_norm)
                            v_t = v_t * (1.0 - friction_mag / v_t_norm)

                        # Transform back to world frame
                        grid_v[i, j, k] = v_c + v_t


@ti.kernel
def g2p():
    """Grid to Particle transfer with deformation gradient update"""
    for p in range(n_particles[None]):
        Xp = (x[p] - origin[None]) * inv_dx[None]
        base = ti.cast(Xp - 0.5, ti.i32)
        fx = Xp - ti.cast(base, ti.f32)

        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

        new_v = ti.Vector.zero(ti.f32, 3)
        new_C = ti.Matrix.zero(ti.f32, 3, 3)

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            idx = base + offset
            if 0 <= idx[0] < GRID_RES and 0 <= idx[1] < GRID_RES and 0 <= idx[2] < GRID_RES:
                weight = w[i][0] * w[j][1] * w[k][2]
                g_v = grid_v[idx]
                dpos = ti.cast(offset, ti.f32) - fx
                new_v += weight * g_v
                new_C += 4.0 * inv_dx[None] * weight * g_v.outer_product(dpos)

        v[p] = new_v
        C[p] = new_C

        # Update position
        x[p] += DT * new_v

        # Update deformation gradient: F_new = (I + dt * C) @ F
        F[p] = (ti.Matrix.identity(ti.f32, 3) + DT * new_C) @ F[p]

        # Clamp deformation gradient to prevent instability
        # (from MPMAvatar: delete particles with high condition number)
        # Here we just clamp singular values
        U, sig, V = ti.svd(F[p])
        sig_clamped = ti.Vector([
            ti.max(0.1, ti.min(sig[0, 0], 4.0)),
            ti.max(0.1, ti.min(sig[1, 1], 4.0)),
            ti.max(0.1, ti.min(sig[2, 2], 4.0))
        ])
        F[p] = U @ ti.Matrix([
            [sig_clamped[0], 0, 0],
            [0, sig_clamped[1], 0],
            [0, 0, sig_clamped[2]]
        ]) @ V.transpose()


@ti.kernel
def apply_kinematic_attachment():
    """
    After G2P, enforce kinematic constraints on attached vertices.
    Boundary vertices (weight > 0.8) follow body exactly.
    Near-boundary vertices (weight > 0) blend between physics and target.
    """
    for p in range(n_particles[None]):
        if attachment_body_idx[p] >= 0:
            target = body_verts[attachment_body_idx[p]] + attachment_offset[p]
            w = attachment_weight[p] * attachment_scale[None]
            if warmup_kinematic[None] != 0:
                w = 1.0
            w = ti.min(1.0, w)

            if w > 0.8:
                # Boundary: fully kinematic - follow body exactly
                x[p] = target
                v[p] = body_vertex_velocities[attachment_body_idx[p]]
                # Reset deformation gradient for kinematically controlled particles
                F[p] = ti.Matrix.identity(ti.f32, 3)
            elif w > 0.0:
                # Near-boundary: blend position for smooth transition
                x[p] = x[p] * (1.0 - w) + target * w
                # Blend velocity too for continuity
                v[p] = v[p] * (1.0 - w) + body_vertex_velocities[attachment_body_idx[p]] * w


@ti.kernel
def apply_distance_failsafe():
    """
    Failsafe: if any particle falls below body, push it back up.
    This prevents catastrophic failure while allowing natural physics.
    """
    for p in range(n_particles[None]):
        if attachment_body_idx[p] >= 0:
            body_y = body_verts[attachment_body_idx[p]][1]
            # If cloth falls more than 30cm below its body attachment point
            if x[p][1] < body_y - 0.30:
                # Push back up (not instant snap, just limit)
                x[p][1] = body_y - 0.30
                # Kill downward velocity
                if v[p][1] < 0:
                    v[p][1] = 0.0


@ti.kernel
def apply_body_pushout(min_thickness: ti.f32):
    """Project particles outside the body along nearest vertex normal"""
    for p in range(n_particles[None]):
        if attachment_body_idx[p] >= 0:
            idx = attachment_body_idx[p]
            n = body_vertex_normals[idx]
            n_len = n.norm()
            if n_len > 1e-6:
                n = n / n_len
                rel = x[p] - body_verts[idx]
                dist = rel.dot(n)
                if dist < min_thickness:
                    x[p] += (min_thickness - dist) * n
                    v_n = v[p].dot(n)
                    if v_n < 0:
                        v[p] -= v_n * n


@ti.kernel
def clear_collision_grid():
    """Clear collision transfer grid"""
    for i, j, k in grid_collider_w:
        grid_collider_v[i, j, k] = ti.Vector.zero(ti.f32, 3)
        grid_collider_n[i, j, k] = ti.Vector.zero(ti.f32, 3)
        grid_collider_w[i, j, k] = 0.0


@ti.kernel
def mesh_to_grid_collision():
    """
    Transfer body mesh face velocity and normal to grid using B-spline weights.
    This is Algorithm 1 from MPMAvatar paper - mesh-to-grid transfer stage.
    """
    for f in range(n_body_faces[None]):
        # Face center position
        xf = body_face_centers[f]
        vf = body_face_velocities[f]
        nf = body_face_normals[f]

        # Convert to grid coordinates
        Xf = (xf - origin[None]) * inv_dx[None]
        base = ti.cast(Xf - 0.5, ti.i32)
        fx = Xf - ti.cast(base, ti.f32)

        # B-spline weights
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

        # Transfer to nearby grid nodes
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            idx = base + offset
            if 0 <= idx[0] < GRID_RES and 0 <= idx[1] < GRID_RES and 0 <= idx[2] < GRID_RES:
                weight = w[i][0] * w[j][1] * w[k][2]
                ti.atomic_add(grid_collider_v[idx], weight * vf)
                ti.atomic_add(grid_collider_n[idx], weight * nf)
                ti.atomic_add(grid_collider_w[idx], weight)


@ti.kernel
def get_stats() -> ti.types.vector(5, ti.f32):
    max_v = 0.0
    min_y = 1e10
    max_y = -1e10
    avg_y = 0.0
    max_F = 0.0

    for p in range(n_particles[None]):
        vel = v[p].norm()
        ti.atomic_max(max_v, vel)
        ti.atomic_min(min_y, x[p][1])
        ti.atomic_max(max_y, x[p][1])
        avg_y += x[p][1]

        # Check deformation gradient condition
        F_norm = F[p].norm()
        ti.atomic_max(max_F, F_norm)

    np_val = float(n_particles[None])
    return ti.Vector([max_v, min_y, max_y, avg_y / np_val, max_F])


def find_boundary_vertices(faces, n_vertices):
    """Find boundary vertices of a mesh"""
    edge_count = defaultdict(int)
    for face in faces:
        edges = [
            tuple(sorted([face[0], face[1]])),
            tuple(sorted([face[1], face[2]])),
            tuple(sorted([face[2], face[0]]))
        ]
        for edge in edges:
            edge_count[edge] += 1

    boundary_edges = [e for e, count in edge_count.items() if count == 1]
    boundary_verts = set()
    for e in boundary_edges:
        boundary_verts.add(e[0])
        boundary_verts.add(e[1])

    return boundary_verts, boundary_edges


def compute_face_data(vertices, faces, prev_vertices=None, frame_dt=1.0/30.0):
    """Compute face centers, normals, and velocities for collision"""
    n_faces = len(faces)
    centers = np.zeros((n_faces, 3), dtype=np.float32)
    normals = np.zeros((n_faces, 3), dtype=np.float32)
    velocities = np.zeros((n_faces, 3), dtype=np.float32)

    for i, face in enumerate(faces):
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        center = (v0 + v1 + v2) / 3.0
        centers[i] = center

        # Normal
        e0, e1 = v1 - v0, v2 - v0
        normal = np.cross(e0, e1)
        n_len = np.linalg.norm(normal)
        if n_len > 1e-10:
            normal = normal / n_len
        normals[i] = normal

        # Velocity from previous frame
        if prev_vertices is not None:
            pv0, pv1, pv2 = prev_vertices[face[0]], prev_vertices[face[1]], prev_vertices[face[2]]
            prev_center = (pv0 + pv1 + pv2) / 3.0
            velocities[i] = (center - prev_center) / frame_dt

    return centers, normals, velocities


def compute_vertex_normals(vertices, faces):
    """Compute per-vertex normals for pushout collision"""
    normals = np.zeros_like(vertices, dtype=np.float32)
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        n = np.cross(v1 - v0, v2 - v0)
        n_len = np.linalg.norm(n)
        if n_len > 1e-10:
            n = n / n_len
        normals[face[0]] += n
        normals[face[1]] += n
        normals[face[2]] += n

    lens = np.linalg.norm(normals, axis=1)
    lens[lens < 1e-10] = 1.0
    normals = (normals.T / lens).T
    return normals


def update_body_collision(vertices, faces, prev_vertices=None, frame_dt=1.0/30.0):
    """Update body mesh collision data"""
    n_verts = len(vertices)
    verts_padded = np.zeros((MAX_BODY_VERTS, 3), dtype=np.float32)
    verts_padded[:n_verts] = vertices
    body_verts.from_numpy(verts_padded)

    # Vertex normals for pushout
    vertex_normals = compute_vertex_normals(vertices, faces)
    normals_padded = np.zeros((MAX_BODY_VERTS, 3), dtype=np.float32)
    normals_padded[:n_verts] = vertex_normals
    body_vertex_normals.from_numpy(normals_padded)

    # Compute face data for collision
    centers, normals, velocities = compute_face_data(vertices, faces, prev_vertices, frame_dt)
    n_faces = len(faces)
    n_body_faces[None] = n_faces

    centers_padded = np.zeros((MAX_BODY_FACES, 3), dtype=np.float32)
    normals_padded = np.zeros((MAX_BODY_FACES, 3), dtype=np.float32)
    velocities_padded = np.zeros((MAX_BODY_FACES, 3), dtype=np.float32)

    centers_padded[:n_faces] = centers
    normals_padded[:n_faces] = normals
    velocities_padded[:n_faces] = velocities

    body_face_centers.from_numpy(centers_padded)
    body_face_normals.from_numpy(normals_padded)
    body_face_velocities.from_numpy(velocities_padded)


def update_body_velocities(vertices, prev_vertices, frame_dt=1.0/30.0):
    """Compute per-vertex body velocities for kinematic attachment"""
    n_verts = len(vertices)
    vel_padded = np.zeros((MAX_BODY_VERTS, 3), dtype=np.float32)

    if prev_vertices is not None:
        velocities = (vertices - prev_vertices) / frame_dt
        vel_padded[:n_verts] = velocities

    body_vertex_velocities.from_numpy(vel_padded)


def setup_smart_attachment(cloth_positions, cloth_faces, body_vertices):
    """
    Auto-detect and setup attachment regions.
    Works for any cloth mesh - generalizable approach.

    ALL particles are attached with varying weights:
    - Boundary vertices close to body: KINEMATIC (weight=1.0)
    - Near-boundary: STRONG gradient (weight 0.5 -> 0.2)
    - Interior: LIGHT attachment (weight 0.15) to prevent falling
    """
    boundary_verts, _ = find_boundary_vertices(cloth_faces, len(cloth_positions))
    print(f"Found {len(boundary_verts)} boundary vertices")

    body_tree = cKDTree(body_vertices)
    n_part = len(cloth_positions)

    # Compute distance to boundary for gradient
    boundary_positions = cloth_positions[list(boundary_verts)]
    boundary_tree = cKDTree(boundary_positions)
    dist_to_boundary, _ = boundary_tree.query(cloth_positions)
    max_boundary_dist = dist_to_boundary.max()

    attach_idx = np.full(MAX_PARTICLES, -1, dtype=np.int32)
    attach_offset = np.zeros((MAX_PARTICLES, 3), dtype=np.float32)
    attach_weight = np.zeros(MAX_PARTICLES, dtype=np.float32)

    BOUNDARY_RADIUS = 0.08  # 8cm from body for kinematic
    GRADIENT_ZONE = 0.16    # 16cm gradient zone from boundary

    n_kinematic = 0
    n_strong = 0
    n_light = 0

    for p in range(n_part):
        dist_body, idx_body = body_tree.query(cloth_positions[p])

        # ALL particles get attached to nearest body vertex
        attach_idx[p] = idx_body
        attach_offset[p] = cloth_positions[p] - body_vertices[idx_body]

        if p in boundary_verts and dist_body < BOUNDARY_RADIUS:
            # Boundary vertex close to body: KINEMATIC
            attach_weight[p] = 1.0
            n_kinematic += 1

        elif dist_to_boundary[p] < GRADIENT_ZONE:
            # Near boundary: STRONG gradient (0.5 at boundary -> 0.2 at edge of zone)
            t = dist_to_boundary[p] / GRADIENT_ZONE
            attach_weight[p] = 0.5 - 0.3 * t  # 0.5 -> 0.2
            n_strong += 1

        else:
            # Interior: LIGHT attachment to prevent falling
            # Weight based on how far from boundary (farther = lighter)
            t = min(dist_to_boundary[p] / max_boundary_dist, 1.0)
            attach_weight[p] = 0.25 * (1.0 - t * 0.4)  # 0.25 -> 0.15
            n_light += 1

    print(f"Smart attachment: {n_kinematic} kinematic, {n_strong} strong, {n_light} light")

    attachment_body_idx.from_numpy(attach_idx)
    attachment_offset.from_numpy(attach_offset)
    attachment_weight.from_numpy(attach_weight)

    return n_kinematic + n_strong + n_light, boundary_verts


def init_particles_from_mesh(mesh, y_offset=0.0):
    """Initialize particles from mesh vertices"""
    global cloth_faces_np
    positions = mesh.vertices.copy()
    positions[:, 1] += y_offset
    cloth_faces_np = mesh.faces.copy()

    n_part = len(positions)
    n_particles[None] = n_part
    print(f"Particles: {n_part}, Faces: {len(cloth_faces_np)}")

    # Setup grid
    pmin, pmax = positions.min(axis=0), positions.max(axis=0)
    extent = (pmax - pmin).max() * 3.0
    center = (pmin + pmax) / 2
    orig = center - extent / 2
    orig[1] = min(orig[1], -0.5)

    dx[None] = extent / GRID_RES
    inv_dx[None] = GRID_RES / extent
    origin[None] = ti.Vector(orig.tolist())
    print(f"Grid: dx={dx[None]:.4f}, extent={extent:.2f}")

    # Initialize particle arrays
    pos_padded = np.zeros((MAX_PARTICLES, 3), dtype=np.float32)
    pos_padded[:n_part] = positions
    x.from_numpy(pos_padded)

    v.from_numpy(np.zeros((MAX_PARTICLES, 3), dtype=np.float32))
    C.from_numpy(np.zeros((MAX_PARTICLES, 3, 3), dtype=np.float32))

    # Initialize deformation gradient to identity
    F_init = np.zeros((MAX_PARTICLES, 3, 3), dtype=np.float32)
    for i in range(n_part):
        F_init[i] = np.eye(3)
    F.from_numpy(F_init)

    # Mass and volume from mesh area
    area = mesh.area
    total_mass = area * DENSITY
    mass_per_particle = total_mass / n_part
    vol_per_particle = area * 0.001 / n_part  # thin shell

    mass_arr = np.zeros(MAX_PARTICLES, dtype=np.float32)
    vol_arr = np.zeros(MAX_PARTICLES, dtype=np.float32)
    mass_arr[:n_part] = mass_per_particle
    vol_arr[:n_part] = vol_per_particle
    p_mass.from_numpy(mass_arr)
    p_vol.from_numpy(vol_arr)

    print(f"Area: {area:.4f} m^2, Mass: {total_mass:.4f} kg")

    return positions


def save_cloth_mesh(path, faces):
    """Save cloth mesh to OBJ file"""
    positions = x.to_numpy()[:n_particles[None]]
    with open(path, 'w') as f:
        for p in positions:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def save_body_mesh(path, vertices, faces):
    """Save body mesh to OBJ file"""
    with open(path, 'w') as f:
        for vert in vertices:
            f.write(f"v {vert[0]:.6f} {vert[1]:.6f} {vert[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def update_attachment_schedule(frame):
    if frame < WARMUP_FRAMES:
        attachment_scale[None] = 1.0
        warmup_kinematic[None] = 1
    else:
        warmup_kinematic[None] = 0
        if DECAY_FRAMES <= 0:
            attachment_scale[None] = MIN_ATTACHMENT_SCALE
        else:
            t = min((frame - WARMUP_FRAMES) / DECAY_FRAMES, 1.0)
            attachment_scale[None] = 1.0 - t * (1.0 - MIN_ATTACHMENT_SCALE)


def main():
    global cloth_faces_np
    print("=" * 60)
    print("MPM Cloth v28 - Kinematic Warmup + Attachment Decay")
    print("Proper MPM + Kinematic boundary + Distance failsafe")
    print("=" * 60)

    data_dir = r"F:\mpm\github\Dreamcloth\data"
    output_dir = r"F:\mpm\github\Dreamcloth\output\running_v28"
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating SMPLX RUNNING animation...")
    runner = SMPLXRunning(data_dir, gender='neutral')
    n_frames = 100
    run_sequence = runner.generate_running_sequence(n_frames, cycles=3.0)

    print("\nLoading quality cloth mesh...")
    # Use quality mesh (not the one with degenerate triangles)
    cloth_path = os.path.join(data_dir, "tshirt-sim.obj")
    if not os.path.exists(cloth_path):
        print(f"Warning: {cloth_path} not found, trying tshirt-decimated.obj")
        cloth_path = os.path.join(data_dir, "tshirt-decimated.obj")
    cloth_mesh = trimesh.load(cloth_path, force='mesh')

    verts, faces = run_sequence[0]
    print(f"Body: {len(verts)} vertices, {len(faces)} faces")

    print("\nInitializing simulation...")
    positions = init_particles_from_mesh(cloth_mesh, y_offset=-0.05)

    # Setup smart attachment - kinematic for boundary, gradient for interior
    n_attached, boundary_verts = setup_smart_attachment(positions, cloth_faces_np, verts)

    # Initialize collision and body velocities
    print("\nSetting up collision...")
    update_body_collision(verts, faces)
    update_body_velocities(verts, None)  # No velocity for first frame
    prev_verts = verts.copy()

    # Initialize attachment schedule
    update_attachment_schedule(0)

    print(f"\n{'='*60}")
    print(f"Parameters: dt={DT:.6f}, substeps={SUBSTEPS}")
    print(f"Material: E={YOUNG_MODULUS}, nu={POISSON_RATIO}")
    print(f"Anisotropic: gamma={SHEAR_STIFFNESS}, kappa={NORMAL_STIFFNESS}")
    print(f"Attachment: warmup={WARMUP_FRAMES} frames, decay={DECAY_FRAMES} frames")
    print(f"Attachment scale min={MIN_ATTACHMENT_SCALE}")
    print(f"Motion: RUNNING (3 cycles)")
    print(f"{'='*60}")

    total_start = time.time()

    for frame in range(n_frames):
        t0 = time.time()

        # Update body
        verts, faces = run_sequence[frame]
        update_body_velocities(verts, prev_verts)  # For kinematic attachment
        update_body_collision(verts, faces, prev_verts)

        # Update attachment schedule
        update_attachment_schedule(frame)

        # Substeps
        for s in range(SUBSTEPS):
            clear_collision_grid()
            mesh_to_grid_collision()
            p2g()
            grid_op()
            g2p()
            apply_kinematic_attachment()  # Enforce boundary constraints
            apply_distance_failsafe()  # Prevent catastrophic falling
            apply_body_pushout(BODY_PUSHOUT)  # Keep cloth outside body

        prev_verts = verts.copy()

        stats = get_stats()
        dt = time.time() - t0

        # Save every 2 frames
        if frame % 2 == 0:
            save_cloth_mesh(os.path.join(output_dir, f"cloth_{frame:04d}.obj"), cloth_faces_np)
            save_body_mesh(os.path.join(output_dir, f"body_{frame:04d}.obj"), verts, faces)

        if frame % 10 == 0:
            print(f"Frame {frame:3d}: v={stats[0]:.2f} m/s, y=[{stats[1]:.3f},{stats[2]:.3f}], "
                  f"F_max={stats[4]:.2f}, time={dt:.2f}s, attach_scale={attachment_scale[None]:.2f}")

        # Check stability
        if stats[0] > 100 or np.isnan(stats[0]):
            print(f"UNSTABLE at frame {frame}! v={stats[0]}")
            break

    total_time = time.time() - total_start
    print(f"\nTotal: {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"Avg: {total_time/n_frames:.2f}s per frame")
    print(f"\nOutput: {output_dir}/")


if __name__ == "__main__":
    main()
# Keep local imports for SMPL-X helper.
