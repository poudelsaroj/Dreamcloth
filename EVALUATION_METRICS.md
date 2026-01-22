# DreamCloth Evaluation Metrics (SIGGRAPH/TOG-ready)

This document defines a **paper-ready evaluation protocol** for DreamCloth (single image -> canonical body/garment -> **MPM** simulation -> rendering -> optional Phase3 refinement). It is designed to:

1) **Compare directly** to recent physics-based clothed avatar / cloth dynamics work (e.g., **PhysAvatar**, **MPMAvatar**, and related papers), and  
2) Add **DreamCloth-specific metrics** that answer the reviewer questions standard PSNR/SSIM/CD tables don’t.

## What DreamCloth outputs (so we pick metrics that fit)

Phase2 (MPM) produces per-frame meshes:

- `output/phase2_sim/video_0000/cloth_0000.obj`
- `output/phase2_sim/video_0000/body_0000.obj`
- …

Phase3 expects the `video_####` folder layout. The current Phase3 dataloader has a placeholder renderer (`phase3/dataloader.py`); for paper evaluation, use a **real renderer** (PyTorch3D / Blender) with **matched cameras**.

## Choose an evaluation track (depends on your ground truth)

Most SIGGRAPH/TOG reviews hinge on whether the evaluation matches the available supervision. Use the track that matches your dataset and be explicit in the paper.

### Track A -- GT 3D meshes available (strongest)
You have per-frame ground-truth cloth meshes (synthetic or registered 4D scans).

- **Use:** CD, F-Score@τ, PVE/V2V, normal error, curvature/bending stats, + all physics plausibility metrics.

### Track B -- GT images/videos + cameras, but no GT cloth meshes
You have real videos (possibly multi-view), calibrated cameras, and can render predictions.

- **Use:** PSNR/SSIM/LPIPS (masked), FID/KID (images), FVD/JEDi (video) + physics plausibility metrics from meshes.

### Track C -- Single image only (no temporal GT)
You only have the input image (common in single-image pipelines).

- **Use:** single-frame render metrics at `t=0` + **physics plausibility** and **stability** metrics over the simulated rollout + DreamCloth-specific metrics (below).

## Tier 1: "baseline-comparable" metrics (PhysAvatar / MPMAvatar style)

These are the safest metrics for a main comparison table because they are widely used and expected.

| Category | Metric | Direction | Comparable to |
|---|---:|:---:|---|
| Geometry | Chamfer Distance (CD) | ↓ | PhysAvatar, MPMAvatar, CLOTH3D-style evals |
| Geometry | F-Score @ τ = 0.001 | ↑ | PhysAvatar, MPMAvatar (τ=1mm in meters) |
| Appearance | LPIPS | ↓ | Most avatar NVS / rendering papers |
| Appearance | PSNR | ↑ | Most avatar NVS / rendering papers |
| Appearance | SSIM | ↑ | Most avatar NVS / rendering papers |

### 1) Chamfer Distance (CD)

**What it measures:** 3D surface accuracy between predicted and GT cloth meshes.

**Definition (symmetric):**

```
CD(X, Y) =
  (1/|X|) Σ_{x∈X} min_{y∈Y} ||x - y||_2  +
  (1/|Y|) Σ_{y∈Y} min_{x∈X} ||y - x||_2
```

**How to compute (per frame):**

1. Load predicted `cloth_t.obj` and GT cloth mesh for the same frame.
2. Uniformly sample surface points (e.g., 10k–100k points) from each mesh.
3. Compute nearest-neighbor distances both ways (KD-tree or PyTorch3D `knn_points`).
4. Report CD in **meters** or **millimeters** (be consistent and state units).

**Reporting:** average CD over frames -> average over sequences (and optionally std / 95% CI).

### 2) F-Score @ τ = 0.001 (1mm in meters)

**What it measures:** a precision/recall-style geometric similarity at a strict threshold.

Let `d(x, Y) = min_{y∈Y} ||x-y||_2`.

```
Precision = |{ x∈X : d(x,Y) < τ }| / |X|
Recall    = |{ y∈Y : d(y,X) < τ }| / |Y|
FScore    = 2 * Precision * Recall / (Precision + Recall)
```

**Critical detail:** τ must match your mesh units. If meshes are in meters, **τ=0.001 = 1mm**.

### 3) LPIPS / PSNR / SSIM (rendered frames)

**What they measure:** image-level similarity between rendered predictions and GT images.

**Best practice (reviewer-proofing):**

- Compute metrics on the **foreground mask** (person + clothing) to prevent background domination.
- Use the **same camera intrinsics/extrinsics** as the GT image(s).
- Keep lighting fixed across methods, or render albedo-only if comparing geometry/shape.

**Reporting:** per-frame values averaged over frames -> sequences -> dataset. Provide LPIPS as the primary perceptual metric when PSNR/SSIM disagree with visuals (common with small misalignment).

## Tier 2: Additional “standard” metrics (common in cloth/scan datasets)

Use these when the dataset supports them; they strengthen the evaluation section without feeling “invented”.

### A) V2V / PVE (requires correspondence)

If predicted and GT cloth share vertex correspondence (same topology / consistent indexing):

```
V2V(t) = (1/N) Σ_i || v̂_i(t) - v_i(t) ||_2
```

If no correspondence, use point-to-surface metrics (CD / surface error) instead.

### B) Normal error / normal consistency

Compute per-vertex or sampled-point normals and report mean angular error (degrees/radians). This helps evaluate wrinkle orientation and shading correctness.

### C) Silhouette IoU (2D)

Render a binary silhouette mask and compute IoU with GT foreground silhouette. This is especially useful when GT is image-only.

## Tier 2: Physics plausibility metrics (no GT required)

These metrics are critical for physics-based cloth papers because they quantify failure modes that PSNR/CD do not capture.

### 1) Cloth-body interpenetration

**Why reviewers care:** penetration is an immediate “physics fail”.

Compute signed distances from cloth vertices (or sampled cloth points) to the body mesh:

- **Penetration rate:** `% points with SDF < 0`
- **Penetration depth:** mean / 95th percentile of `max(0, -SDF)`

**Practical input:** you already have `body_####.obj` and `cloth_####.obj` per frame in `output/phase2_sim/video_####/`.

### 2) Self-intersection severity (cloth–cloth)

Self-intersections correlate strongly with unrealistic cloth.

Options (in increasing sophistication):

- Triangle–triangle intersection **count**
- Total intersection **area** (if computed)
- Intersection “**contour length**” (length of intersection curves)
- Global intersection “**volume**” (requires more geometry processing)

Report mean and worst-case (max) across frames.

### 3) Stretch / strain violation (relative to rest state)

**Goal:** cloth should not behave like rubber.

If topology is consistent across frames, let `e_ij(t)` be an edge length at time `t` and `e_ij(0)` the rest length:

```
StretchRatio_ij(t) = |e_ij(t)| / (|e_ij(0)| + ε)
StretchViolation(t) = mean_ij |StretchRatio_ij(t) - 1|
```

Common summaries:

- mean |stretch−1|
- 95th percentile |stretch−1|
- `% edges with stretch > 1.10` (10% stretch)

### 4) Bending / curvature metrics (wrinkle plausibility)

A simple, robust proxy for bending behavior is dihedral-angle deviation along mesh edges shared by two triangles:

```
BendViolation(t) = mean_edges (θ_ij(t) - θ_ij(0))^2
```

If you want a more “wrinkle-centric” signal, report curvature statistics (mean/percentiles of |mean curvature|).

### 5) Triangle flips / degeneracy (numerical stability)

Count inverted triangles (negative signed area / inconsistent orientation) and near-degenerate triangles. This is an easy "sanity metric" that reviewers trust.

## Tier 2: Temporal consistency metrics (mesh + render)

Physics-based rollouts should be stable and non-jittery.

### 1) Vertex acceleration / jerk (mesh space)

With consistent vertex indexing:

```
v_i(t)   = v_i(t+1) - v_i(t)
a_i(t)   = v_i(t+1) - 2 v_i(t) + v_i(t-1)
Jitter   = mean_{t,i} ||a_i(t)||_2
```

Report mean and 95th percentile. This catches high-frequency "twitching".

### 2) Temporal LPIPS (tLPIPS)

Compute LPIPS between consecutive rendered frames (optionally on flow-warped frames if you have optical flow). This is a practical way to quantify flicker.

## Tier 2: Efficiency + robustness (physics credibility)

These numbers often decide whether reviewers believe the system is usable.

- **Simulation success rate:** fraction of sequences that complete without divergence / NaNs / catastrophic mesh failure.
- **Time per frame / FPS:** wall-clock time for Phase2 (and separately for Phase3 if relevant).
- **Peak memory:** GPU and CPU (if possible).
- **Failure taxonomy:** report what fails (penetration blow-up, self-intersection explosion, solver instability).

## Tier 3: DreamCloth-specific metrics (proposed for this paper)

These metrics are designed to be both **implementable from DreamCloth outputs** and **reviewer-legible** (they answer specific “physics pipeline” concerns).

### DC-1) Cloth-Body Physical Error (DC-PhysErr)

**Motivation:** PhysDiff-style “physical error” metrics are compelling because they quantify concrete failure modes. For cloth, the most reviewer-salient failures are **penetration**, **floating in anchored regions**, and **unphysical slip**.

**Inputs:** `cloth_t.obj`, `body_t.obj` per frame.

**Step 0 (define an anchored set A):**

- Compute cloth–body unsigned distances at `t=0`.
- Define anchored vertices `A = { i : dist(c_i(0), Body(0)) < d_anchor }` (e.g., `d_anchor = 5mm`).

This focuses the metric on regions that should remain near the body (e.g., shoulders/torso for a shirt) and avoids penalizing intentionally loose regions.

**Components (per frame t):**

1) **Penetration depth**

```
PenDepth(t) = mean_i max(0, -SDF_body_t(c_i(t)))
PenRate(t)  = mean_i 1[ SDF_body_t(c_i(t)) < 0 ]
```

2) **Anchored gap (floating)**

```
Gap(t) = mean_{i∈A} max(0, dist(c_i(t), Body(t)) - d_contact)
```

3) **Anchored slip (tangential relative motion)**

Let `b_i(t)` be the closest body point to `c_i(t)`, and `n_i(t)` the body normal there. Define per-frame displacements:

```
Δc_i(t) = c_i(t+1) - c_i(t)
Δb_i(t) = b_i(t+1) - b_i(t)
Slip_i(t) = || (Δc_i(t) - Δb_i(t)) - ((Δc_i(t) - Δb_i(t))·n_i(t)) n_i(t) ||_2
Slip(t) = mean_{i∈A} Slip_i(t)
```

**How to report (recommended):**

- Report the **three components** separately (PenDepth/PenRate, Gap, Slip).
- Optionally report a single composite score:

```
DC-PhysErr = PenDepth/τ_pen + Gap/τ_gap + Slip/τ_slip
```

with thresholds like `τ_pen=2mm`, `τ_gap=5mm`, `τ_slip=5mm/frame` (tune once and keep fixed).

### DC-2) Phase3 “Correction Budget” (how much non-physics is used)

**Motivation:** DreamCloth uses a non-differentiable simulator; Phase3 applies a small differentiable modulation so parameters can be optimized. Reviewers will ask: “Is Phase3 secretly doing the hard part?”

**Definition:** compare the **raw MPM render** `R_t` to the **final output** `F_t`.

**Metrics (masked to garment pixels):**

- `CB-L1`: mean absolute pixel difference `mean |F_t - R_t|`
- `CB-LPIPS`: `LPIPS(F_t, R_t)`
- (Optional) **high-frequency ratio**: fraction of FFT energy of `(F_t - R_t)` above a frequency cutoff

**Interpretation:**

- Low correction budget → physics rollout explains most structure/motion.
- High correction budget → output relies heavily on image-space post-correction (riskier for a physics paper).

**Implementation note:** save both render streams during evaluation (raw render and final render).

### DC-3) Material parameter recovery + identifiability

DreamCloth explicitly exposes material/simulation parameters in `phase3/train.py`:

- `young_modulus`, `poisson_ratio`, `shear_stiffness`, `normal_stiffness`, `density`, `friction`, `damping`

This supports two publishable evaluation directions.

#### DC-3a) Parameter recovery error (synthetic / controlled GT)

If you have synthetic sequences with known GT parameters `p*`:

```
RelErr(p) = |p̂ - p*| / (|p*| + ε)
```

Report mean/median and 95th percentile per parameter. This is directly comparable to physics-parameter estimation evaluations (e.g., PhysAvatar-style claims).

#### DC-3b) Material consistency across motions (no GT required)

**Goal:** estimated material parameters should be **garment-intrinsic**, not motion-specific.

Protocol:

1. Take the same garment instance across K different motions (or subsequences).
2. Estimate parameters independently per motion → `{p̂_k}`.
3. Report coefficient of variation (CV) per parameter:

```
CV(p) = std_k(p̂_k) / (mean_k(p̂_k) + ε)
```

Lower CV is better (more identifiable/consistent).

### DC-4) Wrinkle richness & correctness (curvature-distribution distance)

**Motivation:** CD can be “good” while wrinkles are over-smoothed; PSNR/SSIM can be “bad” while wrinkles are perceptually right. Reviewers often care about wrinkle statistics.

If you have GT cloth meshes (Track A):

1. Compute a per-vertex curvature proxy (e.g., mean curvature magnitude |H|).
2. Build histograms of |H| for prediction and GT.
3. Report a distribution distance per frame (e.g., Wasserstein-1 / EMD) and average over time.

This produces a “wrinkle-centric” metric that complements CD and normal error.

## How to report results (paper structure that reviewers expect)

**Main comparison table (minimum):**

- CD ↓, F-Score@0.001 ↑, LPIPS ↓, PSNR ↑, SSIM ↑
- + Success rate ↑ and time/frame ↓ (strongly recommended for physics papers)

**Physics credibility table / supplement:**

- Penetration depth/rate ↓
- Self-intersection severity ↓
- Stretch violation ↓
- Temporal jitter ↓
- DC-PhysErr components (PenDepth, Gap, Slip) ↓

**DreamCloth-specific credibility (ablation):**

- Phase3 correction budget (CB-L1 / CB-LPIPS) ↓
- Parameter recovery (if synthetic) or parameter consistency across motions (if real)

## Reproducibility checklist (avoid common reviewer objections)

- State **mesh units** (m vs mm) and keep thresholds consistent.
- State whether meshes are aligned by dataset poses/cameras (preferred) vs ICP (often frowned upon unless justified).
- Report per-sequence means (so long sequences don’t dominate).
- Provide standard deviation or 95% confidence intervals for headline metrics.
- For render metrics, specify mask source and whether backgrounds are included.
