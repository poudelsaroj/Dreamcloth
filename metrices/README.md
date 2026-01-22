# `metrices/` (DreamCloth evaluation code)

This folder implements **paper-grade evaluation metrics** for DreamCloth, with a bias toward the real constraint of your system:

- **Input:** a single image  
- **Output:** a **physics-based simulated rollout** (Phase2) as per-frame OBJ meshes  
- **Optional:** rendered frames and/or GT meshes (if you have them)

In a single-image setting, the most defensible evaluation is:

1) **t=0 image consistency** (does the reconstructed canonical avatar match the input?), and  
2) **physics/temporal plausibility** of the simulated rollout (does the cloth behave like cloth and stay outside the body?).

This package computes both, and becomes directly comparable to PhysAvatar/MPMAvatar-style tables when you provide GT meshes/renders.

## What it reads (inputs)

### Required: predicted Phase2 mesh sequence

Point it at a `video_####` folder containing both cloth and body meshes:

```
output/phase2_sim/video_0000/
  cloth_0000.obj
  body_0000.obj
  cloth_0001.obj
  body_0001.obj
  ...
```

### Optional: GT cloth meshes (enables CD + F-score)

If you have ground-truth cloth meshes per frame, provide:

```
<gt_video_dir>/
  cloth_0000.obj
  cloth_0001.obj
  ...
```

### Optional: rendered images (enables PSNR/SSIM/LPIPS)

If you render the predicted frame-0 and want to compare to the **input image**:

- `--input-image`: the original photo
- `--pred-render-dir`: directory containing a rendered prediction image (any common format)
- `--input-mask` (recommended): foreground mask to avoid background dominating PSNR/SSIM

### Optional: Phase3 correction budget (raw vs final render)

If you save both:

- raw MPM render (`--raw-render-dir`)
- final Phase3 render (`--final-render-dir`)

the evaluator reports how much Phase3 changes the raw physics render.

## What it writes (outputs)

It writes a single JSON file containing:

- per-frame metrics under `frames[]`
- dataset/sequence summaries under `summary`

Example output path: `metrics.json`

## Quick start (CLI)

Run:

```
python -m metrices --pred-video-dir output/phase2_sim/video_0000 --out-json metrics.json
```

If you have a single-image target and a rendered prediction:

```
python -m metrices ^
  --pred-video-dir output/phase2_sim/video_0000 ^
  --input-image path/to/input.jpg ^
  --pred-render-dir path/to/rendered_frame0 ^
  --input-mask path/to/mask.png ^
  --out-json metrics.json
```

If you have GT cloth meshes:

```
python -m metrices ^
  --pred-video-dir output/phase2_sim/video_0000 ^
  --gt-video-dir path/to/gt/video_0000 ^
  --out-json metrics.json
```

## Metrics implemented

### A) Single-image-friendly (no GT required)

These are the most important metrics for DreamCloth when the *main input is a single image*.

#### 1) Cloth-body interpenetration

For each frame `t`, compute signed distances from cloth vertices to the body surface.

- `penetration_rate(t)`: fraction of cloth vertices inside the body
- `penetration_depth_mean(t)`: mean penetration depth
- `penetration_depth_p95(t)`: 95th percentile penetration depth

#### 2) Anchored contact quality (gap + slip)

We define an **anchor set** once at `t=0`:

```
A = { i : dist( c_i(0), Body(0) ) < d_anchor }
```

This captures regions that should stay near the body (e.g., shoulders/torso for a shirt), without penalizing loose cloth.

Then per frame:

- `gap_mean(t)`, `gap_p95(t)`:

```
gap_i(t) = max(0, dist(c_i(t), Body(t)) - d_contact), i ∈ A
```

- `contact_rate(t)`: fraction of anchors with `dist(c_i(t), Body(t)) <= d_contact`

- `slip_mean(t)`, `slip_p95(t)` (tangential relative motion in the contact region):

Let `b_i(t)` be the closest body point to `c_i(t)` and `n_i(t)` the body normal there:

```
Δc = c_i(t+1) - c_i(t)
Δb = b_i(t+1) - b_i(t)
Δr = Δc - Δb
slip_i(t) = || Δr - (Δr·n_i(t)) n_i(t) ||_2
```

#### 3) DC-PhysErr (DreamCloth proposed composite)

If penetration + anchored gap + anchored slip exist for a frame:

```
dc_physerr(t) =
  penetration_depth_mean(t) / τ_pen +
  gap_mean(t)              / τ_gap +
  slip_mean(t)             / τ_slip
```

This is a reviewer-friendly single number that penalizes the three most obvious physics failures.

CLI parameters:
- `--dc-tau-pen-mm`, `--dc-tau-gap-mm`, `--dc-tau-slip-mm`

#### 4) Stretch / strain violation (relative to rest)

Assuming consistent topology across frames, for each edge `(i,j)`:

```
ratio_ij(t) = ||c_i(t)-c_j(t)|| / (||c_i(0)-c_j(0)|| + ε)
```

Reported as:
- `stretch_violation_mean(t)` = mean |ratio-1|
- `stretch_violation_p95(t)`
- `stretch_over_threshold_rate(t)` with threshold set by `--stretch-threshold`

#### 5) Bending violation (dihedral proxy)

For each edge shared by two faces, compute the dihedral-angle deviation relative to `t=0`:

```
bend_violation(t) = mean (θ(t) - θ(0))^2
```

Reported as `bend_violation_mean(t)` and `bend_violation_p95(t)`.

#### 6) Triangle stability

- `flipped_face_rate(t)`: fraction of faces whose normal flipped vs `t=0`
- `degenerate_face_rate(t)`: fraction of near-zero-area faces

#### 7) Temporal jitter (mesh acceleration)

For `t-1, t, t+1`:

```
a_i(t) = c_i(t+1) - 2 c_i(t) + c_i(t-1)
jitter(t) = mean_i ||a_i(t)||_2
```

Reported as `jitter_mean(t)` and `jitter_p95(t)`.

#### 8) Self-intersection proxy (self-proximity)

Exact triangle-triangle self-intersection needs extra geometry tooling (often FCL).
This implementation provides a **proxy**:

- sample surface points
- compute nearest-neighbor distance to a point on a different face
- flag unusually close self-proximity

Reported as:
- `self_proximity_rate(t)`
- `self_proximity_depth_mean(t)`

Threshold controlled by `--self-proximity-threshold-mm`.

#### 9) Cloth area preservation (custom, no GT required)

Total cloth surface area should be approximately preserved (most garments are close to inextensible).

- `area_m2(t)`: cloth surface area in m^2
- `area_ratio(t) = area(t) / area(0)`

### B) Baseline-comparable (requires GT)

If you provide `--gt-video-dir`, the evaluator also computes:

#### 1) Chamfer Distance (CD)

With sampled surfaces `X` and `Y`:

```
CD(X,Y) = mean_{x∈X} min_{y∈Y} ||x-y|| + mean_{y∈Y} min_{x∈X} ||y-x||
```

Reported per frame as:
- `chamfer_l2`
- `chamfer_l2_squared`

#### 2) F-Score @ τ

```
Precision = mean_{x∈X} 1[min_{y∈Y} ||x-y|| < τ]
Recall    = mean_{y∈Y} 1[min_{x∈X} ||y-x|| < τ]
F         = 2PR/(P+R)
```

Use `--fscore-threshold-mm 1.0` for the common 1mm threshold (when meshes are in meters).

### C) Image metrics (single-image target)

If you provide `--input-image` and `--pred-render-dir`:

- `psnr`, `ssim`
- `lpips` if the optional `lpips` package is installed

Masking with `--input-mask` is strongly recommended.

## Code organization (subpackages)

- `metrices/io/`: reading images and mesh sequences
- `metrices/metrics/geometry/`: CD + F-score
- `metrices/metrics/physics/`: penetration, anchored gap/slip, stretch/bending, self-proximity proxy, area stats
- `metrices/metrics/temporal/`: jitter
- `metrices/metrics/image/`: PSNR/SSIM and optional LPIPS
- `metrices/evaluator.py`: orchestration and aggregation
- `metrices/cli.py`: CLI entry (`python -m metrices ...`)

## Notes for papers (what to emphasize when input is a single image)

For DreamCloth, reviewers will accept that you cannot compute GT temporal mesh errors on real photos. What they still demand is:

- **t=0 reconstruction quality** (render-to-image)
- **penetration / contact correctness** (penetration + anchored gap/slip + dc_physerr)
- **physical plausibility** (stretch + bending + stability)
- **temporal stability** (jitter + visual flicker if you later add video renders)
- **efficiency/robustness** (log success rate + time/frame from your pipeline runner)

This package covers the first four from saved outputs; add timing logs in your pipeline scripts for the last part.
