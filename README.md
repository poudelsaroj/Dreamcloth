# Dreamcloth

A clothing website project.

## Getting Started

Clone the repository and open the project files to get started.

## Data Files (MPM)

To run the scripts in `mpm/`, put the data files in these locations:

- `F:\mpm\github\Dreamcloth\data\SMPLX_NEUTRAL.npz` (or the gendered equivalent) for the body model used by `mpm\mpm_cloth_v28.py`.
- `F:\mpm\github\Dreamcloth\data\tshirt-sim.obj` (fallback: `tshirt-decimated.obj`) for the cloth mesh used by `mpm\mpm_cloth_v28.py` and `mpm\mpm_cloth_v30_jump.py`.
- `F:\mpm\github\Dreamcloth\data\jump\frame_*.obj` for the jump body frames used by `mpm\mpm_cloth_v30_jump.py`.

Outputs are written to `F:\mpm\github\Dreamcloth\output\running_v28` and `F:\mpm\github\Dreamcloth\output\jump_v30` by default.




## End-to-End Pipeline (PhaseI -> MPM -> Phase3)

The end-to-end training/test scripts live in the repo root:
- `train_end_to_end.py`
- `test_end_to_end.py`

### Required Inputs

PhaseI produces the meshes that drive MPM:
- Body mesh (SMPL-X): `.obj`
- Cloth mesh (Garment): `.obj`

If you used `PhaseI/run_full_pipeline.sh`, the default collected outputs are:
- `PhaseI/full_pipeline_outputs/collected_econ_and_garment3d/econ/` (body)
- `PhaseI/full_pipeline_outputs/collected_econ_and_garment3d/garment3d/` (cloth)

The scripts will auto-detect meshes from those folders, but you can always pass:
- `--body-mesh <path/to/body.obj>`
- `--cloth-mesh <path/to/cloth.obj>`

### Phase2 Output Layout (MPM)

The MPM step writes simulated frames under:
- `output/phase2_sim/video_0000/cloth_0000.obj`
- `output/phase2_sim/video_0000/body_0000.obj`
- ...

This `video_0000` layout is what Phase3 expects.

### Run Training

```bash
python train_end_to_end.py --body-mesh "F:\path\to\body.obj" --cloth-mesh "F:\path\to\cloth.obj"
```

To run PhaseI from the script (requires bash + conda environments):

```bash
python train_end_to_end.py --run-phase1 --phase1-input-image "F:\path\to\image.jpg"
```

### Run Testing (Smoke Test)

```bash
python test_end_to_end.py --body-mesh "F:\path\to\body.obj" --cloth-mesh "F:\path\to\cloth.obj" --mpm-output-dir "F:\mpm\github\Dreamcloth\output\phase2_sim"
```

### Notes

- The MPM simulator is not differentiable. To keep gradients flowing into
  Phase3 parameters, the end-to-end scripts apply a small differentiable
  modulation on top of the rendered MPM video.
- If you already ran MPM, pass `--skip-mpm` to `train_end_to_end.py` and
  point `--mpm-output-dir` at the existing output folder.

## Evaluation (metrics)

- Paper-style metric definitions: `EVALUATION_METRICS.md`
- Metric computation code (single-image-first): `metrices/README.md`

Example:

```bash
python -m metrices --pred-video-dir output/phase2_sim/video_0000 --out-json metrics.json
```

## Phase3 Diffusion Prior Options

By default, Phase3 uses a lightweight local video diffusion model (`phase3/model.py`) as a frozen prior.
If you want to use **Wan2.2 I2V** as the prior instead, see `phase3/README.md`.


## License

MIT License
