# Individual Run Setup (ECON / InstantMesh / Garment3DGen)

This directory contains scripts to set up and run **ECON**, **InstantMesh**, and **Garment3DGen** independently with reproducible package versions, plus small end-to-end runners.

## Files

- **Environment setup**
  - `setup_econ_env.sh` - Creates ECON conda environment and installs all packages with exact versions
  - `setup_instantmesh_env.sh` - Creates InstantMesh conda environment using the pipeline-known-good package versions
  - `setup_garment3d_env.sh` - Creates Garment3DGen conda environment and installs all packages with exact versions

- **Inference / runners**
  - `run_econ.sh` - Run ECON inference on a folder of images (defaults to ECON `examples/`)
  - `run_instantmesh.sh` - Run InstantMesh inference on a single image (defaults to the same demo image ECON uses)
  - `run_instantmesh_and_garment3d.sh` - Run InstantMesh then Garment3DGen using InstantMesh output `.obj` as `--target_mesh`
  - `run_full_pipeline.sh` - Run setup scripts (optional), then ECON → InstantMesh → Garment3DGen on the same image, then collect outputs

## Quick Start

### 0. Clone dependencies (only if missing)

If your repo does **not** already contain these folders:
- `../econ/ECON`
- `../InstantMesh/InstantMesh`
- `../Garment3d/Garment3DGen`

Then run:

```bash
./clone_deps.sh
```

Notes:
- ECON and InstantMesh have default clone URLs.
- Garment3DGen clone URL varies by project, so you may need to set:

```bash
GARMENT3D_REPO_URL="<your-garment3dgen-repo-url>" ./clone_deps.sh
```

### 1. Setup Environments

Run the setup scripts to create the conda environments and install all packages:

```bash
./setup_econ_env.sh
./setup_instantmesh_env.sh
./setup_garment3d_env.sh
```

This script will:
- Check if the environment already exists (asks if you want to recreate it)
- Install packages in a specific order with exact versions where applicable
- Use `--no-deps`, `--no-build-isolation`, and `--break-system-packages` flags where appropriate
- Verify the installation at the end (where possible)

### 2. Download ECON Models (only needed for ECON)

Make sure you have registered and downloaded the required models:

```bash
cd ../econ/ECON
bash fetch_data.sh  # Requires ICON website credentials
```

### 3. Run ECON

```bash
./run_econ.sh [input_dir] [output_dir] [config_file]
```

### 4. Run InstantMesh (single image)

```bash
./run_instantmesh.sh [input_image] [output_dir] [config_yaml]
```

Defaults:
- `input_image`: ECON demo image (`../econ/ECON/examples/demo.jpg`, with fallback)
- `config_yaml`: `../InstantMesh/InstantMesh/configs/instant-mesh-large.yaml`

### 5. Run InstantMesh → Garment3DGen (mesh_target)

```bash
./run_instantmesh_and_garment3d.sh [input_image] [output_root]
```

This runs InstantMesh first, finds the generated `.obj`, then runs Garment3DGen with:
- `--target_mesh <instantmesh_obj>`

### 6. Full pipeline (setup → run → collect)

```bash
./run_full_pipeline.sh [input_image] [output_root]
```

This runs:
- Setup: ECON → InstantMesh → Garment3DGen (skippable with `SKIP_ENV=1`)
- Inference: ECON → InstantMesh → Garment3DGen (`--target_mesh` from InstantMesh)
- Collect: copies ECON + Garment3D outputs into one folder under `output_root`

By default, `run_full_pipeline.sh` will also attempt to run `./clone_deps.sh` if required folders are missing.
To skip that behavior (if you already cloned everything), set:

```bash
SKIP_CLONE=1 ./run_full_pipeline.sh
```

## Setup Script Details

The `setup_econ_env.sh` script installs packages in this order:

1. **PyTorch stack** (CUDA 11.8) - from PyTorch index
2. **Build tools** - setuptools, wheel, packaging, Cython, cmake, ninja
3. **NumPy and SciPy** - foundation packages
4. **PyTorch3D** - 3D processing library
5. **Image processing** - Pillow, OpenCV, scikit-image, imageio
6. **3D mesh processing** - trimesh, open3d, xatlas, fast_simplification, pyembree, Rtree
7. **Deep learning utilities** - pytorch-lightning, kornia, einops, torchmetrics
8. **ML/scientific** - scikit-learn, joblib, onnxruntime
9. **SMPL/SMPL-X** - chumpy
10. **Media processing** - mediapipe, protobuf
11. **Utilities** - termcolor, tqdm, matplotlib, yacs, etc.
12. **Cloud/API** - boto3, huggingface-hub
13. **rembg** - custom fork from git
14. **CUDA packages** - cupy and nvidia packages
15. **Remaining dependencies** - all other packages from pip list

### Installation Flags

The script uses these pip flags strategically:

- `--no-deps` - Skip dependency resolution (installs exact versions only)
- `--no-build-isolation` - Disable build isolation for faster installs
- `--break-system-packages` - Allow installation into system Python if needed

If a package fails with `--no-deps`, the script falls back to installing with dependency resolution.

## Run Script Usage

### Basic Usage

```bash
# Run on ECON examples directory with default settings
./run_econ.sh

# Run on custom input directory
./run_econ.sh /path/to/images /path/to/outputs

# Run with custom config
./run_econ.sh ../econ/ECON/examples ./outputs ../econ/ECON/configs/econ.yaml
```

### Environment Variables

- `ECON_ENV` - Conda environment name (default: `econ`)
- `ECON_DIR` - Path to ECON directory (default: `../econ/ECON`)
- `INSTANTMESH_ENV` - Conda environment name (default: `InstantMesh`)
- `IM_DIR` - Path to InstantMesh directory (default: `../InstantMesh/InstantMesh`)
- `GARMENT3D_ENV` - Conda environment name (default: `garment3d`)
- `G3D_DIR` - Path to Garment3DGen directory (default: `../Garment3d/Garment3DGen`)
- `SAVE_VIDEO` - InstantMesh save video flag (`1` or `0`, default: `1`)
- `G3D_SOURCE_MESH` - Garment3DGen source garment mesh (default: `.../meshes/tshirt.obj`)
- `SKIP_ENV` - Skip environment setup in `run_full_pipeline.sh` (`1` to skip)
- `GPU_DEVICE` - GPU device index (default: `0`)
- `MULTI_PERSON` - Enable multi-person mode (default: `false`)
- `NO_VIS` - Skip visualization steps (default: `false`)
- `LOOP_SMPL` - SMPL optimization loops (default: `50`)
- `PATIENCE` - Early stopping patience (default: `5`)

### Examples

```bash
# Single person, no visualization (faster)
NO_VIS=true ./run_econ.sh

# Run InstantMesh on default ECON demo image
./run_instantmesh.sh

# Run InstantMesh without video
SAVE_VIDEO=0 ./run_instantmesh.sh

# Run InstantMesh -> Garment3DGen end-to-end
./run_instantmesh_and_garment3d.sh

# Full pipeline (skip environment setup if already installed)
SKIP_ENV=1 ./run_full_pipeline.sh

# Multi-person mode
MULTI_PERSON=true ./run_econ.sh

# Use different GPU
GPU_DEVICE=1 ./run_econ.sh

# Custom input and output
./run_econ.sh /path/to/my/images /path/to/my/outputs
```

## Package Versions

All packages are installed with exact versions where possible. InstantMesh uses the versions that are known to work in the pipeline.

- **PyTorch**: 2.0.1+cu118
- **torchvision**: 0.15.2+cu118
- **torchaudio**: 2.0.2+cu118
- **PyTorch3D**: 0.7.4
- **NumPy**: 1.24.4
- **SciPy**: 1.10.1
- **OpenCV**: 4.12.0.88
- **trimesh**: 4.10.1
- **open3d**: 0.19.0
- **kornia**: 0.7.3
- **pytorch-lightning**: 2.4.0
- And 150+ other packages with exact versions

## Troubleshooting

### Environment Already Exists

If the environment already exists, the script will ask if you want to remove it. Answer `y` to recreate, or `n` to use the existing one.

### PyTorch Installation Fails

If PyTorch installation fails, make sure you have CUDA 11.8 installed and accessible. The script installs PyTorch from the PyTorch CUDA index.

### PyTorch3D Build Fails

PyTorch3D takes ~20 minutes to build. If it fails:
1. Make sure PyTorch is installed correctly first
2. Check that you have build tools: `sudo apt-get install build-essential`
3. The script will try without `--no-build-isolation` if needed

### Missing Dependencies

If you get import errors after installation, some packages may need their dependencies resolved. The script has fallback mechanisms, but you can manually install:

```bash
mamba activate econ
pip install --break-system-packages <missing-package>
```

### Missing Models

If you get errors about missing model files:

```bash
cd ../econ/ECON
bash fetch_data.sh
```

## Output

ECON will generate:
- `*_full.obj` - Full mesh reconstruction
- `*_smpl.obj` - SMPL-X body mesh
- Various visualization images and intermediate files

All outputs will be saved to the specified output directory.

For `run_full_pipeline.sh`, outputs are organized under:
- `output_root/econ`
- `output_root/instantmesh`
- `output_root/garment3d`
- `output_root/collected_econ_and_garment3d` (combined)

## Notes

- The setup script installs packages in a specific order to handle dependencies correctly
- Some packages (like PyTorch3D) take a long time to build (~20 minutes)
- The script uses `--no-deps` where possible to ensure exact versions, but falls back to dependency resolution if needed
- All packages are installed with exact versions - no automatic version resolution

