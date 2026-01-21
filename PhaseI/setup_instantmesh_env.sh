#!/usr/bin/env bash
# Script to create InstantMesh conda environment and install all packages with exact versions
#
# This mirrors the style of setup_econ_env.sh / setup_garment3d_env.sh and is designed for
# "known-good" pipeline reproducibility over "latest" compatibility.
#
# Key cautions / gotchas (read before running):
# - CUDA compatibility: prefer installing CUDA toolkit via conda (nvidia channel) to match PyTorch wheels.
# - Compilers: if you need to build any CUDA/C++ extensions (rare with the pinned wheels), you may need an older g++.
# - pip flags: the script uses --no-deps and --no-build-isolation to preserve exact versions; it falls back if needed.
# - If you're using a system python (not conda), pip may require --break-system-packages; we keep it enabled for parity.
#
# Usage:
#   chmod +x ./setup_instantmesh_env.sh
#   ./setup_instantmesh_env.sh
#
# Optional environment variables:
#   INSTANTMESH_ENV="InstantMesh"   (default: InstantMesh)
#   PYTHON_VERSION="3.10"          (default: 3.10)
#   INSTALL_CONDA_CUDA="true"      (default: true) install CUDA toolkit inside conda for best compatibility
#   CUDA_LABEL="cuda-12.1.0"       (default: cuda-12.1.0) nvidia channel label
#
set -euo pipefail

###############################################################################
# Configuration
###############################################################################
INSTANTMESH_ENV="${INSTANTMESH_ENV:-InstantMesh}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
INSTALL_CONDA_CUDA="${INSTALL_CONDA_CUDA:-true}"
CUDA_LABEL="${CUDA_LABEL:-cuda-12.1.0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

###############################################################################
# Helpers
###############################################################################
log() { echo "[INFO] $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }
warn() { echo "[WARN] $*"; }

init_conda() {
  if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    # Prefer mamba shell hook when available; fallback to conda.sh.
    if command -v mamba >/dev/null 2>&1; then
      eval "$(mamba shell hook --shell bash 2>/dev/null)" || true
      local conda_base
      conda_base="$(mamba info --base 2>/dev/null || conda info --base 2>/dev/null || true)"
      if [ -n "${conda_base}" ] && [ -f "${conda_base}/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1090
        source "${conda_base}/etc/profile.d/conda.sh"
      fi
    else
      local conda_base
      conda_base="$(conda info --base 2>/dev/null || true)"
      if [ -n "${conda_base}" ] && [ -f "${conda_base}/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1090
        source "${conda_base}/etc/profile.d/conda.sh"
      fi
    fi
  fi
}

env_exists() {
  conda env list | awk '{print $1}' | grep -Fx "${INSTANTMESH_ENV}" >/dev/null 2>&1
}

run_conda() {
  # Use mamba run if available (faster dependency solver when we do conda installs).
  if command -v mamba >/dev/null 2>&1; then
    mamba run -n "${INSTANTMESH_ENV}" "$@"
  else
    conda run -n "${INSTANTMESH_ENV}" "$@"
  fi
}

run_pip() {
  run_conda pip "$@"
}

run_python() {
  run_conda python "$@"
}

pip_install_exact() {
  # Install pinned packages without dependency resolution first; if it fails, retry with deps.
  # shellcheck disable=SC2068
  run_pip install --no-deps --no-build-isolation --break-system-packages $@ || {
    warn "pip install failed with --no-deps/--no-build-isolation; retrying with dependency resolution..."
    run_pip install --break-system-packages $@
  }
}

###############################################################################
# Main Setup
###############################################################################
init_conda

if env_exists; then
  log "Environment '${INSTANTMESH_ENV}' already exists."
  read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    log "Removing existing environment '${INSTANTMESH_ENV}'..."
    conda env remove -n "${INSTANTMESH_ENV}" -y
  else
    log "Using existing environment. Will use (mamba/conda) run for package installation..."
  fi
fi

if ! env_exists; then
  log "Creating conda environment '${INSTANTMESH_ENV}' with Python ${PYTHON_VERSION}..."
  if command -v mamba >/dev/null 2>&1; then
    mamba create -n "${INSTANTMESH_ENV}" python="${PYTHON_VERSION}" -y
  else
    conda create -n "${INSTANTMESH_ENV}" python="${PYTHON_VERSION}" -y
  fi
fi

log "Using run commands in environment '${INSTANTMESH_ENV}'..."

log "Upgrading pip..."
run_pip install --upgrade pip==25.3 --break-system-packages

###############################################################################
# System-level notes (not executed)
###############################################################################
warn "If you hit build errors, ensure you have system tools:"
warn "  sudo apt-get update && sudo apt-get install -y build-essential git ffmpeg"
warn "If CUDA/C++ builds fail due to compiler version, consider installing an older g++ (e.g. g++-11 or g++-12)."

###############################################################################
# Step 0: (Recommended) install CUDA toolkit into conda for best compatibility
###############################################################################
if [[ "${INSTALL_CONDA_CUDA}" == "true" ]]; then
  log "Step 0: Installing CUDA toolkit into conda env (nvidia/${CUDA_LABEL})..."
  if command -v mamba >/dev/null 2>&1; then
    mamba install -n "${INSTANTMESH_ENV}" -y ninja -c conda-forge || true
    mamba install -n "${INSTANTMESH_ENV}" -y -c "nvidia/label/${CUDA_LABEL}" cuda || {
      warn "Conda CUDA install failed (often OK if you rely on system driver + PyTorch wheel CUDA runtime). Continuing..."
    }
  else
    conda install -n "${INSTANTMESH_ENV}" -y ninja -c conda-forge || true
    conda install -n "${INSTANTMESH_ENV}" -y -c "nvidia/label/${CUDA_LABEL}" cuda || {
      warn "Conda CUDA install failed (often OK if you rely on system driver + PyTorch wheel CUDA runtime). Continuing..."
    }
  fi
else
  warn "Skipping conda CUDA toolkit install (INSTALL_CONDA_CUDA=false)."
fi

###############################################################################
# Step 1: PyTorch stack (CUDA 12.1 wheel index)
###############################################################################
log "Step 1: Installing PyTorch stack (CUDA 12.1 wheels)..."
run_pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121 \
  --break-system-packages || error "Failed to install PyTorch stack"

###############################################################################
# Step 2: xFormers (pinned)
###############################################################################
log "Step 2: Installing xformers..."
run_pip install xformers==0.0.22.post7 --break-system-packages || {
  warn "xformers wheel install failed; retrying without build isolation (may compile, slow)..."
  run_pip install xformers==0.0.22.post7 --no-build-isolation --break-system-packages || {
    error "Failed to install xformers. Ensure CUDA 12.1 compatible torch wheels are installed and you have build tools."
  }
}

###############################################################################
# Step 3: Core pip packages (exact known-good versions from working pipeline)
###############################################################################
log "Step 3: Installing InstantMesh pip packages (known-good pinned versions)..."

# Foundational/utility packages used by many deps
pip_install_exact \
  setuptools==80.9.0 wheel==0.45.1 packaging==25.0 typing_extensions==4.15.0 \
  filelock==3.20.0 fsspec==2025.12.0 tqdm==4.67.1 pyyaml==6.0.3 \
  numpy==1.26.4 scipy==1.15.3 pillow==10.4.0 psutil==7.1.3

# HF / diffusion stack
pip_install_exact \
  transformers==4.35.0 tokenizers==0.14.1 diffusers==0.20.2 accelerate==0.20.3 \
  huggingface-hub==0.17.3 safetensors==0.7.0 regex==2025.11.3

# UI / API
pip_install_exact \
  gradio==3.41.2 gradio_client==0.5.0 fastapi==0.128.0 starlette==0.50.0 uvicorn==0.40.0 \
  httpx==0.28.1 httpcore==1.0.9 h11==0.16.0 websockets==11.0.3 orjson==3.11.5

# Rendering / geometry
pip_install_exact \
  nvdiffrast==0.4.0 xatlas==0.0.11 trimesh==4.11.0 PyMCubes==0.1.6 plyfile==1.1.3 \
  networkx==3.4.2 sympy==1.14.0

# Vision / image + video IO
pip_install_exact \
  opencv-python-headless==4.11.0.86 scikit-image==0.25.2 imageio==2.37.2 imageio-ffmpeg==0.6.0 \
  tifffile==2025.5.10 lazy_loader==0.4

# Background removal / ONNX
pip_install_exact \
  rembg==2.0.69 onnxruntime-gpu==1.23.2 PyMatting==1.1.14

# Training utils
pip_install_exact \
  pytorch-lightning==2.1.2 torchmetrics==1.8.2 lightning-utilities==0.15.2 \
  tensorboard==2.20.0 tensorboard-data-server==0.7.2 protobuf==6.33.4 absl-py==2.3.1 grpcio==1.76.0

# Data + misc
pip_install_exact \
  pandas==2.3.3 python-dateutil==2.9.0.post0 pytz==2025.2 tzdata==2025.3 \
  aiohttp==3.13.3 aiosignal==1.4.0 attrs==25.4.0 frozenlist==1.8.0 multidict==6.7.0 yarl==1.22.0 \
  anyio==4.12.1 idna==3.4 charset-normalizer==2.1.1 certifi==2022.12.7 urllib3==1.26.13 requests==2.28.1

# Bitsandbytes + einsum helper
pip_install_exact bitsandbytes==0.49.1 einops==0.8.1

###############################################################################
# Step 4: Install remaining packages from the known-good pip list (bulk pin)
###############################################################################
# This block is intentionally explicit. If any package here causes conflicts, comment it out and retry.
log "Step 4: Installing remaining pinned packages (bulk)..."
pip_install_exact \
  aiofiles==23.2.1 aiohappyeyeballs==2.6.1 altair==5.5.0 annotated-doc==0.0.4 annotated-types==0.7.0 \
  antlr4-python3-runtime==4.9.3 asttokens==3.0.1 async-timeout==5.0.1 braceexpand==0.1.7 click==8.3.1 \
  coloredlogs==15.0.1 comm==0.2.3 contourpy==1.3.2 cycler==0.12.1 debugpy==1.8.17 decorator==5.2.1 \
  exceptiongroup==1.3.1 executing==2.2.1 ffmpy==1.0.0 flatbuffers==25.12.19 fonttools==4.61.1 \
  humanfriendly==10.0 importlib_metadata==8.7.1 importlib_resources==6.5.2 ipykernel==7.1.0 ipython==8.37.0 \
  jedi==0.19.2 jinja2==3.1.6 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 jupyter_client==8.6.3 \
  jupyter_core==5.9.1 kiwisolver==1.4.9 llvmlite==0.46.0 markdown==3.10 markupsafe==2.1.5 matplotlib==3.10.8 \
  matplotlib-inline==0.2.1 mpmath==1.3.0 narwhals==2.15.0 nest-asyncio==1.6.0 numba==0.63.1 omegaconf==2.3.0 \
  platformdirs==4.5.1 prompt_toolkit==3.0.52 propcache==0.4.1 pydantic==2.12.5 pydantic_core==2.41.5 \
  pydub==0.25.1 pygments==2.19.2 pyparsing==3.3.1 python-multipart==0.0.21 pyzmq==27.1.0 referencing==0.37.0 \
  rpds-py==0.30.0 semantic-version==2.10.0 six==1.17.0 stack-data==0.6.3 tornado==6.5.2 traitlets==5.14.3 \
  triton==2.1.0 typing-inspection==0.4.2 wcwidth==0.2.14 webdataset==1.0.2 werkzeug==3.1.5 zipp==3.23.0

# NVIDIA python packages from the known-good environment (pip wheels).
# Note: these are typically pulled transitively by torch wheels; pinning them helps reproducibility.
log "Step 5: Installing NVIDIA pinned CUDA python packages..."
pip_install_exact \
  nvidia-cublas-cu12==12.8.4.1 nvidia-cuda-cupti-cu12==12.8.90 nvidia-cuda-nvrtc-cu12==12.8.93 \
  nvidia-cuda-runtime-cu12==12.8.90 nvidia-cudnn-cu12==9.10.2.21 nvidia-cufft-cu12==11.3.3.83 \
  nvidia-cufile-cu12==1.13.1.3 nvidia-curand-cu12==10.3.9.90 nvidia-cusolver-cu12==11.7.3.90 \
  nvidia-cusparse-cu12==12.5.8.93 nvidia-cusparselt-cu12==0.7.1 nvidia-nccl-cu12==2.27.5 \
  nvidia-nvjitlink-cu12==12.8.93 nvidia-nvshmem-cu12==3.3.20 nvidia-nvtx-cu12==12.8.90

###############################################################################
# Optional GitHub fallbacks (not run unless needed)
###############################################################################
warn "If you see issues with specific packages, these GitHub fallbacks can help:"
warn "  - nvdiffrast: pip install git+https://github.com/NVlabs/nvdiffrast.git (builds from source)"
warn "  - rembg:      pip install git+https://github.com/danielgatis/rembg.git@v2.0.69"

###############################################################################
# Verification
###############################################################################
log "Verifying installation..."
run_python -c "import torch; print('✓ torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())" || error "torch verification failed"
run_python -c "import xformers; print('✓ xformers:', xformers.__version__)" || warn "xformers import failed"
run_python -c "import gradio; print('✓ gradio:', gradio.__version__)" || error "gradio verification failed"
run_python -c "import onnxruntime; print('✓ onnxruntime:', onnxruntime.__version__)" || warn "onnxruntime verification failed"
run_python -c "import nvdiffrast; print('✓ nvdiffrast: OK')" || warn "nvdiffrast verification failed"
run_python -c "import xatlas; print('✓ xatlas: OK')" || warn "xatlas verification failed"
run_python -c "import rembg; print('✓ rembg:', rembg.__version__ if hasattr(rembg, '__version__') else 'OK')" || warn "rembg verification failed"

log ""
log "=========================================="
log "Installation completed!"
log "Environment: ${INSTANTMESH_ENV}"
log "To activate: conda activate ${INSTANTMESH_ENV}"
log "Repo root:   ${REPO_ROOT}"
log "=========================================="


