#!/usr/bin/env bash
# Script to create ECON conda environment and install all packages with exact versions
# Uses --no-deps, --no-build-isolation, and --break-system-packages where appropriate

set -euo pipefail

###############################################################################
# Configuration
###############################################################################
ECON_ENV="${ECON_ENV:-econ_script}"
PYTHON_VERSION="${PYTHON_VERSION:-3.8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ECON_DIR="${ECON_DIR:-${REPO_ROOT}/econ/ECON}"

###############################################################################
# Helpers
###############################################################################
log() { echo "[INFO] $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }
warn() { echo "[WARN] $*"; }

# Initialize conda/mamba
init_conda() {
  if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    if command -v mamba >/dev/null 2>&1; then
      # Initialize mamba using shell hook
      if [ -z "${MAMBA_ROOT_PREFIX:-}" ]; then
        eval "$(mamba shell hook --shell bash 2>/dev/null)" || {
          # Fallback to conda.sh if mamba hook fails
          local conda_base
          conda_base="$(mamba info --base 2>/dev/null || conda info --base 2>/dev/null)"
          if [ -n "${conda_base}" ] && [ -f "${conda_base}/etc/profile.d/conda.sh" ]; then
            source "${conda_base}/etc/profile.d/conda.sh"
          fi
        }
      fi
    else
      # Fallback to conda
      local conda_base
      conda_base="$(conda info --base 2>/dev/null)"
      if [ -n "${conda_base}" ] && [ -f "${conda_base}/etc/profile.d/conda.sh" ]; then
        source "${conda_base}/etc/profile.d/conda.sh"
      fi
    fi
  fi
}

# Check if environment exists
env_exists() {
  conda env list | awk '{print $1}' | grep -Fx "${ECON_ENV}" >/dev/null 2>&1
}

###############################################################################
# Main Setup
###############################################################################
init_conda

# Check if environment already exists
if env_exists; then
  log "Environment '${ECON_ENV}' already exists."
  read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    log "Removing existing environment '${ECON_ENV}'..."
    conda env remove -n "${ECON_ENV}" -y
  else
    log "Using existing environment. Will use mamba run for package installation..."
  fi
fi

# Create environment if it doesn't exist
if ! env_exists; then
  log "Creating conda environment '${ECON_ENV}' with Python ${PYTHON_VERSION}..."
  
  # Try to use environment.yaml if it exists
  if [ -f "${ECON_DIR}/environment.yaml" ]; then
    log "Using environment.yaml from ECON directory..."
    mamba env create -n "${ECON_ENV}" -f "${ECON_DIR}/environment.yaml"
  else
    log "Creating basic environment..."
    mamba create -n "${ECON_ENV}" python="${PYTHON_VERSION}" -y
  fi
fi

# Note: Using mamba run instead of activation since we're in a subprocess
log "Using mamba run to execute commands in environment '${ECON_ENV}'..."

# Helper function to run pip commands
run_pip() {
  mamba run -n "${ECON_ENV}" pip "$@"
}

# Helper function to run python commands
run_python() {
  mamba run -n "${ECON_ENV}" python "$@"
}

# Upgrade pip first
log "Upgrading pip..."
run_pip install --upgrade pip==24.3.1 --break-system-packages

###############################################################################
# Install packages in order
###############################################################################

# Step 1: Install PyTorch stack (CUDA 11.8)
log "Step 1: Installing PyTorch stack (CUDA 11.8)..."
run_pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  --break-system-packages || {
  error "Failed to install PyTorch"
}

# Step 2: Install build tools and core dependencies
log "Step 2: Installing build tools and core dependencies..."
run_pip install --no-deps --break-system-packages --no-build-isolation \
  setuptools==75.3.0 \
  wheel==0.45.1 \
  packaging==25.0 \
  Cython==3.0.11 \
  cmake==4.2.1 \
  ninja==1.11.1.1 || {
  warn "Some build tools failed with --no-deps, trying with deps..."
  run_pip install --break-system-packages \
    setuptools==75.3.0 \
    wheel==0.45.1 \
    packaging==25.0 \
    Cython==3.0.11 \
    cmake==4.2.1
}

# Step 3: Install NumPy and SciPy (foundation packages)
log "Step 3: Installing NumPy and SciPy..."
run_pip install --no-deps --break-system-packages --no-build-isolation \
  numpy==1.24.4 \
  scipy==1.10.1 || {
  run_pip install --break-system-packages numpy==1.24.4 scipy==1.10.1
}

# Step 4: Install PyTorch3D from git (takes ~20 minutes to build)
log "Step 4: Installing PyTorch3D from git (this will take ~20 minutes)..."
log "PyTorch3D is not available on PyPI, installing from source..."

# Check g++ version - CUDA 11.6 requires g++ < 12.0
GPP_VERSION=$(g++ --version 2>/dev/null | head -n1 | grep -oP '\d+\.\d+' | head -n1 || echo "0")
GPP_MAJOR=$(echo "${GPP_VERSION}" | cut -d. -f1)

if [ "${GPP_MAJOR}" -ge 12 ] 2>/dev/null; then
  warn "g++ version ${GPP_VERSION} is too new for CUDA 11.6 (requires < 12.0)"
  log "Attempting to install g++-11 or g++-10..."
  
  # Try to install g++-11 first
  if command -v apt-get >/dev/null 2>&1; then
    log "Installing g++-11..."
    sudo apt-get update -qq && sudo apt-get install -y g++-11 || {
      log "Trying g++-10..."
      sudo apt-get install -y g++-10 || warn "Could not install compatible g++. Build may fail."
    }
  fi
  
  # Set CXX to use compatible compiler
  if command -v g++-11 >/dev/null 2>&1; then
    export CXX=g++-11
    export CC=gcc-11
    log "Using g++-11 for PyTorch3D build"
  elif command -v g++-10 >/dev/null 2>&1; then
    export CXX=g++-10
    export CC=gcc-10
    log "Using g++-10 for PyTorch3D build"
  else
    warn "No compatible g++ found. PyTorch3D build may fail."
  fi
fi

# Try v0.7.4 tag first, then v0.7.2
run_pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.4 --break-system-packages --no-build-isolation 2>/dev/null || {
  warn "v0.7.4 tag not found, trying v0.7.2..."
  # Ensure CXX is exported for mamba run
  if [ -n "${CXX:-}" ]; then
    CXX="${CXX}" CC="${CC:-${CXX/g++/gcc}}" mamba run -n "${ECON_ENV}" bash -c "export CXX='${CXX}' CC='${CC:-${CXX/g++/gcc}}' && pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2 --break-system-packages --no-build-isolation" || {
      warn "Failed with --no-build-isolation, trying without..."
      CXX="${CXX}" CC="${CC:-${CXX/g++/gcc}}" mamba run -n "${ECON_ENV}" bash -c "export CXX='${CXX}' CC='${CC:-${CXX/g++/gcc}}' && pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2 --break-system-packages" || {
        error "Failed to install PyTorch3D. CUDA 11.6 requires g++ < 12.0. Please install g++-11 or g++-10 and try again."
      }
    }
  else
    run_pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2 --break-system-packages --no-build-isolation 2>/dev/null || {
      warn "Failed with --no-build-isolation, trying without..."
      run_pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2 --break-system-packages || {
        error "Failed to install PyTorch3D. CUDA 11.6 requires g++ < 12.0. Please install g++-11 or g++-10 and try again."
      }
    }
  fi
}

# Step 5: Install image processing packages
log "Step 5: Installing image processing packages..."
run_pip install --no-deps --break-system-packages --no-build-isolation \
  Pillow==9.4.0 \
  opencv-python==4.12.0.88 \
  opencv-contrib-python==4.12.0.88 \
  opencv-python-headless==4.12.0.88 \
  imageio==2.35.1 \
  scikit-image==0.21.0 || {
  run_pip install --break-system-packages \
    Pillow==9.4.0 \
    opencv-python==4.12.0.88 \
    opencv-contrib-python==4.12.0.88 \
    opencv-python-headless==4.12.0.88 \
    imageio==2.35.1 \
    scikit-image==0.21.0
}

# Step 6: Install 3D mesh processing
log "Step 6: Installing 3D mesh processing packages..."
run_pip install --no-deps --break-system-packages --no-build-isolation \
  trimesh==4.10.1 \
  open3d==0.19.0 \
  xatlas==0.0.11 \
  fast_simplification==0.1.7 \
  pyembree==0.1.6 \
  Rtree==1.3.0 || {
  run_pip install --break-system-packages \
    trimesh==4.10.1 \
    open3d==0.19.0 \
    xatlas==0.0.11 \
    fast_simplification==0.1.7 \
    pyembree==0.1.6 \
    Rtree==1.3.0
}

# Step 7: Install deep learning utilities
log "Step 7: Installing deep learning utilities..."
run_pip install --no-deps --break-system-packages --no-build-isolation \
  pytorch-lightning==2.4.0 \
  kornia==0.7.3 \
  kornia_rs==0.1.10 \
  einops==0.8.1 \
  torchmetrics==1.5.2 \
  lightning-utilities==0.11.9 || {
  run_pip install --break-system-packages \
    pytorch-lightning==2.4.0 \
    kornia==0.7.3 \
    kornia_rs==0.1.10 \
    einops==0.8.1 \
    torchmetrics==1.5.2 \
    lightning-utilities==0.11.9
}

# Step 8: Install ML/scientific packages
log "Step 8: Installing ML and scientific packages..."
run_pip install --no-deps --break-system-packages --no-build-isolation \
  scikit-learn==1.3.2 \
  joblib==1.4.2 \
  threadpoolctl==3.5.0 \
  onnxruntime==1.13.1 || {
  run_pip install --break-system-packages \
    scikit-learn==1.3.2 \
    joblib==1.4.2 \
    threadpoolctl==3.5.0 \
    onnxruntime==1.13.1
}

# Step 9: Install SMPL/SMPL-X related
log "Step 9: Installing SMPL/SMPL-X related packages..."
run_pip install --no-deps --break-system-packages --no-build-isolation \
  chumpy==0.70 || {
  run_pip install --break-system-packages chumpy==0.70
}

# Step 10: Install media processing
log "Step 10: Installing media processing packages..."
run_pip install --no-deps --break-system-packages --no-build-isolation \
  mediapipe==0.10.10 \
  protobuf==3.20.3 || {
  run_pip install --break-system-packages mediapipe==0.10.10 protobuf==3.20.3
}

# Step 11: Install utilities
log "Step 11: Installing utility packages..."
run_pip install --no-deps --break-system-packages --no-build-isolation \
  termcolor==2.4.0 \
  tqdm==4.67.1 \
  matplotlib==3.7.5 \
  matplotlib-inline==0.1.7 \
  dataclasses==0.8 \
  PyYAML==6.0.2 \
  yacs==0.1.8 \
  omegaconf==2.3.0 \
  fvcore==0.1.5.post20221221 \
  iopath==0.1.10 || {
  run_pip install --break-system-packages \
    termcolor==2.4.0 \
    tqdm==4.67.1 \
    matplotlib==3.7.5 \
    matplotlib-inline==0.1.7 \
    dataclasses==0.8 \
    PyYAML==6.0.2 \
    yacs==0.1.8 \
    omegaconf==2.3.0 \
    fvcore==0.1.5.post20221221 \
    iopath==0.1.10
}

# Step 12: Install cloud/API packages
log "Step 12: Installing cloud and API packages..."
run_pip install --no-deps --break-system-packages --no-build-isolation \
  boto3==1.37.38 \
  botocore==1.37.38 \
  s3transfer==0.11.5 \
  jmespath==1.0.1 \
  huggingface-hub==0.36.0 \
  hf-xet==1.2.0 || {
  run_pip install --break-system-packages \
    boto3==1.37.38 \
    botocore==1.37.38 \
    s3transfer==0.11.5 \
    jmespath==1.0.1 \
    huggingface-hub==0.36.0 \
    hf-xet==1.2.0
}

# Step 13: Install rembg (custom fork from git)
log "Step 13: Installing rembg (custom fork)..."
run_pip install git+https://github.com/YuliangXiu/rembg.git@5f4386b --break-system-packages || {
  error "Failed to install rembg"
}

# Step 14: Install CUDA packages (if needed)
log "Step 14: Installing CUDA-related packages..."
run_pip install --no-deps --break-system-packages --no-build-isolation \
  cupy==12.3.0 \
  nvidia-cublas-cu12==12.1.3.1 \
  nvidia-cuda-cupti-cu12==12.1.105 \
  nvidia-cuda-nvrtc-cu12==12.1.105 \
  nvidia-cuda-runtime-cu12==12.1.105 \
  nvidia-cudnn-cu12==9.1.0.70 \
  nvidia-cufft-cu12==11.0.2.54 \
  nvidia-curand-cu12==10.3.2.106 \
  nvidia-cusolver-cu12==11.4.5.107 \
  nvidia-cusparse-cu12==12.1.0.106 \
  nvidia-nccl-cu12==2.20.5 \
  nvidia-nvjitlink-cu12==12.9.86 \
  nvidia-nvtx-cu12==12.1.105 || {
  warn "Some CUDA packages failed, continuing..."
}

# Step 15: Install remaining dependencies
log "Step 15: Installing remaining dependencies..."
run_pip install --no-deps --break-system-packages --no-build-isolation \
  absl-py==2.3.1 \
  addict==2.4.0 \
  aiohttp==3.8.1 \
  aiosignal==1.3.1 \
  anyio==3.7.1 \
  appdirs==1.4.4 \
  asttokens==3.0.0 \
  async-timeout==4.0.3 \
  asyncer==0.0.2 \
  attrs==25.3.0 \
  backcall==0.2.0 \
  blinker==1.8.2 \
  Brotli==1.1.0 \
  certifi==2024.8.30 \
  cffi==1.17.0 \
  charset-normalizer==2.1.1 \
  click==8.1.3 \
  colorama==0.4.6 \
  coloredlogs==15.0.1 \
  comm==0.2.2 \
  ConfigArgParse==1.7.1 \
  contourpy==1.1.1 \
  cycler==0.12.1 \
  dash==3.3.0 \
  debugpy==1.8.5 \
  decorator==5.1.1 \
  exceptiongroup==1.3.1 \
  executing==2.1.0 \
  fastapi==0.87.0 \
  fastjsonschema==2.21.2 \
  fastrlock==0.8.2 \
  filelock==3.16.1 \
  filetype==1.2.0 \
  Flask==3.0.3 \
  flatbuffers==25.12.19 \
  fonttools==4.57.0 \
  frozenlist==1.5.0 \
  fsspec==2025.3.0 \
  h11==0.16.0 \
  h2==4.1.0 \
  hpack==4.0.0 \
  humanfriendly==10.0 \
  hyperframe==6.0.1 \
  idna==3.10 \
  ImageHash==4.3.1 \
  importlib_metadata==8.5.0 \
  importlib_resources==6.4.5 \
  ipykernel==6.29.5 \
  ipython==8.12.2 \
  ipywidgets==8.1.8 \
  itsdangerous==2.2.0 \
  jax==0.4.13 \
  jedi==0.19.1 \
  Jinja2==3.1.6 \
  jsonschema==4.23.0 \
  jsonschema-specifications==2023.12.1 \
  jupyter_client==8.6.3 \
  jupyter_core==5.8.1 \
  jupyterlab_widgets==3.0.16 \
  kiwisolver==1.4.7 \
  lazy_loader==0.4 \
  lit==18.1.8 \
  llvmlite==0.41.1 \
  MarkupSafe==2.1.5 \
  ml-dtypes==0.2.0 \
  mpmath==1.3.0 \
  multidict==6.1.0 \
  narwhals==1.42.1 \
  nbformat==5.10.4 \
  nest_asyncio==1.6.0 \
  networkx==3.1 \
  numba==0.58.1 \
  opt_einsum==3.4.0 \
  pandas==2.0.3 \
  parso==0.8.4 \
  pexpect==4.9.0 \
  pickleshare==0.7.5 \
  pkgutil_resolve_name==1.3.10 \
  platformdirs==4.3.6 \
  plotly==6.5.0 \
  pooch==1.6.0 \
  portalocker==2.10.1 \
  prompt_toolkit==3.0.48 \
  propcache==0.2.0 \
  psutil==6.0.0 \
  ptyprocess==0.7.0 \
  pure_eval==0.2.3 \
  pycparser==2.22 \
  pydantic==1.10.26 \
  Pygments==2.18.0 \
  PyMatting==1.1.8 \
  pyparsing==3.1.4 \
  pyquaternion==0.9.9 \
  PySocks==1.7.1 \
  python-dateutil==2.9.0 \
  python-multipart==0.0.5 \
  pytz==2025.2 \
  PyWavelets==1.4.1 \
  pyzmq==26.2.0 \
  referencing==0.35.1 \
  requests==2.32.3 \
  retrying==1.4.2 \
  rpds-py==0.20.1 \
  six==1.16.0 \
  sniffio==1.3.1 \
  sounddevice==0.5.3 \
  stack-data==0.6.2 \
  starlette==0.21.0 \
  sympy==1.13.3 \
  tabulate==0.9.0 \
  tifffile==2023.7.10 \
  triton==2.0.0 \
  typing_extensions==4.12.2 \
  tzdata==2025.3 \
  urllib3==1.26.20 \
  uvicorn==0.20.0 \
  watchdog==2.1.9 \
  wcwidth==0.2.13 \
  Werkzeug==3.0.6 \
  widgetsnbextension==4.0.15 \
  yarl==1.15.2 \
  zipp==3.21.0 \
  zstandard==0.19.0 || {
  warn "Some packages failed with --no-deps, installing with dependency resolution..."
  run_pip install --break-system-packages \
    absl-py==2.3.1 addict==2.4.0 aiohttp==3.8.1 aiosignal==1.3.1 anyio==3.7.1 \
    appdirs==1.4.4 asttokens==3.0.0 async-timeout==4.0.3 asyncer==0.0.2 \
    attrs==25.3.0 backcall==0.2.0 blinker==1.8.2 Brotli==1.1.0 \
    certifi==2024.8.30 cffi==1.17.0 charset-normalizer==2.1.1 click==8.1.3 \
    colorama==0.4.6 coloredlogs==15.0.1 comm==0.2.2 ConfigArgParse==1.7.1 \
    contourpy==1.1.1 cycler==0.12.1 dash==3.3.0 debugpy==1.8.5 \
    decorator==5.1.1 exceptiongroup==1.3.1 executing==2.1.0 fastapi==0.87.0 \
    fastjsonschema==2.21.2 fastrlock==0.8.2 filelock==3.16.1 filetype==1.2.0 \
    Flask==3.0.3 flatbuffers==25.12.19 fonttools==4.57.0 frozenlist==1.5.0 \
    fsspec==2025.3.0 h11==0.16.0 h2==4.1.0 hpack==4.0.0 humanfriendly==10.0 \
    hyperframe==6.0.1 idna==3.10 ImageHash==4.3.1 importlib_metadata==8.5.0 \
    importlib_resources==6.4.5 ipykernel==6.29.5 ipython==8.12.2 \
    ipywidgets==8.1.8 itsdangerous==2.2.0 jax==0.4.13 jedi==0.19.1 \
    Jinja2==3.1.6 jsonschema==4.23.0 jsonschema-specifications==2023.12.1 \
    jupyter_client==8.6.3 jupyter_core==5.8.1 jupyterlab_widgets==3.0.16 \
    kiwisolver==1.4.7 lazy_loader==0.4 lit==18.1.8 llvmlite==0.41.1 \
    MarkupSafe==2.1.5 ml-dtypes==0.2.0 mpmath==1.3.0 multidict==6.1.0 \
    narwhals==1.42.1 nbformat==5.10.4 nest_asyncio==1.6.0 networkx==3.1 \
    numba==0.58.1 opt_einsum==3.4.0 pandas==2.0.3 parso==0.8.4 \
    pexpect==4.9.0 pickleshare==0.7.5 pkgutil_resolve_name==1.3.10 \
    platformdirs==4.3.6 plotly==6.5.0 pooch==1.6.0 portalocker==2.10.1 \
    prompt_toolkit==3.0.48 propcache==0.2.0 psutil==6.0.0 ptyprocess==0.7.0 \
    pure_eval==0.2.3 pycparser==2.22 pydantic==1.10.26 Pygments==2.18.0 \
    PyMatting==1.1.8 pyparsing==3.1.4 pyquaternion==0.9.9 PySocks==1.7.1 \
    python-dateutil==2.9.0 python-multipart==0.0.5 pytz==2025.2 \
    PyWavelets==1.4.1 pyzmq==26.2.0 referencing==0.35.1 requests==2.32.3 \
    retrying==1.4.2 rpds-py==0.20.1 six==1.16.0 sniffio==1.3.1 \
    sounddevice==0.5.3 stack-data==0.6.2 starlette==0.21.0 sympy==1.13.3 \
    tabulate==0.9.0 tifffile==2023.7.10 triton==2.0.0 typing_extensions==4.12.2 \
    tzdata==2025.3 urllib3==1.26.20 uvicorn==0.20.0 watchdog==2.1.9 \
    wcwidth==0.2.13 Werkzeug==3.0.6 widgetsnbextension==4.0.15 yarl==1.15.2 \
    zipp==3.21.0 zstandard==0.19.0
}

###############################################################################
# Verification
###############################################################################
log "Verifying installation..."

run_python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')" || error "PyTorch verification failed"
run_python -c "import torchvision; print(f'✓ torchvision: {torchvision.__version__}')" || error "torchvision verification failed"
run_python -c "import pytorch3d; print(f'✓ PyTorch3D: {pytorch3d.__version__}')" || error "PyTorch3D verification failed"
run_python -c "import trimesh; print('✓ trimesh: OK')" || error "trimesh verification failed"
run_python -c "import open3d; print('✓ open3d: OK')" || error "open3d verification failed"
run_python -c "import kornia; print('✓ kornia: OK')" || error "kornia verification failed"
run_python -c "import rembg; print('✓ rembg: OK')" || error "rembg verification failed"

log ""
log "=========================================="
log "Installation completed successfully!"
log "Environment: ${ECON_ENV}"
log "To activate: mamba activate ${ECON_ENV}"
log "=========================================="

