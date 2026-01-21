#!/usr/bin/env bash
# Script to create Garment3D conda environment and install all packages with exact versions
# Uses --no-deps, --no-build-isolation, and --break-system-packages where appropriate

set -euo pipefail

###############################################################################
# Configuration
###############################################################################
G3D_ENV="${G3D_ENV:-garment3d}"
PYTHON_VERSION="${PYTHON_VERSION:-3.8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
G3D_DIR="${G3D_DIR:-${REPO_ROOT}/Garment3d/Garment3DGen}"

###############################################################################
# Helpers
###############################################################################
log() { echo "[INFO] $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }
warn() { echo "[WARN] $*"; }

# Initialize conda
init_conda() {
  if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    local conda_base
    conda_base="$(conda info --base 2>/dev/null)"
    if [ -n "${conda_base}" ] && [ -f "${conda_base}/etc/profile.d/conda.sh" ]; then
      source "${conda_base}/etc/profile.d/conda.sh"
    fi
  fi
}

# Check if environment exists
env_exists() {
  conda env list | awk '{print $1}' | grep -Fx "${G3D_ENV}" >/dev/null 2>&1
}

# Helper function to run pip commands
run_pip() {
  conda run -n "${G3D_ENV}" pip "$@"
}

# Helper function to run python commands
run_python() {
  conda run -n "${G3D_ENV}" python "$@"
}

###############################################################################
# Main Setup
###############################################################################
init_conda

# Check if environment already exists
if env_exists; then
  log "Environment '${G3D_ENV}' already exists."
  read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    log "Removing existing environment '${G3D_ENV}'..."
    conda env remove -n "${G3D_ENV}" -y
  else
    log "Using existing environment. Will use conda run for package installation..."
  fi
fi

# Create environment if it doesn't exist
if ! env_exists; then
  log "Creating conda environment '${G3D_ENV}' with Python ${PYTHON_VERSION}..."
  conda create -n "${G3D_ENV}" python="${PYTHON_VERSION}" -y
fi

# Note: Using conda run instead of activation since we're in a subprocess
log "Using conda run to execute commands in environment '${G3D_ENV}'..."

# Upgrade pip first
log "Upgrading pip..."
run_pip install --upgrade pip==24.2 --break-system-packages

###############################################################################
# Install packages in order
###############################################################################

# Step 1: Install PyTorch stack (CUDA 12.1)
log "Step 1: Installing PyTorch stack (CUDA 12.1)..."
run_pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu121 \
  --break-system-packages || {
  error "Failed to install PyTorch"
}

# Step 2: Install build tools and core dependencies
log "Step 2: Installing build tools and core dependencies..."
run_pip install --no-deps --break-system-packages --no-build-isolation \
  setuptools==75.1.0 \
  wheel==0.44.0 \
  packaging==25.0 \
  Cython==3.2.4 \
  ninja==1.13.0 || {
  warn "Some build tools failed with --no-deps, trying with deps..."
  run_pip install --break-system-packages \
    setuptools==75.1.0 \
    wheel==0.44.0 \
    packaging==25.0 \
    Cython==3.2.4 \
    ninja==1.13.0
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

# Check g++ version - CUDA 12.1 requires g++ < 13.0 typically, but check
GPP_VERSION=$(g++ --version 2>/dev/null | head -n1 | grep -oP '\d+\.\d+' | head -n1 || echo "0")
GPP_MAJOR=$(echo "${GPP_VERSION}" | cut -d. -f1)

if [ "${GPP_MAJOR}" -ge 13 ] 2>/dev/null; then
  warn "g++ version ${GPP_VERSION} may cause issues. Attempting to install g++-12..."
  
  # Try to install g++-12
  if command -v apt-get >/dev/null 2>&1; then
    log "Installing g++-12..."
    sudo apt-get update -qq && sudo apt-get install -y g++-12 || {
      warn "Could not install g++-12. Build may fail."
    }
  fi
  
  # Set CXX to use compatible compiler
  if command -v g++-12 >/dev/null 2>&1; then
    export CXX=g++-12
    export CC=gcc-12
    log "Using g++-12 for PyTorch3D build"
  fi
fi

# Install PyTorch3D 0.7.9 (newer version for Garment3D)
CXX="${CXX:-}" CC="${CC:-${CXX/g++/gcc}}" conda run -n "${G3D_ENV}" bash -c "export CXX='${CXX:-}' CC='${CC:-}' && pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9 --break-system-packages --no-build-isolation" || {
  warn "PyTorch3D failed with --no-build-isolation, trying without..."
  CXX="${CXX:-}" CC="${CC:-${CXX/g++/gcc}}" conda run -n "${G3D_ENV}" bash -c "export CXX='${CXX:-}' CC='${CC:-}' && pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9 --break-system-packages" || {
    warn "v0.7.9 failed, trying v0.7.4..."
    CXX="${CXX:-}" CC="${CC:-${CXX/g++/gcc}}" conda run -n "${G3D_ENV}" bash -c "export CXX='${CXX:-}' CC='${CC:-}' && pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.4 --break-system-packages" || {
      error "Failed to install PyTorch3D. Please check g++ version compatibility."
    }
  }
}

# Step 5: Install torch geometric extensions (need CUDA 12.1 wheels)
log "Step 5: Installing PyTorch Geometric extensions..."
run_pip install --no-deps --break-system-packages --no-build-isolation \
  torch_cluster==1.6.3+pt22cu121 \
  torch_scatter==2.1.2+pt22cu121 \
  torch_sparse==0.6.18+pt22cu121 \
  torch_spline_conv==1.2.2+pt22cu121 \
  -f https://data.pyg.org/whl/torch-2.2.0+cu121.html || {
  warn "Some torch extensions failed with --no-deps, trying with deps..."
  run_pip install --break-system-packages \
    torch_cluster==1.6.3+pt22cu121 \
    torch_scatter==2.1.2+pt22cu121 \
    torch_sparse==0.6.18+pt22cu121 \
    torch_spline_conv==1.2.2+pt22cu121 \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
}

# Step 6: Install image processing packages
log "Step 6: Installing image processing packages..."
run_pip install --no-deps --break-system-packages --no-build-isolation \
  Pillow==10.4.0 \
  imageio==2.35.1 \
  imageio-ffmpeg==0.5.1 || {
  run_pip install --break-system-packages \
    Pillow==10.4.0 \
    imageio==2.35.1 \
    imageio-ffmpeg==0.5.1
}

# Step 7: Install 3D mesh processing
log "Step 7: Installing 3D mesh processing packages..."
run_pip install --no-deps --break-system-packages --no-build-isolation \
  trimesh==4.11.0 \
  pymeshlab==2023.12.post2 || {
  run_pip install --break-system-packages \
    trimesh==4.11.0 \
    pymeshlab==2023.12.post2
}

# Step 8: Install deep learning utilities
log "Step 8: Installing deep learning utilities..."
run_pip install --no-deps --break-system-packages --no-build-isolation \
  kornia==0.7.3 \
  kornia_rs==0.1.10 || {
  run_pip install --break-system-packages \
    kornia==0.7.3 \
    kornia_rs==0.1.10
}

# Step 9: Install ML/HuggingFace packages
log "Step 9: Installing ML and HuggingFace packages..."
run_pip install --no-deps --break-system-packages --no-build-isolation \
  transformers==4.46.3 \
  tokenizers==0.20.3 \
  datasets==3.1.0 \
  huggingface-hub==0.36.0 \
  hf-xet==1.2.0 \
  safetensors==0.5.3 || {
  run_pip install --break-system-packages \
    transformers==4.46.3 \
    tokenizers==0.20.3 \
    datasets==3.1.0 \
    huggingface-hub==0.36.0 \
    hf-xet==1.2.0 \
    safetensors==0.5.3
}

# Step 10: Install SMPL/SMPL-X related
log "Step 10: Installing SMPL/SMPL-X related packages..."
run_pip install --no-deps --break-system-packages --no-build-isolation \
  smplx==0.1.28 || {
  run_pip install --break-system-packages smplx==0.1.28
}

# Step 11: Install utilities
log "Step 11: Installing utility packages..."
run_pip install --no-deps --break-system-packages --no-build-isolation \
  termcolor==2.4.0 \
  tqdm==4.67.1 \
  matplotlib==3.7.5 \
  matplotlib-inline==0.1.7 \
  PyYAML==6.0.3 \
  easydict==1.13 \
  fire==0.7.1 \
  tensorboard==2.14.0 \
  tensorboard-data-server==0.7.2 || {
  run_pip install --break-system-packages \
    termcolor==2.4.0 \
    tqdm==4.67.1 \
    matplotlib==3.7.5 \
    matplotlib-inline==0.1.7 \
    PyYAML==6.0.3 \
    easydict==1.13 \
    fire==0.7.1 \
    tensorboard==2.14.0 \
    tensorboard-data-server==0.7.2
}

# Step 12: Install cloud/API packages
log "Step 12: Installing cloud and API packages..."
run_pip install --no-deps --break-system-packages --no-build-isolation \
  boto3==1.37.38 \
  botocore==1.37.38 \
  s3transfer==0.11.5 \
  jmespath==1.0.1 || {
  run_pip install --break-system-packages \
    boto3==1.37.38 \
    botocore==1.37.38 \
    s3transfer==0.11.5 \
    jmespath==1.0.1
}

# Step 13: Install CLIP (from git)
log "Step 13: Installing OpenAI CLIP..."
run_pip install git+https://github.com/openai/CLIP.git --break-system-packages || {
  error "Failed to install CLIP"
}

# Step 14: Install nvdiffrast (build from local if available)
log "Step 14: Installing nvdiffrast..."
if [ -d "${G3D_DIR}/packages/nvdiffrast" ]; then
  log "Building nvdiffrast from local source..."
  conda run -n "${G3D_ENV}" bash -c "cd '${G3D_DIR}/packages/nvdiffrast' && pip install . --break-system-packages" || {
    warn "Local nvdiffrast build failed, trying from PyPI..."
    run_pip install nvdiffrast==0.4.0 --break-system-packages
  }
else
  log "Installing nvdiffrast from PyPI..."
  run_pip install nvdiffrast==0.4.0 --break-system-packages || {
    warn "nvdiffrast installation failed, continuing..."
  }
fi

# Step 15: Install special packages (libigl, cholespy, etc.)
log "Step 15: Installing special packages..."
run_pip install --no-deps --break-system-packages --no-build-isolation \
  libigl==2.4.1 \
  cholespy==2.1.0 \
  pyglm==2.8.3 \
  resize-right==0.0.2 || {
  run_pip install --break-system-packages \
    libigl==2.4.1 \
    cholespy==2.1.0 \
    pyglm==2.8.3 \
    resize-right==0.0.2
}

# Step 16: Install CUDA packages
log "Step 16: Installing CUDA-related packages..."
run_pip install --no-deps --break-system-packages --no-build-isolation \
  nvidia-cublas-cu12==12.1.3.1 \
  nvidia-cuda-cupti-cu12==12.1.105 \
  nvidia-cuda-nvrtc-cu12==12.1.105 \
  nvidia-cuda-runtime-cu12==12.1.105 \
  nvidia-cudnn-cu12==8.9.2.26 \
  nvidia-cufft-cu12==11.0.2.54 \
  nvidia-curand-cu12==10.3.2.106 \
  nvidia-cusolver-cu12==11.4.5.107 \
  nvidia-cusparse-cu12==12.1.0.106 \
  nvidia-nccl-cu12==2.19.3 \
  nvidia-nvjitlink-cu12==12.9.86 \
  nvidia-nvtx-cu12==12.1.105 || {
  warn "Some CUDA packages failed, continuing..."
}

# Step 17: Install remaining dependencies
log "Step 17: Installing remaining dependencies..."
run_pip install --no-deps --break-system-packages --no-build-isolation \
  absl-py==2.3.1 \
  aiohappyeyeballs==2.4.4 \
  aiohttp==3.10.11 \
  aiosignal==1.3.1 \
  annoy==1.17.3 \
  appdirs==1.4.4 \
  asttokens==3.0.1 \
  async-timeout==5.0.1 \
  attrs==25.3.0 \
  backcall==0.2.0 \
  certifi==2026.1.4 \
  charset-normalizer==3.4.4 \
  clip==1.0 \
  contourpy==1.1.1 \
  cycler==0.12.1 \
  datasets==3.1.0 \
  decorator==5.2.1 \
  dill==0.3.8 \
  executing==2.2.1 \
  fashion-clip==0.2.2 \
  filelock==3.16.1 \
  fonttools==4.57.0 \
  frozenlist==1.5.0 \
  fsspec==2024.9.0 \
  ftfy==6.2.3 \
  google-auth==2.47.0 \
  google-auth-oauthlib==1.0.0 \
  grpcio==1.70.0 \
  idna==3.11 \
  importlib_metadata==8.5.0 \
  importlib_resources==6.4.5 \
  iopath==0.1.10 \
  ipyplot==1.1.2 \
  ipython==8.12.3 \
  jedi==0.19.2 \
  Jinja2==3.1.6 \
  kiwisolver==1.4.7 \
  Markdown==3.7 \
  MarkupSafe==2.1.5 \
  mpmath==1.3.0 \
  multidict==6.1.0 \
  multiprocess==0.70.16 \
  networkx==3.1 \
  oauthlib==3.3.1 \
  pandas==2.0.3 \
  parso==0.8.5 \
  pexpect==4.9.0 \
  pickleshare==0.7.5 \
  portalocker==3.2.0 \
  prompt_toolkit==3.0.52 \
  propcache==0.2.0 \
  protobuf==5.29.5 \
  ptyprocess==0.7.0 \
  pure_eval==0.2.3 \
  pyarrow==17.0.0 \
  pyasn1==0.6.1 \
  pyasn1_modules==0.4.2 \
  Pygments==2.19.2 \
  pyparsing==3.1.4 \
  python-dateutil==2.9.0.post0 \
  python-dotenv==1.0.1 \
  pytz==2025.2 \
  regex==2024.11.6 \
  requests==2.32.4 \
  requests-oauthlib==2.0.0 \
  rsa==4.9.1 \
  shortuuid==1.0.13 \
  six==1.17.0 \
  stack-data==0.6.3 \
  sympy==1.13.3 \
  traitlets==5.14.3 \
  typing_extensions==4.13.2 \
  tzdata==2025.3 \
  urllib3==1.26.20 \
  validators==0.34.0 \
  wcwidth==0.2.14 \
  Werkzeug==3.0.6 \
  xxhash==3.6.0 \
  yarl==1.15.2 \
  zipp==3.20.2 || {
  warn "Some packages failed with --no-deps, installing with dependency resolution..."
  run_pip install --break-system-packages \
    absl-py==2.3.1 aiohappyeyeballs==2.4.4 aiohttp==3.10.11 aiosignal==1.3.1 \
    annoy==1.17.3 appdirs==1.4.4 asttokens==3.0.1 async-timeout==5.0.1 \
    attrs==25.3.0 backcall==0.2.0 certifi==2026.1.4 charset-normalizer==3.4.4 \
    clip==1.0 contourpy==1.1.1 cycler==0.12.1 datasets==3.1.0 \
    decorator==5.2.1 dill==0.3.8 executing==2.2.1 fashion-clip==0.2.2 \
    filelock==3.16.1 fonttools==4.57.0 frozenlist==1.5.0 fsspec==2024.9.0 \
    ftfy==6.2.3 google-auth==2.47.0 google-auth-oauthlib==1.0.0 grpcio==1.70.0 \
    idna==3.11 importlib_metadata==8.5.0 importlib_resources==6.4.5 \
    iopath==0.1.10 ipyplot==1.1.2 ipython==8.12.3 jedi==0.19.2 Jinja2==3.1.6 \
    kiwisolver==1.4.7 Markdown==3.7 MarkupSafe==2.1.5 mpmath==1.3.0 \
    multidict==6.1.0 multiprocess==0.70.16 networkx==3.1 oauthlib==3.3.1 \
    pandas==2.0.3 parso==0.8.5 pexpect==4.9.0 pickleshare==0.7.5 \
    portalocker==3.2.0 prompt_toolkit==3.0.52 propcache==0.2.0 protobuf==5.29.5 \
    ptyprocess==0.7.0 pure_eval==0.2.3 pyarrow==17.0.0 pyasn1==0.6.1 \
    pyasn1_modules==0.4.2 Pygments==2.19.2 pyparsing==3.1.4 \
    python-dateutil==2.9.0.post0 python-dotenv==1.0.1 pytz==2025.2 \
    regex==2024.11.6 requests==2.32.4 requests-oauthlib==2.0.0 rsa==4.9.1 \
    shortuuid==1.0.13 six==1.17.0 stack-data==0.6.3 sympy==1.13.3 \
    traitlets==5.14.3 typing_extensions==4.13.2 tzdata==2025.3 urllib3==1.26.20 \
    validators==0.34.0 wcwidth==0.2.14 Werkzeug==3.0.6 xxhash==3.6.0 \
    yarl==1.15.2 zipp==3.20.2
}

###############################################################################
# Verification
###############################################################################
log "Verifying installation..."

run_python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')" || error "PyTorch verification failed"
run_python -c "import torchvision; print(f'✓ torchvision: {torchvision.__version__}')" || error "torchvision verification failed"
run_python -c "import pytorch3d; print(f'✓ PyTorch3D: {pytorch3d.__version__}')" || error "PyTorch3D verification failed"
run_python -c "import trimesh; print('✓ trimesh: OK')" || error "trimesh verification failed"
run_python -c "import clip; print('✓ CLIP: OK')" || error "CLIP verification failed"
run_python -c "import kornia; print('✓ kornia: OK')" || error "kornia verification failed"
run_python -c "import nvdiffrast; print('✓ nvdiffrast: OK')" || warn "nvdiffrast verification failed (may not be critical)"

log ""
log "=========================================="
log "Installation completed successfully!"
log "Environment: ${G3D_ENV}"
log "To activate: conda activate ${G3D_ENV}"
log "=========================================="


