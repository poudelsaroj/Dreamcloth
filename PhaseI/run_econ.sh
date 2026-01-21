#!/usr/bin/env bash
# Script to run ECON paper inference
# Requires the econ conda environment to be set up

set -euo pipefail

###############################################################################
# Configuration
###############################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ECON_DIR="${ECON_DIR:-${REPO_ROOT}/econ/ECON}"
ECON_ENV="${ECON_ENV:-econ_script}"

# Default paths
INPUT_DIR="${1:-${ECON_DIR}/examples}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/econ_outputs}"
CONFIG_FILE="${3:-${ECON_DIR}/configs/econ.yaml}"

# Options
GPU_DEVICE="${GPU_DEVICE:-0}"
MULTI_PERSON="${MULTI_PERSON:-false}"  # set to "true" for multi-person
NO_VIS="${NO_VIS:-false}"              # set to "true" to skip visualization
LOOP_SMPL="${LOOP_SMPL:-50}"
PATIENCE="${PATIENCE:-5}"

###############################################################################
# Helpers
###############################################################################
log() { echo "[INFO] $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }
warn() { echo "[WARN] $*"; }

# Initialize conda/mamba
init_conda() {
  if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    local conda_base
    if command -v mamba >/dev/null 2>&1; then
      conda_base="$(mamba info --base 2>/dev/null || conda info --base)"
    else
      conda_base="$(conda info --base)"
    fi
    if [ -n "${conda_base}" ] && [ -f "${conda_base}/etc/profile.d/conda.sh" ]; then
      source "${conda_base}/etc/profile.d/conda.sh"
    fi
  fi
}

###############################################################################
# Validation
###############################################################################
if [ ! -d "${ECON_DIR}" ]; then
  error "ECON directory not found: ${ECON_DIR}"
fi

if [ ! -d "${INPUT_DIR}" ]; then
  error "Input directory not found: ${INPUT_DIR}"
fi

if [ ! -f "${CONFIG_FILE}" ]; then
  error "Config file not found: ${CONFIG_FILE}"
fi

# Check if environment exists
init_conda
if ! conda env list | grep -q "^${ECON_ENV} "; then
  error "Conda environment '${ECON_ENV}' not found. Please run ./setup_econ_env.sh first."
fi

# Check for required model files
if [ ! -f "${ECON_DIR}/data/body_models/smplx/SMPLX_NEUTRAL.pkl" ]; then
  warn "SMPL-X model files may be missing. Run fetch_data.sh if needed."
fi

###############################################################################
# Main execution
###############################################################################
log "Starting ECON inference"
log "  Input directory: ${INPUT_DIR}"
log "  Output directory: ${OUTPUT_DIR}"
log "  Config file: ${CONFIG_FILE}"
log "  GPU device: ${GPU_DEVICE}"
log "  Multi-person: ${MULTI_PERSON}"
log "  No visualization: ${NO_VIS}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Build command arguments
ARGS=(
  -gpu "${GPU_DEVICE}"
  -loop_smpl "${LOOP_SMPL}"
  -patience "${PATIENCE}"
  -in_dir "${INPUT_DIR}"
  -out_dir "${OUTPUT_DIR}"
  -cfg "${CONFIG_FILE}"
)

if [ "${MULTI_PERSON}" = "true" ]; then
  ARGS+=(-multi)
fi

if [ "${NO_VIS}" = "true" ]; then
  ARGS+=(-novis)
fi

# Run ECON inference
(
  cd "${ECON_DIR}" || exit 1
  init_conda
  mamba activate "${ECON_ENV}"
  
  log "Running: python -m apps.infer ${ARGS[*]}"
  python -m apps.infer "${ARGS[@]}"
)

log "ECON inference completed!"
log "Results saved to: ${OUTPUT_DIR}"

# List output files
if [ -d "${OUTPUT_DIR}" ]; then
  log "Output files:"
  find "${OUTPUT_DIR}" -type f \( -name "*.obj" -o -name "*.png" -o -name "*.jpg" \) | head -20
fi

