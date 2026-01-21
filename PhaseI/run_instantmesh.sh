#!/usr/bin/env bash
# Script to run InstantMesh inference on a single image (same demo image ECON uses by default).
# Requires the InstantMesh conda environment to be set up (see: setup_instantmesh_env.sh).
#
# Usage:
#   ./run_instantmesh.sh [input_image] [output_dir] [config_yaml]
#
# Examples:
#   ./run_instantmesh.sh
#   ./run_instantmesh.sh /path/to/img.jpg ./instantmesh_outputs
#   SAVE_VIDEO=0 ./run_instantmesh.sh
#
set -euo pipefail

###############################################################################
# Configuration
###############################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IM_DIR="${IM_DIR:-${REPO_ROOT}/InstantMesh/InstantMesh}"
IM_ENV="${INSTANTMESH_ENV:-InstantMesh}"

# Default demo image from ECON examples (same heuristic as Pipeline/run_econ_and_instantmesh.sh).
ECON_DIR="${ECON_DIR:-${REPO_ROOT}/econ/ECON}"
DEFAULT_INPUT_1="${ECON_DIR}/examples/304e9c4798a8c3967de7c74c24ef2e38.jpg"
DEFAULT_INPUT_2="${ECON_DIR}/examples/demo.jpg"

INPUT_IMAGE="${1:-${DEFAULT_INPUT_2}}"
if [ ! -f "${INPUT_IMAGE}" ] && [ -f "${DEFAULT_INPUT_1}" ]; then
  INPUT_IMAGE="${DEFAULT_INPUT_1}"
fi

OUTPUT_DIR="${2:-${SCRIPT_DIR}/instantmesh_outputs}"
CONFIG_YAML="${3:-${IM_DIR}/configs/instant-mesh-large.yaml}"

# Options
SAVE_VIDEO="${SAVE_VIDEO:-1}" # set to 0 to skip video

###############################################################################
# Helpers
###############################################################################
log() { echo "[INFO] $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }
warn() { echo "[WARN] $*"; }

init_conda() {
  if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    local conda_base
    if command -v mamba >/dev/null 2>&1; then
      conda_base="$(mamba info --base 2>/dev/null || conda info --base)"
    else
      conda_base="$(conda info --base)"
    fi
    if [ -n "${conda_base}" ] && [ -f "${conda_base}/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1090
      source "${conda_base}/etc/profile.d/conda.sh"
    fi
  fi
}

###############################################################################
# Validation
###############################################################################
if ! command -v conda >/dev/null 2>&1; then
  error "conda not found in PATH"
fi

if [ ! -d "${IM_DIR}" ]; then
  error "InstantMesh directory not found: ${IM_DIR}"
fi

if [ ! -f "${INPUT_IMAGE}" ]; then
  error "Input image not found: ${INPUT_IMAGE}"
fi

if [ ! -f "${CONFIG_YAML}" ]; then
  error "Config YAML not found: ${CONFIG_YAML}"
fi

init_conda
if ! conda env list | awk '{print $1}' | grep -Fx "${IM_ENV}" >/dev/null 2>&1; then
  error "Conda environment '${IM_ENV}' not found. Please run ./setup_instantmesh_env.sh first."
fi

###############################################################################
# Main execution
###############################################################################
log "Starting InstantMesh inference"
log "  Input image:     ${INPUT_IMAGE}"
log "  Output directory:${OUTPUT_DIR}"
log "  Config:          ${CONFIG_YAML}"
log "  Save video:      ${SAVE_VIDEO}"

mkdir -p "${OUTPUT_DIR}"

IM_SAVE_VIDEO_FLAG=()
if [ "${SAVE_VIDEO}" = "1" ]; then
  IM_SAVE_VIDEO_FLAG+=(--save_video)
fi

(
  cd "${IM_DIR}" || exit 1
  init_conda
  conda activate "${IM_ENV}"

  log "Running: python ${IM_DIR}/run.py ${CONFIG_YAML} ${INPUT_IMAGE} --output_path ${OUTPUT_DIR} ${IM_SAVE_VIDEO_FLAG[*]:-}"
  python "${IM_DIR}/run.py" \
    "${CONFIG_YAML}" \
    "${INPUT_IMAGE}" \
    --output_path "${OUTPUT_DIR}" \
    "${IM_SAVE_VIDEO_FLAG[@]}"
)

log "InstantMesh inference completed!"
log "Results saved to: ${OUTPUT_DIR}"

if [ -d "${OUTPUT_DIR}" ]; then
  log "Sample output files:"
  find "${OUTPUT_DIR}" -type f \( -name "*.obj" -o -name "*.glb" -o -name "*.mp4" -o -name "*.png" -o -name "*.jpg" \) | head -30
fi


