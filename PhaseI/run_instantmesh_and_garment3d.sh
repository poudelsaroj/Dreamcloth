#!/usr/bin/env bash
# Run InstantMesh -> Garment3DGen on a single image.
# - Uses the same default demo image ECON uses (econ/ECON/examples/demo.jpg) if no input is provided.
# - Runs InstantMesh first, then uses its generated OBJ as Garment3DGen --target_mesh (mesh_target).
#
# Usage:
#   ./run_instantmesh_and_garment3d.sh [input_image] [output_root]
#
# Examples:
#   ./run_instantmesh_and_garment3d.sh
#   ./run_instantmesh_and_garment3d.sh /path/to/img.jpg ./outputs_im_g3d
#   SAVE_VIDEO=0 ./run_instantmesh_and_garment3d.sh
#
# Env vars:
#   INSTANTMESH_ENV (default: InstantMesh)
#   GARMENT3D_ENV   (default: garment3d)
#   IM_DIR          (default: ../InstantMesh/InstantMesh)
#   G3D_DIR         (default: ../Garment3d/Garment3DGen)
#   ECON_DIR        (for default image lookup; default: ../econ/ECON)
#   G3D_SOURCE_MESH (default: ${G3D_DIR}/meshes/tshirt.obj)
#   SAVE_VIDEO=1|0  (default: 1)
#
set -euo pipefail
trap 'echo "[ERROR] Failed at ${BASH_SOURCE}:${LINENO}" >&2' ERR

###############################################################################
# Configuration
###############################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IM_DIR="${IM_DIR:-${REPO_ROOT}/InstantMesh/InstantMesh}"
G3D_DIR="${G3D_DIR:-${REPO_ROOT}/Garment3d/Garment3DGen}"
ECON_DIR="${ECON_DIR:-${REPO_ROOT}/econ/ECON}"

IM_ENV="${INSTANTMESH_ENV:-InstantMesh}"
G3D_ENV="${GARMENT3D_ENV:-garment3d}"

SAVE_VIDEO="${SAVE_VIDEO:-1}"

DEFAULT_INPUT_1="${ECON_DIR}/examples/304e9c4798a8c3967de7c74c24ef2e38.jpg"
DEFAULT_INPUT_2="${ECON_DIR}/examples/demo.jpg"
INPUT_IMAGE="${1:-${DEFAULT_INPUT_2}}"
if [ ! -f "${INPUT_IMAGE}" ] && [ -f "${DEFAULT_INPUT_1}" ]; then
  INPUT_IMAGE="${DEFAULT_INPUT_1}"
fi

OUTPUT_ROOT="${2:-${SCRIPT_DIR}/im_g3d_outputs}"
IM_OUT="${OUTPUT_ROOT}/instantmesh"
G3D_OUT="${OUTPUT_ROOT}/garment3d"

###############################################################################
# Helpers
###############################################################################
log() { echo "[INFO] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }
have_env() { conda env list | awk '{print $1}' | grep -Fx "$1" >/dev/null 2>&1; }

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

find_instantmesh_obj() {
  # InstantMesh writes: <output_path>/<config_name>/meshes/<name>.obj
  local base_name
  base_name="$(basename "${INPUT_IMAGE}")"
  base_name="${base_name%.*}"

  local expected="${IM_OUT}/instant-mesh-large/meshes/${base_name}.obj"
  if [ -f "${expected}" ]; then
    echo "${expected}"
    return 0
  fi

  local found
  found="$(find "${IM_OUT}" -type f -name "${base_name}.obj" 2>/dev/null | head -n 1 || true)"
  if [ -n "${found}" ] && [ -f "${found}" ]; then
    echo "${found}"
    return 0
  fi

  die "Could not find InstantMesh output .obj for '${base_name}'. Looked under: ${IM_OUT}"
}

###############################################################################
# Validation
###############################################################################
command -v conda >/dev/null 2>&1 || die "conda not found in PATH"
init_conda

[ -d "${IM_DIR}" ] || die "IM_DIR not found: ${IM_DIR}"
[ -d "${G3D_DIR}" ] || die "G3D_DIR not found: ${G3D_DIR}"
[ -f "${INPUT_IMAGE}" ] || die "Input image not found: ${INPUT_IMAGE}"

have_env "${IM_ENV}" || die "InstantMesh env '${IM_ENV}' not found. Run ./setup_instantmesh_env.sh first."
have_env "${G3D_ENV}" || die "Garment3D env '${G3D_ENV}' not found. Run ./setup_garment3d_env.sh first."

G3D_SOURCE_MESH="${G3D_SOURCE_MESH:-${G3D_DIR}/meshes/tshirt.obj}"
[ -f "${G3D_SOURCE_MESH}" ] || die "Garment3D source mesh not found: ${G3D_SOURCE_MESH}"

###############################################################################
# Run InstantMesh
###############################################################################
log "[1/2] Running InstantMesh on: ${INPUT_IMAGE}"
mkdir -p "${IM_OUT}" "${G3D_OUT}"

IM_SAVE_VIDEO_FLAG=()
if [ "${SAVE_VIDEO}" = "1" ]; then
  IM_SAVE_VIDEO_FLAG+=(--save_video)
fi

(
  cd "${IM_DIR}" || exit 1
  init_conda
  conda activate "${IM_ENV}"
  python "${IM_DIR}/run.py" \
    "${IM_DIR}/configs/instant-mesh-large.yaml" \
    "${INPUT_IMAGE}" \
    --output_path "${IM_OUT}" \
    "${IM_SAVE_VIDEO_FLAG[@]}"
)

IM_OBJ="$(find_instantmesh_obj)"
log "InstantMesh target mesh: ${IM_OBJ}"

###############################################################################
# Run Garment3DGen (mesh_target)
###############################################################################
log "[2/2] Running Garment3DGen with InstantMesh mesh as --target_mesh"

(
  init_conda
  conda activate "${G3D_ENV}"
  python "${G3D_DIR}/main.py" \
    --config "${G3D_DIR}/example_config.yml" \
    --output_path "${G3D_OUT}" \
    --mesh "${G3D_SOURCE_MESH}" \
    --target_mesh "${IM_OBJ}"
)

log "Done."
log "InstantMesh outputs -> ${IM_OUT}"
log "Garment3DGen out    -> ${G3D_OUT}"


