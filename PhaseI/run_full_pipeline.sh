#!/usr/bin/env bash
# End-to-end IndividualRun pipeline:
# 1) setup ECON env
# 2) setup InstantMesh env
# 3) setup Garment3DGen env
# 4) run ECON on one image
# 5) run InstantMesh on the same image
# 6) run Garment3DGen using InstantMesh output OBJ as --target_mesh
# 7) collect ECON + Garment3D outputs into one directory
#
# Usage:
#   ./run_full_pipeline.sh [input_image] [output_root]
#
# Examples:
#   ./run_full_pipeline.sh
#   ./run_full_pipeline.sh /path/to/img.jpg ./all_outputs
#   SKIP_ENV=1 SAVE_VIDEO=0 ./run_full_pipeline.sh
#
# Env vars:
#   SKIP_ENV=1                 Skip environment setup steps
#   SKIP_CLONE=1               Skip dependency cloning (if econ/InstantMesh/Garment3d are already present)
#   GARMENT3D_REPO_URL         Used by clone_deps.sh if Garment3DGen is missing
#   ECON_ENV                   (default: econ_script) - must match setup_econ_env.sh / run_econ.sh defaults if you changed them
#   INSTANTMESH_ENV            (default: InstantMesh)
#   GARMENT3D_ENV              (default: garment3d)
#   SAVE_VIDEO=1|0             (default: 1) for InstantMesh
#   G3D_SOURCE_MESH            (default: ${G3D_DIR}/meshes/tshirt.obj)
#   ECON_DIR / IM_DIR / G3D_DIR Override repo submodule paths if needed
#
set -euo pipefail
trap 'echo "[ERROR] Failed at ${BASH_SOURCE}:${LINENO}" >&2' ERR

###############################################################################
# Configuration
###############################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SKIP_ENV="${SKIP_ENV:-0}"
SKIP_CLONE="${SKIP_CLONE:-0}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"

ECON_ENV="${ECON_ENV:-econ_script}"
IM_ENV="${INSTANTMESH_ENV:-InstantMesh}"
G3D_ENV="${GARMENT3D_ENV:-garment3d}"

ECON_DIR="${ECON_DIR:-${REPO_ROOT}/econ/ECON}"
IM_DIR="${IM_DIR:-${REPO_ROOT}/InstantMesh/InstantMesh}"
G3D_DIR="${G3D_DIR:-${REPO_ROOT}/Garment3d/Garment3DGen}"

# Default demo image from ECON examples (same heuristic as other scripts).
DEFAULT_INPUT_1="${ECON_DIR}/examples/304e9c4798a8c3967de7c74c24ef2e38.jpg"
DEFAULT_INPUT_2="${ECON_DIR}/examples/demo.jpg"
INPUT_IMAGE="${1:-${DEFAULT_INPUT_2}}"
if [ ! -f "${INPUT_IMAGE}" ] && [ -f "${DEFAULT_INPUT_1}" ]; then
  INPUT_IMAGE="${DEFAULT_INPUT_1}"
fi

OUTPUT_ROOT="${2:-${SCRIPT_DIR}/full_pipeline_outputs}"
ECON_OUT="${OUTPUT_ROOT}/econ"
IM_OUT="${OUTPUT_ROOT}/instantmesh"
G3D_OUT="${OUTPUT_ROOT}/garment3d"
COLLECTED_OUT="${OUTPUT_ROOT}/collected_econ_and_garment3d"

###############################################################################
# Helpers
###############################################################################
log() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*" >&2; }
die() { echo "[ERROR] $*" >&2; exit 1; }

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

have_env() { conda env list | awk '{print $1}' | grep -Fx "$1" >/dev/null 2>&1; }

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

collect_outputs() {
  rm -rf "${COLLECTED_OUT}"
  mkdir -p "${COLLECTED_OUT}/econ" "${COLLECTED_OUT}/garment3d"

  if [ -d "${ECON_OUT}" ]; then
    # Prefer rsync if available for a clean mirrored copy; fallback to cp -r.
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --delete "${ECON_OUT}/" "${COLLECTED_OUT}/econ/"
    else
      cp -r "${ECON_OUT}/." "${COLLECTED_OUT}/econ/" || true
    fi
  fi

  if [ -d "${G3D_OUT}" ]; then
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --delete "${G3D_OUT}/" "${COLLECTED_OUT}/garment3d/"
    else
      cp -r "${G3D_OUT}/." "${COLLECTED_OUT}/garment3d/" || true
    fi
  fi
}

###############################################################################
# Validation
###############################################################################
command -v conda >/dev/null 2>&1 || die "conda not found in PATH"
init_conda

# Optional: clone deps if missing
if [ "${SKIP_CLONE}" != "1" ]; then
  if [ ! -d "${ECON_DIR}" ] || [ ! -d "${IM_DIR}" ] || [ ! -d "${G3D_DIR}" ]; then
    log "[setup] Some dependency folders are missing; running clone_deps.sh"
    # Pass through GARMENT3D_REPO_URL/GARMENT3D_REF if user set them
    GARMENT3D_REPO_URL="${GARMENT3D_REPO_URL:-}" GARMENT3D_REF="${GARMENT3D_REF:-main}" \
      ECON_REPO_URL="${ECON_REPO_URL:-}" ECON_REF="${ECON_REF:-main}" \
      INSTANTMESH_REPO_URL="${INSTANTMESH_REPO_URL:-}" INSTANTMESH_REF="${INSTANTMESH_REF:-main}" \
      bash "${SCRIPT_DIR}/clone_deps.sh"
  fi
else
  log "[setup] Skipping dependency cloning (SKIP_CLONE=1)"
fi

[ -d "${ECON_DIR}" ] || die "ECON_DIR not found: ${ECON_DIR}"
[ -d "${IM_DIR}" ] || die "IM_DIR not found: ${IM_DIR}"
[ -d "${G3D_DIR}" ] || die "G3D_DIR not found: ${G3D_DIR}"
[ -f "${INPUT_IMAGE}" ] || die "Input image not found: ${INPUT_IMAGE}"

G3D_SOURCE_MESH="${G3D_SOURCE_MESH:-${G3D_DIR}/meshes/tshirt.obj}"
[ -f "${G3D_SOURCE_MESH}" ] || die "Garment3D source mesh not found: ${G3D_SOURCE_MESH}"

mkdir -p "${OUTPUT_ROOT}" "${ECON_OUT}" "${IM_OUT}" "${G3D_OUT}"

###############################################################################
# 1) Setup envs (optional)
###############################################################################
if [ "${SKIP_ENV}" != "1" ]; then
  log "[setup] ECON environment"
  ECON_ENV="${ECON_ENV}" bash "${SCRIPT_DIR}/setup_econ_env.sh"

  log "[setup] InstantMesh environment"
  INSTANTMESH_ENV="${IM_ENV}" bash "${SCRIPT_DIR}/setup_instantmesh_env.sh"

  log "[setup] Garment3DGen environment"
  G3D_ENV="${G3D_ENV}" bash "${SCRIPT_DIR}/setup_garment3d_env.sh"
else
  log "[setup] Skipping environment setup (SKIP_ENV=1)"
fi

# Ensure envs exist
have_env "${ECON_ENV}" || die "ECON env '${ECON_ENV}' not found"
have_env "${IM_ENV}" || die "InstantMesh env '${IM_ENV}' not found"
have_env "${G3D_ENV}" || die "Garment3D env '${G3D_ENV}' not found"

###############################################################################
# 2) Run ECON
###############################################################################
log "[run 1/3] ECON"
ECON_ENV="${ECON_ENV}" ECON_DIR="${ECON_DIR}" \
  bash "${SCRIPT_DIR}/run_econ.sh" "$(dirname "${INPUT_IMAGE}")" "${ECON_OUT}" "${ECON_DIR}/configs/econ.yaml"

###############################################################################
# 3) Run InstantMesh (same image)
###############################################################################
log "[run 2/3] InstantMesh"
INSTANTMESH_ENV="${IM_ENV}" IM_DIR="${IM_DIR}" ECON_DIR="${ECON_DIR}" SAVE_VIDEO="${SAVE_VIDEO}" \
  bash "${SCRIPT_DIR}/run_instantmesh.sh" "${INPUT_IMAGE}" "${IM_OUT}" "${IM_DIR}/configs/instant-mesh-large.yaml"

IM_OBJ="$(find_instantmesh_obj)"
log "InstantMesh target mesh: ${IM_OBJ}"

###############################################################################
# 4) Run Garment3DGen using InstantMesh OBJ as mesh_target
###############################################################################
log "[run 3/3] Garment3DGen"
(
  init_conda
  conda activate "${G3D_ENV}"
  python "${G3D_DIR}/main.py" \
    --config "${G3D_DIR}/example_config.yml" \
    --output_path "${G3D_OUT}" \
    --mesh "${G3D_SOURCE_MESH}" \
    --target_mesh "${IM_OBJ}"
)

###############################################################################
# 5) Collect outputs
###############################################################################
log "[collect] ECON + Garment3D outputs"
collect_outputs

log "Done."
log "All outputs         -> ${OUTPUT_ROOT}"
log "Collected (econ+g3d) -> ${COLLECTED_OUT}"


