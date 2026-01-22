#!/bin/bash

# Usage: ./RUNME.sh ["A person is dancing"]
# This script assumes you already have the conda environment and requirements set up.
# All steps are idempotent: additional downloads/installations are only performed if missing.

ENV_NAME="momask"
MODEL_DIR="models"
CLIP_PKG="clip"
CHUMPY_PKG="chumpy"
# 1. Check if environment exists
if ! conda env list | grep -q "^${ENV_NAME}\s"; then
    echo "Conda environment '${ENV_NAME}' not found. Creating from environment.yml..."
    conda env create -f environment.yml
else
    echo "Conda environment '${ENV_NAME}' already exists. Skipping creation."
fi

# 2. Activate environment (assumes conda is initialized)
echo "Activating environment '${ENV_NAME}'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# 3. Install CLIP if not already installed
if ! python -c "import ${CLIP_PKG}" 2>/dev/null; then
    echo "CLIP not found. Installing CLIP..."
    pip install git+https://github.com/openai/CLIP.git
else
    echo "CLIP already installed. Skipping."
fi

# 3. Install CLIP if not already installed
if ! python -c "import ${CHUMPY_PKG}" 2>/dev/null; then
    echo "Chumpy not found. Installing Chumpy..."
    pip install chumpy
else
    echo "Chumpy already installed. Skipping."
fi

# 4. Install rendering tools if not already installed
RENDER_PKGS=(bvh trimesh open3d opencv-python)
for pkg in "${RENDER_PKGS[@]}"; do
    if ! pip show $pkg >/dev/null 2>&1; then
        echo "$pkg not found. Installing $pkg..."
        pip install $pkg
    else
        echo "$pkg already installed. Skipping."
    fi
done

# 5. Download models FOR INFERENCE (only if not already present)
DOWNLOAD_MODELS=0

# Check if MODEL_DIR exists and is non-empty
if [ ! -d "${MODEL_DIR}" ] || [ -z "$(ls -A ${MODEL_DIR})" ]; then
    DOWNLOAD_MODELS=1
else
    # Check if 't2m' folder exists inside MODEL_DIR, and has at least one non-empty subfolder with files
    T2M_DIR="${MODEL_DIR}/t2m"
    if [ ! -d "${T2M_DIR}" ]; then
        DOWNLOAD_MODELS=1
    else
        # For every subfolder of t2m, check if it contains files
        EMPTY_SUBFOLDER=0
        for SUBFOLDER in "${T2M_DIR}"/*; do
            if [ -d "$SUBFOLDER" ]; then
                # Does this subfolder have at least one file?
                if [ -z "$(ls -A "$SUBFOLDER")" ]; then
                    EMPTY_SUBFOLDER=1
                    break
                fi
            fi
        done
        if [ "$EMPTY_SUBFOLDER" -eq 1 ]; then
            DOWNLOAD_MODELS=1
        fi
    fi
fi

if [ "$DOWNLOAD_MODELS" -eq 1 ]; then
    echo "Models or required t2m checkpoints not found or are empty. Downloading gdown before downloading models..."
    pip install --upgrade --no-cache-dir gdown
    echo "Downloading models..."
    if ! bash prepare/download_models.sh; then
        echo "download_models.sh failed. Exiting."
        exit 1
    fi
    echo "Models downloaded successfully."
else
    echo "Models and t2m checkpoints already present. Skipping model download."
fi

# ONLY FOR RETRAINING AND EVALUATING THE MODEL
# bash prepare/download_evaluator.sh
# bash prepare/download_glove.sh

# 6. Test
TEXT_PROMPT=${1:-"A person is jogging in a treadmill."}
echo "Generating motion from text: \"$TEXT_PROMPT\" ..."
python gen_t2m.py --text_prompt "$TEXT_PROMPT"
if [ $? -ne 0 ]; then
    echo "gen_t2m.py failed. Exiting."
    exit 1
fi

echo "Converting BVH to SMPL..."
python bvh_to_smpl.py
if [ $? -ne 0 ]; then
    echo "bvh_to_smpl.py failed. Exiting."
    exit 1
fi

echo "Converting frames to motion..."
python frame_to_motion.py
if [ $? -ne 0 ]; then
    echo "frame_to_motion.py failed. Exiting."
    exit 1
fi


echo "Done!"