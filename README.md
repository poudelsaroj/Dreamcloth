# MoMask - Text to Motion Inference

This repository contains the inference pipeline for generating 3D human motion animations from text descriptions.

## Quick Start

### Usage

Run the inference pipeline using `RUNME.sh`. The first argument is the text prompt describing the motion you want to generate:

```bash
./RUNME.sh "A person is dancing"
```

If no text prompt is provided, it will use the default prompt: "A person is jogging in a treadmill."

### What the Script Does

The `RUNME.sh` script automates the entire inference pipeline:

1. **Environment Setup**: Creates/activates the conda environment and installs required dependencies
2. **Model Download**: Downloads pre-trained models if not already present
3. **Motion Generation**: Generates motion from text using `gen_t2m.py`
4. **BVH to SMPL Conversion**: Converts BVH files to SMPL meshes using `bvh_to_smpl.py`
5. **Video Generation**: Renders frames and creates the final `animation.mp4` using `frame_to_motion.py`

### Output Files

After running the script, you'll find:

- **BVH files**: `generation/text2motion/animations/0/` - Motion data in BVH format
- **SMPL meshes**: `smpl_meshes_from_bvh/` - 3D mesh files (`.obj` format)
- **Rendered frames**: `rendered_frames/` - Individual frame images (`.png` format)
- **Final animation**: `animation.mp4` - The final video output

### Prerequisites

- Conda (for environment management)
- Python 3.x
- CUDA-capable GPU (recommended for faster inference)

### Example Usage

```bash
# Generate motion for dancing
./RUNME.sh "A person is dancing"

# Generate motion for running
./RUNME.sh "A person is running on a treadmill"

# Use default prompt. Text Prompt "A person is running on a treadmill"
./RUNME.sh
```

### Notes

- The script is idempotent: it will only download/install missing components
- All steps are automated - just provide the text prompt and wait for the animation!
- The generated animation will be saved as `animation.mp4` in the current directory

