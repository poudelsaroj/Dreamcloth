# Dreamcloth

A clothing website project.

## Getting Started

Clone the repository and open the project files to get started.

## Data Files (MPM)

To run the scripts in `mpm/`, put the data files in these locations:

- `F:\mpm\github\Dreamcloth\data\SMPLX_NEUTRAL.npz` (or the gendered equivalent) for the body model used by `mpm\mpm_cloth_v28.py`.
- `F:\mpm\github\Dreamcloth\data\tshirt-sim.obj` (fallback: `tshirt-decimated.obj`) for the cloth mesh used by `mpm\mpm_cloth_v28.py` and `mpm\mpm_cloth_v30_jump.py`.
- `F:\mpm\github\Dreamcloth\data\jump\frame_*.obj` for the jump body frames used by `mpm\mpm_cloth_v30_jump.py`.

Outputs are written to `F:\mpm\github\Dreamcloth\output\running_v28` and `F:\mpm\github\Dreamcloth\output\jump_v30` by default.




## License

MIT License
