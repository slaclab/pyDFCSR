# Checkpoint-enabled pyDFCSR scripts

This branch contains modified `pyDFCSR_2D` scripts with checkpoint save/load support.  
**This is not a standalone repo.** Install the main pyDFCSR branch first, then overwrite with these files.

## Contents

```text
pyDFCSR_2D/
├── CSR.py
├── deposit.py
├── beams.py
└── params.py
example_configs/
├── config_beams_save_ckpt_debug.yaml
└── config_beams_load_ckpt_debug.yaml
```

## Changes vs original

### `CSR.py`
- **Checkpoint save/load**: `save_checkpoint()` and `load_checkpoint()` save and restore the complete simulation state (beam, DF history, interpolant buffers, runtime CSR scalars) into a named HDF5 group. Restarts are numerically exact.
- **Fixed LAB-frame envelope**: `_update_envelope_limits_for_source_and_CSR()` computes a fixed bounding box that covers the full particle distribution across the trajectory, replacing the per-step σ-relative grid of the original. This is used for both deposition and CSR mesh, and makes the density grids consistent across steps (required for CNN training data).
- **LAB-frame CSR mesh**: `get_CSR_mesh()` builds the observation mesh directly in LAB coordinates using the envelope bounds, instead of chirp-subtracted x′ coordinates.
- **Wake gate mask**: before calling `apply_wakes`, the kick array is masked to zero outside `kick_xlim`/`kick_zlim` σ windows, suppressing numerical noise at grid edges.
- **`stop_step`**: cleanly terminates the run and writes statistics after a fixed number of steps.

### `deposit.py`
- **`set_absolute_bounds(bounds)`**: new method. When called by `CSR.py`, `get_DF` uses the externally supplied LAB envelope instead of recomputing σ-relative bounds each step.
- **`_safe_savgol_2d`**: replaces bare `savgol_filter` calls; guards against filter window exceeding array size (avoids crashes on small grids).
- **`build_interpolant`**: guards against single-slice edge case (`nx == 1`) that caused division by zero.

## New YAML parameters

```yaml
CSR_computation:
  stop_step: 200          # stop after this many steps (null = run to end)
  kick_xlim: 3.0          # gate mask half-width in σ_x for kick application (default: xlim)
  kick_zlim: 3.0          # gate mask half-width in σ_z for kick application (default: zlim)

  restart:
    enabled: true         # master switch
    save: true            # write checkpoints during this run
    load: false           # resume from a saved checkpoint
    save_every_dipole: start,start+2,end  # which steps per dipole to checkpoint
                          # tokens: start, start+N, mid, end, end-N
    checkpoint_path: /path/to/restart.h5  # auto-generated if omitted
    checkpoint_group: b1le_1_step5        # group to load; uses latest if omitted
    save_wakes_range: 0   # write wakes only for N steps after restart (0 = all)
    save_xz: false        # save particle x-z scatter in wakes file
    save_xpx: false       # save x-px phase space in wakes file
    save_model: false     # embed loaded checkpoint into wakes HDF5 root
    strict: true          # raise error if checkpoint file missing on load
```

## Typical workflow

**Save run** — generate checkpoints:
```yaml
restart:
  enabled: true
  save: true
  load: false
  save_every_dipole: start,start+2,start+4
```

**Load run** — resume from checkpoint:
```yaml
restart:
  enabled: true
  save: false
  load: true
  checkpoint_path: /path/to/<write_name>-restart.h5
  checkpoint_group: b1le_1_step5
```
To list available checkpoint groups:
```python
import h5py
with h5py.File("restart.h5", "r") as f:
    print(list(f.keys()))          # all saved groups
    print(f.attrs["latest_group"]) # most recently saved
```
