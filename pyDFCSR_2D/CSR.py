_CSR_DEBUG = 0  

import os
import time
import re
from bmadx import  Drift, SBend, Quadrupole, Sextupole
from .tools import dict2hdf5
import h5py
import numpy as np



from .beams import Beam
# from .deposit import histogram_cic_1d, histogram_cic_2d
from .deposit import DF_tracker
from .interp1D import interpolate1D
from .interp3D import interpolate3D
from .lattice import Lattice  # , get_referece_traj
from .params import Integration_params, CSR_params
# from .physical_constants import c, e, qe, me, MC2
#from line_profiler_pycharm import profile
# from .tools import (find_nearest_ind, full_path, isotime, plot_2D_contour,
#                     plot_surface)
from .tools import full_path, isotime
from .yaml_parser import parse_yaml




class CSR2D:
    """
    The main class to calculate 2D CSR
    """

    def __init__(self, input_file=None, parallel = False):

        self.timestamp = isotime()
        if input_file:
            self.parse_input(input_file)
            self.input_file = input_file
        self.formation_length = None
        self.initialization()  # process the initial beam



        self.prefix = f'{self.CSR_params.write_name}-{self.timestamp}'

        if parallel:
            self.init_MPI()
        else:
            self.parallel = False


    def parse_input(self, input_file):
        input = parse_yaml(input_file)
        self.check_input_consistency(input)
        self.input = input
        self.beam = Beam(input['input_beam'])
        self.lattice = Lattice(input['input_lattice'])

        if 'particle_deposition' in input:
            self.DF_tracker = DF_tracker(input['particle_deposition'])
        else:
            self.DF_tracker = DF_tracker()

        if 'CSR_integration' in input:
            self.integration_params = Integration_params(input['CSR_integration'])
        else:
            self.integration_params = Integration_params()

        if 'CSR_computation' in input:
            self.CSR_params = CSR_params(input['CSR_computation'])
        else:
            self.CSR_params = CSR_params()

        # NEW: restart/checkpoint configuration
        # AFTER:
        self._parse_restart_config(input.get('CSR_computation', {}))
        wd = (self.CSR_params.workdir or "").rstrip("/")
        self.CSR_params.workdir = wd


        if wd:
            os.makedirs(wd, exist_ok=True)


    # ---------------------------------------------------------------------
    # Restart / checkpointing helpers
    # ---------------------------------------------------------------------
    def _write_deque_arrays(self, grp, name, dq):
        """Store a deque/list of numpy arrays into an HDF5 group."""
        sub = grp.create_group(name)
        try:
            n = len(dq)
        except Exception:
            dq = list(dq)
            n = len(dq)
        sub.attrs["len"] = int(n)
        for i, arr in enumerate(list(dq)):
            # Try regular conversion first; catch ragged/inhomogeneous sequences
            if isinstance(arr, np.ndarray):
                converted = arr
                is_ragged = False
            else:
                try:
                    converted = np.asarray(arr)
                    is_ragged = (converted.dtype == object)
                except (ValueError, TypeError):
                    converted = None
                    is_ragged = True

            if not is_ragged and converted is not None:
                if converted.ndim == 0:
                    sub.attrs[f"{i:04d}"] = converted.item()
                else:
                    sub.create_dataset(f"{i:04d}", data=converted, compression="gzip")
            else:
                # Ragged/inhomogeneous — save each element separately in a subgroup
                row_grp = sub.create_group(f"{i:04d}")
                row_grp.attrs["ragged"] = True
                row_grp.attrs["nrows"] = len(arr)
                for j, row in enumerate(arr):
                    row_grp.create_dataset(f"{j:04d}", data=np.asarray(row, dtype=float), compression="gzip")
    
    def _read_deque_arrays(self, grp, name):
        """Read a deque/list of numpy arrays previously stored by _write_deque_arrays."""
        sub = grp[name]
        n = int(sub.attrs.get("len", 0))
        out = []
        for i in range(n):
            key = f"{i:04d}"
            if key in sub.attrs:
                # Scalar saved as attribute (ndim==0 case)
                out.append(np.asarray(sub.attrs[key]))
            elif key in sub:
                item = sub[key]
                if isinstance(item, h5py.Group) and item.attrs.get("ragged", False):
                    nrows = int(item.attrs["nrows"])
                    rows = [np.asarray(item[f"{j:04d}"]) for j in range(nrows)]
                    out.append(rows)
                else:
                    out.append(np.asarray(item))
            else:
                raise KeyError(f"_read_deque_arrays: key '{key}' not found in attrs or datasets of group '{sub.name}'")
        return out
    
    def _save_df_tracker_state(self, grp):
        """Serialize DF_tracker state needed to continue CSR history."""
        dft = self.DF_tracker
    
        # Scalar params / config
        for k in [
            "xbins", "zbins", "xlim", "zlim",
            "filter_order", "filter_window",
            "upper_limit", "velocity_threhold",
        ]:
            if hasattr(dft, k):
                try:
                    grp.attrs[k] = getattr(dft, k)
                except Exception:
                    pass
    
        # Absolute bounds used for deposition (if any)
        if hasattr(dft, "absolute_bounds") and (dft.absolute_bounds is not None):
            try:
                grp.attrs["absolute_bounds"] = np.asarray(dft.absolute_bounds, dtype=float)
            except Exception:
                pass
    
        # Current DF fields (if present)
        # Current DF fields (if present)
        for k in [
            "sigma_x", "sigma_z", "t", "start_time", "end_time",
            "x_grids", "z_grids",
            "density", "density_x", "density_z",
            "vx", "vx_x",
        ]:
            if hasattr(dft, k) and getattr(dft, k) is not None:
                arr = np.asarray(getattr(dft, k))
                if arr.ndim == 0:  # scalar — save as HDF5 attribute, not dataset
                    grp.attrs[k] = arr.item()
                else:
                    grp.create_dataset(k, data=arr, compression="gzip")
    
        # Logs (history)
        for name in ["slope_log", "DF_log", "sigma_x_log", "sigma_z_log", "time_log"]:
            if hasattr(dft, name) and getattr(dft, name) is not None:
                self._write_deque_arrays(grp, name, getattr(dft, name))
    
        # Interpolant buffers
        for name in ["time_interp", "density_interp", "density_x_interp", "density_z_interp", "vx_interp", "vx_x_interp"]:
            if hasattr(dft, name) and getattr(dft, name) is not None:
                self._write_deque_arrays(grp, name, getattr(dft, name))
    
        # Interpolant grids and indices
        for k in ["interp_start", "interp_end", "x_grid_interp", "z_grid_interp", "sigma_x_interp", "sigma_z_interp"]:
            if hasattr(dft, k) and getattr(dft, k) is not None:
                v = getattr(dft, k)
                if np.isscalar(v):
                    try:
                        grp.attrs[k] = v
                    except Exception:
                        pass
                else:
                    grp.create_dataset(k, data=np.asarray(v), compression="gzip")
    
    def _load_df_tracker_state(self, grp):
        """Restore DF_tracker state saved by _save_df_tracker_state."""
        from collections import deque
        dft = self.DF_tracker
    
        # Restore attrs (mostly config scalars)
        for k, v in grp.attrs.items():
            try:
                setattr(dft, k, v)
            except Exception:
                pass
    
        # Datasets
        for k in [
            "sigma_x", "sigma_z", "t", "start_time", "end_time",
            "x_grids", "z_grids",
            "density", "density_x", "density_z",
            "vx", "vx_x",
            "x_grid_interp", "z_grid_interp",
            "sigma_x_interp", "sigma_z_interp",
        ]:
            if k in grp:
                setattr(dft, k, np.asarray(grp[k]))
            elif k in grp.attrs:
                setattr(dft, k, grp.attrs[k])
    
        # Deques
        for name in [
            "slope_log", "DF_log", "sigma_x_log", "sigma_z_log", "time_log",
            "time_interp", "density_interp", "density_x_interp", "density_z_interp", "vx_interp", "vx_x_interp",
        ]:
            if name in grp:
                arrs = self._read_deque_arrays(grp, name)
                setattr(dft, name, deque(arrs))
    
        # Interp indices
        for k in ["interp_start", "interp_end"]:
            if k in grp.attrs:
                try:
                    setattr(dft, k, float(grp.attrs[k]))
                except Exception:
                    pass
    
        # Only rebuild if interpolant buffers were NOT restored
        # Always rebuild the interpolant after loading.
        # The saved buffers are restored, but derived fields like
        # data_density_interp/min_x/delta_x/etc. are not saved explicitly.
        try:
            dft.build_interpolant()
        except Exception as e:
            if _CSR_DEBUG and ((not self.parallel) or (getattr(self, "rank", 0) == 0)):
                print(f"⚠️  DF_tracker.build_interpolant() failed during restart load: {e}")
    
    # ---------------------------------------------------------------------
    # Restart / checkpointing configuration & core I/O
    # ---------------------------------------------------------------------
    def _parse_restart_config(self, csr_config):
        """Parse restart/checkpoint configuration from YAML under CSR_computation.restart."""
        cfg = {}
        if isinstance(csr_config, dict):
            cfg = csr_config.get("restart", {}) or {}
        if not isinstance(cfg, dict):
            cfg = {}

        self.restart_enabled = bool(cfg.get("enabled", False))
        self.restart_load = bool(cfg.get("load", cfg.get("resume", False)))
        self.restart_save = bool(cfg.get("save", True))
        self.restart_keep_history = bool(cfg.get("keep_history", False))
        self.restart_strict = bool(cfg.get("strict", True))

        self.restart_checkpoint_path = cfg.get("checkpoint_path", None)
        self.restart_checkpoint_group = cfg.get("checkpoint_group", None)

        self.restart_save_every_dipole = cfg.get("save_every_dipole", "none")
        self.restart_save_wakes_range = int(cfg.get("save_wakes_range", 0) or 0)
        self.restart_save_xz  = bool(cfg.get("save_xz",  False))
        self.restart_save_xpx = bool(cfg.get("save_xpx", False))
        self.restart_save_model = bool(cfg.get("save_model", False))


        self._restart_loaded = False
        self._restart_next_step = None
        self._restart_target_ele_index = 0
        self._restart_target_local_step = 0
        self._restart_wake_start_step = None
        self._restart_wake_end_step = None
        self._restart_wake_prefix = None

    def _default_checkpoint_path(self):
        outdir = self.CSR_params.workdir
        os.makedirs(outdir, exist_ok=True)
        return os.path.join(outdir, f"{self.CSR_params.write_name}-restart.h5")
    
    def _get_checkpoint_path(self):
        return self.restart_checkpoint_path or self._default_checkpoint_path()

    
    
    def _replace_last_df_state_with_current(self):
        """
        Make the saved DF/interpolant state consistent with the current post-kick beam.
        This is used only right before save_checkpoint().
        """
        from collections import deque
    
        dft = self.DF_tracker
    
        # Recompute DF from current post-kick beam on the current envelope
        self._update_envelope_limits_for_source_and_CSR()
        dft.set_absolute_bounds(self._envelope_bounds)
        dft.get_DF(
            x=self.beam.x, z=self.beam.z,
            px=self.beam.px, pz=self.beam.pz, t=self.beam.position
        )
    
        # Replace only the most recent DF_log entry
        if hasattr(dft, "DF_log") and len(dft.DF_log) > 0:
            dft.DF_log[-1] = (
                np.asarray(dft.x_grids).copy(),
                np.asarray(dft.z_grids).copy(),
                np.asarray(dft.density).copy(),
                np.asarray(dft.vx).copy(),
                np.asarray(dft.density_x).copy(),
                np.asarray(dft.density_z).copy(),
                np.asarray(dft.vx_x).copy(),
            )
    
        # Replace matching scalar/history logs
        if hasattr(dft, "time_log") and len(dft.time_log) > 0:
            dft.time_log[-1] = float(self.beam.position)
            dft.end_time = float(self.beam.position)
    
        if hasattr(dft, "sigma_x_log") and len(dft.sigma_x_log) > 0:
            dft.sigma_x_log[-1] = float(dft.sigma_x)
    
        if hasattr(dft, "sigma_z_log") and len(dft.sigma_z_log) > 0:
            dft.sigma_z_log[-1] = float(dft.sigma_z)
    
        if hasattr(dft, "slope_log") and len(dft.slope_log) > 0:
            dft.slope_log[-1] = np.asarray(self.beam.slope).copy()
    
        # Rebuild interpolant buffers from corrected history
        dft.time_interp      = deque()
        dft.density_interp   = deque()
        dft.density_x_interp = deque()
        dft.density_z_interp = deque()
        dft.vx_interp        = deque()
        dft.vx_x_interp      = deque()
    
        dft.append_interpolant(
            formation_length=self.formation_length,
            n_formation_length=self.integration_params.n_formation_length
        )
        dft.build_interpolant()
                
    def _parse_dipole_step_expressions(self, exprs):
        """Return list of expression tokens (e.g. ['start+1','mid','end-5'])."""
        if exprs is None:
            return []
        if isinstance(exprs, str):
            s = exprs.strip()
            if s.lower() in ("none", "null", "false", ""):
                return []
            return [p.strip() for p in s.split(",") if p.strip()]
        if isinstance(exprs, (list, tuple)):
            out = []
            for v in exprs:
                if v is None:
                    continue
                sv = str(v).strip()
                if sv and (sv.lower() not in ("none", "null", "false")):
                    out.append(sv)
            return out
        return []
    
    def _eval_dipole_expr(self, expr, start_step, end_step):
        """Evaluate one expression against dipole start/end global step indices (inclusive).
        Returns an integer global step (or None if invalid/out of range).
        """
        expr = str(expr).strip().lower()
        steps = end_step - start_step + 1
        if steps <= 0:
            return None
        mid_step = start_step + (steps // 2)
    
        base = None
        offset = 0
    
        if expr.startswith("start"):
            base = start_step
            tail = expr[len("start"):]
        elif expr.startswith("mid"):
            base = mid_step
            tail = expr[len("mid"):]
        elif expr.startswith("end"):
            base = end_step
            tail = expr[len("end"):]
        else:
            # allow explicit integer
            try:
                v = int(expr)
                return v
            except Exception:
                return None
    
        tail = tail.strip()
        if tail:
            m = re.match(r"^([+-])\s*(\d+)$", tail)
            if not m:
                return None
            sign = 1 if m.group(1) == "+" else -1
            offset = sign * int(m.group(2))
    
        s = base + offset
        if s < start_step or s > end_step:
            return None
        return int(s)
    
    def _compute_ckpt_save_steps(self):
        """Compute set of global step indices at which to write checkpoints."""
        exprs = self._parse_dipole_step_expressions(self.restart_save_every_dipole)
        if not exprs:
            return set()
    
        steps_set = set()
        # Use lattice_config order and steps_per_element to derive global step ranges per dipole
        elems = list(self.lattice.lattice_config.keys())[1:]
        global_step = 1  # this matches run() initialization
        for ele_count, ele in enumerate(elems):
            nsteps = int(self.lattice.steps_per_element[ele_count])
            typ = self.lattice.lattice_config[ele].get("type", None)
            if nsteps <= 0:
                continue
    
            dip_start = global_step
            dip_end = global_step + nsteps - 1
    
            if typ == "dipole":
                for ex in exprs:
                    s = self._eval_dipole_expr(ex, dip_start, dip_end)
                    if s is not None:
                        steps_set.add(int(s))
    
            global_step += nsteps
    
        return steps_set
    
    def _locate_element_for_global_step(self, target_step):
        """Map a global step index to (ele_index, local_step_index, ele_name)."""
        elems = list(self.lattice.lattice_config.keys())[1:]
        global_step = 1
        for ele_index, ele in enumerate(elems):
            nsteps = int(self.lattice.steps_per_element[ele_index])
            if nsteps <= 0:
                continue
            if global_step <= target_step <= (global_step + nsteps - 1):
                local = int(target_step - global_step)
                return int(ele_index), int(local), str(ele)
            global_step += nsteps
        # If out of range, return end
        return int(len(elems) - 1), int(max(self.lattice.steps_per_element[-1] - 1, 0)), str(elems[-1])


    def _write_restart_group_to_h5(self, hf, completed_step, ele_name):
        """
        Write one restart/checkpoint group into an already-open HDF5 file handle.
        The layout matches the normal restart H5 structure.
        """
        group_key = f"{ele_name}_step{completed_step}"
    
        # Root attrs: make this file restartable like a standard restart file
        hf.attrs["format"] = "CSR2D_restart_v2"
        hf.attrs["write_name"] = str(self.CSR_params.write_name)
        hf.attrs["latest_step"] = int(completed_step)
        hf.attrs["latest_group"] = group_key
    
        # Overwrite existing group if needed
        if group_key in hf:
            del hf[group_key]
    
        grp = hf.create_group(group_key)
        grp.attrs["completed_step"] = int(completed_step)
        grp.attrs["next_step"] = int(completed_step + 1)
        grp.attrs["ele_name"] = str(ele_name)
        grp.attrs["timestamp_saved"] = float(time.time())
    
        # Beam state
        gb = grp.create_group("beam")
        for k in ["x", "px", "y", "py", "z", "pz"]:
            try:
                gb.create_dataset(
                    k,
                    data=np.asarray(getattr(self.beam.particle, k)),
                    compression="gzip"
                )
            except Exception:
                pass
    
        if hasattr(self.beam.particle, "t"):
            gb.attrs["particle_t"] = float(self.beam.particle.t)
        if hasattr(self.beam.particle, "p0c"):
            gb.attrs["particle_p0c"] = float(self.beam.particle.p0c)
        if hasattr(self.beam.particle, "mc2"):
            gb.attrs["particle_mc2"] = float(self.beam.particle.mc2)
    
        gb.attrs["position"] = float(self.beam.position)
        gb.attrs["step"] = int(getattr(self.beam, "step", completed_step))
    
        # CSR state
        gs = grp.create_group("csr_state")
        for k in ["inbend", "afterbend"]:
            if hasattr(self, k):
                gs.attrs[k] = bool(getattr(self, k))
        for k in ["formation_length", "R_rec", "phi_rec", "CSR_scaling"]:
            if hasattr(self, k) and getattr(self, k) is not None:
                try:
                    gs.attrs[k] = float(getattr(self, k))
                except Exception:
                    pass
    
        # Statistics
        gst = grp.create_group("statistics")
        for k, v in self.statistics.items():
            if isinstance(v, dict):
                sub = gst.create_group(k)
                for kk, vv in v.items():
                    sub.create_dataset(kk, data=np.asarray(vv), compression="gzip")
            else:
                gst.create_dataset(k, data=np.asarray(v), compression="gzip")
    
        # DF tracker
        gdf = grp.create_group("DF_tracker")
        self._save_df_tracker_state(gdf)
    
        # Runtime CSR state
        grt = grp.create_group("runtime_state")
        self._save_runtime_state(grt)
    
        return group_key
        

    
    
    
    def save_checkpoint(self, completed_step, ele_name):
        """Save checkpoint as a named group inside the shared restart HDF5 file."""
        if (not getattr(self, "restart_enabled", False)) or (not getattr(self, "restart_save", False)):
            return
        if self.parallel and getattr(self, "rank", 0) != 0:
            return
    
        ckpt_path = self._get_checkpoint_path()
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    
        with h5py.File(ckpt_path, "a") as f:
            group_key = self._write_restart_group_to_h5(f, completed_step, ele_name)
    
        # optional debug verification
        with h5py.File(ckpt_path, "r") as fchk:
            gchk = fchk[group_key]["beam"]
            x_chk  = np.asarray(gchk["x"])
            px_chk = np.asarray(gchk["px"])
    
            if (not self.parallel) or (getattr(self, "rank", 0) == 0):
                print(
                    "[POST-WRITE CHECK] "
                    f"sum_x={np.sum(x_chk):.16e}, "
                    f"sum_px={np.sum(px_chk):.16e}, "
                    f"sum_x2={np.sum(x_chk*x_chk):.16e}, "
                    f"sum_px2={np.sum(px_chk*px_chk):.16e}, "
                    f"sum_xpx={np.sum(x_chk*px_chk):.16e}"
                )
    
        if _CSR_DEBUG and ((not self.parallel) or (getattr(self, "rank", 0) == 0)):
            print(f"💾 Saved checkpoint group '{group_key}' -> {ckpt_path}")

    def _save_model_to_wakes_root(self, wakes_filename, completed_step, ele_name):
        """
        Save checkpoint/restart-style state directly into the ROOT of the wakes H5 file,
        so the wakes file itself is restartable like a normal restart H5.
        """
        if self.parallel and getattr(self, "rank", 0) != 0:
            return
    
        max_retries = 5
        retry_delay = 1.0
    
        for attempt in range(max_retries):
            try:
                with h5py.File(wakes_filename, "a") as hf:
                    group_key = self._write_restart_group_to_h5(hf, completed_step, ele_name)
    
                    # Optional provenance attrs
                    hf.attrs["restart_model_saved"] = True
                    hf.attrs["restart_model_source"] = "in_file_root"
    
                if _CSR_DEBUG and ((not self.parallel) or (getattr(self, "rank", 0) == 0)):
                    print(f"💾 Saved restart model '{group_key}' into wakes file root: {wakes_filename}")
                return
    
            except (BlockingIOError, OSError) as e:
                if attempt < max_retries - 1:
                    print(f"File lock error during model save on attempt {attempt + 1}, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"Failed to save restart model into wakes file after {max_retries} attempts: {e}")
                    return
                
    def _embed_loaded_checkpoint_in_wakes_file(self, wakes_filename):
        """
        Copy the loaded restart checkpoint group into the wake HDF5 file once,
        under /loaded_checkpoint/<checkpoint_group>.
        """
        if not getattr(self, "_restart_loaded", False):
            return

        ckpt_path = self._get_checkpoint_path()
        if (not ckpt_path) or (not os.path.exists(ckpt_path)):
            return

        group_key = getattr(self, "restart_checkpoint_group", None)
        if not group_key:
            return

        with h5py.File(ckpt_path, "r") as fsrc, h5py.File(wakes_filename, "a") as fdst:
            if group_key not in fsrc:
                print(f"WARNING: checkpoint group '{group_key}' not found in {ckpt_path}; not embedding into wakes file.")
                return

            parent = fdst.require_group("loaded_checkpoint")

            # overwrite if already present
            if group_key in parent:
                del parent[group_key]

            fsrc.copy(fsrc[group_key], parent, name=group_key)

            parent.attrs["source_checkpoint_file"] = str(ckpt_path)
            parent.attrs["source_checkpoint_group"] = str(group_key)

        if _CSR_DEBUG and ((not self.parallel) or (getattr(self, "rank", 0) == 0)):
            print(f"📦 Embedded checkpoint group '{group_key}' into {wakes_filename}")
            
    def load_checkpoint(self, ckpt_path=None):
        """Load checkpoint and restore beam/DF/history so we can continue from next_step."""
        if self.parallel:
            raise NotImplementedError(
                "Restart/load is currently only safe in serial mode. "
                "Rank-wise broadcast of beam/DF state has not been implemented yet."
            )
    
        ckpt_path = ckpt_path or self._get_checkpoint_path()
        if (not ckpt_path) or (not os.path.exists(ckpt_path)):
            if getattr(self, "restart_strict", True):
                raise FileNotFoundError(f"Restart requested but checkpoint not found: {ckpt_path}")
            if _CSR_DEBUG and ((not self.parallel) or (getattr(self, "rank", 0) == 0)):
                print(f"⚠️  Restart checkpoint not found, starting fresh: {ckpt_path}")
            return False
    
        with h5py.File(ckpt_path, "r") as f:
            fmt = str(f.attrs.get("format", ""))

            has_embedded_checkpoint = (
                "loaded_checkpoint" in f and isinstance(f["loaded_checkpoint"], h5py.Group)
            )
            
            has_shard_checkpoint = any(
                k.startswith("sample_") and isinstance(f[k], h5py.Group) and "checkpoint" in f[k]
                for k in f.keys()
            )
            
            if (
                fmt not in ("CSR2D_restart_v1", "CSR2D_restart_v2")
                and not has_embedded_checkpoint
                and not has_shard_checkpoint
                and getattr(self, "restart_strict", True)
            ):
                raise ValueError(f"Unrecognized checkpoint format: {fmt}")

            # Find which group to load: user can specify a step via
            requested_group = getattr(self, "restart_checkpoint_group", None)

            if requested_group is not None:
                # Case 1: standard restart file, checkpoint group at root
                if requested_group in f:
                    group_key = requested_group
                    grp = f[group_key]

                # Case 2: wakes file with embedded checkpoint under /loaded_checkpoint/<group>
                elif ("loaded_checkpoint" in f) and (requested_group in f["loaded_checkpoint"]):
                    group_key = requested_group
                    grp = f["loaded_checkpoint"][group_key]

                # Case 3: finetune shard file — find sample_XXXX whose checkpoint_group attr
                # matches requested_group, then load from sample_XXXX/checkpoint/
                else:
                    # Case 3: finetune shard — prefer direct sample_key lookup to avoid
                    # collisions where multiple samples share the same checkpoint_group name
                    shard_sample_key = None
                    key_hint = getattr(self, "restart_checkpoint_sample_key", None)

                    if (key_hint and key_hint in f
                            and isinstance(f[key_hint], h5py.Group)
                            and "checkpoint" in f[key_hint]):
                        # Direct match via sample key — unambiguous
                        shard_sample_key = key_hint
                    else:
                        # Fallback: search by checkpoint_group (may find wrong sample
                        # if multiple samples share the same group name)
                        for k in sorted(f.keys()):
                            if (
                                k.startswith("sample_")
                                and isinstance(f[k], h5py.Group)
                                and str(f[k].attrs.get("checkpoint_group", "")) == requested_group
                            ):
                                shard_sample_key = k
                                break

                    if shard_sample_key is not None and "checkpoint" in f[shard_sample_key]:
                        group_key = requested_group
                        grp = f[shard_sample_key]["checkpoint"]
                        if _CSR_DEBUG and ((not self.parallel) or (getattr(self, "rank", 0) == 0)):
                            print(f"   🗂️  Case 3: loading from shard "
                                  f"{shard_sample_key}/checkpoint/ "
                                  f"(checkpoint_group='{requested_group}')")
                    else:
                        available_root = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
                        available_embedded = []
                        if "loaded_checkpoint" in f and isinstance(f["loaded_checkpoint"], h5py.Group):
                            available_embedded = [
                                k for k in f["loaded_checkpoint"].keys()
                                if isinstance(f["loaded_checkpoint"][k], h5py.Group)
                            ]
                        available_shard = [
                            f"{k}(cg={f[k].attrs.get('checkpoint_group','')})"
                            for k in sorted(f.keys())
                            if k.startswith("sample_") and isinstance(f[k], h5py.Group)
                        ]
                        raise KeyError(
                            f"Checkpoint group '{requested_group}' not found in {ckpt_path}.\n"
                            f"Available root groups: {available_root}\n"
                            f"Available embedded checkpoint groups: {available_embedded}\n"
                            f"Available shard samples: {available_shard}"
                        )

            else:
                # default: try latest_group at root first
                group_key = str(f.attrs.get("latest_group", ""))

                if group_key and (group_key in f):
                    grp = f[group_key]

                # optional fallback: if latest_group is absent at root, check embedded checkpoints
                elif group_key and ("loaded_checkpoint" in f) and (group_key in f["loaded_checkpoint"]):
                    grp = f["loaded_checkpoint"][group_key]

                else:
                    available_root = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
                    available_embedded = []
                    if "loaded_checkpoint" in f and isinstance(f["loaded_checkpoint"], h5py.Group):
                        available_embedded = [
                            k for k in f["loaded_checkpoint"].keys()
                            if isinstance(f["loaded_checkpoint"][k], h5py.Group)
                        ]

                    raise KeyError(
                        f"No usable latest_group attr found in {ckpt_path}.\n"
                        f"Available root groups: {available_root}\n"
                        f"Available embedded checkpoint groups: {available_embedded}\n"
                        f"Specify one via restart.checkpoint_group in YAML."
                    )

            next_step = int(grp.attrs["next_step"])
            completed_step = int(grp.attrs["completed_step"])
            ele_name = str(grp.attrs["ele_name"])

            if _CSR_DEBUG and ((not self.parallel) or (getattr(self, "rank", 0) == 0)):
                print(f"🔁 Loading checkpoint group '{group_key}' "
                      f"(completed_step={completed_step}, ele='{ele_name}', resuming at step {next_step})")

            # Beam
            # Beam — x/px/y/py/z/pz are read-only properties on Beam,
            # must reconstruct self.beam.particle directly
            gb = grp["beam"]
            coords = {k: np.asarray(gb[k]) for k in ["x", "px", "y", "py", "z", "pz"] if k in gb}

            if _CSR_DEBUG and ((not self.parallel) or (getattr(self, "rank", 0) == 0)):
                print(
                    "[RAW H5 LOAD CHECK] "
                    f"sum_x={np.sum(coords['x']):.16e}, "
                    f"sum_px={np.sum(coords['px']):.16e}, "
                    f"sum_x2={np.sum(coords['x']*coords['x']):.16e}, "
                    f"sum_px2={np.sum(coords['px']*coords['px']):.16e}, "
                    f"sum_xpx={np.sum(coords['x']*coords['px']):.16e}"
                )
            
            if _CSR_DEBUG and ((not self.parallel) or (getattr(self, "rank", 0) == 0)):
                print(f"[DEBUG] coord keys = {list(coords.keys())}, len(coords) = {len(coords)}")
            
            required = ["x", "px", "z", "pz"]
            missing_required = [k for k in required if k not in coords]
            if missing_required:
                raise ValueError(
                    f"Checkpoint is missing required beam coordinates: {missing_required}. "
                    f"Available keys={list(coords.keys())}"
                )
            
            from bmadx import Particle
            
            # Never use gb.attrs.get(..., self.beam.particle.t) here:
            # the fallback expression is evaluated immediately.
            
            if "particle_t" in gb.attrs:
                t_saved = float(gb.attrs["particle_t"])
            else:
                t_saved = float(getattr(self.beam.particle, "t", 0.0))
            
            if "particle_p0c" in gb.attrs:
                p0c_saved = float(gb.attrs["particle_p0c"])
            else:
                p0c_saved = float(getattr(self.beam.particle, "p0c", 0.0))
            
            if "particle_mc2" in gb.attrs:
                mc2_saved = float(gb.attrs["particle_mc2"])
            else:
                mc2_saved = float(getattr(self.beam.particle, "mc2", 0.0))
            
            # y/py may be absent in older checkpoints; fall back to current beam particle
            y_saved  = coords["y"]  if "y"  in coords else np.asarray(self.beam.particle.y)
            py_saved = coords["py"] if "py" in coords else np.asarray(self.beam.particle.py)
            
            self.beam.particle = Particle(
                coords["x"], coords["px"],
                y_saved, py_saved,
                coords["z"], coords["pz"],
                t_saved, p0c_saved, mc2_saved
            )
            
            if _CSR_DEBUG and ((not self.parallel) or (getattr(self, "rank", 0) == 0)):
                print(
                    "[AFTER PARTICLE ASSIGN] "
                    f"sum_x={np.sum(self.beam.x):.16e}, "
                    f"sum_px={np.sum(self.beam.px):.16e}, "
                    f"sum_x2={np.sum(self.beam.x*self.beam.x):.16e}, "
                    f"sum_px2={np.sum(self.beam.px*self.beam.px):.16e}, "
                    f"sum_xpx={np.sum(self.beam.x*self.beam.px):.16e}"
                )
            
            # Restore cached beam-derived state first
            # DO NOT restore cached beam-derived state; it is corrupting the beam
            # self._load_beam_cached_state(gb)
            
            # Restore only simple scalar bookkeeping
            if "position" in gb.attrs:
                try:
                    self.beam.position = float(gb.attrs["position"])
                except Exception:
                    pass
            
            if "step" in gb.attrs:
                try:
                    self.beam.step = int(gb.attrs["step"])
                except Exception:
                    pass
                    
            # NEW DEBUG: confirm restart bookkeeping values
            if _CSR_DEBUG and ((not self.parallel) or (getattr(self, "rank", 0) == 0)):
                print(
                    "[LOAD BOOKKEEPING] "
                    f"position={getattr(self.beam, 'position', None)}, "
                    f"step={getattr(self.beam, 'step', None)}"
                )            
            # Recompute all derived beam quantities from the restored particle state
            self.beam.update_status()
            
            if _CSR_DEBUG and ((not self.parallel) or (getattr(self, "rank", 0) == 0)):
                print(
                    "[AFTER UPDATE_STATUS] "
                    f"sum_x={np.sum(self.beam.x):.16e}, "
                    f"sum_px={np.sum(self.beam.px):.16e}, "
                    f"sum_x2={np.sum(self.beam.x*self.beam.x):.16e}, "
                    f"sum_px2={np.sum(self.beam.px*self.beam.px):.16e}, "
                    f"sum_xpx={np.sum(self.beam.x*self.beam.px):.16e}, "
                    f"emit_x_beam={self.beam.twiss['emit_x']:.16e}"
                )

            # ---------------- DEBUG CHECK ----------------
            if _CSR_DEBUG and ((not self.parallel) or (getattr(self, "rank", 0) == 0)):
            
                print(
                    "[LOAD MOMENTS] "
                    f"sum_x={np.sum(self.beam.x):.16e}, "
                    f"sum_px={np.sum(self.beam.px):.16e}, "
                    f"sum_x2={np.sum(self.beam.x*self.beam.x):.16e}, "
                    f"sum_px2={np.sum(self.beam.px*self.beam.px):.16e}, "
                    f"sum_xpx={np.sum(self.beam.x*self.beam.px):.16e}, "
                    f"emit_x_beam={self.beam.twiss['emit_x']:.16e}"
                )
                
            # Recompute derived geometry from restored beam
            # Restore runtime CSR state if present
            if "runtime_state" in grp:
                self._load_runtime_state(grp["runtime_state"])
            
            # Only recompute envelope bounds if they were not restored
            if not hasattr(self, "_envelope_bounds") or self._envelope_bounds is None:
                self._update_envelope_limits_for_source_and_CSR()

            # CSR state
            if "csr_state" in grp:
                gs = grp["csr_state"]
                for k in ["inbend", "afterbend"]:
                    if k in gs.attrs:
                        setattr(self, k, bool(gs.attrs[k]))
                for k in ["formation_length", "R_rec", "phi_rec", "CSR_scaling"]:
                    if k in gs.attrs:
                        setattr(self, k, float(gs.attrs[k]))

            # Statistics
            if "statistics" in grp:
                gst = grp["statistics"]
                for k in gst.keys():
                    item = gst[k]
                    if isinstance(item, h5py.Group):
                        if k not in self.statistics:
                            self.statistics[k] = {}
                        for kk in item.keys():
                            self.statistics[k][kk] = np.asarray(item[kk])
                    else:
                        self.statistics[k] = np.asarray(item)

            # DF tracker
            # DF tracker
            if "DF_tracker" in grp:
                self._load_df_tracker_state(grp["DF_tracker"])
            
            # ---------------- DIRECT SAVE vs LOAD CHECK ----------------
            if (not self.parallel) or (getattr(self, "rank", 0) == 0):
                try:
                    emit_saved = self.statistics['twiss']['emit_x'][completed_step]
                    print(f"[CHECKPOINT STATS] emit_x_saved={emit_saved:.6e}")
                    print(f"[COMPARE] delta_emit_x={self.beam.twiss['emit_x'] - emit_saved:.6e}")
                except Exception as e:
                    print(f"[CHECKPOINT STATS] could not compare emit_x: {e}")
    
        # Determine where to restart in lattice
        ei, li, ename = self._locate_element_for_global_step(next_step)
        self._restart_loaded = True
        self._restart_next_step = next_step
        self._restart_target_ele_index = ei
        self._restart_target_local_step = li
        self._restart_first_step_pending = True
    
        if _CSR_DEBUG and ((not self.parallel) or (getattr(self, "rank", 0) == 0)):
            print(f"   Lattice lookup: element='{ename}' (ele_index={ei}), next_step={next_step}")
    
        return True

    def _save_beam_cached_state(self, grp):
        """
        Save as much already-computed Beam state as possible so restart can avoid
        recomputing derived quantities.
        """
        # Scalars commonly used later in CSR.py
        scalar_names = [
            "_mean_x", "_mean_z",
            "_sigma_x", "_sigma_z",
            "sigma_energy", "mean_energy",
            "init_gamma", "init_energy",
        ]
        for name in scalar_names:
            if hasattr(self.beam, name):
                try:
                    grp.attrs[name] = float(getattr(self.beam, name))
                except Exception:
                    pass
    
        # slope is array-like
        if hasattr(self.beam, "_slope") and getattr(self.beam, "_slope") is not None:
            try:
                grp.create_dataset("_slope", data=np.asarray(self.beam._slope), compression="gzip")
            except Exception:
                pass
    
        # twiss dict if available
        try:
            tw = self.beam.twiss
            gtw = grp.create_group("twiss")
            for k, v in tw.items():
                try:
                    gtw.attrs[k] = float(v)
                except Exception:
                    pass
        except Exception:
            pass
    
    
    def _load_beam_cached_state(self, grp):
        """
        Restore cached Beam state directly, avoiding update_status() when possible.
        Returns True if cached state restoration looks complete enough to skip
        update_status(); otherwise returns False.
        """
        restored_any = False
    
        for name, val in grp.attrs.items():
            if name in [
                "_mean_x", "_mean_z",
                "_sigma_x", "_sigma_z",
                "sigma_energy", "mean_energy",
                "init_gamma", "init_energy",
            ]:
                try:
                    setattr(self.beam, name, float(val))
                    restored_any = True
                except Exception:
                    pass
    
        if "_slope" in grp:
            try:
                self.beam._slope = np.asarray(grp["_slope"])
                restored_any = True
            except Exception:
                pass
    
        if "twiss" in grp:
            try:
                tw = {}
                for k, v in grp["twiss"].attrs.items():
                    tw[k] = float(v)
                # Store on a private cache if Beam uses one; harmless otherwise
                self.beam._twiss = tw
                restored_any = True
            except Exception:
                pass
    
        # Decide whether we have enough to skip update_status()
        needed = ["_mean_x", "_mean_z", "_sigma_x", "_sigma_z", "_slope"]
        have_enough = all(hasattr(self.beam, name) for name in needed)
    
        return bool(restored_any and have_enough)
 
    def _save_runtime_state(self, grp):
        """
        Save runtime-calculated CSR state so restart can resume without rebuilding it.
        """
        # Envelope bounds
        if hasattr(self, "_envelope_bounds") and self._envelope_bounds is not None:
            grp.create_dataset("_envelope_bounds", data=np.asarray(self._envelope_bounds), compression="gzip")
    
        # Kick stats
        if hasattr(self, "_kick_stats") and self._kick_stats is not None:
            gk = grp.create_group("_kick_stats")
            for k, v in self._kick_stats.items():
                try:
                    gk.attrs[k] = float(v)
                except Exception:
                    pass
    
        # CSR mesh / wake metadata
        for name in [
            "CSR_xmesh", "CSR_zmesh",
            "CSR_xrange_transformed", "CSR_zrange",
            "dE_dct", "x_kick",
        ]:
            if hasattr(self, name) and getattr(self, name) is not None:
                try:
                    grp.create_dataset(name, data=np.asarray(getattr(self, name)), compression="gzip")
                except Exception:
                    pass
    
        if hasattr(self, "wake_shape") and getattr(self, "wake_shape", None) is not None:
            try:
                grp.attrs["wake_shape_0"] = int(self.wake_shape[0])
                grp.attrs["wake_shape_1"] = int(self.wake_shape[1])
            except Exception:
                pass
    
        if hasattr(self, "_wake_bounds_lab") and getattr(self, "_wake_bounds_lab", None) is not None:
            try:
                grp.create_dataset("_wake_bounds_lab", data=np.asarray(self._wake_bounds_lab), compression="gzip")
            except Exception:
                pass
    
    
    def _load_runtime_state(self, grp):
        """
        Restore runtime-calculated CSR state directly.
        """
        if "_envelope_bounds" in grp:
            self._envelope_bounds = tuple(np.asarray(grp["_envelope_bounds"]).tolist())
    
        if "_kick_stats" in grp:
            self._kick_stats = {k: float(v) for k, v in grp["_kick_stats"].attrs.items()}
    
        for name in [
            "CSR_xmesh", "CSR_zmesh",
            "CSR_xrange_transformed", "CSR_zrange",
            "dE_dct", "x_kick",
        ]:
            if name in grp:
                setattr(self, name, np.asarray(grp[name]))
    
        if ("wake_shape_0" in grp.attrs) and ("wake_shape_1" in grp.attrs):
            self.wake_shape = (int(grp.attrs["wake_shape_0"]), int(grp.attrs["wake_shape_1"]))
    
        if "_wake_bounds_lab" in grp:
            self._wake_bounds_lab = tuple(np.asarray(grp["_wake_bounds_lab"]).tolist())
  
        
    def _update_envelope_limits_for_source_and_CSR(self, margin=0.05, z_samples=64):
        """
        ONLY computes and stores self._envelope_bounds (and optional debug scalars).
        MUST NOT do restart/load logic, DF deposition, CSR scaling, wake writing, etc.
        """
        import numpy as np
    
        mean_x_lab = float(np.mean(self.beam.x))
        mean_z_lab = float(np.mean(self.beam.z))
        sig_x_now  = float(np.std(self.beam.x))
        sig_z_now  = float(np.std(self.beam.z))
    
        cfg_xlim = float(getattr(self, "_cfg_xlim_lab", self.DF_tracker.xlim))
        cfg_zlim = float(getattr(self, "_cfg_zlim_lab", self.DF_tracker.zlim))
    
        xhalf_base = cfg_xlim * max(sig_x_now, 1e-16)
        zhalf_base = cfg_zlim * max(sig_z_now, 1e-16)
    
        x_base_min, x_base_max = mean_x_lab - xhalf_base, mean_x_lab + xhalf_base
        z_base_min, z_base_max = mean_z_lab - zhalf_base, mean_z_lab + zhalf_base
    
        # Use same definition as grid: x' = x_lab − p(z_lab)
        xprime_particles = self.beam.x - np.polyval(self.beam.slope, self.beam.z)
        
        mean_xp = float(np.mean(xprime_particles))
        sig_xp  = float(np.std(xprime_particles))
    
        kx = float(getattr(self.CSR_params, "kick_xlim", cfg_xlim))
        kz = float(getattr(self.CSR_params, "kick_zlim", cfg_zlim))
    
        xp_min, xp_max   = mean_xp - kx * sig_xp, mean_xp + kx * sig_xp
        z_min_k, z_max_k = mean_z_lab - kz * sig_z_now, mean_z_lab + kz * sig_z_now
    
        pcoef   = self.beam.slope
        z_probe = np.linspace(z_min_k, z_max_k, max(3, int(z_samples)))
        pvals   = np.polyval(pcoef, z_probe)
        pmin, pmax = float(np.min(pvals)), float(np.max(pvals))
    
        xk_min_lab = min(xp_min + pmin, xp_min + pmax, xp_max + pmin, xp_max + pmax)
        xk_max_lab = max(xp_min + pmin, xp_min + pmax, xp_max + pmin, xp_max + pmax)
    
        xmin = min(x_base_min, xk_min_lab)
        xmax = max(x_base_max, xk_max_lab)
        zmin = min(z_base_min, z_min_k)
        zmax = max(z_base_max, z_max_k)
    
        xhalf = max(xmax - mean_x_lab, mean_x_lab - xmin) * (1.0 + margin)
        zhalf = max(zmax - mean_z_lab, mean_z_lab - zmin) * (1.0 + margin)
    
        self._envelope_bounds = (float(mean_x_lab - xhalf), float(mean_x_lab + xhalf),
                                 float(mean_z_lab - zhalf), float(mean_z_lab + zhalf))



    def initialization(self):
        """
        Prepare the run: anchor baseline ROI, compute straightening, and
        ensure the first deposition uses the global LAB envelope.
        """
        import numpy as np
    
        # Anchor baseline LAB half-widths using INITIAL stats
        sig_x0 = float(np.std(self.beam.x))
        sig_z0 = float(np.std(self.beam.z))
        cfg_xlim = float(self.DF_tracker.xlim)
        cfg_zlim = float(self.DF_tracker.zlim)
    
        self._baseline_xhalf_lab = cfg_xlim * max(sig_x0, 1e-16)
        self._baseline_zhalf_lab = cfg_zlim * max(sig_z0, 1e-16)
        self._cfg_xlim_lab = cfg_xlim
        self._cfg_zlim_lab = cfg_zlim
        self._sig_x0 = sig_x0
        self._sig_z0 = sig_z0
    
        self.init_statistics()
    
        self._update_envelope_limits_for_source_and_CSR()
        self.DF_tracker.set_absolute_bounds(self._envelope_bounds)
        self.DF_tracker.get_DF(
            x=self.beam.x, z=self.beam.z,
            px=self.beam.px, pz=self.beam.pz, t=self.beam.position
        )
        self.DF_tracker.append_DF()
        self.DF_tracker.append_interpolant(
            formation_length=float('inf'),
            n_formation_length=self.integration_params.n_formation_length
        )
    
        self.CSR_scaling = 8.98755e3 * self.beam.charge

    def init_statistics(self):
        Nstep = self.lattice.total_steps
        self.statistics = {}
        self.statistics['twiss'] = {'alpha_x': np.zeros(Nstep),
                                    'beta_x': np.zeros(Nstep),
                                    'gamma_x': np.zeros(Nstep),
                                    'emit_x': np.zeros(Nstep),
                                    'eta_x': np.zeros(Nstep),
                                    'etap_x': np.zeros(Nstep),
                                    'norm_emit_x': np.zeros(Nstep),
                                    'alpha_y': np.zeros(Nstep),
                                    'beta_y': np.zeros(Nstep),
                                    'gamma_y': np.zeros(Nstep),
                                    'emit_y': np.zeros(Nstep),
                                    'eta_y': np.zeros(Nstep),
                                    'etap_y': np.zeros(Nstep),
                                    'norm_emit_y': np.zeros(Nstep)}
    
        self.statistics['slope'] = np.zeros((Nstep, 2))
        self.statistics['sigma_x'] = np.zeros(Nstep)
        self.statistics['sigma_z'] = np.zeros(Nstep)
        self.statistics['sigma_energy'] = np.zeros(Nstep)
        self.statistics['mean_x']  = np.zeros(Nstep)
        self.statistics['mean_z'] = np.zeros(Nstep)
        self.statistics['mean_energy'] = np.zeros(Nstep)
    
        self.update_statistics(step = 0)
    
    
        self.inbend = False
        self.afterbend = False
        self.R_rec = None
        self.phi_rec = None
    
    def init_MPI(self):
        self.parallel = True
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        mpi_size = comm.Get_size()
        work_size = self.CSR_params.xbins * self.CSR_params.zbins
        ave, res = divmod(work_size, mpi_size)
        self.count = [ave + 1 if p < res else ave for p in range(mpi_size)]
        displ = [sum(self.count[:p]) for p in range(mpi_size)]
        self.displ = np.array(displ)
    
    def check_input_consistency(self, input):
        # Todo: need modification if dipole_config.yaml format changed
        self.required_inputs = ['input_beam', 'input_lattice']
    
        allowed_params = self.required_inputs + ['particle_deposition', 'distribution_interpolation', 'CSR_integration',
                                                 'CSR_computation']
        for input_param in input:
            assert input_param in allowed_params, f'Incorrect param given to {self.__class__.__name__}.__init__(**kwargs): {input_param}\nAllowed params: {allowed_params}'
    
        # Make sure all required parameters are specified
        for req in self.required_inputs:
            assert req in input, f'Required input parameter {req} to {self.__class__.__name__}.__init__(**kwargs) was not found.'
    
    def get_formation_length(self, R, sigma_z, phi = 0.0, inbend=True):
        if inbend:
            self.formation_length = (24 * (R ** 2) * sigma_z) ** (1 / 3)
        else:
            self.formation_length = (3*R**2*phi**4)/(4*(-6*sigma_z + R*phi**3))
    
    def get_bmadx_element(self, ele,  DL, entrance = False, exit = False):
        input_dic = self.lattice.lattice_config[ele].copy()
        input_dic.pop('nsep')
        L = input_dic.pop('L')
        type = input_dic.pop('type')
    
    
        if type == 'dipole':
            if 'angle' in input_dic.keys():
                angle = input_dic.pop('angle')
                G = angle / L
    
            if 'G' in input_dic.keys():
                G = input_dic.pop('G')
    
            if 'E1' in input_dic.keys():
                E1 = input_dic.pop('E1')
            else:
                E1 = 0
    
            if 'E2' in input_dic.keys():
                E2 = input_dic.pop('E2')
            else:
                E2 = 0
    
            if 'FRINGE_AT' in input_dic.keys():
                FRINGE_AT = input_dic.pop('FRINGE_AT')
    
    
            if entrance and exit:
                element = SBend(L = DL, P0C = self.beam.init_energy, G = G, E1 = E1, E2 = E2, FRINGE_AT = FRINGE_AT, **input_dic)
    
            elif entrance:
                element = SBend(L=DL, P0C=self.beam.init_energy, G=G, E1=E1, E2=0.0, FRINGE_AT = "entrance_end", **input_dic)
    
    
            elif exit:
                element = SBend(L=DL, P0C=self.beam.init_energy, G=G, E1=0.0, E2=E2, FRINGE_AT = "exit_end", **input_dic)
    
            else:
                element = SBend(L=DL, P0C=self.beam.init_energy, G=G, E1=0.0, E2=0.0, FRINGE_AT = "no_end", **input_dic)
    
        elif type == 'drift':
            element = Drift(L = DL)
    
        elif type == 'quad':
            K1 = input_dic.pop('K1')
            element = Quadrupole(L=DL, K1=K1, **input_dic)
    
        elif type == 'sextupole':
            K2 = input_dic.pop('K2')
            element = Sextupole(L=DL, K2=K2, **input_dic)
        if _CSR_DEBUG:
            print(element)
        return element
    
    #    @profile
    


    
    def run(self, stop_time = None, debug = False):
        #print("DEBUG: ENTERED RUN METHOD - THIS SHOULD ALWAYS PRINT")
        if (not self.parallel) or (self.rank == 0):
            print('Starting the DFCSR run')
    
        step_count = 1
        stop_step = getattr(self.CSR_params, "stop_step", None)
        ckpt_save_steps = set()  # populated if restart.save is enabled
        self._run_wall_start = time.time()
        
        # --- restart / checkpoint load (rank0 only) ---
        if bool(getattr(self, 'restart_enabled', False)) and bool(getattr(self, 'restart_load', False)):
            loaded = self.load_checkpoint()
            if loaded:
                step_count = int(self._restart_next_step)
                print(f"[RESTART CHECK] beam.position={self.beam.position:.6f}, "
                      f"inbend={self.inbend}, formation_length={self.formation_length:.4f}, "
                      f"len(DF_log)={len(self.DF_tracker.DF_log)}, "
                      f"len(time_log)={len(self.DF_tracker.time_log)}, "
                      f"time_log[-1]={list(self.DF_tracker.time_log)[-1]:.6f}, "
                      f"end_time={getattr(self.DF_tracker, 'end_time', 'NOT SET')}, "
                      f"sigma_x_interp={self.DF_tracker.sigma_x_interp}, "
                      f"emit_x[{step_count-1}]={self.statistics['twiss']['emit_x'][step_count-1]:.6e}")
    
    
        # --- restart: wakes-range bookkeeping (only meaningful if we actually loaded) ---
        if getattr(self, "_restart_loaded", False):
            s0 = int(self._restart_next_step)
            rng = int(getattr(self, "restart_save_wakes_range", 0) or 0)
            if rng > 0:
                self._restart_wake_start_step = s0
                self._restart_wake_end_step = s0 + rng
                ckpt_group = getattr(self, "restart_checkpoint_group", None) or f"step{s0}"
                csr_suffix = "CSRon" if bool(getattr(self.CSR_params, "apply_CSR", 1)) else "CSRoff"
                self._restart_wake_prefix = f"restart_{ckpt_group}_{csr_suffix}_{self.prefix}"
    
        # --- restart: compute checkpoint save steps in dipoles ---
        ckpt_save_steps = set()
        if getattr(self, "restart_enabled", False) and getattr(self, "restart_save", False):
            ckpt_save_steps = self._compute_ckpt_save_steps()
        DL = self.lattice.step_size
        ele_count = 0
        skip_ele = False
        ele_prev = None
        type_prev = None

        elems = list(self.lattice.lattice_config.keys())[1:]
        start_ele_index = int(getattr(self, '_restart_target_ele_index', 0) or 0) if getattr(self, '_restart_loaded', False) else 0
        
        # If we restarted mid-lattice, set ele_prev so boundary handling can reference it safely
        if getattr(self, '_restart_loaded', False):
            if start_ele_index > 0:
                ele_prev = elems[start_ele_index - 1]
                try:
                    type_prev = self.lattice.lattice_config[ele_prev].get('type', None)
                except Exception:
                    type_prev = None

        if not getattr(self, '_restart_loaded', False):
            self.inbend = False
            self.afterbend = False
            self.formation_length = 0.0
    
        # Save initial particle cloud and dipole dumps only if restart is NOT enabled
        _do_particle_dumps = (
            not getattr(self, 'restart_enabled', False)
            and getattr(self.CSR_params, 'write_beam', None)
        )
        if _do_particle_dumps:
            self.dump_beam(label='initial', step=0, element='initial', location='initial')
        dump_map = self._compute_dipole_dump_map() if _do_particle_dumps else {}
    



        for ele_count, ele in enumerate(elems[start_ele_index:], start=start_ele_index):
    
    
            self.lattice.update(ele)
            # Todo: add sextupole, maybe Bmad Tracking?
    
            # -----------------------load current lattice params-----------------#
            # Pre-process the lattice params
            L = self.lattice.lattice_config[ele]['L']
            type = self.lattice.lattice_config[ele]['type']
            steps = self.lattice.steps_per_element[ele_count]
            R = float('inf')
    
            ####### A step over the boundary of the elements, deal with the part of the step in the previous element
            _is_restart_ele = (getattr(self, '_restart_loaded', False) and
                               ele_count == start_ele_index)
            if (not skip_ele) and ele_count > 0 and not _is_restart_ele:
                DL_1 = self.lattice.distance[ele_count - 1] - self.beam.position
                
                #Todo: Bmadx seems to have some problems when DL is very
                if DL_1 > 1.0e-6:
                # calculate the part in the previous element
                    element = self.get_bmadx_element(ele=ele_prev, DL=DL_1, exit=True)
                    self.beam.track(element, DL_1, update_step=False)
                else:
                    DL_1 = 0.0
            # If no steps inside an element
            if steps == 0:    #If one step over the whole element
                skip_ele = True
                element = self.get_bmadx_element(ele=ele,  DL=L, exit=True, entrance = True)
                self.beam.track(element, L, update_step=False)
    
            if type == 'dipole':
                angle = self.lattice.lattice_config[ele]['angle']
                R = L / angle
    
                self.inbend = True
                self.afterbend = True
                self.R_rec = R
                self.phi_rec = angle
    
                if not _is_restart_ele:
                    self.get_formation_length(R=R, sigma_z=5*self.beam.sigma_z, inbend = True)
    
            else:  # If not in a bend
                self.inbend = False
    
                if not _is_restart_ele:
                    if self.afterbend:
                        self.get_formation_length(R=self.R_rec, sigma_z=5 * self.beam.sigma_z, inbend=True)
    
                    else:  # if it is the first drift in the lattice
                        self.formation_length += L
    
            distance_in_current_ele = 0.0
            # -----------------------tracking---------------------------------
            step_start = 0
            if getattr(self, '_restart_loaded', False) and (ele_count == start_ele_index):
                step_start = int(getattr(self, '_restart_target_local_step', 0) or 0)
            for step in range(step_start, steps):
                time0  = time.time()
                # NEW: keep Beam.step consistent with global step_count
                self.beam.step = int(step_count)    

                # Deal with boundary condition. A step over the boundary of two adjacent elements
                if (step == 0) and (ele_count > 0):
                    # If enter a new element, split the step
    
                    DL_2 = self.lattice._positions_record[step_count] - self.lattice.distance[ele_count - 1]
    
                    # calculate the part in the new element
                    element = self.get_bmadx_element(ele = ele,  DL = DL_2, entrance = True)
                    self.beam.track(element, DL_2)
                    self.beam.step = int(step_count)
                    distance_in_current_ele += DL_2
                    skip_ele = False    # Reset the flag
    
                else:
                    element = self.get_bmadx_element(ele = ele,  DL = DL)
                    # Propagate beam for one step
                    self.beam.track(element, DL)
                    self.beam.step = int(step_count)
                    distance_in_current_ele += DL
    
                if debug or self.CSR_params.compute_CSR:
                    self._update_envelope_limits_for_source_and_CSR()
                    self.DF_tracker.set_absolute_bounds(self._envelope_bounds)
                    self.DF_tracker.get_DF(x=self.beam.x, z=self.beam.z,
                                           px=self.beam.px, pz=self.beam.pz, t=self.beam.position)
                    self.DF_tracker.append_DF()
                    self.DF_tracker.append_interpolant(
                        formation_length=self.formation_length,
                        n_formation_length=self.integration_params.n_formation_length)
                    self.DF_tracker.build_interpolant()
                    # _restart_first_step_pending no longer needed: sigma_x_interp is
                    # correctly saved/restored from checkpoint, so append_interpolant
                    # behaves identically to the save run without any forcing.
                    if getattr(self, '_restart_first_step_pending', False):
                        self._restart_first_step_pending = False
                        
      
                # Geometry guard: suppress physics CSR and CNN (via CSR_blocker)
                # when geometry makes CSR meaningless. use_cnn (line 3726), physics
                # CSR, and kick application (line 3841) all live inside the
                # compute_CSR and not CSR_blocker block, so both physics and CNN
                # are co-gated automatically.
                #
                # Case 1: not yet in or after any bend — density history is empty,
                #         CSR source is zero regardless of what is computed.
                # Case 2: far past last bend — wakes have decayed beyond relevance.
                _not_in_or_after_bend = (
                    not getattr(self, 'inbend', False) and
                    not getattr(self, 'afterbend', False)
                )
                _far_from_bend = (
                    getattr(self, 'afterbend', False) and
                    not getattr(self, 'inbend', False) and
                    self.formation_length is not None and
                    distance_in_current_ele > 3 * self.formation_length
                )
                if _not_in_or_after_bend or _far_from_bend:
                    CSR_blocker = True
                    if (not self.parallel) or (self.rank == 0):
                        _reason = (
                            "not in/after any bend"
                            if _not_in_or_after_bend
                            else f"far from bend "
                                 f"(dist={distance_in_current_ele:.3f} > "
                                 f"3×FL={3*self.formation_length:.3f})"
                        )
                        print(f"[step {step_count}] CSR_blocker=True ({_reason}), "
                              f"skipping CSR")
                else:
                    CSR_blocker = False
                
                # DEBUG STATEMENTS START HERE
                '''
                #print("DEBUG: Step {}, checking CSR conditions:".format(step_count))
                #print("DEBUG: self.CSR_params.compute_CSR = {}".format(self.CSR_params.compute_CSR))
                #print("DEBUG: CSR_blocker = {}".format(CSR_blocker))
                #print("DEBUG: Combined condition (compute_CSR and not CSR_blocker) = {}".format(self.CSR_params.compute_CSR and (not CSR_blocker)))
                '''
                wrote_step_output = False

                if self.CSR_params.compute_CSR and (not CSR_blocker):
                    '''
                    #print("DEBUG: Entered CSR computation block for step {}".format(step_count))
                    #print("DEBUG: step = {}, lattice.nsep[ele_count] = {}".format(step, self.lattice.nsep[ele_count]))
                    #print("DEBUG: step % nsep = {}".format(step % self.lattice.nsep[ele_count]))
                    #print("DEBUG: Condition (step % nsep == 0) = {}".format(step % self.lattice.nsep[ele_count] == 0))
                    '''
                    if step % self.lattice.nsep[ele_count] == 0:
                        # Build CSR mesh on exact LAB rectangle
                        self.get_CSR_mesh()

                   
                        if self.parallel:
                            self.calculate_2D_CSR_parallel()
                        else:
                            self.calculate_2D_CSR()

                              
                        # Apply CSR kick to the beam
                        if self.CSR_params.apply_CSR:
                            nx, nz = self.wake_shape
                            dE2d = np.asarray(self.dE_dct).reshape(nx, nz)
                            XK2d = np.asarray(self.x_kick).reshape(nx, nz)

                            x_axis = self.CSR_xrange_transformed
                            z_axis = self.CSR_zrange
                            mean_xp = self._kick_stats["mean_xp"]
                            sig_xp  = self._kick_stats["sig_xp"]
                            mean_z  = self._kick_stats["mean_z"]
                            sig_z   = self._kick_stats["sig_z"]

                            tiny = 1e-16
                            pvals = np.polyval(self.beam.slope, z_axis)
                            Xprime_grid = x_axis[:, None] - pvals[None, :]
                            mx = (Xprime_grid - mean_xp) / max(sig_xp, tiny)
                            mz = (z_axis - mean_z) / max(sig_z, tiny)
                            gate_mask = (np.abs(mx) <= self.CSR_params.kick_xlim) & \
                                        (np.abs(mz)[None, :] <= self.CSR_params.kick_zlim)
                            print(f"  gate_mask: {gate_mask.sum()}/{gate_mask.size} active | "
                                  f"sig_xp={sig_xp:.3e} sig_z={sig_z:.3e} kick_xlim={self.CSR_params.kick_xlim}")
                            dE_apply = dE2d * gate_mask
                            XK_apply = XK2d * gate_mask if int(self.CSR_params.transverse_on) else XK2d

                            # ── DEBUG BLOCK 4: beam state before/after kick ────
                            self.beam.apply_wakes(
                                dE_apply, XK_apply,
                                x_axis, z_axis,
                                DL * self.lattice.nsep[ele_count],
                                self.CSR_params.transverse_on
                            )
                                  
                        # Write wakes if requested
                        if getattr(self.CSR_params, 'write_wakes', False):
                            restart_range_active = (
                                getattr(self, '_restart_wake_start_step', None) is not None
                                and getattr(self, '_restart_wake_end_step', None) is not None
                                and int(getattr(self, 'restart_save_wakes_range', 0) or 0) > 0
                            )

                            if restart_range_active:
                                if (step_count >= self._restart_wake_start_step) and (step_count < self._restart_wake_end_step):
                                    self.write_wakes(
                                        save_fields=True,
                                        save_xz=bool(getattr(self, "restart_save_xz", False)),
                                        save_xpx=bool(getattr(self, "restart_save_xpx", False))
                                    )
                                    wrote_step_output = True
                            else:
                                self.write_wakes(
                                    save_fields=True,
                                    save_xz=bool(getattr(self, "restart_save_xz", False)),
                                    save_xpx=bool(getattr(self, "restart_save_xpx", False))
                                )
                                wrote_step_output = True


                        current_step_size = DL * self.lattice.nsep[ele_count]

                # NEW: if compute_CSR is off (or no field output was written), still write restart-range output with metadata/xz only
                if (not wrote_step_output) and getattr(self.CSR_params, 'write_wakes', False):
                    restart_range_active = (
                        getattr(self, '_restart_wake_start_step', None) is not None
                        and getattr(self, '_restart_wake_end_step', None) is not None
                        and int(getattr(self, 'restart_save_wakes_range', 0) or 0) > 0
                    )

                    if restart_range_active:
                        if (step_count >= self._restart_wake_start_step) and (step_count < self._restart_wake_end_step):
                            self.write_wakes(
                                save_fields=False,
                                save_xz=bool(getattr(self, "restart_save_xz", False)),
                                save_xpx=bool(getattr(self, "restart_save_xpx", False))
                            )
    
                # NEW: save 6D particles at dipole start/mid/end steps
                info = dump_map.get(step_count)
                if info is not None:
                    ele_name, where = info
                    self.dump_beam(label=f'{ele_name}_{where}_step{step_count}',
                                   step=step_count, element=ele_name, location=where)
    
                
    
                # recording statistics at each step
                self.update_statistics(step=step_count)
                
                if not self.parallel or self.rank == 0:
                    print("Finish step {}, s = {},  in {} seconds".format(
                        step_count, self.beam.position, time.time() - time0
                    ))
                
                # checkpoint save block stays here
                if getattr(self, 'restart_enabled', False) and getattr(self, 'restart_save', False):
                    if step_count in ckpt_save_steps:
                        self._replace_last_df_state_with_current()
                
                        if _CSR_DEBUG and ((not self.parallel) or (getattr(self, "rank", 0) == 0)):
                            print(
                                "[SAVE MOMENTS] "
                                f"sum_x={np.sum(self.beam.x):.16e}, "
                                f"sum_px={np.sum(self.beam.px):.16e}, "
                                f"sum_x2={np.sum(self.beam.x*self.beam.x):.16e}, "
                                f"sum_px2={np.sum(self.beam.px*self.beam.px):.16e}, "
                                f"sum_xpx={np.sum(self.beam.x*self.beam.px):.16e}, "
                                f"emit_x_beam={self.beam.twiss['emit_x']:.16e}, "
                                f"emit_x_saved={self.statistics['twiss']['emit_x'][step_count]:.16e}"
                            )
                
                        self.save_checkpoint(completed_step=step_count, ele_name=ele)
                
                # stop by step count after this step is fully completed/saved
                if stop_step is not None and step_count >= int(stop_step):
                    if _CSR_DEBUG and ((not self.parallel) or (getattr(self, "rank", 0) == 0)):
                        print(f"Stopping early at step {step_count} due to CSR_computation.stop_step={stop_step}")
                    self.write_statistics()
                    return
                # existing position-based stop
                step_count += 1
                if stop_time and self.beam.position > stop_time:
                    self.write_statistics()
                    return
            ele_prev = ele
            type_prev = type
            # ele_count is driven by enumerate(elems, ...)
        self.write_statistics()

    
    
    def get_CSR_mesh(self):
        """
        Build CSR observation mesh directly in the LAB frame on the exact
        envelope bounds stored in self._envelope_bounds = (xmin_lab, xmax_lab, zmin_lab, zmax_lab).
    
        Side effects:
          - self.CSR_xmesh, self.CSR_zmesh : flattened LAB coordinates at mesh points
          - self.CSR_xrange_transformed     : 1D LAB x-axis (length xbins)  [name kept for compat]
          - self.CSR_zrange                 : 1D LAB z-axis (length zbins)
          - self.wake_shape                 : (xbins, zbins)
          - self._wake_bounds_lab           : (xmin_lab, xmax_lab, zmin_lab, zmax_lab)
          - self._kick_stats                : dict of current means/sigmas (for gating)
        """
        import numpy as np
    
        # Current stats (for gating/debug only)
        xprime = self.beam.x_transform
        mean_xp = float(np.mean(xprime))
        sig_xp  = float(np.std(xprime))
        mean_z  = float(np.mean(self.beam.z))
        sig_z   = float(np.std(self.beam.z))
        self._kick_stats = dict(mean_xp=mean_xp, sig_xp=sig_xp, mean_z=mean_z, sig_z=sig_z)
    
        # Envelope bounds (required; fall back to a σ-window only if truly absent)
        # Envelope bounds (preferred). If absent, fall back to a LAB σ-window using LAB sigmas.
        if not hasattr(self, "_envelope_bounds"):
            mean_x_lab = float(np.mean(self.beam.x))
            mean_z_lab = float(np.mean(self.beam.z))
            sig_x_lab  = float(np.std(self.beam.x))
            sig_z_lab  = float(np.std(self.beam.z))
        
            xmin_lab = mean_x_lab - float(self.CSR_params.xlim) * sig_x_lab
            xmax_lab = mean_x_lab + float(self.CSR_params.xlim) * sig_x_lab
            zmin_lab = mean_z_lab - float(self.CSR_params.zlim) * sig_z_lab
            zmax_lab = mean_z_lab + float(self.CSR_params.zlim) * sig_z_lab
        else:
            xmin_lab, xmax_lab, zmin_lab, zmax_lab = self._envelope_bounds
    
        # 1D LAB axes on the envelope rectangle
        x_axis = np.linspace(xmin_lab, xmax_lab, int(self.CSR_params.xbins))
        z_axis = np.linspace(zmin_lab, zmax_lab, int(self.CSR_params.zbins))
    
        # Rectangular LAB mesh (no shear here; kernel can reconstruct x' later as x - p(z))
        Xlab, Z = np.meshgrid(x_axis, z_axis, indexing='ij')  # shapes (xbins, zbins)
    
        # Flatten LAB coordinates for kernel evaluation
        self.CSR_xmesh = Xlab.ravel(order='C')
        self.CSR_zmesh = Z.ravel(order='C')
    
        # Cache axes and shape (NOTE: x axis is LAB; name kept for compat)
        self.CSR_xrange_transformed = x_axis
        self.CSR_zrange             = z_axis
        self.wake_shape             = (len(x_axis), len(z_axis))
    
        # Record the LAB bounds actually used (matches envelope exactly)
        self._wake_bounds_lab = (float(x_axis[0]), float(x_axis[-1]),
                                 float(z_axis[0]), float(z_axis[-1]))
    
    
    



    
#    @profile
    def calculate_2D_CSR(self):

        N = self.CSR_params.xbins*self.CSR_params.zbins
        self.dE_dct = np.zeros((N,))
        self.x_kick = np.zeros((N,))

        start_time = time.time()
        for i in range(N):

            #if i == 210:
            #    print(i)

            #if i%int(N//10) == 0:
            #    print('Complete', str(np.round(i/N*100,2)), '%')

            s = self.beam.position + self.CSR_zmesh[i]
            x = self.CSR_xmesh[i]

            self.dE_dct[i], self.x_kick[i] = self.get_CSR_wake(s,x)

        self.dE_dct = self.dE_dct.reshape((self.CSR_params.xbins, self.CSR_params.zbins))
        self.x_kick = self.x_kick.reshape((self.CSR_params.xbins, self.CSR_params.zbins))

    def calculate_2D_CSR_parallel(self):
        work_size= self.CSR_params.xbins * self.CSR_params.zbins
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        start = int(self.displ[rank])
        local_size = int(self.count[rank])

        self.dE_dct = np.zeros((work_size,))
        self.x_kick = np.zeros((work_size,))

        dE_dct_local = np.zeros((local_size,))
        x_kick_local = np.zeros(local_size, )

        start_time = time.time()
        for i in range(local_size):
            k  = start + i
            # if i == 210:
            #    print(i)

            # if i%int(N//10) == 0:
            #    print('Complete', str(np.round(i/N*100,2)), '%')

            s = self.beam.position + self.CSR_zmesh[k]
            x = self.CSR_xmesh[k]

            dE_dct_local[i], x_kick_local[i] = self.get_CSR_wake(s,x)

        comm.Allgatherv(dE_dct_local, [self.dE_dct, self.count, self.displ, MPI.DOUBLE])
        comm.Allgatherv(x_kick_local, [self.x_kick, self.count, self.displ, MPI.DOUBLE])

        self.dE_dct = self.dE_dct.reshape((self.CSR_params.xbins, self.CSR_params.zbins))
        self.x_kick = self.x_kick.reshape((self.CSR_params.xbins, self.CSR_params.zbins))

#    @profile
    def get_CSR_wake(self, s, x, debug = False):

        t = self.beam.position

        # ── DEBUG BLOCK 5: first call per step only ────────────────────────
        _dbg_cur_step = int(getattr(self.beam, 'step', -1))
        if getattr(self, '_dbg_wake_printed_step', -1) != _dbg_cur_step:
            self._dbg_wake_printed_step = _dbg_cur_step
            print(f"[DBG5 get_CSR_wake step={_dbg_cur_step}] "
                  f"beam.position={t:.4f} | s={s:.4f} | "
                  f"len(DF_log)={len(getattr(self.DF_tracker,'DF_log',[]))} | "
                  f"len(interpolant_log)={len(getattr(self.DF_tracker,'interpolant_log',[]))} | "
                  f"formation_length={getattr(self,'formation_length',float('nan')):.4f} | "
                  f"inbend={self.inbend} | sigma_z={self.beam._sigma_z:.4e}")

        sigma_z = self.beam._sigma_z
        sigma_x = self.beam._sigma_x
        tan_theta = self.beam._slope[0]

        #TODO： why?
        x0 = (s-t)*self.beam._slope[0]
        xmean = self.beam._mean_x

        
        ######### For Debug ########################################################## 
        if np.abs(tan_theta) <= 1:  # if theta <45 degre, the chirp band can be ignored. theta is the angle in z-x plane
            ignore_vx = False
        else:
            ignore_vx = False

        ############################################################################

        chirp_band = False

        if np.abs(tan_theta) <= 1:  # if chirp is small, the chirp band can be ignored. theta is the angle in z-x plane
            s2 = s - 500 * sigma_z
            s3 = s - 20*sigma_z
            s4 = s + 5 * sigma_z
            x1_w = x0 - 20 * sigma_x
            x2_w = x0 + 20 * sigma_x

            x1_n = x0 - 10 * sigma_x
            x2_n = x0 + 10 * sigma_x

        else:
            chirp_band = True
            if tan_theta > 0:
                tan_alpha = -2 * tan_theta / (1 - tan_theta ** 2)  # alpha = pi - 2 theta, tan_alpha > 0
                d = (10 * sigma_x + xmean - x) / tan_alpha
                
                s4 = s + 3 * sigma_z
                s3 = np.max((0, s - d))
                s2 = s3 - 200 * sigma_z

                # area 1
                x1_l = x + 0.1 * sigma_x
                x1_r = x + 10 * sigma_x
        
                # area 2
                x2_l = x - 3 * sigma_x
                x2_r = x1_l

                # area 3
                x3_l = x0 - 5 * sigma_x
                x3_r = x0 + 5 * sigma_x

                x4_l = x0 - 20 * sigma_x
                x4_r = x0 + 20 * sigma_x


            else:
                tan_alpha = 2 * tan_theta / (1 - tan_theta ** 2)
                d = -(xmean - x - 10 * sigma_x) / tan_alpha
                
                s4 = s + 3 * sigma_z
                s3 = np.max((0, s - d))
                s2 = s3 - 200 * sigma_z

                # area 1
                x1_l = x - 10 * sigma_x
                x1_r = x - 1 * sigma_x
                
                # area 2
                x2_l = x1_r
                x2_r = x + 3 *sigma_x
  
                # area 3
                x3_l = x0 - 5 * sigma_x
                x3_r = x0 + 5 * sigma_x

                x4_l = x0 - 20 * sigma_x
                x4_r = x0 + 20 * sigma_x
        
        s1 = np.max((0, s2 - self.integration_params.n_formation_length * self.formation_length))
       
        if chirp_band:
            sp1 = np.linspace(s1, s2, self.integration_params.zbins)
            sp2 = np.linspace(s2, s3, self.integration_params.zbins)
            sp3 = np.linspace(s3, s4, self.integration_params.zbins)
            xp1 = np.linspace(x1_l, x1_r, self.integration_params.xbins)
            xp2 = np.linspace(x2_l, x2_r, self.integration_params.xbins)
            xp3 = np.linspace(x3_l, x3_r, self.integration_params.xbins)
            xp4 = np.linspace(x4_l, x4_r, 2*self.integration_params.xbins)

            [xp_mesh1, sp_mesh1] = np.meshgrid(xp4, sp1, indexing='ij')
            [xp_mesh2, sp_mesh2] = np.meshgrid(xp3, sp2, indexing = 'ij')
            [xp_mesh3, sp_mesh3] = np.meshgrid(xp1, sp3, indexing='ij')
            [xp_mesh4, sp_mesh4] = np.meshgrid(xp2, sp3, indexing='ij')

            CSR_integrand_z1, CSR_integrand_x1 = self.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh1, sp=sp_mesh1, ignore_vx = ignore_vx)
            dE_dct1 = -self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_z1, x=xp4, axis=0), x=sp1)
            x_kick1 = self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_x1, x=xp4, axis=0), x=sp1)

            CSR_integrand_z2, CSR_integrand_x2 = self.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh2, sp=sp_mesh2, ignore_vx = ignore_vx)
            dE_dct2 = -self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_z2, x=xp3, axis=0), x=sp2)
            x_kick2 = self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_x2, x=xp3, axis=0), x=sp2)

            CSR_integrand_z3, CSR_integrand_x3 = self.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh3, sp=sp_mesh3, ignore_vx = ignore_vx)
            dE_dct3 = -self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_z3, x=xp1, axis=0), x=sp3)
            x_kick3 = self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_x3, x=xp1, axis=0), x=sp3)

            CSR_integrand_z4, CSR_integrand_x4 = self.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh4, sp=sp_mesh4, ignore_vx = ignore_vx)
            dE_dct4 = -self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_z4, x=xp2, axis=0), x=sp3)
            x_kick4 = self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_x4, x=xp2, axis=0), x=sp3)

            if debug:
                return xp1, xp2, xp3, xp4, sp1, sp2, sp3,  CSR_integrand_z1, CSR_integrand_x1, CSR_integrand_z2, CSR_integrand_x2, CSR_integrand_z3, CSR_integrand_x3, CSR_integrand_z4, CSR_integrand_x4
            else:
                return dE_dct1 + dE_dct2 + dE_dct3 + dE_dct4, x_kick1 + x_kick2 + x_kick3 + x_kick4

        else:
            sp1 = np.linspace(s1, s2, self.integration_params.zbins)
            sp2 = np.linspace(s2, s3, self.integration_params.zbins)
            sp3 = np.linspace(s3, s4, self.integration_params.zbins)
            xp_w = np.linspace(x1_w, x2_w, 2*self.integration_params.xbins)
            xp_n = np.linspace(x1_n, x2_n, self.integration_params.xbins)

            [xp_mesh1, sp_mesh1] = np.meshgrid(xp_w, sp1, indexing='ij')
            [xp_mesh2, sp_mesh2] = np.meshgrid(xp_n, sp2, indexing='ij')
            [xp_mesh3, sp_mesh3] = np.meshgrid(xp_n, sp3, indexing='ij')

            CSR_integrand_z1, CSR_integrand_x1 = self.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh1, sp=sp_mesh1, ignore_vx = ignore_vx)
            dE_dct1 = -self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_z1, x=xp_w, axis=0), x=sp1)
            x_kick1 = self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_x1, x=xp_w, axis=0), x=sp1)

            CSR_integrand_z2, CSR_integrand_x2 = self.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh2, sp=sp_mesh2, ignore_vx = ignore_vx)
            dE_dct2 = -self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_z2, x=xp_n, axis=0), x=sp2)
            x_kick2 = self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_x2, x=xp_n, axis=0), x=sp2)

            CSR_integrand_z3, CSR_integrand_x3 = self.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh3, sp=sp_mesh3, ignore_vx = ignore_vx)
            dE_dct3 = -self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_z3, x=xp_n, axis=0), x=sp3)
            x_kick3 = self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_x3, x=xp_n, axis=0), x=sp3)
            
            if debug:
                return xp_w, xp_n,  sp1, sp2, sp3, CSR_integrand_z1, CSR_integrand_x1,CSR_integrand_z2, CSR_integrand_x2,CSR_integrand_z3, CSR_integrand_x3
            else:
                return dE_dct1 + dE_dct2 + dE_dct3, x_kick1 + x_kick2 + x_kick3
          
          
    def get_CSR_integrand(self,s ,x, t, sp, xp, ignore_vx = False):

        #vx = self.DF_tracker.F_vx([t, x, s - t])
        vx = interpolate3D(xval=np.array([t]), yval=np.array([x]), zval=np.array([s-t]),
                             data=self.DF_tracker.data_vx_interp,
                             min_x=self.DF_tracker.min_x, min_y=self.DF_tracker.min_y,
                             min_z=self.DF_tracker.min_z,
                             delta_x=self.DF_tracker.delta_x, delta_y=self.DF_tracker.delta_y,
                             delta_z=self.DF_tracker.delta_z)[0]

        sp_flat = sp.ravel()
        xp_flat = xp.ravel()


        X0_s = interpolate1D(xval = np.array([s]), data = self.lattice.coords[:, 0], min_x = self.lattice.min_x,
                             delta_x = self.lattice.delta_x)[0]
        X0_sp = interpolate1D(xval = sp_flat, data = self.lattice.coords[:, 0], min_x = self.lattice.min_x,
                              delta_x = self.lattice.delta_x)
        Y0_s = interpolate1D(xval = np.array([s]), data = self.lattice.coords[:, 1], min_x = self.lattice.min_x,
                             delta_x = self.lattice.delta_x)[0]
        Y0_sp = interpolate1D(xval = sp_flat, data = self.lattice.coords[:, 1], min_x = self.lattice.min_x,
                              delta_x = self.lattice.delta_x)
        n_vec_s_x = interpolate1D(xval = np.array([s]), data = self.lattice.n_vec[:, 0], min_x = self.lattice.min_x,
                                  delta_x = self.lattice.delta_x)[0]
        n_vec_sp_x =interpolate1D(xval = sp_flat, data = self.lattice.n_vec[:, 0], min_x = self.lattice.min_x,
                                  delta_x = self.lattice.delta_x)
        n_vec_s_y = interpolate1D(xval=np.array([s]), data=self.lattice.n_vec[:, 1], min_x=self.lattice.min_x,
                                  delta_x=self.lattice.delta_x)[0]
        n_vec_sp_y = interpolate1D(xval=sp_flat, data=self.lattice.n_vec[:, 1], min_x=self.lattice.min_x,
                                   delta_x=self.lattice.delta_x)
        tau_vec_s_x = interpolate1D(xval=np.array([s]), data=self.lattice.tau_vec[:, 0], min_x=self.lattice.min_x,
                                  delta_x=self.lattice.delta_x)[0]
        tau_vec_sp_x = interpolate1D(xval=sp_flat, data=self.lattice.tau_vec[:, 0], min_x=self.lattice.min_x,
                                   delta_x=self.lattice.delta_x)
        tau_vec_s_y = interpolate1D(xval=np.array([s]), data=self.lattice.tau_vec[:, 1], min_x=self.lattice.min_x,
                                  delta_x=self.lattice.delta_x)[0]
        tau_vec_sp_y = interpolate1D(xval=sp_flat, data=self.lattice.tau_vec[:, 1], min_x=self.lattice.min_x,
                                   delta_x=self.lattice.delta_x)


        r_minus_rp_x = X0_s - X0_sp + x * n_vec_s_x - xp_flat * n_vec_sp_x
        r_minus_rp_y = Y0_s - Y0_sp + x * n_vec_s_y - xp_flat * n_vec_sp_y
        r_minus_rp = np.sqrt(r_minus_rp_x**2 + r_minus_rp_y**2)


        #rho_sp = self.lattice.F_rho(sp_flat)
        rho_sp = np.zeros(sp_flat.shape)
        for count in range(self.lattice.Nelement):
            if count == 0:
                rho_sp[sp_flat < self.lattice.distance[count]] = self.lattice.rho[count]
            else:
                rho_sp[(sp_flat < self.lattice.distance[count]) & (sp_flat >= self.lattice.distance[count - 1])] = self.lattice.rho[count]

        t_ret = t - r_minus_rp

        #density_ret = self.DF_tracker.F_density(np.array([t_ret, xp_flat, sp_flat - t_ret]).T)
        #density_x_ret = self.DF_tracker.F_density_x(np.array([t_ret, xp_flat, sp_flat- t_ret]).T)
        #density_z_ret = self.DF_tracker.F_density_z(np.array([t_ret, xp_flat, sp_flat- t_ret]).T)
        #vx_ret = self.DF_tracker.F_vx(np.array([t_ret, xp_flat, sp_flat- t_ret]).T)
        #vx_x_ret = self.DF_tracker.F_vx_x(np.array([t_ret, xp_flat, sp_flat- t_ret]).T)

        density_ret = interpolate3D(xval = t_ret, yval = xp_flat, zval = sp_flat - t_ret,
                                  data = self.DF_tracker.data_density_interp,
                                  min_x = self.DF_tracker.min_x, min_y = self.DF_tracker.min_y,  min_z = self.DF_tracker.min_z,
                                  delta_x = self.DF_tracker.delta_x, delta_y = self.DF_tracker.delta_y, delta_z = self.DF_tracker.delta_z)

        density_x_ret = interpolate3D(xval=t_ret, yval=xp_flat, zval=sp_flat - t_ret,
                                  data=self.DF_tracker.data_density_x_interp,
                                  min_x=self.DF_tracker.min_x, min_y=self.DF_tracker.min_y, min_z=self.DF_tracker.min_z,
                                  delta_x=self.DF_tracker.delta_x, delta_y=self.DF_tracker.delta_y,
                                  delta_z=self.DF_tracker.delta_z)

        density_z_ret = interpolate3D(xval=t_ret, yval=xp_flat, zval=sp_flat - t_ret,
                                    data=self.DF_tracker.data_density_z_interp,
                                    min_x=self.DF_tracker.min_x, min_y=self.DF_tracker.min_y,
                                    min_z=self.DF_tracker.min_z,
                                    delta_x=self.DF_tracker.delta_x, delta_y=self.DF_tracker.delta_y,
                                    delta_z=self.DF_tracker.delta_z)

        vx_ret = interpolate3D(xval=t_ret, yval=xp_flat, zval=sp_flat - t_ret,
                                    data=self.DF_tracker.data_vx_interp,
                                    min_x=self.DF_tracker.min_x, min_y=self.DF_tracker.min_y,
                                    min_z=self.DF_tracker.min_z,
                                    delta_x=self.DF_tracker.delta_x, delta_y=self.DF_tracker.delta_y,
                                    delta_z=self.DF_tracker.delta_z)

        vx_x_ret = interpolate3D(xval=t_ret, yval=xp_flat, zval=sp_flat - t_ret,
                             data=self.DF_tracker.data_vx_x_interp,
                             min_x=self.DF_tracker.min_x, min_y=self.DF_tracker.min_y,
                             min_z=self.DF_tracker.min_z,
                             delta_x=self.DF_tracker.delta_x, delta_y=self.DF_tracker.delta_y,
                             delta_z=self.DF_tracker.delta_z)

        ## Todo: More accurate vx, maybe add vs
        vs = 1.0
        vs_ret = 1.0
        vs_s_ret = 0.0
        vx_t = 0.0
        vs_t = 0.0
        
        # Physics approximation: zero out normal velocity components
        vx = 0.0
        vx_ret = 0.0
        vx_x_ret = 0.0
        
        if ignore_vx:
            vx = 0.0
            vx_x_ret = 0.0
            vx_ret = 0.0
        # else: expect vx, vx_ret, vx_x_ret to have been set earlier if you want nonzero normal velocity
        
        scale_term = 1.0 + xp_flat * rho_sp
        
        velocity_x = vs * tau_vec_s_x + vx * n_vec_s_x
        velocity_y = vs * tau_vec_s_y + vx * n_vec_s_y
        
        velocity_ret_x = vs_ret * tau_vec_sp_x + vx_ret * n_vec_sp_x
        velocity_ret_y = vs_ret * tau_vec_sp_y + vx_ret * n_vec_sp_y
        
        # ∇ρ_ret components (note the /scale_term along τ')
        nabla_density_ret_x = density_x_ret * n_vec_sp_x + (density_z_ret / scale_term) * tau_vec_sp_x
        nabla_density_ret_y = density_x_ret * n_vec_sp_y + (density_z_ret / scale_term) * tau_vec_sp_y
        
        div_velocity = vs_s_ret + vx_x_ret  # ???
        
        # ---- CSR_numerator terms (use vdot to avoid name collision) ----
        vdot = velocity_x * velocity_ret_x + velocity_y * velocity_ret_y
        CSR_numerator1 = scale_term * ((velocity_x - vdot * velocity_ret_x) * nabla_density_ret_x +
                                       (velocity_y - vdot * velocity_ret_y) * nabla_density_ret_y)
        CSR_numerator2 = -scale_term * vdot * density_ret * div_velocity
        
        # ---- SAFE divide guard for r_minus_rp ----
        eps = 1e-15
        with np.errstate(divide='ignore', invalid='ignore'):
            safe = np.where(np.abs(r_minus_rp) > eps, r_minus_rp, np.inf)
            inv  = 1.0 / safe
            inv2 = inv * inv
            inv3 = inv2 * inv
            # old: CSR_integrand_z = CSR_numerator1 / r_minus_rp + CSR_numerator2 / r_minus_rp
            CSR_integrand_z = (CSR_numerator1 + CSR_numerator2) * inv
        
        # ---- geometric factors for W1..W3 (use rdot_nnm for clarity) ----
        n_minus_np_x = n_vec_s_x - n_vec_sp_x
        n_minus_np_y = n_vec_s_y - n_vec_sp_y
        rdot_nnm = r_minus_rp_x * n_minus_np_x + r_minus_rp_y * n_minus_np_y  # (r−r')·(n−n')
        
        # part2: n·τ'
        part2 = n_vec_s_x * tau_vec_sp_x + n_vec_s_y * tau_vec_sp_y
        
        # ∂ρ/∂t_ret
        partial_density = -(velocity_ret_x * nabla_density_ret_x + velocity_ret_y * nabla_density_ret_y) - \
                           density_ret * div_velocity
        
        with np.errstate(divide='ignore', invalid='ignore'):
            # old:
            # W1 = scale_term * part1 / (r_minus_rp**3) * density_ret
            # W2 = scale_term * part1 / (r_minus_rp**2) * partial_density
            # W3 = -scale_term * part2 / r_minus_rp * partial_density
            W1 = scale_term * rdot_nnm * inv3 * density_ret
            W2 = scale_term * rdot_nnm * inv2 * partial_density
            W3 = -scale_term * part2 * inv * partial_density
            # Optional: zero exact self-term
            # mask = (np.abs(r_minus_rp) > eps)
            # W1 *= mask; W2 *= mask; W3 *= mask
        
        CSR_integrand_x = W1 + W2 + W3
        # CSR_integrand_x = W1
        
        CSR_integrand_x = CSR_integrand_x.reshape(xp.shape)
        CSR_integrand_z = CSR_integrand_z.reshape(xp.shape)
        
        return CSR_integrand_z, CSR_integrand_x



    # Replace your dump_beam with this minimal extension
    def dump_beam(self, label, *, step=None, element=None, location=None):
        if self.parallel and self.rank != 0:
            return
    
        path = self.CSR_params.workdir
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, f'{self.prefix}-particles-{label}.h5')
    
        if os.path.isfile(filename):
            os.remove(filename)
            print("Existing file " + filename + " deleted.")
    
        print(f"Beam at position {self.beam.position} is written to {filename}")
        self.beam.particle_group.write(filename)
    
        # NEW: tag attrs on the root
        import h5py
        with h5py.File(filename, 'a') as hf:
            hf.attrs['step']          = int(step if step is not None else getattr(self.beam, 'step', -1))
            hf.attrs['timestep_size'] = float(self.lattice.step_size)
            if element is not None:
                hf.attrs['element']   = str(element)
            if location is not None:
                hf.attrs['location']  = str(location)   # 'initial', 'start', 'mid', 'end'



    def _write_xz_particles_to_group(self, g):
        """
        Save current particle x/z coordinates for later scatter plotting.
        Writes into step group as /particles_xz/{x,z}.
        """
        try:
            gp = g.create_group("particles_xz")
            gp.attrs["description"] = "Particle x-z coordinates for scatter plotting"
            gp.attrs["x_unit"] = "m"
            gp.attrs["z_unit"] = "m"
            gp.create_dataset("x", data=np.asarray(self.beam.x), compression="gzip")
            gp.create_dataset("z", data=np.asarray(self.beam.z), compression="gzip")
        except Exception as e:
            print(f"WARNING: failed to save particles_xz: {e}")

    def _write_xpx_particles_to_group(self, g):
        """
        Save current particle x/px coordinates for later scatter plotting.
        Writes into step group as /particles_xpx/{x,px}.
        """
        try:
            gp = g.create_group("particles_xpx")
            gp.attrs["description"] = "Particle x-px coordinates for scatter plotting"
            gp.attrs["x_unit"]  = "m"
            gp.attrs["px_unit"] = "rad"   # bmadx px is normalized divergence (dimensionless/rad)
            gp.create_dataset("x",  data=np.asarray(self.beam.x),  compression="gzip")
            gp.create_dataset("px", data=np.asarray(self.beam.px), compression="gzip")
        except Exception as e:
            print(f"WARNING: failed to save particles_xpx: {e}")
            

            
    def write_wakes(self, save_fields=True, save_xz=False, save_xpx=False):
        ##print("DEBUG: write_wakes() called! Step: {}".format(self.beam.step))
        
        if self.parallel and self.rank != 0:
            ##print("DEBUG: Exiting write_wakes() - parallel rank {} != 0".format(self.rank))
            return
        
        ##print("DEBUG: write_wakes() proceeding with rank {}".format(getattr(self, 'rank', 'non-parallel')))
        
        path = self.CSR_params.workdir
        os.makedirs(path, exist_ok=True)
        wakes_prefix = getattr(self, '_restart_wake_prefix', None) or self.prefix
        filename = os.path.join(path, '{}-wakes.h5'.format(wakes_prefix))
        current_ele = str(getattr(self.lattice, "current_element", "unknown"))
        
        # restart-range case: save only wakes, and embed loaded checkpoint once
        restart_only_wakes = (
            getattr(self, '_restart_wake_start_step', None) is not None
            and getattr(self, '_restart_wake_end_step', None) is not None
            and int(getattr(self, 'restart_save_wakes_range', 0) or 0) > 0
        )
    
        # **ADDED: Import time for retry mechanism**
        import time
        
        if (self.beam.step == 1) or (not os.path.isfile(filename)):
            if os.path.isfile(filename):
                os.remove(filename)
                print("Existing file " + filename + " deleted.")
            print("Wakes written to ", filename)
            
            # **MODIFIED: Add retry mechanism for initial metadata write**
            max_retries = 5
            retry_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    # **NEW CODE: Add comprehensive metadata with retry protection**
                    with h5py.File(filename, 'w') as hf_init:
                        # Basic simulation metadata
                        hf_init.attrs['timestep_size'] = self.lattice.step_size
                        hf_init.attrs['config_file'] = self.input_file if hasattr(self, 'input_file') else 'unknown'
                        hf_init.attrs['lattice_file'] = (
                            self.input.get('input_lattice', {}).get('lattice_input_file', 'unknown') 
                            if hasattr(self, 'input') else 'unknown'
                        )
                                        
                        # Dipole elements metadata
                        dipole_group = hf_init.create_group('dipole_elements')
                        
                        # Find all dipole elements and their start/end steps
                        current_position = 0.0  # Track position in lattice
                        current_step = 0
                        element_count = 0
                        
                        for ele_name in list(self.lattice.lattice_config.keys())[1:]:  # Skip first element
                            ele_config = self.lattice.lattice_config[ele_name]
                            ele_type = ele_config['type']
                            ele_length = ele_config['L']
                            steps_in_element = self.lattice.steps_per_element[element_count]
                            
                            if ele_type == 'dipole':
                                
                                start_step = current_step + 1  # +1 because step counting starts at 1
                                end_step_dipole = current_step + steps_in_element
                                
                                # **SIMPLIFIED: Calculate end step for dipole_end + 1m**
                                dipole_end_position = current_position + ele_length
                                position_plus_1m = dipole_end_position + 1.0  # Add 1 meter
                                end_step_plus_1m = int(np.ceil(position_plus_1m / self.lattice.step_size))
                                
                                # Make sure we don't exceed total steps
                                end_step_plus_1m = min(end_step_plus_1m, self.lattice.total_steps)
                                if _CSR_DEBUG:
                                    print(f"Found dipole: {ele_name}, steps {start_step}-{end_step_dipole}, +1m at step {end_step_plus_1m}")
                                
                                # Save dipole metadata
                                dipole_subgroup = dipole_group.create_group(ele_name)
                                dipole_subgroup.attrs['start_step'] = start_step
                                dipole_subgroup.attrs['end_step_dipole'] = end_step_dipole
                                dipole_subgroup.attrs['end_step_plus_1m'] = end_step_plus_1m
                                dipole_subgroup.attrs['length'] = ele_length
                                dipole_subgroup.attrs['angle'] = ele_config.get('angle', 0.0)
                                dipole_subgroup.attrs['dipole_end_position'] = dipole_end_position
                                dipole_subgroup.attrs['position_plus_1m'] = position_plus_1m
                                
                            current_position += ele_length
                            current_step += steps_in_element
                            element_count += 1
                            
                        # Store beam dump step configuration
                        hf_init.attrs['beam_dump_config'] = str(self.CSR_params.write_beam)
                        if isinstance(self.CSR_params.write_beam, list):
                            hf_init.attrs['beam_dump_start_step'] = min(self.CSR_params.write_beam)
                            hf_init.attrs['beam_dump_end_step'] = max(self.CSR_params.write_beam)
                            hf_init.attrs['beam_dump_steps'] = self.CSR_params.write_beam
                        elif self.CSR_params.write_beam == 'all':
                            hf_init.attrs['beam_dump_start_step'] = 1
                            hf_init.attrs['beam_dump_end_step'] = self.lattice.total_steps
                            hf_init.attrs['beam_dump_steps'] = 'all'
                        else:
                            hf_init.attrs['beam_dump_start_step'] = -1
                            hf_init.attrs['beam_dump_end_step'] = -1
                            hf_init.attrs['beam_dump_steps'] = 'none'
                    
                    # **ADDED: Break out of retry loop on success**
                    # If requested, embed the originally loaded checkpoint once
                    # when the wakes file is first created.
                    if bool(getattr(self, "restart_save_model", False)):
                        self._embed_loaded_checkpoint_in_wakes_file(filename)
                        
                    break
                    
                except (BlockingIOError, OSError) as e:
                    if attempt < max_retries - 1:
                        print(f"File lock error during metadata write on attempt {attempt + 1}, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        print(f"Failed to write metadata after {max_retries} attempts: {e}")
                        return  # Give up and exit the method
    
        # **MODIFIED: Add retry mechanism for main data write**
        max_retries = 5
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                with h5py.File(filename, 'a') as hf:
                    step = self.beam.step
                    groupname = 'step_' + str(step)

                    if groupname in hf:
                        del hf[groupname]
                    g = hf.create_group(groupname)

                    # --- Save step attributes ---
                    g.attrs['step'] = step
                    g.attrs['position'] = self.beam.position
                    g.attrs['mean_gamma'] = self.beam.init_gamma
                    g.attrs['beam_energy'] = self.beam.init_energy
                    g.attrs['element'] = self.lattice.current_element
                    g.attrs['charge'] = self.beam.charge
                    g.attrs['fields_saved'] = bool(save_fields)
                    g.attrs['xz_saved'] = bool(save_xz)

                    # Save wakes only if requested and arrays exist
                    if save_fields and hasattr(self, "dE_dct") and hasattr(self, "x_kick") and hasattr(self, "CSR_xmesh") and hasattr(self, "CSR_zmesh"):
                        g1 = g.create_group('longitudinal')
                        g1.attrs['unit'] = 'MeV/m'
                        g1.create_dataset('x_grids', data=self.CSR_xmesh.reshape(self.dE_dct.shape))
                        g1.create_dataset('z_grids', data=self.CSR_zmesh.reshape(self.dE_dct.shape))
                        g1.create_dataset('dE_dct', data=self.dE_dct)

                        g2 = g.create_group('transverse')
                        g2.attrs['unit'] = 'MeV/m'
                        g2.create_dataset('x_grids', data=self.CSR_xmesh.reshape(self.dE_dct.shape))
                        g2.create_dataset('z_grids', data=self.CSR_zmesh.reshape(self.dE_dct.shape))
                        g2.create_dataset('xkicks', data=self.x_kick)

                    # NEW: save x-z particles even if no fields were computed
                    if save_xz:
                        self._write_xz_particles_to_group(g)

                    if save_xpx:
                        self._write_xpx_particles_to_group(g)

                    if save_xpx and 'phase_space_xpx' not in g:
                        # Independent flag — works in all modes including restart_only_wakes
                        try:
                            from .deposit import histogram_cic_2d
                            from scipy.signal import savgol_filter
                            g5 = g.create_group('phase_space_xpx')
                            g5.attrs['unit'] = '1/(m × eV/c)'
                            g5.attrs['x_unit'] = 'm'
                            g5.attrs['px_unit'] = 'eV/c'
                            sigma_px = np.std(self.beam.px)
                            mean_px  = np.mean(self.beam.px)
                            x_grids_xpx = np.linspace(
                                self.DF_tracker.xmean - self.DF_tracker.xlim * self.DF_tracker.sigma_x,
                                self.DF_tracker.xmean + self.DF_tracker.xlim * self.DF_tracker.sigma_x,
                                self.DF_tracker.xbins)
                            px_grids = np.linspace(
                                mean_px - 5 * sigma_px,
                                mean_px + 5 * sigma_px,
                                self.DF_tracker.xbins)
                            density_xpx = histogram_cic_2d(
                                q1=self.beam.x, q2=self.beam.px,
                                w=np.ones(self.beam.x.shape),
                                nbins_1=self.DF_tracker.xbins,
                                bins_start_1=self.DF_tracker.xmean - self.DF_tracker.xlim * self.DF_tracker.sigma_x,
                                bins_end_1=self.DF_tracker.xmean   + self.DF_tracker.xlim * self.DF_tracker.sigma_x,
                                nbins_2=self.DF_tracker.xbins,
                                bins_start_2=mean_px - 5 * sigma_px,
                                bins_end_2=mean_px   + 5 * sigma_px)
                            fw = self.DF_tracker.filter_window
                            fo = self.DF_tracker.filter_order
                            if fw > 1 and fw <= min(density_xpx.shape):
                                density_xpx = savgol_filter(
                                    savgol_filter(density_xpx, fw, fo, axis=0), fw, fo, axis=1)
                            dsum = np.trapz(np.trapz(density_xpx, x_grids_xpx, axis=0), px_grids)
                            if dsum > 0:
                                density_xpx /= dsum
                            g5.create_dataset('density',   data=density_xpx)
                            g5.create_dataset('x_grids',   data=x_grids_xpx)
                            g5.create_dataset('px_grids',  data=px_grids)
                        except Exception as e:
                            print(f"ERROR saving x-px phase space (save_xpx flag): {e}")
        
                    # === NEW: Save 3σ mesh limits as metadata ===
                    if hasattr(self, '_meta'):
                        g.attrs['x3_min'] = self._meta['x3_min']
                        g.attrs['x3_max'] = self._meta['x3_max']
                        g.attrs['z3_min'] = self._meta['z3_min']
                        g.attrs['z3_max'] = self._meta['z3_max']
        

                    
            
                    # Save XZ charge density grid
                    if save_fields and (not restart_only_wakes):
                        # Save XZ charge density grid
                        try:
                            g3 = g.create_group('charge_density_xz')
                            g3.attrs['unit'] = 'nC/m^2'
                            g3.attrs['xbins'] = self.DF_tracker.xbins
                            g3.attrs['zbins'] = self.DF_tracker.zbins
                            g3.attrs['xlim'] = self.DF_tracker.xlim
                            g3.attrs['zlim'] = self.DF_tracker.zlim
                            g3.create_dataset('density', data=self.DF_tracker.density)
                            g3.create_dataset('x_grids', data=self.DF_tracker.x_grids)
                            g3.create_dataset('z_grids', data=self.DF_tracker.z_grids)
                        except Exception as e:
                            import traceback

                        # Save Z-PZ phase space
                        try:
                            from scipy.signal import savgol_filter
                            
                            g4 = g.create_group('phase_space_zpz')
                            g4.attrs['unit'] = '1/(m × eV/c)'
                            g4.attrs['z_unit'] = 'm'
                            g4.attrs['pz_unit'] = 'eV/c'
                            g4.attrs['zbins'] = self.DF_tracker.zbins
                            g4.attrs['pzbins'] = self.DF_tracker.zbins
                            g4.attrs['zlim'] = self.DF_tracker.zlim
                            g4.attrs['pzlim'] = 5
                            
                            sigma_pz = np.std(self.beam.pz)
                            mean_pz = np.mean(self.beam.pz)
                            
                            z_grids_zpz = np.linspace(
                                self.DF_tracker.zmean - self.DF_tracker.zlim * self.DF_tracker.sigma_z,
                                self.DF_tracker.zmean + self.DF_tracker.zlim * self.DF_tracker.sigma_z,
                                self.DF_tracker.zbins
                            )
                            pz_grids = np.linspace(
                                mean_pz - 5 * sigma_pz,
                                mean_pz + 5 * sigma_pz,
                                self.DF_tracker.zbins
                            )
                            
                            from .deposit import histogram_cic_2d
                            density_zpz = histogram_cic_2d(
                                q1=self.beam.z, q2=self.beam.pz, w=np.ones(self.beam.z.shape),
                                nbins_1=self.DF_tracker.zbins,
                                bins_start_1=self.DF_tracker.zmean - self.DF_tracker.zlim * self.DF_tracker.sigma_z,
                                bins_end_1=self.DF_tracker.zmean + self.DF_tracker.zlim * self.DF_tracker.sigma_z,
                                nbins_2=self.DF_tracker.zbins,
                                bins_start_2=mean_pz - 5 * sigma_pz,
                                bins_end_2=mean_pz + 5 * sigma_pz
                            )
                            
                            filter_window = self.DF_tracker.filter_window
                            filter_order = self.DF_tracker.filter_order
                            
                            if filter_window > 1 and filter_window <= min(density_zpz.shape):
                                density_zpz = savgol_filter(
                                    x=savgol_filter(
                                        x=density_zpz,
                                        window_length=filter_window,
                                        polyorder=filter_order,
                                        axis=0
                                    ),
                                    window_length=filter_window,
                                    polyorder=filter_order,
                                    axis=1
                                )
                            
                            dsum_zpz = np.trapz(np.trapz(density_zpz, z_grids_zpz, axis=0), pz_grids)
                            if dsum_zpz > 0:
                                density_zpz /= dsum_zpz
                            
                            g4.create_dataset('density', data=density_zpz)
                            g4.create_dataset('z_grids', data=z_grids_zpz)
                            g4.create_dataset('pz_grids', data=pz_grids)
                            
                            g4['z_grids'].attrs['unit'] = 'm'
                            g4['z_grids'].attrs['long_name'] = 'Longitudinal position'
                            g4['pz_grids'].attrs['unit'] = 'eV/c'
                            g4['pz_grids'].attrs['long_name'] = 'Longitudinal momentum'
                            g4['density'].attrs['unit'] = '1/(m × eV/c)'
                            g4['density'].attrs['long_name'] = 'Phase space density'
                            
                        except Exception as e:
                            print(f"ERROR saving z-pz phase space: {e}")
                
                        # Save X-PX phase space
                        try:
                            from .deposit import histogram_cic_2d
                            from scipy.signal import savgol_filter
                            g5 = g.create_group('phase_space_xpx')
                            g5.attrs['unit'] = '1/(m × eV/c)'
                            g5.attrs['x_unit'] = 'm'
                            g5.attrs['px_unit'] = 'eV/c'
                            g5.attrs['xbins'] = self.DF_tracker.xbins
                            g5.attrs['pxbins'] = self.DF_tracker.xbins
                            g5.attrs['xlim'] = self.DF_tracker.xlim
                            g5.attrs['pxlim'] = 5
                            
                            sigma_px = np.std(self.beam.px)
                            mean_px = np.mean(self.beam.px)
                            
                            x_grids_xpx = np.linspace(
                                self.DF_tracker.xmean - self.DF_tracker.xlim * self.DF_tracker.sigma_x,
                                self.DF_tracker.xmean + self.DF_tracker.xlim * self.DF_tracker.sigma_x,
                                self.DF_tracker.xbins
                            )
                            px_grids = np.linspace(
                                mean_px - 5 * sigma_px,
                                mean_px + 5 * sigma_px,
                                self.DF_tracker.xbins
                            )
                            
                            density_xpx = histogram_cic_2d(
                                q1=self.beam.x, q2=self.beam.px, w=np.ones(self.beam.x.shape),
                                nbins_1=self.DF_tracker.xbins,
                                bins_start_1=self.DF_tracker.xmean - self.DF_tracker.xlim * self.DF_tracker.sigma_x,
                                bins_end_1=self.DF_tracker.xmean + self.DF_tracker.xlim * self.DF_tracker.sigma_x,
                                nbins_2=self.DF_tracker.xbins,
                                bins_start_2=mean_px - 5 * sigma_px,
                                bins_end_2=mean_px + 5 * sigma_px
                            )
                            
                            filter_window = self.DF_tracker.filter_window
                            filter_order = self.DF_tracker.filter_order
                            
                            if filter_window > 1 and filter_window <= min(density_xpx.shape):
                                density_xpx = savgol_filter(
                                    x=savgol_filter(
                                        x=density_xpx,
                                        window_length=filter_window,
                                        polyorder=filter_order,
                                        axis=0
                                    ),
                                    window_length=filter_window,
                                    polyorder=filter_order,
                                    axis=1
                                )
                            
                            dsum_xpx = np.trapz(np.trapz(density_xpx, x_grids_xpx, axis=0), px_grids)
                            if dsum_xpx > 0:
                                density_xpx /= dsum_xpx
                            
                            g5.create_dataset('density', data=density_xpx)
                            g5.create_dataset('x_grids', data=x_grids_xpx)
                            g5.create_dataset('px_grids', data=px_grids)
                            
                            g5['x_grids'].attrs['unit'] = 'm'
                            g5['x_grids'].attrs['long_name'] = 'Transverse position'
                            g5['px_grids'].attrs['unit'] = 'eV/c'
                            g5['px_grids'].attrs['long_name'] = 'Transverse momentum'
                            g5['density'].attrs['unit'] = '1/(m × eV/c)'
                            g5['density'].attrs['long_name'] = 'Phase space density'
                            
                        except Exception as e:
                            print(f"ERROR saving x-px phase space: {e}")
                        
                
                        
                        

                    pass
                break
                
            except (BlockingIOError, OSError) as e:
                if attempt < max_retries - 1:
                    print(f"File lock error during data write on attempt {attempt + 1}, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Failed to write step data after {max_retries} attempts: {e}")
                    return  # Give up and exit the method
                
            
#    @profile
    def update_statistics(self, step):
        twiss = self.beam.twiss
        self.statistics['twiss']['alpha_x'][step] = twiss['alpha_x']
        self.statistics['twiss']['beta_x'][step] = twiss['beta_x']
        self.statistics['twiss']['gamma_x'][step] = twiss['gamma_x']
        self.statistics['twiss']['emit_x'][step] = twiss['emit_x']
        self.statistics['twiss']['eta_x'][step] = twiss['eta_x']
        self.statistics['twiss']['etap_x'][step] = twiss['etap_x']
        self.statistics['twiss']['norm_emit_x'][step] = twiss['norm_emit_x']
        self.statistics['twiss']['alpha_y'][step] = twiss['alpha_y']
        self.statistics['twiss']['beta_y'][step] = twiss['beta_y']
        self.statistics['twiss']['gamma_y'][step] = twiss['gamma_y']
        self.statistics['twiss']['emit_y'][step] = twiss['emit_y']
        self.statistics['twiss']['eta_y'][step] = twiss['eta_y']
        self.statistics['twiss']['etap_y'][step] = twiss['etap_y']
        self.statistics['twiss']['norm_emit_y'][step] = twiss['norm_emit_y']
        self.statistics['slope'][step, :] = self.beam._slope
        self.statistics['sigma_x'][step] = self.beam._sigma_x
        self.statistics['sigma_z'][step] = self.beam._sigma_z
        self.statistics['sigma_energy'][step] = self.beam.sigma_energy
        self.statistics['mean_x'][step] = self.beam._mean_x
        self.statistics['mean_z'][step] = self.beam._mean_z
        self.statistics['mean_energy'][step] = self.beam.mean_energy
        

    
    
    def write_statistics(self):

        if self.parallel and self.rank != 0:
            return

        path = self.CSR_params.workdir
        os.makedirs(path, exist_ok=True)

        if getattr(self, 'restart_enabled', False) and getattr(self, 'restart_load', False):
            ckpt_group = getattr(self, 'restart_checkpoint_group', None) or 'unknown'
            csr_suffix = "CSRon" if bool(getattr(self.CSR_params, "apply_CSR", 1)) else "CSRoff"
            restart_tag = f'restart_{ckpt_group}_{csr_suffix}'
            filename = os.path.join(path, f'{restart_tag}_{self.prefix}-statistics.h5')
        else:
            filename = os.path.join(path, f'{self.prefix}-statistics.h5')


        if os.path.isfile(filename):
            os.remove(filename)
            print("Existing file " + filename + " deleted.")
        print("Statistics written to ", filename)

        with h5py.File(filename, 'w') as hf:
            hf.create_dataset(name = 'step_positions', data = self.lattice.steps_record, shape = self.lattice.steps_record.shape)
            hf.create_dataset(name='coords', data=self.lattice.coords)
            hf.create_dataset(name='n_vec', data=self.lattice.n_vec)
            hf.create_dataset(name='tau_vec', data=self.lattice.tau_vec)
            dict2hdf5(hf, self.statistics)
            t0     = getattr(self, '_run_wall_start', None)
            t_wall = (time.time() - t0) if t0 is not None else float('nan')

            tg = hf.create_group('timing')
            tg.attrs['prefix']              = str(self.prefix)
            tg.attrs['run_start_unix']      = float(t0) if t0 is not None else float('nan')
            tg.attrs['run_wall_time_sec']   = float(t_wall)
            tg.attrs['restarted_from_group']= str(getattr(self, 'restart_checkpoint_group', 'none') or 'none')
            tg.attrs['restart_step']        = int((self._restart_next_step or 1) - 1)
            print(f"⏱️  Timing stored in statistics h5 — wall={t_wall:.2f}s")



    

    def _compute_dipole_dump_map(self):
        """
        Returns a dict: {step_number: (element_name, loc)} with loc in {'start','mid','end'}.
        Uses lattice.lattice_config and steps_per_element.
        """
        dump_map = {}
        current_step = 0
        current_position = 0.0
        element_count = 0
    
        # Skip the first key if it's a pseudo-element (mimic existing iteration)
        for ele_name in list(self.lattice.lattice_config.keys())[1:]:
            ele_cfg = self.lattice.lattice_config[ele_name]
            ele_type = ele_cfg['type']
            ele_len  = ele_cfg['L']
            steps_in_ele = self.lattice.steps_per_element[element_count]
    
            if ele_type == 'dipole':
                start_step = current_step + 1            # step counting starts at 1
                end_step   = current_step + steps_in_ele
                mid_step   = (start_step + end_step) // 2
    
                dump_map[start_step] = (ele_name, 'start')
                dump_map[mid_step]   = (ele_name, 'mid')
                dump_map[end_step]   = (ele_name, 'end')
    
            current_step     += steps_in_ele
            current_position += ele_len
            element_count    += 1
    
        return dump_map