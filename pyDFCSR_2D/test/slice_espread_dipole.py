"""Unchirped beam through a 1 m, 1 rad dipole: does CSR increase the slice energy spread?
Runs dipole_chirp0 with CSR on and off from the same (Hammersley) initial beam.
At every step (after the CSR kick) records, into beam_<tag>.npz (no macroparticles):
  - slice mean E, sigma_E (raw and linear-detrended), sigma_x, mean x in 40 final-z slices over +-3 sigma_z
  - charge-density histograms of (z, delta) and (x, x') over +-5 sigma of the current beam
Wakes (2D mesh per step) are copied to wakes_<tag>.h5."""
import os, sys, glob, shutil, numpy as np
from pyDFCSR_2D import CSR2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'benchmark_results', 'slice_espread_dipole')
os.chdir(os.path.join(HERE, '..', 'example'))
tag = sys.argv[1]  # 'on' or 'off'
case = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != 'base' else None   # e.g. 'sx5' -> input/sesp_sx5_config.yaml
guard = '--guard' in sys.argv
name = (tag if case is None else f'{case}_{tag}') + ('_guard' if guard else '')
NSLICE, ZR, NH, HR = 40, 3.0, 200, 5.0

sim = CSR2D(input_file='input/dipole_chirp0_config.yaml' if case is None else f'input/sesp_{case}_config.yaml')
sim.CSR_params.write_name = f'slice_espread_{name}'
sim.prefix = f'{sim.CSR_params.write_name}-{sim.timestamp}'   # prefix is fixed in __init__; concurrent runs collide otherwise
sim.CSR_params.write_wakes = (tag == 'on')
sim.CSR_params.write_beam = None
if tag == 'off':
    sim.CSR_params.compute_CSR = 0
    sim.CSR_params.apply_CSR = 0
Q = sim.beam.charge
keys = ['s', 'zc', 'n', 'mE', 'sE', 'sEd', 'sx', 'mx', 'H_zd', 'ez', 'ed', 'H_xxp', 'ex', 'exp']
rec = {k: [] for k in keys}

def hist2(a, b):
    ea = np.linspace(a.mean() - HR * a.std(), a.mean() + HR * a.std(), NH + 1)
    eb = np.linspace(b.mean() - HR * b.std(), b.mean() + HR * b.std(), NH + 1)
    H, _, _ = np.histogram2d(a, b, bins=[ea, eb])
    return H * Q / len(a) / np.diff(ea)[0] / np.diff(eb)[0], ea, eb   # C per unit area

def record(step):
    p = sim.beam.particle; E0 = p.p0c
    z = p.z - p.z.mean(); sz = z.std()
    edges = np.linspace(-ZR * sz, ZR * sz, NSLICE + 1); idx = np.digitize(z, edges) - 1
    cols = {k: np.zeros(NSLICE) for k in ['n', 'mE', 'sE', 'sEd', 'sx', 'mx']}
    for i in range(NSLICE):
        m = idx == i; zi, ei, xi = z[m], p.pz[m] * E0, p.x[m]
        cols['n'][i] = m.sum(); cols['mE'][i] = ei.mean(); cols['sE'][i] = ei.std()
        cols['sEd'][i] = np.std(ei - np.polyval(np.polyfit(zi, ei, 1), zi))  # linear-detrended
        cols['sx'][i] = xi.std(); cols['mx'][i] = xi.mean()
    rec['s'].append(sim.beam.position); rec['zc'].append(0.5 * (edges[1:] + edges[:-1]))
    for k, v in cols.items(): rec[k].append(v)
    for (h, e1, e2), names in [(hist2(p.z, p.pz), ('H_zd', 'ez', 'ed')),
                               (hist2(p.x, p.px), ('H_xxp', 'ex', 'exp'))]:
        rec[names[0]].append(h); rec[names[1]].append(e1); rec[names[2]].append(e2)

if guard:
    # DF_tracker only re-grids its history when sigma_x or sigma_z changes by more than 2x. A beam
    # that grows 5 -> 9.8 um (ratio 1.96) then outruns a +-5 sigma_x_interp grid, the density is
    # cut off at the grid edge, and the wake goes spurious (+53 MeV/m at step 5 for sigma_x = 5 um).
    # Force a re-grid whenever the newest deposition grid is not inside the interpolation grid.
    df = sim.DF_tracker; _append = df.append_interpolant
    def append_interpolant(*a, **k):
        if df.x_grid_interp is not None and (df.x_grids[0] < df.x_grid_interp[0] or df.x_grids[-1] > df.x_grid_interp[-1]
                                             or df.z_grids[0] < df.z_grid_interp[0] or df.z_grids[-1] > df.z_grid_interp[-1]):
            df.sigma_x_interp = None
        _append(*a, **k)
    df.append_interpolant = append_interpolant
orig = sim.update_statistics
def update_statistics(step):
    orig(step); record(step)
sim.update_statistics = update_statistics
record(0)   # step-0 statistics were taken in the constructor
sim.run()
np.savez(os.path.join(OUT, f'beam_{name}.npz'), p0c=sim.beam.particle.p0c, charge=Q,
         **{k: np.array(v) for k, v in rec.items()})
if tag == 'on':
    shutil.copy(f'output/{sim.prefix}-wakes.h5', os.path.join(OUT, f'wakes_{name}.h5'))
print('saved')
