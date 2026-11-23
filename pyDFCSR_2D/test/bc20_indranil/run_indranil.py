"""
Indranil's two BC20 beams ("gain" and "loss", 1M particles, BEGBC20) on his lattice variant
(example/BC20_Indranil/lattice_bc20w_c_0.060000_000000.yaml), tracked with the current reference
recipe of progress_facet_bc20.md 9n:

  deposition  peak frame, 100 x 300, z smoothing 6.5 bins capped at 0.1 FWHM of the peak
  wake mesh   charge mesh 12 x 40, x axis on the 0.2% quantiles
  kicks       csr1d from a 20k-subsample 1D pre-pass, tol 0.005, h_max 0.5 m, kappa_refine 4, h_exit 0.1 m
  near field  near_adapt + near_adapt_x (eps 0.02, 4 x' rounds), degenerate band margin 3

    cd pyDFCSR_2D/example
    mpirun -np 3 python ../test/bc20_indranil/run_indranil.py gain        (pydfcsr-mpi)
    mpirun -np 3 python ../test/bc20_indranil/run_indranil.py loss
  variants: <beam>_1d (CSR1D, same schedule), <beam>_off (CSR off), <beam>_k2 (kick_tol 0.0025, h_exit
  0.05: ~2x the kicks, for the schedule convergence check)

The beams are the openPMD files converted to bmad-x text (convert in /tmp/convert_indranil.py:
P0C = mean p, z from position/z or -c(t - <t>)); the files' own pmd_beamphysics format predates the
installed reader. Writes benchmark_results/bc20_indranil/runs/<name>.npz in the run_dfcsr_bc20 format
(exit moments, final beam, statistics), wakes.h5, statistics.h5.
"""
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(ROOT, '..'))
sys.path.insert(0, os.path.join(ROOT, 'test', 'facet_bc20'))
DATA = os.path.join(ROOT, 'example', 'BC20_Indranil')
OUT = os.path.join(ROOT, 'test', 'benchmark_results', 'bc20_indranil', 'runs')
LATTICE = os.path.join(DATA, 'lattice_bc20w_c_0.060000_000000.yaml')
BEAMS = {'gain': 'beam_gain_bmadx.txt', 'loss': 'beam_loss_bmadx.txt'}
Q = {'gain': 1.5996736e-9, 'loss': None}          # filled from the conversion log by main (loss read from file)

import run_dfcsr_bc20 as RD  # noqa: E402

DEP = dict(RD.P_DEP, smoothing_sigma_z=6.5, smoothing_res_fwhm=0.1)
MESH = dict(RD.Q_MESH, zbins=40, mesh_x_quantile=0.002)
INTEG = dict(near_adapt=True, near_adapt_x=True, far_adapt_x=True)
KP = dict(RD.KP_REF)
KP2 = dict(KP, kick_tol=0.0025, kick_h_exit=0.05)
# the loss beam is 65 um long with a sharp 42 kA peak on a broad body: the peak window (+-3 FWHM) kept
# 20% of the charge and the 1D estimate put 85% of the loss in the dropped 80% (first kick, log_loss_peak).
# It gets the quantile frame (whole bunch; 300 bins over ~80 um = 0.27 um cells), gain the peak frame.
DEP_BEAM = {'gain': DEP, 'loss': dict(DEP, frame='quantile')}
# and a 12 x 80 wake mesh for loss: with 40 z nodes the finest spacing (0.79 um) was coarser than FWHM/3 of
# its 1.1-1.3 um peak (CSR2D warning at the first kick)
MESH_BEAM = {'gain': MESH, 'loss': dict(MESH, zbins=80)}
RUNS = {}
for b in BEAMS:
    RUNS[b] = dict(cls='2d', beam=b, transverse=1, sc=KP, dep=DEP_BEAM[b])
    RUNS[f'{b}_k2'] = dict(cls='2d', beam=b, transverse=1, sc=KP2, dep=DEP_BEAM[b])
    RUNS[f'{b}_1d'] = dict(cls='1d', beam=b, sc=KP)
    RUNS[f'{b}_off'] = dict(cls='1d', beam=b, sc=KP, csr=dict(compute_CSR=0, apply_CSR=0))


def beam_meta(b):
    """charge and P0C of the converted beam: the conversion log is not kept, recompute from the h5"""
    import glob
    import h5py
    f = sorted(glob.glob(os.path.join(DATA, f'beam_{b}_*.h5')))[0]
    h = h5py.File(f, 'r')
    p = np.sqrt(sum(h['momentum/' + k][()] ** 2 for k in 'xyz'))
    return float(np.sum(h['weight'][()])), float(p.mean())


def config(name):
    r = RUNS[name]
    charge, p0c = beam_meta(r['beam'])
    cfg = yaml.safe_load(open(RD.BASE))
    cfg['input_beam'] = dict(style='from_file', beamfile=os.path.join(DATA, BEAMS[r['beam']]), charge=charge, energy=p0c)
    cfg['input_lattice'] = dict(lattice_input_file=LATTICE)
    cfg['step_control'] = dict(mode='auto', kick_interval='midpoint', n_sub=400, **r.get('sc', {}))
    cfg['CSR_computation'].update(workdir=OUT, write_name=name, write_wakes=True, write_beam=None,
                                  compute_CSR=1, apply_CSR=1, transverse_on=r.get('transverse', 1))
    cfg['CSR_computation'].update(MESH_BEAM[r['beam']])
    cfg['CSR_computation'].update(r.get('csr', {}))
    cfg['particle_deposition'] = dict(r.get('dep', DEP))
    cfg['CSR_integration'] = dict(RD.PIN_2D_INTEG, **INTEG)
    if r['cls'] == '1d':
        cfg['CSR_1D'] = dict(nbins=300, nsmooth=2, drift_cutoff=0.0, bin_mode='peak', bins_per_fwhm=20)
    os.makedirs(os.path.join(OUT, 'inputs'), exist_ok=True)
    f = os.path.join(OUT, 'inputs', f'{name}.yaml')
    yaml.safe_dump(cfg, open(f, 'w'), sort_keys=False)
    return f


def main(name):
    # reuse run_dfcsr_bc20.main's run/record machinery with this module's RUNS and config
    RD.RUNS.update(RUNS)
    RD.OUT = OUT
    RD.config = config
    RD.main(name)


if __name__ == '__main__':
    main(sys.argv[1])
