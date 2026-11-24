"""
DFCSR runs of BC20 (BEGBC20 -> ENDBC20), from example/input/facet_bc20_config.yaml.

    cd pyDFCSR_2D/example
    conda run -n pydfcsr python ../test/facet_bc20/run_dfcsr_bc20.py dfcsr_1d
    mpirun -np 6 python ../test/facet_bc20/run_dfcsr_bc20.py dfcsr_2d          (pydfcsr-mpi env)

  dfcsr_1d       CSR1D (Saldin/Stupakov 1D, line density, nbins 300, nsmooth 2), no drift cutoff
  dfcsr_2d       CSR2D, longitudinal + transverse
  dfcsr_2d_noT   CSR2D, transverse force off
  dfcsr_2d_q     CSR2D, quantile frame + 0.2 um z bins + charge-equidistributed wake mesh
  dfcsr_2d_q_noT the same, transverse force off
  dfcsr_1d_nb<N> CSR1D with N bins instead of 300 (z-resolution check)
  dfcsr_1d_q<N>  CSR1D, bin_mode quantile: N bins across the central 90% of the charge
  dfcsr_1d_pk<M> CSR1D, bin_mode peak: M bins per FWHM of the current peak
All three use the same step schedule (step_control auto, midpoint kicks), so with CSR off they
track bit-identically. The 2D settings are pinned below, so a later change of code defaults
does not change what these runs mean.

Writes test/benchmark_results/facet_bc20/runs/<name>.npz: the beam at every element exit (the
exit node of every element is a schedule node in mode auto), the step statistics, run time.
"""
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(ROOT, '..'))
OUT = os.path.join(ROOT, 'test', 'benchmark_results', 'facet_bc20', 'runs')
BASE = os.path.join(ROOT, 'example', 'input', 'facet_bc20_config.yaml')

# the 2D defaults as of 5fb726b (2026-11-06), set explicitly
PIN_2D_INTEG = dict(n_formation_length=1, xbins=200, zbins=200, drift_cutoff=0.0, tangent_bands=True,
                    drift_reach=True, drift_snap_thin=0.01, far_fine_max=500, band_jacobian_min=0.2,
                    prev_bend_reach=True, prev_bend_grade=0.02, near_cell=0.25,
                    # shipped-vs-by-parts check at 3 mesh points per kick (~one wake point of cost): a
                    # disagreement above 1e-3 flags a deposition problem at the kick where it starts
                    # (progress_facet_bc20.md 11i)
                    ibp_check=3)
PIN_2D_DEP = dict(method='bspline_comoving', xbins=300, zbins=300, xlim=5, zlim=5, smoothing_sigma=3.0,
                  poly_degree=1, velocity_threhold=1000, vx_model='affine', curvature=True, curv_zlim=4.0)
RUNS = {
    'dfcsr_1d': dict(cls='1d'),
    'dfcsr_2d': dict(cls='2d', transverse=1),
    'dfcsr_2d_noT': dict(cls='2d', transverse=0),
}
# 2D with the quantile frame (window [z_0.005, z_0.995] + 10% margins, taper) at 0.2 um z bins
# and the charge-equidistributed wake mesh: the 2D runs above cannot resolve the ~3 um spike
Q_DEP = dict(frame='quantile', frame_quantile=0.005, frame_margin=0.1, xbins=150, zbins=1200,
             smoothing_sigma_z=1.0)
Q_MESH = dict(mesh_mode='charge', xbins=12, zbins=80)
RUNS['dfcsr_2d_q'] = dict(cls='2d', transverse=1, dep=Q_DEP, csr=Q_MESH)
RUNS['dfcsr_2d_q_noT'] = dict(cls='2d', transverse=0, dep=Q_DEP, csr=Q_MESH)
# the same with kicks placed by the 1D CSR profile (kick_placement csr1d), calibrated in 1D
RUNS['dfcsr_2d_q_kp'] = dict(cls='2d', transverse=1, dep=Q_DEP, csr=Q_MESH,
                             sc=dict(kick_placement='csr1d', kick_tol=0.005, kick_h_max=0.5))
RUNS['dfcsr_2d_q_kp2'] = dict(cls='2d', transverse=1, dep=Q_DEP, csr=Q_MESH,
                              sc=dict(kick_placement='csr1d', kick_tol=0.01, kick_h_max=0.25))
# density-adaptive z smoothing (smoothing_adaptive): the 440-kick schedule and the 220-kick one
Q_DEP_AD = dict(Q_DEP, smoothing_adaptive=True, smoothing_min_particles=100, smoothing_sigma_z_max=16.0)
RUNS['dfcsr_2d_qa'] = dict(cls='2d', transverse=1, dep=Q_DEP_AD, csr=Q_MESH)
RUNS['dfcsr_2d_qa_kp'] = dict(cls='2d', transverse=1, dep=Q_DEP_AD, csr=Q_MESH,
                              sc=dict(kick_placement='csr1d', kick_tol=0.005, kick_h_max=0.5))
RUNS['dfcsr_2d_qa_kp_1M'] = dict(cls='2d', transverse=1, dep=Q_DEP_AD, csr=Q_MESH,
                                 beam='input/facet_bc20_beam_1000000.txt',
                                 sc=dict(kick_placement='csr1d', kick_tol=0.005, kick_h_max=0.5))
# peak frame (window around the current peak, the pedestal dropped as a source), 1M particles; the
# window is ~ +-5 FWHM at the end, so 300 z bins give ~25 bins per FWHM
P_DEP = dict(PIN_2D_DEP, frame='peak', frame_quantile=0.005, frame_margin=0.1, peak_halfwidth=3.0,
             peak_taper=2.0, xbins=100, zbins=300, smoothing_sigma=2.0)
RUNS['dfcsr_2d_peak_1M'] = dict(cls='2d', transverse=1, dep=P_DEP, csr=Q_MESH,
                                beam='input/facet_bc20_beam_1000000.txt',
                                sc=dict(kick_placement='csr1d', kick_tol=0.005, kick_h_max=0.5))
# the peak frame with more z smoothing (smoothing_sigma_z bins; at the first kick a bin is 2.1 um, so 6.5
# bins = the old rms grid's ~14 um effective) and the wake mesh at 12 x 40, to remove the narrow wake of the
# input beam's hard tail edge and of the frozen 20k shot noise (progress_facet_bc20.md 9i)
for _sz in (2, 4, 6.5, 10):
    RUNS[f'dfcsr_2d_peak_1M_sz{_sz:g}_m40'] = dict(cls='2d', transverse=1, dep=dict(P_DEP, smoothing_sigma_z=float(_sz)),
                                                  csr=dict(Q_MESH, zbins=40), beam='input/facet_bc20_beam_1000000.txt',
                                                  sc=dict(kick_placement='csr1d', kick_tol=0.005, kick_h_max=0.5))
# 6.5-bin z smoothing capped at 0.1 FWHM of the peak (smoothing_res_fwhm), the 12 x 40 charge mesh
# with its x axis on the 0.2% quantiles (mesh_x_quantile), and ~115 kicks (tol 0.02, h 0.75 m)
RUNS['dfcsr_2d_peak_1M_sz6.5cap_m40_k115'] = dict(
    cls='2d', transverse=1, dep=dict(P_DEP, smoothing_sigma_z=6.5, smoothing_res_fwhm=0.1),
    csr=dict(Q_MESH, zbins=40, mesh_x_quantile=0.002), beam='input/facet_bc20_beam_1000000.txt',
    sc=dict(kick_placement='csr1d', kick_tol=0.02, kick_h_max=0.75))
# the converged-reference run (progress_facet_bc20.md 9m): 1D-guided kicks from a 20k-subsample pre-pass,
# plus a kick per 4 refined steps where the auto profile refines (waists, edges) and <= 10 cm spacing for
# R*phi after each bend exit; adaptive near-region nodes (near_adapt + near_adapt_x, defaults eps 0.02)
KP_REF = dict(kick_placement='csr1d', kick_tol=0.005, kick_h_max=0.5, kick_prepass_particles=20000,
              kick_kappa_refine=4, kick_h_exit=0.1)
RUNS['dfcsr_2d_peak_1M_ref'] = dict(cls='2d', transverse=1, dep=dict(P_DEP, smoothing_sigma_z=6.5, smoothing_res_fwhm=0.1),
                                    csr=dict(Q_MESH, zbins=40, mesh_x_quantile=0.002), beam='input/facet_bc20_beam_1000000.txt',
                                    sc=KP_REF, integ=dict(near_adapt=True, near_adapt_x=True))
# the same 1M runs at coarser deposition grids (x 200, z 600 and 300)
for _nz in (600, 300):
    RUNS[f'dfcsr_2d_qa_kp_1M_z{_nz}'] = dict(cls='2d', transverse=1, dep=dict(Q_DEP_AD, xbins=200, zbins=_nz),
                                            csr=Q_MESH, beam='input/facet_bc20_beam_1000000.txt',
                                            sc=dict(kick_placement='csr1d', kick_tol=0.005, kick_h_max=0.5))
# 1D with coarser z bins (bins span the beam's full z extent: 209 um at ENDBC20), to see how much
# of the 1D-vs-2D difference a 2D-like z resolution makes (2D wake mesh dz is ~11 um after B2RE)
for _nb in (1200, 600, 150, 80, 40, 20):
    RUNS[f'dfcsr_1d_nb{_nb}'] = dict(cls='1d', nbins=_nb)
# the same with the bin width from the central 90% of the charge, or from the current peak's FWHM
for _nb in (75, 150, 300, 600, 1200):
    RUNS[f'dfcsr_1d_q{_nb}'] = dict(cls='1d', nbins=_nb, csr1d=dict(bin_mode='quantile', bin_quantile=0.05))
for _m in (5, 10, 20, 40, 80):
    RUNS[f'dfcsr_1d_pk{_m}'] = dict(cls='1d', csr1d=dict(bin_mode='peak', bins_per_fwhm=_m))
# smoothing off (nsmooth 0), to compare the effective resolution with Tao's at equal bin width
for _nb in (150, 300):
    RUNS[f'dfcsr_1d_nb{_nb}_ns0'] = dict(cls='1d', nbins=_nb, nsmooth=0)
# kick-schedule calibration in 1D (peak 20/FWHM bins throughout): a kick at every snapshot as
# the reference, and kick_placement csr1d (kicks where the 1D wake is) at several tolerances
PK20 = dict(bin_mode='peak', bins_per_fwhm=20)
RUNS['dfcsr_1d_pk20_k1'] = dict(cls='1d', csr1d=PK20, sc=dict(kappa=1.0))
RUNS['dfcsr_1d_pk20_k05'] = dict(cls='1d', csr1d=PK20, sc=dict(kappa=0.5))     # every snapshot
for _tol in (0.04, 0.02, 0.01, 0.005):
    for _hk in (1.0, 0.5, 0.25):
        RUNS[f'dfcsr_1d_pk20_kp{_tol:g}_h{_hk:g}'] = dict(
            cls='1d', csr1d=PK20, sc=dict(kick_placement='csr1d', kick_tol=_tol, kick_h_max=_hk))
# ~110-kick candidates between the grid above (tol 0.02 / h 1 m gives 101, tol 0.04 / h 0.5 m 130)
for _tol, _hk in ((0.02, 0.75), (0.03, 0.75), (0.03, 1.0), (0.015, 1.0)):
    RUNS[f'dfcsr_1d_pk20_kp{_tol:g}_h{_hk:g}'] = dict(
        cls='1d', csr1d=PK20, sc=dict(kick_placement='csr1d', kick_tol=_tol, kick_h_max=_hk))
# the 20 000-particle beam upsampled to 1M (upsample_beam.py): CSR off, 1D and 2D on the same
# CSR-guided 220-kick schedule as dfcsr_1d_pk20_kp0.005_h0.5 and dfcsr_2d_qa_kp
B1M = 'input/facet_bc20_beam_1000000.txt'
KP = dict(kick_placement='csr1d', kick_tol=0.005, kick_h_max=0.5)
RUNS['nocsr_1M'] = dict(cls='1d', beam=B1M, csr=dict(compute_CSR=0, apply_CSR=0), sc=KP, csr1d=PK20)
RUNS['dfcsr_1d_1M'] = dict(cls='1d', beam=B1M, sc=KP, csr1d=PK20)
RUNS['dfcsr_1d_recheck'] = dict(cls='1d')       # dfcsr_1d again: the default must be unchanged
# shot-noise check: a random half of the particles (A, or the complementary B), same total charge
for _h in ('A', 'B'):
    for _m in (10, 20, 40, 80):
        RUNS[f'dfcsr_1d_pk{_m}_half{_h}'] = dict(cls='1d', half=_h,
                                                 csr1d=dict(bin_mode='peak', bins_per_fwhm=_m))
    for _nb in (300, 1200):
        RUNS[f'dfcsr_1d_nb{_nb}_half{_h}'] = dict(cls='1d', half=_h, nbins=_nb)


def half_indices(h):
    """indices of half A (a random 10 000 of the 20 000, seed 0) or of the complementary half B"""
    a = np.sort(np.random.default_rng(0).permutation(20000)[:10000])
    return a if h == 'A' else np.setdiff1d(np.arange(20000), a)


def config(name):
    r = RUNS[name]
    cfg = yaml.safe_load(open(BASE))
    cfg['input_beam']['beamfile'] = os.path.join(ROOT, 'example', r.get('beam', cfg['input_beam']['beamfile']))
    cfg['input_lattice']['lattice_input_file'] = os.path.join(
        ROOT, 'example', cfg['input_lattice']['lattice_input_file'])
    if r.get('half'):
        full = np.loadtxt(cfg['input_beam']['beamfile'])
        assert full.shape[0] == 20000
        cfg['input_beam']['beamfile'] = os.path.join(OUT, 'inputs', f'facet_bc20_beam_half{r["half"]}.txt')
        os.makedirs(os.path.dirname(cfg['input_beam']['beamfile']), exist_ok=True)
        np.savetxt(cfg['input_beam']['beamfile'], full[half_indices(r['half'])])
    cfg['step_control'] = dict(mode='auto', kick_interval='midpoint', n_sub=400, **r.get('sc', {}))
    cfg['CSR_computation'].update(workdir=OUT, write_name=name, write_wakes=True, write_beam=None,
                                  compute_CSR=1, apply_CSR=1, transverse_on=r.get('transverse', 1))
    cfg['particle_deposition'] = dict(PIN_2D_DEP, **r.get('dep', {}))
    cfg['CSR_computation'].update(r.get('csr', {}))
    cfg['CSR_integration'] = dict(PIN_2D_INTEG, **r.get('integ', {}))
    if r['cls'] == '1d':
        cfg['CSR_1D'] = dict(nbins=r.get('nbins', 300), nsmooth=r.get('nsmooth', 2), drift_cutoff=0.0, **r.get('csr1d', {}))
    os.makedirs(os.path.join(OUT, 'inputs'), exist_ok=True)
    f = os.path.join(OUT, 'inputs', f'{name}.yaml')
    yaml.safe_dump(cfg, open(f, 'w'), sort_keys=False)
    return f


def main(name):
    parallel = False
    rank = 0
    if RUNS[name]['cls'] == '2d':
        try:
            from mpi4py import MPI
            parallel = MPI.COMM_WORLD.Get_size() > 1
            rank = MPI.COMM_WORLD.Get_rank()
        except ImportError:
            pass
    f = os.path.join(OUT, 'inputs', f'{name}.yaml')
    if rank == 0:
        os.makedirs(OUT, exist_ok=True)
        f = config(name)
    if parallel:
        MPI.COMM_WORLD.Barrier()

    if RUNS[name]['cls'] == '2d':
        from pyDFCSR_2D import CSR2D
        sim = CSR2D(input_file=f, parallel=parallel)
    else:
        from pyDFCSR_2D.csr1d import CSR1D
        sim = CSR1D(input_file=f)

    ends = np.asarray(sim.lattice.distance, float)
    keys = [k for k in sim.lattice.lattice_config if k != 'step_size']
    rec = {}
    upd = sim.update_statistics
    # beyond 200k particles the beam at every element exit would be GBs: keep the moments there
    # (mean, 6x6 covariance, peak current and FWHM on 0.5 um bins) and the final beam only
    big = sim.beam.particle.x.size > 200_000

    def update_statistics(step):
        upd(step)
        k = int(np.argmin(np.abs(ends - sim.beam.position)))
        if abs(ends[k] - sim.beam.position) < 1e-9:
            p = sim.beam.particle
            c = np.array([p.x, p.px, p.y, p.py, p.z, p.pz])
            if not big:
                rec[k] = c
            else:
                from scipy.ndimage import gaussian_filter1d
                h, _ = np.histogram(c[4], bins=np.arange(c[4].min(), c[4].max() + 0.5e-6, 0.5e-6))
                I = gaussian_filter1d(h.astype(float), 1.0) * sim.beam.charge / c.shape[1] / (0.5e-6 / 299792458.0)
                half = np.flatnonzero(I > I.max() / 2)
                rec[k] = (c.mean(1), np.cov(c), I.max(), (half[-1] - half[0] + 1) * 0.5e-6)
                if k == len(keys) - 1:
                    rec['end'] = c
    sim.update_statistics = update_statistics

    t0 = time.time()
    sim.run()
    if rank != 0:
        return
    missing = [keys[k] for k in range(len(keys)) if k not in rec]
    assert not missing, f'no node at the exit of {missing}'
    st = {k: v for k, v in sim.statistics.items() if not isinstance(v, dict)}
    tw = {f'twiss_{k}': v for k, v in sim.statistics['twiss'].items()}
    if big:
        beam = dict(coords_end=rec['end'], exit_mean=np.stack([rec[k][0] for k in range(len(keys))]),
                    exit_cov=np.stack([rec[k][1] for k in range(len(keys))]),
                    exit_ipk=np.array([rec[k][2] for k in range(len(keys))]),
                    exit_fwhm=np.array([rec[k][3] for k in range(len(keys))]))
    else:
        beam = dict(coords=np.stack([rec[k] for k in range(len(keys))]))
    np.savez(os.path.join(OUT, f'{name}.npz'), names=np.array(keys), s=ends, p0c=sim.beam.init_energy,
             step_s=np.array(sim.lattice.steps_record), runtime=time.time() - t0, prefix=sim.prefix,
             n_kicks=sim.lattice.schedule.n_kick, **beam, **st, **tw)
    print(f'{name}: done in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
