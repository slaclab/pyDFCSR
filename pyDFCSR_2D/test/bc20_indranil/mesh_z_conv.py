import os, sys, time, numpy as np
HERE='/Users/jytang/Library/CloudStorage/GoogleDrive-jytang@stanford.edu/My Drive/Stanford/research/FEL/pyDFCSR_claude/pyDFCSR_near_adapt/pyDFCSR_2D/test/bc20_indranil'
sys.path.insert(0, os.path.join(HERE, '..', '..', '..')); sys.path.insert(0, os.path.join(HERE, '..', 'facet_bc20'))
import dump_kick_state as D
from scipy.interpolate import RegularGridInterpolator
D.RUNS = os.path.join(HERE, '..', 'benchmark_results', 'bc20_indranil', 'runs')
sim = D.load_state(D.state_path('loss', 0.15)); cp = sim.CSR_params; pos = sim.beam.position; ip = sim.integration_params; ip.xbins = 100; ip.reach_straight = 0.0
z = sim.beam.z; xt = sim.beam.x_transform - sim._mesh_ridge(z)
kicks = {}
for q, nz in ((None, 80), (0.001, 80), (0.001, 160), (0.001, 320)):
    cp.mesh_z_quantile = q; cp.zbins = nz; sim.get_CSR_mesh()
    zr, xr = np.asarray(sim.CSR_zrange), np.asarray(sim.CSR_xrange_transformed)
    t1 = time.time(); r = np.array([sim.get_CSR_wake(pos + zz, xx) for zz, xx in zip(sim.CSR_zmesh, sim.CSR_xmesh)]); dt = time.time() - t1
    W = r[:, 0].reshape(xr.size, zr.size)
    zc = np.clip(z, zr[0], zr[-1]) if q else z
    k = RegularGridInterpolator((xr, zr), W, fill_value=0.0, bounds_error=False)(np.array([np.clip(xt, xr[0], xr[-1]), zc]).T)
    kicks[(q, nz)] = k
    print(f'q={q} zbins={nz}: median dz {np.median(np.diff(zr))*1e6:.2f} um, min dz {np.diff(zr).min()*1e6:.2f}, {dt:.0f} s, kick mean {k.mean():+.5f} rms {k.std():.4f} min {k.min():+.3f}', flush=True)
ref = kicks[(0.001, 320)]
for key, k in kicks.items():
    print(f'   {key}: rms(kick - ref320) = {np.std(k - ref):.4f} MeV/m = {np.std(k-ref)/ref.std()*100:.1f}% of kick rms; mean diff {k.mean()-ref.mean():+.5f}')
