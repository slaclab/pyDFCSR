import os, sys, time, numpy as np
HERE='/Users/jytang/Library/CloudStorage/GoogleDrive-jytang@stanford.edu/My Drive/Stanford/research/FEL/pyDFCSR_claude/pyDFCSR_near_adapt/pyDFCSR_2D/test/bc20_indranil'
sys.path.insert(0, os.path.join(HERE, '..', '..', '..')); sys.path.insert(0, os.path.join(HERE, '..', 'facet_bc20'))
import dump_kick_state as D
from scipy.interpolate import RegularGridInterpolator
D.RUNS = os.path.join(HERE, '..', 'benchmark_results', 'bc20_indranil', 'runs')
for run, st in (('loss', 0.15), ('loss', 37.74)):
    sim = D.load_state(D.state_path(run, st)); cp = sim.CSR_params; pos = sim.beam.position; ip = sim.integration_params; ip.xbins = 100
    for k, v in (('seam_ratio', 2.0), ('prehistory', 0.0), ('reach_straight', 0.0), ('reach_slip_max', 5.0)):
        if not hasattr(ip, k): setattr(ip, k, v)
    if not hasattr(cp, 'mesh_z_quantile'): cp.mesh_z_quantile = None
    z = sim.beam.z; xt = sim.beam.x_transform - sim._mesh_ridge(z)
    out = {}
    for q in (None, 0.001):
        cp.mesh_z_quantile = q; sim.get_CSR_mesh()
        zr, xr = np.asarray(sim.CSR_zrange), np.asarray(sim.CSR_xrange_transformed)
        t1 = time.time(); r = np.array([sim.get_CSR_wake(pos + zz, xx) for zz, xx in zip(sim.CSR_zmesh, sim.CSR_xmesh)]); dt = time.time() - t1
        W = r[:, 0].reshape(xr.size, zr.size)
        zc = np.clip(z, zr[0], zr[-1]) if q else z
        kick = RegularGridInterpolator((xr, zr), W, fill_value=0.0, bounds_error=False)(np.array([np.clip(xt, xr[0], xr[-1]), zc]).T)
        frac = np.histogram(z, bins=zr)[0] / z.size
        out[q] = kick
        print(f'{run} s={pos:.2f} mesh_z_quantile {q}: z axis [{zr[0]*1e6:+.1f}, {zr[-1]*1e6:+.1f}] um, median dz {np.median(np.diff(zr))*1e6:.2f} um, empty cells {(frac < 5e-4).sum()}, {dt:.0f} s; per-particle kick: mean {kick.mean():+.5f} MeV/m, rms {kick.std():.5f}, min {kick.min():+.4f}', flush=True)
    a, b = out[None], out[0.001]
    core = (z > np.quantile(z, 0.001)) & (z < np.quantile(z, 0.999))
    print(f'   kick(q=0.001) - kick(all): rms {np.std(b - a):.5f} MeV/m ({np.std(b-a)/a.std()*100:.1f}% of the kick rms), max |diff| {np.abs(b-a).max():.4f}; mean kick change {(b.mean()-a.mean())/abs(a.mean())*100:+.2f}%; tail particles ({(~core).sum()}): mean kick {a[~core].mean():+.4f} -> {b[~core].mean():+.4f}')
