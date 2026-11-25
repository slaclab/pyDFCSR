import os, sys, numpy as np
T = '/Users/jytang/Library/CloudStorage/GoogleDrive-jytang@stanford.edu/My Drive/Stanford/research/FEL/pyDFCSR_claude/pyDFCSR_near_adapt/pyDFCSR_2D/test'
sys.path.insert(0, os.path.join(T, '..', '..')); sys.path.insert(0, os.path.join(T, 'facet_bc20')); sys.path.insert(0, T)
import dump_kick_state as D, plot_bc20_wakes as P, chicane_wake_plots as W
P.RUNS = os.path.join(T, 'benchmark_results', 'bc20_indranil', 'runs')
D.RUNS = os.path.join(T, 'benchmark_results', 'bc20_indranil', 'runs')
sim = D.load_state(D.state_path('gain_v2', 0.75)); ip = sim.integration_params; pos = sim.beam.position
nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins
xi = np.asarray(sim.CSR_xrange_transformed); h = np.gradient(xi); n, L = xi.size, xi[-1]-xi[0]
wx = np.clip(1.0/((n-1)*h) - 0.3/L, 0, None)*h; wx /= wx.sum()
D1 = W.load1d_full('gain_1d'); d = W.nearest1d(D1, pos, tol=0.1)
print(f'gain s={pos:.3f} (B1LE, a = {pos - sim.bend_start_rec:.3f} m), sigma_z {sim.beam._sigma_z*1e6:.0f} um, sigma_x {sim.beam._sigma_x*1e6:.0f} um; 1D at {d["s"]:.3f}: <W> {W.mean_w1d(d):+.5f}, max|W| {np.abs(d["W"]).max():.4f}')
orig = sim._integrate_xi_region
OFF = {'vx': False}
def patched(s, x, t, sp, ignore_vx, *a, **k):
    return orig(s, x, t, sp, ignore_vx or OFF['vx'], *a, **k)
sim._integrate_xi_region = patched
for pre, vxoff, lab in ((45.0, False, 'v2 (prehistory 45)'), (0.0, False, 'prehistory 0 (v1 boundary)'), (45.0, True, 'v2, velocity terms off'), (0.0, True, 'prehistory 0, velocity terms off')):
    ip.prehistory = pre; OFF['vx'] = vxoff
    r = np.array([sim.get_CSR_wake(pos + z, x) for z, x in zip(sim.CSR_zmesh, sim.CSR_xmesh)])
    Wm = r[:, 0].reshape(nx, nz); rr = dict(z=sim.CSR_zmesh[:nz], wx=wx, W=Wm)
    print(f'   {lab:34s}: <W> {W.mean_w(rr):+.5f}, max|W(z)| {np.abs(P.proj(rr, Wm)).max():.4f} MeV/m', flush=True)
