import os, sys, time, numpy as np
T='/Users/jytang/Library/CloudStorage/GoogleDrive-jytang@stanford.edu/My Drive/Stanford/research/FEL/pyDFCSR_claude/pyDFCSR_near_adapt/pyDFCSR_2D/test'
sys.path.insert(0, os.path.join(T, '..', '..')); sys.path.insert(0, os.path.join(T, 'facet_bc20')); sys.path.insert(0, T)
import dump_kick_state as D
D.RUNS = os.path.join(T, 'benchmark_results', 'chicane_2d_vs_1d', 'closed', 'runs')
run, st = '2dR_pre2_rs2', 0.6
if sys.argv[1] == 'dump':
    D.main(run, st); sys.exit()
import chicane_wake_plots as W
import plot_bc20_wakes as P
sim = D.load_state(D.state_path(run, st)); ip = sim.integration_params; pos = sim.beam.position
nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins
D1 = W.load1d_full('1d_full'); d1 = W.nearest1d(D1, pos, tol=0.05)
print(f'B1 exit state s={pos:.4f}: a={pos - sim.bend_start_rec:.3f} m, R={abs(sim.R_rec):.2f}, L_f {sim.formation_length:.3f}, sigma_z {sim.beam._sigma_z*1e6:.0f} um; 1D at {d1["s"]:.3f}: <W> {W.mean_w1d(d1):+.5f}, max|W| {np.abs(d1["W"]).max():.4f}')
ip.prehistory = 20.0
xi = np.asarray(sim.CSR_xrange_transformed); h = np.gradient(xi); n, L = xi.size, xi[-1]-xi[0]
wx = np.clip(1.0/((n-1)*h) - 0.3/L, 0, None)*h; wx /= wx.sum()
def mesh(label):
    t1 = time.time(); r = np.array([sim.get_CSR_wake(pos + z, x) for z, x in zip(sim.CSR_zmesh, sim.CSR_xmesh)]); dt = time.time() - t1
    Wm = r[:, 0].reshape(nx, nz); rr = dict(z=sim.CSR_zmesh[:nz], wx=wx, W=Wm)
    (s1, s3), _ = sim._layout_bounds(pos, sim.CSR_xmesh[nz // 2])
    print(f'   {label:34s}: reach {pos - s1:5.2f} m (straight {-s1 + sim.bend_start_rec:5.2f} m), far nodes {sim._last_region_nodes[0]:4d}, {dt:3.0f} s, <W> {W.mean_w(rr):+.5f}, max|W| {np.abs(P.proj(rr, Wm)).max():.4f}', flush=True)
print('--- graded straight-section region (reach_straight in units of R phi), far layout unchanged:')
ip.reach_slip_max = 2.0; ip.n_formation_length = 1; ip.far_zbins = 200
for rs in (0.0, 5.0, 10.0, 20.0, 40.0, 100.0):
    ip.reach_straight = rs
    t1 = time.time(); r = np.array([sim.get_CSR_wake(pos + z, x) for z, x in zip(sim.CSR_zmesh, sim.CSR_xmesh)]); dt = time.time() - t1
    Wm = r[:, 0].reshape(nx, nz); rr = dict(z=sim.CSR_zmesh[:nz], wx=wx, W=Wm)
    g = sim._geo_reach(pos); nn = len(sim._straight_region_nodes(pos, pos - g, pos - 1.41)) if g else 0
    print(f'   reach_straight {rs:5g}: straight to {g - 0.5 if g else 0:6.2f} m before the entrance, extension nodes {nn:3d}, {dt:3.0f} s, <W> {W.mean_w(rr):+.5f}, max|W| {np.abs(P.proj(rr, Wm)).max():.4f}', flush=True)
