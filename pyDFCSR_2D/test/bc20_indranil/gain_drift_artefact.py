import os, sys, time, numpy as np
T = '/Users/jytang/Library/CloudStorage/GoogleDrive-jytang@stanford.edu/My Drive/Stanford/research/FEL/pyDFCSR_claude/pyDFCSR_near_adapt/pyDFCSR_2D/test'
sys.path.insert(0, os.path.join(T, '..', '..')); sys.path.insert(0, os.path.join(T, 'facet_bc20')); sys.path.insert(0, T)
import dump_kick_state as D, plot_bc20_wakes as P, chicane_wake_plots as W
D.RUNS = os.path.join(T, 'benchmark_results', 'bc20_indranil', 'runs')
sim = D.load_state(D.state_path('gain_v2', 1.25)); ip = sim.integration_params; pos = sim.beam.position
nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins
R, phi = abs(sim.R_rec), abs(sim.phi_rec); a = pos - sim.bend_start_rec; Lb = R * phi
print(f'gain_v2 s={pos:.4f}: inbend {sim.inbend} afterbend {sim.afterbend}; a={a:.3f}, x past exit {a - Lb:.3f}, R={R:.2f}, Rphi={Lb:.3f}; sigma_z {sim.beam._sigma_z*1e6:.0f} um; slip {(R*phi**3/6 + (a-Lb)*phi**2/2)*1e6:.0f} um; geo reach {sim._geo_reach(pos)}; lo {sim._straight_region_lo(pos)}; time_log[0] {sim.DF_tracker.time_log[0]}; prehistory {ip.prehistory}')
xi = np.asarray(sim.CSR_xrange_transformed); h = np.gradient(xi); n, L = xi.size, xi[-1]-xi[0]
wx = np.clip(1.0/((n-1)*h) - 0.3/L, 0, None)*h; wx /= wx.sum()
def run(label):
    t1 = time.time(); r = np.array([sim.get_CSR_wake(pos + z, x) for z, x in zip(sim.CSR_zmesh, sim.CSR_xmesh)]); dt = time.time()-t1
    Wm = r[:, 0].reshape(nx, nz); rr = dict(z=sim.CSR_zmesh[:nz], wx=wx, W=Wm)
    print(f'   {label:40s}: <W> {W.mean_w(rr):+.5f}, max|W(z)| {np.abs(P.proj(rr, Wm)).max():.4f} MeV/m, max|W| any point {np.abs(Wm).max():.3f}  ({dt:.0f} s)', flush=True)
    return Wm
ip.prehistory, ip.reach_straight = 45.0, 40.0; Wfull = run('as run: prehistory 45, reach_straight 40')
ip.reach_straight = 0.0; run('prehistory 45, reach_straight 0')
ip.prehistory = 0.0; run('prehistory 0, reach_straight 0 (= v1 integration)')
ip.prehistory, ip.reach_straight = 45.0, 40.0
# per-region split at the worst point
k = int(np.abs(Wfull).argmax()); s, x = pos + sim.CSR_zmesh[k], sim.CSR_xmesh[k]
dbg = sim.get_CSR_wake(s, x, debug=True)
print(f'worst point z={sim.CSR_zmesh[k]*1e6:+.0f} um x={x*1e6:+.0f} um; regions: far s\' [{dbg["sp"][0][0]:.3f}, {dbg["sp"][0][-1]:.3f}] ({len(dbg["sp"][0])}), near [{dbg["sp"][1][0]:.4f}, {dbg["sp"][1][-1]:.4f}]')
for i, (spm, iz) in enumerate(zip(dbg['sp_mesh'], dbg['integrand_z'])):
    sp = spm[0] if np.ndim(spm) == 2 else spm
    xp = dbg['xp_mesh'][i]
    col = np.trapz(iz, xp, axis=0) if np.ndim(iz) == 2 else iz
    tot = -sim.CSR_scaling * np.trapz(col, sp) if np.ndim(col) == 1 and col.size == sp.size else np.nan
    j = int(np.abs(col).argmax()) if np.ndim(col) == 1 else 0
    print(f'   part {i}: s\' [{sp.min():+.3f}, {sp.max():+.3f}], {sp.size} cols, contribution {tot:+.4f} MeV/m, largest |column| {np.abs(col).max():.3e} at s\'={sp[j]:+.4f}')
