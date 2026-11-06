"""prev_bend_reach at a frozen state inside a bend: print the bend's own far region [s1, s3], the previous bend's
entrance pb and the extra region [pb, s1]; integrate the extra region alone at a few observers, and compare with
the 1D exit tail of the previous bend at the same s (case D with the previous bend's R, phi, d past its exit)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from d3_wake_diagnostic import build, wake_mesh
from pyDFCSR_2D.csr1d import line_density, wake_after_bend
s_stop = float(sys.argv[1]); prev_exit = float(sys.argv[2]); R, phim = 10.356107660455486, 0.0483
sim = build(s_stop, {'CSR_integration': dict(prev_bend_reach=True, band_jacobian_min=0.2)}); t = sim.beam.position; b = sim.beam
print(f's = {t:.4f} ({sim.lattice.current_element}), inbend {sim.inbend}, prev_bend_start {getattr(sim, "prev_bend_start", None)}, '
      f'bend_start_rec {sim.bend_start_rec}, L_f {sim.formation_length:.3f}, history {sim.DF_tracker.time_log[0]:.3f}..{sim.DF_tracker.time_log[-1]:.3f}')
xm, zm = wake_mesh(sim); X, Z = xm.reshape(12, 40), zm.reshape(12, 40)
S, ds, lam, dl = line_density(b.z, 300, abs(b.charge) / 1.602176634e-19, 2)
W1 = wake_after_bend(S, ds, lam, dl, R, phim, (t - prev_exit) / R) / 1e6
for r, c in ((6, 15), (6, 20), (6, 25), (1, 20), (10, 20)):
    x, z = X[r, c], Z[r, c]; s = t + z
    (s1, s3), (_, s4) = sim._layout_bounds(s, x); pb = sim._prev_bend_reach_start()
    sp = sim._far_region_nodes(s, x, t, pb, s1, transient=True)
    dE = sim._integrate_xi_region(s, x, t, sp, False, None, scale_bands=True)[0]
    # how much of the extra region is live (causal)?
    alive = sim._band_alive(s, x, t, sp)
    live = sp[alive] if alive.any() else np.array([np.nan])
    print(f'  row {r} col {c} (z/sz {(z - b._mean_z)/b._sigma_z:+.1f}): own far region s\' = {s1:.3f}..{s3:.3f}; extra [{pb:.3f}, {s1:.3f}] with {sp.size} nodes, '
          f'live s\' {np.nanmin(live):.3f}..{np.nanmax(live):.3f}; extra region dE {dE:+.4f} MeV/m; 1D tail of the previous bend at this z {np.interp(z, S, W1):+.4f}', flush=True)
