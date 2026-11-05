"""Re-time single wake maps on frozen chicane states (CSR off), default far layout vs far_fine_max 500, drift_reach on."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from d3_wake_diagnostic import build, wake_mesh, evaluate
for s_stop in [float(v) for v in sys.argv[1:]]:
    sim = build(s_stop, {'CSR_integration': dict(n_formation_length=1.0, drift_reach=True, drift_snap_thin=0.01, drift_cutoff=0.0)})
    xm, zm = wake_mesh(sim); out = []
    for lab, p in (('default', {}), ('cap 500', dict(far_fine_max=500))):
        t0 = time.time(); W = evaluate(sim, xm, zm, p); out.append((lab, time.time() - t0, sim._last_region_nodes[0], W))
    d = np.linalg.norm(out[1][3] - out[0][3]) / np.linalg.norm(out[0][3])
    print(f's = {sim.beam.position:.4f} ({sim.lattice.current_element}, exit transient {sim._exit_transient()}): '
          + '; '.join(f'{l}: {t:.1f} s for the 12x40 map, far nodes {n}' for l, t, n, _ in out) + f'; rel L2 cap vs default {d:.1e}', flush=True)
