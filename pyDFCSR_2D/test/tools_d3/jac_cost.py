"""Cost of the far-band Jacobian scaling: one full 12x40 map at frozen states, band_jacobian_min 0 vs 0.2, with the
far fine-window node count. Run alone, on an idle machine."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from d3_wake_diagnostic import build, wake_mesh, evaluate
for s_stop in [float(v) for v in sys.argv[1:]]:
    sim = build(s_stop); xm, zm = wake_mesh(sim); out = []
    for j in (0.0, 0.2):
        t0 = time.time(); W = evaluate(sim, xm, zm, dict(band_jacobian_min=j)); dt = time.time() - t0
        out.append(f'jac {j}: {dt:5.1f} s ({dt/W.size*1e3:.0f} ms/pt), far nodes {sim._last_region_nodes[0]}')
    print(f's = {sim.beam.position:.3f} ({sim.lattice.current_element}): ' + '; '.join(out), flush=True)
