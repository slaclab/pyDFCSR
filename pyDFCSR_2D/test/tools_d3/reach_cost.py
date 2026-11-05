import sys, os, time, numpy as np
sys.path.insert(0, os.getcwd())
from d3_wake_diagnostic import build, wake_mesh, evaluate
import cProfile, pstats, io
for nfl in (1.0, 6.0):
    sim = build(8.566, {'CSR_integration': dict(n_formation_length=nfl)})
    tr = sim.DF_tracker; n = len(tr.time_log)
    mem = tr._ring.nbytes / 1e6 if tr._ring is not None else 0
    xm, zm = wake_mesh(sim)
    t0 = time.time(); evaluate(sim, xm[:48], zm[:48]); dt = time.time() - t0
    pr = cProfile.Profile(); pr.enable(); evaluate(sim, xm[:24], zm[:24]); pr.disable()
    st = io.StringIO(); pstats.Stats(pr, stream=st).sort_stats('cumulative').print_stats(14)
    print(f'n_fl {nfl}: {n} snapshots, ring {mem:.0f} MB; 48 wake points {dt:.1f} s')
    print('\n'.join(l for l in st.getvalue().splitlines() if 'CSR.py' in l or 'interp3D' in l)[:2500])
