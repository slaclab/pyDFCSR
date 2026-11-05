"""+-3 sigma_z spikes: z-cut on mesh row 6 at a frozen state for deposition variants; clipped charge,
spike size (excess over a 9-point median), and change of the core (|z| < 2 sigma_z) vs the default."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from d3_wake_diagnostic import build, wake_mesh, evaluate, OUT
from scipy.ndimage import median_filter
s_stop = float(sys.argv[1]); names = sys.argv[2].split(';')
V = {'default': {}, 'fit_zlim 5': dict(fit_zlim=5.0), 'vx_model none': dict(vx_model='none'),
     'xlim 7, xbins 420': dict(xlim=7, xbins=420), 'xlim 9, xbins 540': dict(xlim=9, xbins=540),
     'xlim 7, xbins 300': dict(xlim=7, xbins=300), 'xlim 8, xbins 480': dict(xlim=8, xbins=480),
     'xlim 8, xbins 300': dict(xlim=8, xbins=300)}
zc = np.linspace(-4, 4, 81); core = np.abs(zc) < 2; ref = None; out = {}
for n in names:
    sim = build(s_stop, {'particle_deposition': V[n]} if V[n] else None)
    b = sim.beam; xm, zm = wake_mesh(sim); X, Z = xm.reshape(12, 40), zm.reshape(12, 40)
    zz = b._mean_z + zc * b._sigma_z; xx = np.interp(zz, Z[6], X[6])
    t0 = time.time(); w = evaluate(sim, xx, zz); dt = time.time() - t0
    d = w - median_filter(w, 9, mode='nearest'); sp = np.abs(d[np.abs(np.abs(zc) - 3) < 0.4]).max()
    out[n] = w
    if ref is None: ref = w
    print(f'{s_stop} {n:20s} clipped {sim.DF_tracker.clipped*100:.3f}%  spike at |z|~3: {sp:.4f} MeV/m ({sp/np.abs(w).max()*100:.1f}% of peak)  '
          f'core change vs default {np.linalg.norm((w-ref)[core])/np.linalg.norm(ref[core]):.4f}  {dt:.0f} s', flush=True)
tag = sys.argv[3] if len(sys.argv) > 3 else ''
np.savez(os.path.join(OUT, f'zspike_fix_{s_stop:.4f}{tag}.npz'), zc=zc, names=np.array(list(out)), cuts=np.array(list(out.values())))
