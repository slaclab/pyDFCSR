import os, sys, time, numpy as np
HERE='/Users/jytang/Library/CloudStorage/GoogleDrive-jytang@stanford.edu/My Drive/Stanford/research/FEL/pyDFCSR_claude/pyDFCSR_near_adapt/pyDFCSR_2D/test/bc20_indranil'
sys.path.insert(0, os.path.join(HERE, '..', '..', '..')); sys.path.insert(0, os.path.join(HERE, '..', 'facet_bc20'))
import dump_kick_state as D
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D.RUNS = os.path.join(HERE, '..', 'benchmark_results', 'bc20_indranil', 'runs')
sim = D.load_state(D.state_path('loss', 37.74)); ip = sim.integration_params; pos = sim.beam.position
nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins; ip.xbins = 100
zm, xm = sim.CSR_zmesh, sim.CSR_xmesh
# the worst point: evaluate the row of mesh points at z ~ -65 um with 200 and 400 far columns
cols = np.flatnonzero(np.abs(zm[:nz] - (-64.9e-6)) < 0.3e-6)
iz = cols[0]; idx = np.arange(nx) * nz + iz
W = {}
for fz in (200, 400):
    ip.far_zbins = fz
    W[fz] = np.array([sim.get_CSR_wake(pos + zm[k], xm[k])[0] for k in idx])
d = np.abs(W[400] - W[200]); i = d.argmax(); k = idx[i]
print(f'row z={zm[k]*1e6:+.1f} um: W(200) by x row:', np.array2string(W[200], precision=2), '\n   W(400)-W(200):', np.array2string(W[400]-W[200], precision=2))
print(f'worst: x={xm[k]*1e6:+.1f} um, W 200/400 = {W[200][i]:.3f}/{W[400][i]:.3f}')
(s1, s3), (_, s4) = sim._layout_bounds(pos + zm[k], xm[k])
print(f'far region s1={s1:.3f} s3={s3:.3f} (bend entrance 36.845), s={pos+zm[k]:.3f}')
fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
res = {}
for fz in (200, 400, 1600):
    ip.far_zbins = fz
    t1 = time.time(); dbg = sim.get_CSR_wake(pos + zm[k], xm[k], debug=True); dt = time.time() - t1
    xp, sp, Iz = dbg['xp_mesh'][0], dbg['sp_mesh'][0], dbg['integrand_z'][0]
    if isinstance(Iz, np.ndarray) and Iz.ndim == 2:
        col = np.trapz(Iz, xp, axis=0); spc = sp[0]
    else:  # ragged: per column
        col = np.array([np.trapz(iz_, xp_) for iz_, xp_ in zip(Iz, xp)]); spc = np.array([s_[0] if np.ndim(s_) else s_ for s_ in sp])
    tot = -sim.CSR_scaling * np.trapz(col, spc)
    res[fz] = (spc, col)
    print(f'far_zbins {fz}: {len(spc)} cols, far-region W = {tot:+.4f} MeV/m, {dt:.0f} s')
    ax[0].plot((pos + zm[k]) - spc, -sim.CSR_scaling * col, '.-', ms=2, lw=0.6, label=f'{fz}: far W {tot:+.3f}')
ax[0].set_ylabel('column integral  -C ∫K dx′  [MeV/m per m]'); ax[0].legend(); ax[0].axvline(pos + zm[k] - 36.845, color='k', ls=':', label='bend entrance')
# cumulative difference 200 vs 1600, on the 1600 grid
spf, cf = res[1600]
for fz in (200, 400):
    spc, col = res[fz]; ci = np.interp(spf, spc, col)
    cum = -sim.CSR_scaling * np.cumsum(np.gradient(spf) * (ci - cf))
    ax[1].plot((pos + zm[k]) - spf, cum, label=f'cumulative (W{fz} − W1600) from the far end')
ax[1].legend(); ax[1].set_ylabel('MeV/m'); ax[1].axvline(pos + zm[k] - 36.845, color='k', ls=':')
# where does the fine column integral vary fastest? local scale = |col| / |dcol/ds'|
dc = np.gradient(cf, spf); ax[2].semilogy((pos + zm[k]) - spf, np.abs(cf) / np.maximum(np.abs(dc), 1e-300), '.', ms=2)
ax[2].set_ylabel('|I| / |dI/ds′|  [m]'); ax[2].set_xlabel('u = s − s′ [m]'); ax[2].axhline(1.98/400, color='r', ls='--', label='5 mm cell'); ax[2].legend()
fig.suptitle(f'loss B2RE s={pos:.2f}: far-region column integrals at mesh point z={zm[k]*1e6:+.1f}, x={xm[k]*1e6:+.1f} um')
out = os.path.join(HERE, '..', 'benchmark_results', 'bc20_indranil', 'far_column_integrand_b2re.png'); fig.savefig(out, dpi=120); print('saved', out)
