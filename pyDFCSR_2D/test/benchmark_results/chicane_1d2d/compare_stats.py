"""Chicane, 1D vs 2D: bunch length, energy, emittance, beam size and slice emittance, CSR on and off.
Final-beam statistics are computed the SAME way for both codes, from the particles:
  2D: example/output/ts_chic_{on,off}-<first run>-particles-end.h5 (openPMD, all 1e6 particles)
  1D: 1d_chic{,_off}_nsb500.mat Xs = [x, x', z, delta] (1 mm bend slices, converged)
Emittance: projected sqrt(det cov(x,x')) and dispersion-corrected (x, x' with the linear fit in delta
removed; same as csr_calc's emitxc and pyDFCSR's twiss eta-subtraction). Normalised by gamma0 = 9784.8.
Slice emittance: 40 slices over +-3 sigma_z, dispersion-corrected within each slice, core |z| < 2 sigma_z.
Along s: pyDFCSR statistics.h5 (sigma_z, sigma_x, sigma_energy, twiss norm_emit_x) and csr_calc's sigZ/sigX/sigE."""
import glob, numpy as np, h5py, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pmd_beamphysics import ParticleGroup
OUTD = '../../../example/output'; G0 = 5e9 / 510998.95; E0 = 5e9
BENDS = [(0.1, 0.6002), (5.606, 6.1062), (7.1062, 7.6064), (12.6122, 13.1124)]
def from_pmd(f):
    P = ParticleGroup(f); p0 = P.p.mean() if False else 5e9
    return dict(x=P.x, xp=P.px / P.pz, z=P.z, d=P.p / p0 - 1)          # pmd z = bmad z for this run (t-based output, s fixed)
def from_mat(f):
    X = h5py.File(f)['Xs'][()]; return dict(x=X[:, 0], xp=X[:, 1], z=X[:, 2], d=X[:, 3])
def emit(x, xp, d=None):
    if d is not None:
        x = x - np.polyval(np.polyfit(d, x, 1), d); xp = xp - np.polyval(np.polyfit(d, xp, 1), d)
    c = np.cov(x, xp); return np.sqrt(max(np.linalg.det(c), 0)) * G0
def stats(b):
    z = b['z'] - b['z'].mean(); sz = z.std(); ed = np.linspace(-3 * sz, 3 * sz, 41); idx = np.digitize(z, ed) - 1
    se = []; w = []
    for i in range(40):
        m = idx == i
        if abs(0.5 * (ed[i] + ed[i + 1])) > 2 * sz or m.sum() < 1000: continue
        se.append(emit(b['x'][m], b['xp'][m], b['d'][m])); w.append(m.sum())
    I = np.histogram(z, 200)[0].max() / (np.ptp(z) / 200) * 1e-9 / np.sum(np.ones_like(z)) * 2.998e8   # peak current, 1 nC
    return dict(sz=sz * 1e6, sE=b['d'].std() * E0 / 1e6, mE=b['d'].mean() * E0 / 1e6, sx=b['x'].std() * 1e6,
                en=emit(b['x'], b['xp']) * 1e6, enc=emit(b['x'], b['xp'], b['d']) * 1e6, slen=np.average(se, weights=w) * 1e6,
                eta=np.polyfit(b['d'], b['x'], 1)[0] * 1e3, Ipk=I / 1e3)
pre = sorted(glob.glob(f'{OUTD}/ts_chic_on-2026-10-29T10_30_12*-particles-end.h5'))[0].split('ts_chic_on-')[1].split('-particles')[0]
beams = {'2D on': from_pmd(f'{OUTD}/ts_chic_on-{pre}-particles-end.h5'), '2D off': from_pmd(f'{OUTD}/ts_chic_off-{pre}-particles-end.h5'),
         '1D on': from_mat('1d_chic_nsb500.mat'), '1D off': from_mat('1d_chic_off_nsb500.mat')}
# consistency: the 2D particle dump must match the npz used for the slice analysis
A = np.load('2d_chic_on.npz'); assert np.allclose(np.sort(beams['2D on']['z']), np.sort(A['zf']), atol=1e-9), 'pmd z != npz z'
S = {k: stats(v) for k, v in beams.items()}
rows = [('sz', 'rms bunch length sigma_z', 'um'), ('Ipk', 'peak current', 'kA'), ('mE', 'mean energy change', 'MeV'), ('sE', 'rms energy spread', 'MeV'),
        ('sx', 'rms beam size sigma_x', 'um'), ('eta', 'residual dispersion dx/ddelta', 'mm'),
        ('en', 'projected norm. emittance', 'um'), ('enc', 'dispersion-corrected norm. emittance', 'um'), ('slen', 'core slice norm. emittance', 'um')]
L = [__doc__, f"initial: sigma_z 200 um, norm. emittance {emit(A['x0'], A['xp0'])*1e6:.3f} um", '',
     f"{'final beam (s = 13.31 m)':40s} {'2D off':>9s} {'2D on':>9s} {'CSR':>9s} | {'1D off':>9s} {'1D on':>9s} {'CSR':>9s}"]
for k, name, u in rows:
    a, b, c, d = S['2D off'][k], S['2D on'][k], S['1D off'][k], S['1D on'][k]
    L.append(f"{name + ' [' + u + ']':40s} {a:9.3f} {b:9.3f} {b - a:+9.3f} | {c:9.3f} {d:9.3f} {d - c:+9.3f}")
open('compare_stats.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L[1:]))
# along s
st = {t: h5py.File(sorted(glob.glob(f'{OUTD}/ts_chic_{t}-{pre}-statistics.h5'))[0]) for t in ['on', 'off']}
one = {t: h5py.File(f) for t, f in [('on', '1d_chic_nsb500.mat'), ('off', '1d_chic_off_nsb500.mat')]}
fig, ax = plt.subplots(2, 2, figsize=(13, 8), layout='constrained'); ax = ax.ravel()
for t, ls in [('on', '-'), ('off', ':')]:
    h = st[t]; s2 = h['step_positions'][()]; m = one[t]; s1 = m['s'][()].ravel() + 0.1
    ax[0].plot(s2, h['sigma_z'][()] * 1e6, ls, c='C3', label=f'2D CSR {t}'); ax[0].plot(s1, m['sigZ'][()].ravel() * 1e6, ls, c='C0', label=f'1D CSR {t}')
    ax[1].plot(s2, h['sigma_x'][()] * 1e3, ls, c='C3', label=f'2D CSR {t}'); ax[1].plot(s1, m['sigX'][()].ravel() * 1e3, ls, c='C0', label=f'1D CSR {t}')
    ax[2].plot(s2, h['sigma_energy'][()] / 1e6, ls, c='C3', label=f'2D CSR {t}'); ax[2].plot(s1, m['sigE'][()].ravel() * E0 / 1e6, ls, c='C0', label=f'1D CSR {t}')
    ax[3].plot(s2, h['twiss/norm_emit_x'][()] * 1e6, ls, c='C3', label=f'2D CSR {t} (η-corrected)')
for a, yl, ttl in zip(ax, ['σz [µm]', 'σx [mm]', 'σE [MeV]', 'εn,x [µm]'], ['bunch length', 'beam size (dominated by dispersion inside)', 'energy spread', '2D dispersion-corrected emittance along s']):
    [a.axvspan(*bb, color='0.92', zorder=-1) for bb in BENDS]; a.set(xlabel='s [m]', ylabel=yl, title=ttl); a.legend(fontsize=8)
ax[0].set_yscale('log'); ax[3].set_ylim(0, None)
plt.savefig('compare_stats.png', dpi=110)
