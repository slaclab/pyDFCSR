"""
Why does the CSR change of the core slice energy spread go negative through B3-D3-B4 and turn positive
at the end of B4 (nominal chicane, along_s.png)? Exact decomposition at every node.

Tracking never changes pz, so each particle's energy is  delta = delta0 + K,  with delta0 its initial
pz and K the CSR kick accumulated so far. Within each core slice of the CSR-on beam (80 slices over
+-3 sigma_z, quadratic chirp removed, |z| < 2 sigma_z, population-weighted):
    var(delta) = var(delta0) + var(K) + 2 cov(delta0, K)
var(delta0) on the CSR-on slices differs from the CSR-off slice variance only because CSR has moved
particles in z (which particles share a slice), so
    sigma_on^2 - sigma_off^2 = [var_on-slices(delta0) - sigma_off^2] + var(K) + 2 cov(delta0, K)
              =  re-slicing                      +  kick spread  + kick-energy correlation
K is also split by where the kick was taken: before B3, B3, D3, B4, Df.

Usage (pydfcsr env):
  python slice_decomposition_chicane.py run <1d|2d|2d_noT|2d_new|2d_new_noT>   -> chicane_2d_vs_1d/slice_decomp_<name>.npz
  python slice_decomposition_chicane.py plot                 -> slice_decomp.png, slice_decomp.txt
"""
import contextlib
import io
import os
import sys
import time

import numpy as np

ARGS = sys.argv[1:]
sys.argv = sys.argv[:1]                      # chicane_2d_vs_1d reads --shear from argv at import
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chicane_2d_vs_1d as C                 # noqa: E402

OUT = C.OUT
REGIONS = ['before B3', 'B3', 'D3', 'B4', 'Df']
R_EDGES = [0.0, 7.1062, 7.6064, 12.6122, 13.1124, 13.3125]


def region_of(pos):
    for j in range(len(REGIONS)):
        if pos <= R_EDGES[j + 1] + 1e-9:
            return j
    return len(REGIONS) - 1


def slice_cov(z, cols, E0, ns=80, zr=3.0, core=2.0, nmin=500):
    """
    Population-weighted covariance matrix, over the core slices of `z`, of the residuals of each column
    (times E0) after a quadratic fit in z within the slice. Moment-based (bincount), so it is cheap enough
    for every node on 1e6 particles; the diagonal for a single column equals C.slice_variance.
    """
    zc = z - z.mean(); sz = zc.std()
    ed = np.linspace(-zr * sz, zr * sz, ns + 1); w = ed[1] - ed[0]
    idx = np.digitize(zc, ed) - 1
    ok = (idx >= 0) & (idx < ns); i = idx[ok]
    ctr = 0.5 * (ed[1:] + ed[:-1])
    n = np.bincount(i, minlength=ns).astype(float)
    use = (np.abs(ctr) < core * sz) & (n > nmin)
    u = (zc[ok] - ctr[i]) / w
    Y = []
    for c in cols:
        y = c[ok] * E0
        Y.append(y - (np.bincount(i, y, ns) / np.maximum(n, 1))[i])      # centre per slice first
    M = np.stack([np.bincount(i, u ** k, ns) for k in range(5)])
    Ainv = np.linalg.inv(np.stack([[M[0], M[1], M[2]], [M[1], M[2], M[3]], [M[2], M[3], M[4]]])
                         .transpose(2, 0, 1)[use])
    P = [np.stack([np.bincount(i, y * u ** k, ns) for k in range(3)]).T[use] for y in Y]
    J = len(Y); out = np.zeros((J, J))
    for a in range(J):
        for b in range(a, J):
            raw = np.bincount(i, Y[a] * Y[b], ns)[use]
            proj = np.einsum('si,sij,sj->s', P[a], Ainv, P[b])
            out[a, b] = out[b, a] = (raw - proj).sum()
    return out / n[use].sum()


def run(name):
    f = C.config(name)
    if C.RUNS[name]['cls'] == '2d':
        from pyDFCSR_2D import CSR2D as Cls
    else:
        from pyDFCSR_2D.csr1d import CSR1D as Cls
    t0 = time.time()
    sim = Cls(input_file=f)
    sim.CSR_params.write_wakes = False
    sim.dump_beam = lambda label: None
    E0 = sim.beam.init_energy
    d0 = sim.beam.particle.pz.copy()
    N = d0.size
    Kr = np.zeros((len(REGIONS), N))

    def add(k):
        Kr[region_of(sim.beam.position)] += k
    if C.RUNS[name]['cls'] == '2d':
        orig = sim.beam.apply_wakes

        def apply_wakes(*a, **kw):
            pz0 = sim.beam.particle.pz.copy(); orig(*a, **kw); add(sim.beam.particle.pz - pz0)
        sim.beam.apply_wakes = apply_wakes
    else:
        orig = sim.apply_wake_1d

        def apply_wake_1d(S, W, L):
            k = orig(S, W, L); add(k); return k
        sim.apply_wake_1d = apply_wake_1d

    rec_s, rec_C = [], []
    upd = sim.update_statistics

    def update_statistics(step):
        upd(step)
        rec_s.append(sim.beam.position)
        rec_C.append(slice_cov(sim.beam.particle.z, [d0, *Kr], E0))
    sim.update_statistics = update_statistics
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run()
    p = sim.beam.particle
    assert np.allclose(p.pz, d0 + Kr.sum(0), rtol=0, atol=1e-12), 'pz changed by something other than CSR'
    np.savez(os.path.join(OUT, f'slice_decomp_{name}.npz'), s=np.array(rec_s), C=np.array(rec_C),
             regions=np.array(REGIONS), E0=E0, runtime=time.time() - t0)
    print(f'{name}: done in {(time.time() - t0) / 60:.1f} min', flush=True)


def terms(name):
    d = np.load(os.path.join(OUT, f'slice_decomp_{name}.npz'))
    off = np.load(os.path.join(OUT, 'off.npz'))
    s, Cm = d['s'], d['C']
    assert np.allclose(off['slice_s'][1:], s, atol=1e-9), f'{name}: nodes differ from the CSR-off run'
    v_off = off['slice_var'][1:]
    v_d0 = Cm[:, 0, 0]
    v_K = Cm[:, 1:, 1:].sum(axis=(1, 2))
    cov = Cm[:, 0, 1:].sum(axis=1)                         # cov(delta0, K)
    v_on = Cm.sum(axis=(1, 2))
    cov_r = Cm[:, 0, 1:]                                   # cov(delta0, K_region)
    rho = cov / np.sqrt(v_d0 * v_K + 1e-300)
    return dict(s=s, v_off=v_off, v_on=v_on, v_d0=v_d0, v_K=v_K, cov=cov, cov_r=cov_r, rho=rho,
                x=np.sqrt(v_K / v_d0), reslice=v_d0 - v_off,
                rel=100 * (np.sqrt(v_on / v_off) - 1))


def plot():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    names = [n for n in ('2d', '2d_noT', '2d_new', '2d_new_noT', '1d') if os.path.isfile(os.path.join(OUT, f'slice_decomp_{n}.npz'))]
    T = {n: terms(n) for n in names}
    lab = {'2d': '2D (old default)', '2d_noT': '2D, no transverse force (old default)', '2d_new': '2D, both fixes',
           '2d_new_noT': '2D, both fixes, no transverse force', '1d': '1D'}
    L = [__doc__.split('Usage')[0].strip(), '']
    # consistency: v_on from the decomposition vs the slice_var recorded independently in the main runs
    for n in names:
        r = np.load(os.path.join(OUT, f'{n}.npz'))
        L.append(f'{n}: decomposition sigma_on vs independent run slice_var: max rel diff '
                 f'{np.max(np.abs(np.sqrt(T[n]["v_on"] / r["slice_var"][1:]) - 1)):.1e} '
                 f'(same particles if CSR on deterministic)')
    pts = [('B3 middle', 7.356), ('B3 exit', 7.6064), ('D3 middle', 10.0), ('B4 entrance', 12.6122),
           ('B4 1/4', 12.737), ('B4 1/2', 12.862), ('B4 3/4', 12.987), ('B4 9/10', 13.062), ('B4 exit', 13.1124), ('end', 13.3124)]
    for n in names:
        t = T[n]
        L += ['', f'== {lab[n]} ==  (variances as % of the CSR-off slice variance sigma_off^2 at the same s)',
              f"{'':12s} {'s':>7s} {'sig_off':>8s} {'dsig/sig':>9s} | {'re-slice':>9s} {'var K':>8s} {'2cov':>8s} | "
              f"{'rho':>6s} {'x=sK/sd0':>8s} | 2cov by kick region: " + ' '.join(f'{r:>9s}' for r in REGIONS)]
        for lb, at in pts:
            k = int(np.argmin(np.abs(t['s'] - at)))
            v0 = t['v_off'][k]
            L.append(f"{lb:12s} {t['s'][k]:7.3f} {np.sqrt(v0) / 1e3:7.1f}k {t['rel'][k]:+8.2f}% | "
                     f"{100 * t['reslice'][k] / v0:+8.2f}% {100 * t['v_K'][k] / v0:+7.2f}% {100 * 2 * t['cov'][k] / v0:+7.2f}% | "
                     f"{t['rho'][k]:+6.2f} {t['x'][k]:8.3f} | " + ' '.join(f'{100 * 2 * c / v0:+8.2f}%' for c in t['cov_r'][k]))
    open(os.path.join(OUT, 'slice_decomp.txt'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

    fig, ax = plt.subplots(4, len(names), figsize=(6.3 * len(names), 15), layout='constrained', squeeze=False)
    for j, n in enumerate(names):
        t = T[n]; s = t['s']; v0 = t['v_off']
        a = ax[0, j]
        a.semilogy(s, np.sqrt(v0) / 1e3, 'k:', label='CSR off'); a.semilogy(s, np.sqrt(t['v_on']) / 1e3, c='C3', label='CSR on')
        a.semilogy(s, np.sqrt(t['v_K']) / 1e3, c='C2', label='σ_K: spread of the CSR kicks in a slice')
        a.set(title=f'{lab[n]}: core slice σE and kick spread', ylabel='[keV]'); a.legend(fontsize=7)
        a = ax[1, j]
        a.plot(s, 100 * t['reslice'] / v0, c='C0', label='re-slicing')
        a.plot(s, 100 * t['v_K'] / v0, c='C2', label='var K')
        a.plot(s, 100 * 2 * t['cov'] / v0, c='C1', label='2 cov(δ0, K)')
        a.plot(s, 100 * (t['v_on'] / v0 - 1), 'k', lw=2, label='total: σ²_on/σ²_off − 1')
        a.axhline(0, c='k', lw=0.5); a.set(title='terms of σ²_on − σ²_off, relative to σ²_off', ylabel='[%]')
        a.set_ylim(-60, 60); a.legend(fontsize=7)
        a = ax[2, j]
        a.plot(s, t['rho'], c='C1', label='ρ = corr(δ0, K) within a slice')
        a.plot(s, t['x'], c='C2', label='x = σ_K / σ_δ0')
        a.axhline(0, c='k', lw=0.5); a.set(title='correlation and size of the kicks within a slice'); a.set_ylim(-1.1, 1.6); a.legend(fontsize=7)
        a = ax[3, j]
        for r, c in zip(range(len(REGIONS)), ['0.5', 'C3', 'C0', 'C2', 'C4']):
            a.plot(s, 100 * 2 * t['cov_r'][:, r] / v0, c=c, label=f'2 cov(δ0, K from {REGIONS[r]})')
        a.axhline(0, c='k', lw=0.5); a.set(title='2 cov(δ0, K) split by where the kick was taken', ylabel='[% of σ²_off]', xlabel='s [m]')
        a.set_ylim(-60, 30); a.legend(fontsize=7)
        for a in ax[:, j]:
            [a.axvspan(lo, hi, color='0.92', zorder=-1) for lo, hi, _ in C.BENDS]; a.set_xlim(6.9, 13.35)
    plt.savefig(os.path.join(OUT, 'slice_decomp.png'), dpi=100); plt.close()


if __name__ == '__main__':
    if ARGS[0] == 'run':
        run(ARGS[1])
    else:
        plot()
