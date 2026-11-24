"""
Wake maps, z- and x-projected wakes and the along-the-lattice view of the benchmark chicane
(chicane_2d_vs_1d.py runs 2dR / 2dR_noT / 1d_full), reusing the BC20 plot machinery.

    PYTHONPATH=<repo root> conda run -n pydfcsr python pyDFCSR_2D/test/chicane_wake_plots.py [run] [cmp] [run1d]

Writes to benchmark_results/chicane_2d_vs_1d/closed/: wake_maps_long_<run>.png, wake_maps_trans_<run>.png,
wake_zproj_<run>.png (vs cmp and the 1D run at the same kick), wake_xproj_<run>.png (W and F projected on
the transverse mesh axis xi), wakes_along_s_<run>.png, and wake_table_<run>.txt (per pick: charge-weighted
mean W of 2D and 1D, their ratio, max |W|, max |F|).
"""
import os
import sys

import h5py
import matplotlib
import numpy as np
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'facet_bc20'))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
import plot_bc20_wakes as P  # noqa: E402

RES = os.path.join(HERE, 'benchmark_results', 'chicane_2d_vs_1d', 'closed')
P.RES, P.RUNS = RES, os.path.join(RES, 'runs')
LAT = os.path.join(HERE, '..', 'example', 'input', 'chicane_lattice.yaml')


def boundaries():
    lat = yaml.safe_load(open(LAT))
    s, out = 0.0, {}
    for k, v in lat.items():
        if isinstance(v, dict) and 'L' in v:
            out[k] = (s, s + float(v['L']), v['type'])
            s += float(v['L'])
    return out


def picks(B):
    f = lambda e, q: B[e][0] + q * (B[e][1] - B[e][0])
    return [(f('B1', 0.5), 'B1 mid'), (f('B1', 1.0), 'B1 exit'), (B['B1'][1] + 0.3, 'D1 +0.3 m'), (f('D1', 0.5), 'D1 mid'),
            (f('B2', 0.5), 'B2 mid'), (f('B2', 1.0), 'B2 exit'), (f('D2', 0.5), 'D2 mid'),
            (f('B3', 0.5), 'B3 mid'), (f('B3', 1.0), 'B3 exit'), (B['B3'][1] + 0.3, 'D3 +0.3 m'), (f('D3', 0.3), 'D3 30%'),
            (f('D3', 0.6), 'D3 60%'), (f('D3', 0.95), 'D3 end'), (f('B4', 0.33), 'B4 1/3'), (f('B4', 0.67), 'B4 2/3'),
            (f('B4', 1.0), 'B4 exit'), (f('Df', 0.95), 'Df end')]


def wz_of(r):
    """node-density weights along z (the charge mesh equidistributes 0.3 uniform + 0.7 charge)"""
    z = r['z']
    h = np.gradient(z)
    n, L = z.size, z[-1] - z[0]
    w = np.clip(1.0 / ((n - 1) * h) - P.UNIFORM_FRAC / L, 0, None) * h
    return w / w.sum()


def load1d_full(run):
    f = h5py.File(P.newest(f'{run}-2*-wakes1d.h5'), 'r')
    out = {}
    for k in f:
        g = f[k]
        z, W, lam = g['longitudinal_1d/z_grid'][()], g['longitudinal_1d/dE_dct'][()] / 1e6, g['longitudinal_1d/lam'][()]
        out[round(float(g.attrs['position']), 6)] = dict(z=z, W=W, lam=lam, s=float(g.attrs['position']))
    return out


def nearest1d(D1, s, tol=0.15):
    k = min(D1, key=lambda q: abs(q - s))
    return D1[k] if abs(k - s) < tol else None


def mean_w(r):
    return float(np.sum(r['wx'][:, None] * wz_of(r)[None, :] * r['W']))


def mean_w1d(d):
    return float(np.sum(d['lam'] * d['W']) / np.sum(d['lam']))


def zproj(R, Ro, D1, PK, labels, fname):
    nc = 6
    nr = int(np.ceil(len(PK) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(2.3 * nc, 1.75 * nr + 0.9), squeeze=False)
    for ax, (s, lab) in zip(axs.flat, PK):
        r = R[P.nearest(R, s)]
        zm = P.zmid(r)
        ax.axhline(0, color=P.GRID, lw=0.8)
        d = nearest1d(D1, r['s'])
        if d is not None:
            ax.plot((d['z'] - zm) * 1e6, d['W'], color=P.COL['1d'], lw=1.3, ls=(0, (3, 1.5)))
        ro = Ro[P.nearest(Ro, s)]
        ax.plot((ro['z'] - zm) * 1e6, P.proj(ro, ro['W']), color=P.COL['old'], lw=1.1)
        ax.plot((r['z'] - zm) * 1e6, P.proj(r, r['W']), color=P.COL['q'], lw=1.4)
        ax.set_title(f"{lab}, s = {r['s']:.2f} m", loc='left', pad=1.5, fontsize=8.5)
        ax.set_xlim((r['z'][0] - zm) * 1e6, (r['z'][-1] - zm) * 1e6)
        ax.tick_params(labelsize=6, length=1.5)
    for ax in axs.flat[len(PK):]:
        ax.axis('off')
    for ax in axs[:, 0]:
        ax.set_ylabel('W [MeV/m]')
    for ax in axs[-1]:
        ax.set_xlabel('z − z_mid [µm]  (head at +z)')
    h = [plt.Line2D([], [], color=P.COL['q'], lw=1.4), plt.Line2D([], [], color=P.COL['old'], lw=1.1),
         plt.Line2D([], [], color=P.COL['1d'], lw=1.3, ls=(0, (3, 1.5)))]
    fig.legend(h, [f'{labels[0]} (charge-weighted over ξ)', labels[1], f'{labels[2]}, same kick'],
               loc='upper right', ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle('Benchmark chicane: longitudinal wake projected on z', x=0.01, ha='left', fontsize=10, color=P.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.5, w_pad=0.4)
    fig.savefig(os.path.join(RES, fname), dpi=130)
    plt.close(fig)
    print('wrote', fname)


def xproj(R, Ro, PK, labels, fname):
    """W and F projected on the transverse mesh axis xi (weighted by the z node density = charge)"""
    nc = 6
    nr = int(np.ceil(len(PK) / nc))
    fig, axs = plt.subplots(2 * nr, nc, figsize=(2.3 * nc, 1.6 * 2 * nr + 0.9), squeeze=False)
    for j, (s, lab) in enumerate(PK):
        r = R[P.nearest(R, s)]
        ro = Ro[P.nearest(Ro, s)]
        wz, wzo = wz_of(r), wz_of(ro)
        row, col = 2 * (j // nc), j % nc
        for ax, key in ((axs[row, col], 'W'), (axs[row + 1, col], 'F')):
            ax.axhline(0, color=P.GRID, lw=0.8)
            ax.plot(ro['xi'] * 1e6, (wzo[None, :] * ro[key]).sum(1), color=P.COL['old'], lw=1.1)
            ax.plot(r['xi'] * 1e6, (wz[None, :] * r[key]).sum(1), color=P.COL['q'], lw=1.4, marker='.', ms=3)
            ax.tick_params(labelsize=6, length=1.5)
            ax.set_title(f"{key}(ξ): {lab}, s = {r['s']:.2f} m", loc='left', pad=1.5, fontsize=8)
    for ax in axs.flat:
        if not ax.lines:
            ax.axis('off')
    for i in range(nr):
        axs[2 * i, 0].set_ylabel('W(ξ) [MeV/m]')
        axs[2 * i + 1, 0].set_ylabel('F(ξ) [MeV/m]')
    for ax in axs[-1]:
        ax.set_xlabel('ξ = x − x(z) fit [µm]')
    h = [plt.Line2D([], [], color=P.COL['q'], lw=1.4, marker='.'), plt.Line2D([], [], color=P.COL['old'], lw=1.1)]
    fig.legend(h, [f'{labels[0]} (charge-weighted over z)', labels[1]], loc='upper right', ncol=2, frameon=False, fontsize=8.5)
    fig.suptitle('Benchmark chicane: wakes projected on the transverse axis ξ (12 charge-mesh rows on the 0.2–99.8% span)',
                 x=0.01, ha='left', fontsize=10, color=P.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=0.5, w_pad=0.4)
    fig.savefig(os.path.join(RES, fname), dpi=130)
    plt.close(fig)
    print('wrote', fname)


def along_s(R, Ro, D1, B, labels, fname):
    s = np.array([r['s'] for r in R])
    zg = np.linspace(-700, 700, 281)
    rel = lambda r, M: np.interp(zg, (r['z'] - P.zmid(r)) * 1e6, P.proj(r, M), left=np.nan, right=np.nan)
    Wq = np.array([rel(r, r['W']) for r in R])
    Fq = np.array([rel(r, r['F']) for r in R])
    W1 = np.full_like(Wq, np.nan)
    m2, m1 = np.array([mean_w(r) for r in R]), np.full(len(R), np.nan)
    for i, r in enumerate(R):
        d = nearest1d(D1, r['s'], tol=0.02)
        if d is not None:
            W1[i] = np.interp(zg, (d['z'] - P.zmid(r)) * 1e6, d['W'], left=np.nan, right=np.nan)
            m1[i] = mean_w1d(d)
    xg = np.linspace(-1, 1, 25)       # xi normalised to the mesh span, for the x-projection along s
    Fx = np.array([np.interp(xg, 2 * (r['xi'] - r['xi'].mean()) / np.ptp(r['xi']), (wz_of(r)[None, :] * r['F']).sum(1)) for r in R])
    Wx = np.array([np.interp(xg, 2 * (r['xi'] - r['xi'].mean()) / np.ptp(r['xi']), (wz_of(r)[None, :] * r['W']).sum(1)) for r in R])
    peak = lambda A: np.nanmax(np.abs(A), axis=1)
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(7, 1, height_ratios=[0.2, 0.9, 0.9, 1, 1, 1, 1], hspace=0.16)
    axL = fig.add_subplot(gs[0])
    axs = [fig.add_subplot(gs[i], sharex=axL) for i in range(1, 7)]
    for e, (a, b, t) in B.items():
        if t == 'dipole':
            axL.axvspan(a, b, color='#c3c2b7', lw=0)
            axL.text(0.5 * (a + b), 0.5, e, ha='center', va='center', fontsize=8.5, color=P.INK)
    axL.set_yticks([])
    axL.set_title('lattice (dark: dipoles)', loc='left', fontsize=9)
    ax = axs[0]
    ax.semilogy(s, peak(Wq), color=P.COL['q'], lw=1.4, label=labels[0])
    ax.semilogy([r['s'] for r in Ro], [np.abs(P.proj(r, r['W'])).max() for r in Ro], color=P.COL['old'], lw=1.1, label=labels[1])
    ax.semilogy(s, peak(W1), color=P.COL['1d'], lw=1.2, ls='--', label=labels[2])
    ax.set_ylabel('max |W(z)| [MeV/m]')
    ax.legend(fontsize=8, frameon=False, loc='upper left', ncol=3)
    ax.grid(color=P.GRID, lw=0.5)
    ax = axs[1]
    ax.plot(s, -m2, color=P.COL['q'], lw=1.4, label=f'{labels[0]}: −⟨W⟩ (charge-weighted mean = energy loss rate)')
    ax.plot([r['s'] for r in Ro], [-mean_w(r) for r in Ro], color=P.COL['old'], lw=1.1, label=labels[1])
    ax.plot(s, -m1, color=P.COL['1d'], lw=1.2, ls='--', label=labels[2])
    ax.set_yscale('symlog', linthresh=0.01)
    ax.set_ylabel('−⟨W⟩ [MeV/m]')
    ax.legend(fontsize=8, frameon=False, loc='upper left', ncol=3)
    ax.grid(color=P.GRID, lw=0.5)
    for ax, M, g, lab in ((axs[2], Wq, zg, f'{labels[0]}: W(z) projected on z, each kick normalised to its max |W|'),
                          (axs[3], W1, zg, f'{labels[2]}: W(z), each kick normalised'),
                          (axs[4], Fq, zg, f'{labels[0]}: transverse force F(z) projected on z, each kick normalised'),
                          (axs[5], Fx, xg, f'{labels[0]}: F(ξ) projected on z, each kick normalised; vertical axis ξ over the mesh span')):
        N = M / np.nanmax(np.abs(M), axis=1, keepdims=True)
        ax.pcolormesh(s, g, N.T, cmap=P.CMAP, norm=TwoSlopeNorm(0, -1, 1), shading='nearest', rasterized=True)
        ax.set_ylabel('z − z_mid [µm]' if g is zg else 'ξ / half span')
        ax.set_title(lab, loc='left', fontsize=9)
    for ax in [axL] + axs[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    axs[-1].set_xlabel('s [m]')
    fig.suptitle(f'Benchmark chicane wakes along the lattice, {labels[0]} ({len(s)} kicks)', x=0.01, ha='left',
                 fontsize=11, color=P.INK, y=0.995)
    fig.savefig(os.path.join(RES, fname), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('wrote', fname)
    return Wx


def table(R, Ro, D1, PK, run, cmp_run, fname):
    lines = [f'{"pick":10s} {"s [m]":>7s} {"ele":>4s} | {"<W> 2D":>9s} {"<W> noT":>9s} {"<W> 1D":>9s} {"2D/1D":>6s} | '
             f'{"max|W| 2D":>10s} {"max|W| 1D":>10s} | {"max|F| 2D":>10s} {"<F>":>8s}   [MeV/m]']
    for s, lab in PK:
        r, ro = R[P.nearest(R, s)], Ro[P.nearest(Ro, s)]
        d = nearest1d(D1, r['s'])
        m2, mo = mean_w(r), mean_w(ro)
        m1 = mean_w1d(d) if d is not None else np.nan
        wz = wz_of(r)
        mf = float(np.sum(r['wx'][:, None] * wz[None, :] * r['F']))
        lines.append(f'{lab:10s} {r["s"]:7.3f} {r["ele"]:>4s} | {m2:+9.4f} {mo:+9.4f} {m1:+9.4f} {m2 / m1 if m1 else np.nan:6.2f} | '
                     f'{np.abs(P.proj(r, r["W"])).max():10.4f} {np.abs(d["W"]).max() if d is not None else np.nan:10.4f} | '
                     f'{np.abs(P.proj(r, r["F"])).max():10.4f} {mf:+8.4f}')
    txt = '\n'.join(lines)
    open(os.path.join(RES, fname), 'w').write(f'{run} vs {cmp_run} vs 1d_full, charge-weighted projections at the picked kicks\n' + txt + '\n')
    print(txt)


def main(run='2dR', cmp_run='2dR_noT', run1d='1d_full'):
    B = boundaries()
    PK = picks(B)
    labels = (f'2D {run}', f'2D {cmp_run}', f'1D {run1d}')
    P.LABEL.update({run: labels[0], cmp_run: labels[1], run1d: labels[2]})
    P.RUN = run
    R, Ro = P.load2d(run), P.load2d(cmp_run)
    D1 = load1d_full(run1d)
    P.maps(R, 'W', 'Longitudinal wake dE/ds', f'wake_maps_long_{run}.png', picks=PK)
    P.maps(R, 'F', 'Transverse force', f'wake_maps_trans_{run}.png', picks=PK)
    zproj(R, Ro, D1, PK, labels, f'wake_zproj_{run}.png')
    xproj(R, Ro, PK, labels, f'wake_xproj_{run}.png')
    along_s(R, Ro, D1, B, labels, f'wakes_along_s_{run}.png')
    table(R, Ro, D1, PK, run, cmp_run, f'wake_table_{run}.txt')


if __name__ == '__main__':
    main(*sys.argv[1:])
