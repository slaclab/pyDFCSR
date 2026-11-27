"""
BC20 wakes along the lattice, from the wakes.h5 files of the 2D runs and wakes1d.h5 of the 1D run.

    conda run -n pydfcsr python pyDFCSR_2D/test/facet_bc20/plot_bc20_wakes.py [run [compare [run1d]]]

run (default dfcsr_2d_q) is mapped and projected; compare (default dfcsr_2d) and run1d (default
dfcsr_1d_pk20) are drawn with it in the projections. Files for a run other than the default get
the suffix _<run>.

Writes to benchmark_results/facet_bc20/:
  bc20_wake_maps_long.png, _trans.png   2D maps W(z, xi) of the quantile-frame run at selected kicks
  bc20_wake_maps_spike.png              the same, zoomed onto the spike, after B2RE
  bc20_wake_zproj.png                   W projected on z at the same kicks: 2D quantile, old 2D, 1D
  bc20_wakes_along_s.png                every kick: |W|max vs s, and W(z) waterfalls (each kick
                                        normalised to its own max) for 2D quantile, 1D, and F(z)

The projection on z is a charge-weighted average over the x rows of the mesh, with x weights
independent of z: for mesh_mode charge from the mesh's own node density (the axis equidistributes
uniform_frac/L + (1 - uniform_frac) * lambda_x, so lambda_x ~ (1/((n-1) h) - uniform_frac/L) /
(1 - uniform_frac)); for the uniform mesh a Gaussian of sigma = span/8 (xlim 4).
"""
import glob
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'benchmark_results', 'facet_bc20')
RUNS = os.path.join(RES, 'runs')
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
CMAP = LinearSegmentedColormap.from_list('div', ['#0d366b', '#2a78d6', '#9ec5f4', '#f0efec',
                                                 '#f4a3a2', '#e34948', '#7a1f1e'])
COL = {'q': '#4a3aa7', 'old': '#1baf7a', '1d': '#eb6834'}
LABEL = {'dfcsr_2d_q': '2D quantile frame', 'dfcsr_2d': '2D, old rms grids',
         'dfcsr_2d_qa': '2D quantile, adaptive smoothing (440 kicks)',
         'dfcsr_2d_qa_kp': '2D quantile, adaptive smoothing, 220 kicks',
         'dfcsr_1d_pk20': '1D (peak 20/FWHM)', 'dfcsr_1d_pk20_k05': '1D (peak 20/FWHM), every snapshot',
         'dfcsr_2d_peak_1M': '2D peak frame, 1M particles', 'dfcsr_1d_1M': '1D, 1M particles',
         'dfcsr_2d_peak_1M_sz6.5cap_m40_k115': '2D peak frame 1M, z smoothing 6.5 bins capped, 12x40 mesh, 116 kicks',
         'dfcsr_2d_peak_1M_sz6.5cap_m40_k115_b1reW1': '2D peak 1M, 116 kicks; F = W1 only from the B1RE entrance',
         'dfcsr_2d_peak_1M_ref': '2D peak 1M reference: hybrid 269 kicks, adaptive near nodes'}
RUN, CMP, RUN1 = 'dfcsr_2d_q', 'dfcsr_2d', 'dfcsr_1d_pk20'
UNIFORM_FRAC = 0.3
# kicks to show: (target s [m], short label)
PICK = [(0.55, 'B1LE mid'), (1.06, 'B1LE exit'), (3.0, 'drift D1'), (11.3, 'B2LE mid'),
        (12.2, 'B2LE exit'), (14.5, 'quads Q3'), (18.0, 'drift D5'), (24.1, 'B3LE exit'),
        (25.4, 'B3RE exit'), (31.0, 'drift D5'), (36.6, 'Q2ER exit'), (37.2, 'B2RE 1/4'),
        (37.7, 'B2RE 1/2'), (38.2, 'B2RE 3/4'), (38.65, 'B2RE exit'), (40.0, 'drift'),
        (41.5, 'wiggler'), (45.5, 'drift D1ET2B'), (48.4, 'B1RE 1/3'), (48.75, 'B1RE 2/3'),
        (49.08, 'B1RE exit')]
plt.rcParams.update({'font.size': 8, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'axes.edgecolor': '#c3c2b7', 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.labelcolor': INK2, 'axes.titlesize': 8, 'axes.titlecolor': INK,
                     'axes.spines.top': False, 'axes.spines.right': False})


def newest(pattern):
    f = sorted(glob.glob(os.path.join(RUNS, pattern)), key=os.path.getmtime)
    return f[-1]


def load2d(run):
    f = h5py.File(newest(f'{run}-2*-wakes.h5'), 'r')
    out = []
    for k in sorted(f, key=lambda k: f[k].attrs['position']):
        g = f[k]
        Z, X = g['longitudinal/z_grids'][()], g['longitudinal/x_grids'][()]
        z = Z[0]
        if 'x_axis_transformed' in g:
            xi = g['x_axis_transformed'][()]
            h = np.gradient(xi)
            n, L = xi.size, xi[-1] - xi[0]
            wx = np.clip(1.0 / ((n - 1) * h) - UNIFORM_FRAC / L, 0, None) * h
        else:
            xi = (X - X.mean(0, keepdims=True))[:, len(z) // 2]
            wx = np.exp(-0.5 * ((xi - xi.mean()) / (np.ptp(xi) / 8)) ** 2)
        wx = wx / wx.sum()
        out.append(dict(s=float(g.attrs['position']), ele=str(g.attrs['element']), z=z, xi=xi, wx=wx,
                        charge='x_axis_transformed' in g,
                        W=g['longitudinal/dE_dct'][()], F=g['transverse/xkicks'][()]))
    return out


def load1d(run):
    f = h5py.File(newest(f'{run}-2*-wakes1d.h5'), 'r')
    return {round(float(f[k].attrs['position']), 6): (f[k]['longitudinal_1d/z_grid'][()],
                                                      f[k]['longitudinal_1d/dE_dct'][()] / 1e6) for k in f}


def proj(r, M):
    return (r['wx'][:, None] * M).sum(0)


def nearest(R, s):
    return int(np.argmin([abs(r['s'] - s) for r in R]))


def zmid(r):
    return 0.5 * (r['z'][0] + r['z'][-1])


def maps(R, key, label, fname, zoom=None, picks=PICK):
    nc = 7
    nr = int(np.ceil(len(picks) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(2.15 * nc, 1.85 * nr + 0.8), squeeze=False)
    for ax, (s, lab) in zip(axs.flat, picks):
        r = R[nearest(R, s)]
        M = r[key]
        z = (r['z'] - zmid(r)) * 1e6
        xi = r['xi'] * 1e6
        if zoom is not None:                # centre on the strongest |W| column of the projection
            zc = z[np.argmax(np.abs(proj(r, r['W'])))]
            m = np.abs(z - zc) < zoom
            z, M = z[m], M[:, m]
        a = np.abs(M).max() or 1.0
        ax.pcolormesh(z, xi, M / a, cmap=CMAP, vmin=-1, vmax=1, shading='nearest', rasterized=True)
        ax.set_title(f"{lab}, s = {r['s']:.2f} m", loc='left', pad=1.5)
        ax.text(0.03, 0.04, f'{M.min():+.3g} … {M.max():+.3g}', transform=ax.transAxes, fontsize=6.5,
                color=INK, bbox=dict(boxstyle='round,pad=0.1', fc=SURF, ec='none', alpha=0.85))
        ax.tick_params(labelsize=6, length=1.5)
    for ax in axs.flat[len(picks):]:
        ax.axis('off')
    for ax in axs[:, 0]:
        ax.set_ylabel('ξ [µm]')
    for ax in axs[-1]:
        ax.set_xlabel('z − z_mid [µm]  (head at +z)')
    mesh = ('mesh nodes follow the charge' if R[0].get('charge') else
            'uniform mesh over +-4 rms')
    fig.suptitle(f'{label}, {LABEL.get(RUN, RUN)}\neach panel normalised to its own max |value| '
                 f'(red > 0, blue < 0; range in MeV/m); {mesh}',
                 x=0.01, ha='left', fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if nr < 3 else 0.96), h_pad=0.5, w_pad=0.4)
    fig.savefig(os.path.join(RES, fname), dpi=130)
    plt.close(fig)
    print('wrote', fname)


def zproj(Rq, Ro, D1, fname):
    nc = 7
    nr = int(np.ceil(len(PICK) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(2.15 * nc, 1.7 * nr + 0.9), squeeze=False)
    for ax, (s, lab) in zip(axs.flat, PICK):
        r = Rq[nearest(Rq, s)]
        zm = zmid(r)
        ax.axhline(0, color=GRID, lw=0.8)
        # the 1D run's kick nearest to this one (its schedule differs); the offset is in the title
        s1 = min(D1, key=lambda k: abs(k - r['s']))
        d = D1[s1] if abs(s1 - r['s']) < 0.15 else None
        if d is not None:
            ax.plot((d[0] - zm) * 1e6, d[1], color=COL['1d'], lw=1.3, ls=(0, (3, 1.5)))
        ro = Ro[nearest(Ro, s)]
        ax.plot((ro['z'] - zm) * 1e6, proj(ro, ro['W']), color=COL['old'], lw=1.2)
        ax.plot((r['z'] - zm) * 1e6, proj(r, r['W']), color=COL['q'], lw=1.4)
        off = f" (1D at {s1:.2f})" if d is not None and abs(s1 - r['s']) > 5e-3 else ''
        ax.set_title(f"{lab}, s = {r['s']:.2f} m{off}", loc='left', pad=1.5)
        ax.set_xlim((r['z'][0] - zm) * 1e6, (r['z'][-1] - zm) * 1e6)
        ax.tick_params(labelsize=6, length=1.5)
    for ax in axs.flat[len(PICK):]:
        ax.axis('off')
    for ax in axs[:, 0]:
        ax.set_ylabel('W [MeV/m]')
    for ax in axs[-1]:
        ax.set_xlabel('z − z_mid [µm]')
    h = [plt.Line2D([], [], color=COL['q'], lw=1.4), plt.Line2D([], [], color=COL['old'], lw=1.2),
         plt.Line2D([], [], color=COL['1d'], lw=1.3, ls=(0, (3, 1.5)))]
    fig.legend(h, [f'{LABEL.get(RUN, RUN)} (charge-weighted over ξ)', f'{LABEL.get(CMP, CMP)}, nearest kick',
                   f'{LABEL.get(RUN1, RUN1)}, nearest kick (within 15 cm)'], loc='upper right', ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle('BC20 longitudinal wake projected on z', x=0.01, ha='left', fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=0.5, w_pad=0.4)
    fig.savefig(os.path.join(RES, fname), dpi=130)
    plt.close(fig)
    print('wrote', fname)


def along_s(Rq, Ro, D1, fname):
    s = np.array([r['s'] for r in Rq])
    zg = np.linspace(-300, 300, 241)
    Wq = np.array([np.interp(zg, (r['z'] - zmid(r)) * 1e6, proj(r, r['W']), left=np.nan, right=np.nan)
                   for r in Rq])
    Fq = np.array([np.interp(zg, (r['z'] - zmid(r)) * 1e6, proj(r, r['F']), left=np.nan, right=np.nan)
                   for r in Rq])
    # the 1D run on its OWN kicks (its schedule can differ from the 2D one), placed on the 2D kick-number axis
    # by interpolating s; z relative to the nearest 2D kick's mesh centre
    idx = np.arange(len(s))
    s1 = np.array(sorted(D1)) if D1 else np.zeros(0)
    W1 = np.full((s1.size, zg.size), np.nan)
    for i, k in enumerate(s1):
        d = D1[k]; r = Rq[nearest(Rq, k)]
        W1[i] = np.interp(zg, (d[0] - zmid(r)) * 1e6, d[1], left=np.nan, right=np.nan)
    x1 = np.interp(s1, s, idx)
    peak = lambda A: np.nanmax(np.abs(A), axis=1)
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(5, 1, height_ratios=[0.22, 0.9, 1, 1, 1], hspace=0.14)
    axL = fig.add_subplot(gs[0])
    axs = [fig.add_subplot(gs[i], sharex=axL) for i in range(1, 5)]
    ele = [r['ele'] for r in Rq]
    ele = [e.upper().split('_')[0] for e in ele]          # Indranil's lattices use lower-case, split names
    for i, e in enumerate(ele):
        if e.startswith('B') or e.startswith('WIG'):
            axL.axvspan(i - 0.5, i + 0.5, color='#c3c2b7' if e.startswith('B') else '#e4e3df', lw=0)
    for e in ('B1LE', 'B2LE', 'B3LE', 'B3RE', 'B2RE', 'B1RE'):
        j = [i for i, x in enumerate(ele) if x == e]
        if j:
            axL.text(np.mean(j), 0.5, e, ha='center', va='center', fontsize=8.5, color=INK)
    axL.set_yticks([])
    axL.set_title('lattice (dark: dipoles, light: vertical wiggler); horizontal axis is the kick number', loc='left', fontsize=9)
    ax = axs[0]
    ax.semilogy(idx, peak(Wq), color=COL['q'], lw=1.4, label=LABEL.get(RUN, RUN))
    ax.semilogy(idx, [np.abs(proj(Ro[nearest(Ro, x)], Ro[nearest(Ro, x)]['W'])).max() for x in s],
                color=COL['old'], lw=1.2, label=LABEL.get(CMP, CMP))
    ax.semilogy(x1, peak(W1), color=COL['1d'], lw=1.2, ls='--', label=LABEL.get(RUN1, RUN1))
    ax.set_ylabel('max |W(z)| [MeV/m]')
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    ax.grid(color=GRID, lw=0.5)
    for ax, M, lab in ((axs[1], Wq, f'{LABEL.get(RUN, RUN)}: longitudinal W(z), each kick normalised to its max |W|'),
                       (axs[2], W1, '1D: W(z), each kick normalised'),
                       (axs[3], Fq, f'{LABEL.get(RUN, RUN)}: transverse force F(z), each kick normalised')):
        N = M / np.nanmax(np.abs(M), axis=1, keepdims=True)
        xs = x1 if M is W1 else idx
        if xs.size:
            ax.pcolormesh(xs, zg, N.T, cmap=CMAP, norm=TwoSlopeNorm(0, -1, 1), shading='nearest', rasterized=True)
        ax.set_ylabel('z − z_mid [µm]')
        ax.set_title(lab, loc='left', fontsize=9)
    for ax in [axL] + axs[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    t = np.arange(0, len(s), 15)
    axs[-1].set_xticks(t)
    axs[-1].set_xticklabels([f'{s[i]:.1f}' for i in t], rotation=60, fontsize=7)
    axs[-1].set_xlabel('s [m] at the kick')
    fig.suptitle(f'BC20 wakes along the lattice, {RUN} ({len(s)} kicks)', x=0.01, ha='left',
                 fontsize=11, color=INK, y=0.995)
    fig.savefig(os.path.join(RES, fname), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('wrote', fname)


def main():
    global RUN, CMP, RUN1
    import sys
    RUN, CMP, RUN1 = (sys.argv[1:] + [RUN, CMP, RUN1][len(sys.argv[1:]):])[:3]
    tag = '' if RUN == 'dfcsr_2d_q' else f'_{RUN}'
    Rq, Ro, D1 = load2d(RUN), load2d(CMP), load1d(RUN1)
    print(f'{RUN}: {len(Rq)} kicks; {CMP}: {len(Ro)}; {RUN1}: {len(D1)} 1D kicks, '
          f'{sum(round(r["s"], 6) in D1 for r in Rq)} matched')
    maps(Rq, 'W', 'Longitudinal wake dE/ds', f'bc20_wake_maps_long{tag}.png')
    maps(Rq, 'F', 'Transverse force', f'bc20_wake_maps_trans{tag}.png')
    late = [p for p in PICK if p[0] >= 38.2]
    maps(Rq, 'W', 'Longitudinal wake, zoomed on the spike (±12 µm)', f'bc20_wake_maps_spike{tag}.png',
         zoom=12.0, picks=late)
    zproj(Rq, Ro, D1, f'bc20_wake_zproj{tag}.png')
    along_s(Rq, Ro, D1, f'bc20_wakes_along_s{tag}.png')


if __name__ == '__main__':
    main()
