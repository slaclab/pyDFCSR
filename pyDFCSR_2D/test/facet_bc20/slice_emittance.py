"""
Slice emittance at ENDBC20 for the BC20 runs (same 20 000 particles in every run).

    conda run -n pydfcsr python pyDFCSR_2D/test/facet_bc20/slice_emittance.py

Each run's own beam is cut into equal-charge z slices (NSLICE slices of N/NSLICE particles), so
the spike, which holds ~45% of the charge in ~3 um, gets about half the slices. Per slice:
  raw        eps_n = gamma * sqrt(det cov(x, px))
  corrected  the same after removing the slice's own linear x-delta and px-delta correlation
             (BC20 leaves eta_x = 16 mm at ENDBC20, so the slice energy spread, which CSR
             raises, maps into x even within a slice)
gamma from each slice's mean energy. Writes slice_emittance.txt and
benchmark_results/facet_bc20/slice_emittance.png.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import compare_csr_runs as C

HERE = os.path.dirname(os.path.abspath(__file__))
NSLICE = 40
RUNS = [('CSR off', 'off', '#52514e', '-'),
        ('Tao 1D CSR (n_bin 128)', 'tao_csr', '#2a78d6', '-'),
        ('pyDFCSR 1D (peak 20/FWHM)', 'dfcsr_1d_pk20', '#eb6834', '--'),
        ('pyDFCSR 2D, old grids', 'dfcsr_2d', '#1baf7a', '-'),
        ('pyDFCSR 2D quantile frame', 'dfcsr_2d_q', '#4a3aa7', '-'),
        ('pyDFCSR 2D quantile, no transverse', 'dfcsr_2d_q_noT', '#008300', ':')]


def slices(c, p0c, n=NSLICE):
    x, px, z, pz = c[0], c[1], c[4], c[5]
    order = np.argsort(z)
    out = []
    for idx in np.array_split(order, n):
        g = (1 + pz[idx].mean()) * p0c / C.ME
        raw = np.sqrt(np.linalg.det(np.cov(x[idx], px[idx]))) * g
        d = pz[idx] - pz[idx].mean()
        A = np.column_stack([np.ones(d.size), d])
        res = lambda u: u - A @ np.linalg.lstsq(A, u, rcond=None)[0]
        cor = np.sqrt(np.linalg.det(np.cov(res(x[idx]), res(px[idx])))) * g
        zs = z[idx]
        out.append((zs.mean(), zs.min(), zs.max(), raw, cor, d.std()))
    return np.array(out)


def main():
    s, names, types, p0c, data, init, rt = C.load()
    for _, f, _, _ in RUNS:
        if f not in data:
            data[f] = np.load(os.path.join(C.RUNS, f + '.npz'))['coords']
    Q = 1.6e-9 / NSLICE
    res = {f: slices(data[f][-1], p0c) for _, f, _, _ in RUNS}
    zc0 = data['off'][-1][4].mean()

    # summary: charge-weighted mean over slices (all slices hold equal charge), and over the
    # spike = slices narrower than 1 um (current above ~40 kA)
    L = [f'Slice emittance at ENDBC20, {NSLICE} equal-charge slices of {20000 // NSLICE} particles; '
         'eps_n in um. spike = slices shorter than 1 um.',
         f'{"run":>36s} {"mean raw":>9s} {"mean corr":>10s} {"spike raw":>10s} {"spike corr":>11s} '
         f'{"n spike":>8s} {"projected 1st":>14s}']
    for lab, f, _, _ in RUNS:
        r = res[f]
        sp = (r[:, 2] - r[:, 1]) < 1e-6
        L.append(f'{lab:>36s} {r[:, 3].mean()*1e6:9.2f} {r[:, 4].mean()*1e6:10.2f} '
                 f'{r[sp, 3].mean()*1e6:10.2f} {r[sp, 4].mean()*1e6:11.2f} {sp.sum():8d} '
                 f'{C.emit_n(data[f][-1], p0c, 1)*1e6:14.2f}')
    txt = '\n'.join(L)
    print(txt)
    open(os.path.join(HERE, 'slice_emittance.txt'), 'w').write(txt + '\n')

    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': C.INK2, 'axes.labelcolor': C.INK,
                         'xtick.color': C.INK2, 'ytick.color': C.INK2, 'text.color': C.INK})
    fig, axes = plt.subplots(3, 2, figsize=(12, 8.6), sharex='col', facecolor=C.SURF,
                             gridspec_kw=dict(width_ratios=(1.5, 1)))
    for col, (lo, hi) in enumerate(((-140e-6, 60e-6), (25e-6, 45e-6))):
        for lab, f, c, ls in RUNS:
            r = res[f]
            zz = (r[:, 0] - zc0) * 1e6
            cur = Q / ((r[:, 2] - r[:, 1]) / C.C0) / 1e3
            axes[0, col].plot(zz, cur, color=c, ls=ls, lw=1.6, marker='.', ms=4, label=lab)
            axes[1, col].plot(zz, r[:, 3] * 1e6, color=c, ls=ls, lw=1.6, marker='.', ms=4)
            axes[2, col].plot(zz, r[:, 4] * 1e6, color=c, ls=ls, lw=1.6, marker='.', ms=4)
        axes[0, col].set_ylabel('slice current [kA]')
        axes[1, col].set_ylabel('slice εnx, raw [µm]')
        axes[2, col].set_ylabel('slice εnx, slice dispersion\nremoved [µm]')
        axes[2, col].set_xlabel('z − ⟨z⟩ of the CSR-off beam [µm]  (head at +z)')
        axes[2, col].set_xlim(lo * 1e6, hi * 1e6)
        for ax in axes[:, col]:
            C.style(ax)
    axes[0, 0].legend(fontsize=8, frameon=False, loc='upper left')
    axes[0, 0].set_title(f'ENDBC20, {NSLICE} equal-charge slices', loc='left', fontsize=10)
    axes[0, 1].set_title('the spike', loc='left', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(C.RES, 'slice_emittance.png'), dpi=140, facecolor=C.SURF)
    plt.close(fig)


if __name__ == '__main__':
    main()
