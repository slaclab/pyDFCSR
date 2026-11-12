"""
Beam along BC20 for the 1M runs, at every element exit (exit_mean, exit_cov, exit_ipk, exit_fwhm):
mean energy change against CSR off, eps_nx with the linear x-delta and px-delta correlation removed
(from the 6x6 covariance), peak current and FWHM of the highest peak (0.5 um bins).

    conda run -n pydfcsr python pyDFCSR_2D/test/facet_bc20/plot_along_s_1M.py [run ...]

Writes benchmark_results/facet_bc20/bc20_along_s_1M.png.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'benchmark_results', 'facet_bc20')
RUNS = os.path.join(RES, 'runs')
ME = 0.51099895e6
DEFAULT = [('2D peak 1M: 6.5-bin z smoothing capped, 12×40 mesh, x on 0.2% quantiles, 116 kicks',
            'dfcsr_2d_peak_1M_sz6.5cap_m40_k115', '#4a3aa7', '-'),
           ('2D peak 1M: 2-bin z smoothing, 12×80 mesh, 222 kicks', 'dfcsr_2d_peak_1M', '#1baf7a', '-'),
           ('1D, 1M', 'dfcsr_1d_1M', '#eb6834', '--'),
           ('CSR off, 1M', 'nocsr_1M', '#898781', ':')]


def emit1(S, pz_mean, p0c):
    """eps_n from a 6x6 covariance with the linear dependence on delta (index 5) removed"""
    A = S[np.ix_([0, 1], [0, 1])]
    b = S[np.ix_([0, 1], [5])]
    C = A - b @ b.T / S[5, 5]
    return np.sqrt(np.linalg.det(C)) * (1 + pz_mean) * p0c / ME


def main(series):
    off = np.load(os.path.join(RUNS, 'nocsr_1M.npz'))
    p0c = float(off['p0c'])
    s = off['s']
    fig, axs = plt.subplots(4, 1, figsize=(11, 9.5), sharex=True)
    for lab, f, c, ls in series:
        d = np.load(os.path.join(RUNS, f + '.npz'))
        assert np.allclose(d['s'], s)
        dE = (d['exit_mean'][:, 5] - off['exit_mean'][:, 5]) * p0c / 1e6
        en = [emit1(S, m[5], p0c) * 1e6 for S, m in zip(d['exit_cov'], d['exit_mean'])]
        kw = dict(color=c, ls=ls, lw=1.6 if ls == '-' else 1.3, label=lab)
        if f != 'nocsr_1M':
            axs[0].plot(s, dE, **kw)
        axs[1].plot(s, en, **kw)
        axs[2].semilogy(s, d['exit_ipk'] / 1e3, **kw)
        axs[3].semilogy(s, d['exit_fwhm'] * 1e6, **kw)
    names = list(off['names'])
    ends = np.asarray(s)
    starts = np.concatenate([[0.0], ends[:-1]])
    for a in axs:
        for k, n in enumerate(names):
            if n.startswith('B') and n[1].isdigit():
                a.axvspan(starts[k], ends[k], color='#c3c2b7', alpha=0.35, lw=0)
        a.grid(color='#e4e3df', lw=0.5)
    for k, n in enumerate(names):
        if n.startswith('B') and n[1].isdigit():
            axs[0].text(0.5 * (starts[k] + ends[k]), 0.02, n, transform=axs[0].get_xaxis_transform(),
                        ha='center', fontsize=7, color='#52514e')
    axs[0].set_ylabel('⟨ΔE⟩ vs CSR off [MeV]')
    axs[1].set_ylabel('εnx, linear dispersion\nremoved [µm]')
    axs[2].set_ylabel('peak current [kA]')
    axs[3].set_ylabel('FWHM of the peak [µm]')
    axs[3].set_xlabel('s from BEGBC20 [m]  (bends shaded)')
    axs[0].legend(fontsize=7.5, frameon=False, loc='lower left')
    fig.suptitle('BC20 with the 1M beam, at every element exit', x=0.01, ha='left', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'bc20_along_s_1M.png'), dpi=140)
    print('wrote bc20_along_s_1M.png')


if __name__ == '__main__':
    main(DEFAULT)
