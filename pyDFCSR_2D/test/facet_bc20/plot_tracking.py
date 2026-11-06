"""Plot compare_tracking.npz -> benchmark_results/facet_bc20/tracking_vs_bmad.png"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'benchmark_results', 'facet_bc20')
SERIES = ('#2a78d6', '#eb6834', '#1baf7a')       # reference categorical slots 1-3
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def main():
    d = np.load(os.path.join(HERE, 'compare_tracking.npz'))
    s, mx, typ, rms = d['s'], d['maxdiff'], d['types'], d['beam_rms']
    s0 = np.concatenate([[0.0], s[:-1]])
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                         'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK})
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, facecolor=SURF)
    panels = [((0, 2, 4), ('x', 'y', 'z'), 'max |pyDFCSR - Tao|  [m]'),
              ((1, 3, 5), ('px', 'py', 'pz'), 'max |pyDFCSR - Tao|  [1]')]
    for ax, (idx, names, ylab) in zip(axes, panels):
        ax.set_facecolor(SURF)
        for a, b, t in zip(s0, s, typ):
            if t in ('dipole', 'vbend'):
                ax.axvspan(a, b, color='#d9d8d3' if t == 'dipole' else '#ecebe7', lw=0, zorder=0)
        for c, i, n in zip(SERIES, idx, names):
            y = np.maximum(mx[:, i], 1e-18)
            ax.plot(s, y, color=c, lw=2, marker='o', ms=3, label=n, zorder=3)
            ax.annotate(f'{n}  ({mx[-1, i]:.1e}, rms beam {rms[i]:.1e})', (s[-1], y[-1]),
                        xytext=(6, 0), textcoords='offset points', va='center', color=INK2, fontsize=8.5)
        ax.set_yscale('log')
        ax.set_ylabel(ylab)
        ax.grid(True, which='major', color=GRID, lw=0.8)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.legend(loc='upper left', frameon=False, ncol=3)
    axes[0].set_title('FACET-II BC20, CSR off: pyDFCSR (bmad-x) vs Bmad/Tao, 20000 particles, '
                      'every element exit\nshaded: horizontal bends (dark), vertical wiggler (light)',
                      loc='left', fontsize=10.5)
    axes[1].set_xlabel('s from BEGBC20  [m]')
    axes[1].set_xlim(0, s[-1] * 1.32)
    fig.tight_layout()
    fn = os.path.join(OUT, 'tracking_vs_bmad.png')
    fig.savefig(fn, dpi=150, facecolor=SURF)
    print(fn)


if __name__ == '__main__':
    main()
