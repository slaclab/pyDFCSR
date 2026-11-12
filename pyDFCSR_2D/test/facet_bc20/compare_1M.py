"""
BC20 runs on the 1M upsampled beam, at ENDBC20 and by section, against CSR off (nocsr_1M).

    conda run -n pydfcsr python pyDFCSR_2D/test/facet_bc20/compare_1M.py [run ...]

Energy loss by section from the mean pz at the element exits (exit_mean), the rest from the final
beam (coords_end): rms of the CSR-induced energy change, projected eps_nx with 1st / 3rd order
dispersion removed (compare_csr_runs.emit_n), and the spike slice emittance (slice_emittance:
40 equal-charge slices, spike = slices shorter than 1 um; raw / slice-dispersion-corrected).
Writes compare_1M.txt.
"""
import os
import sys

import numpy as np

import compare_csr_runs as C
from slice_emittance import slices

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'benchmark_results', 'facet_bc20', 'runs')
DEFAULT = ['dfcsr_2d_peak_1M_sz6.5cap_m40_k115', 'dfcsr_2d_peak_1M', 'dfcsr_1d_1M']
# section ends (element exits): -> Q2ER, B2RE, drift + wiggler (to the B1RE entrance), B1RE
ENDS = ('Q2ER', 'B2RE', 'DM11Q_D1ET2B', 'B1RE')


def row(name, off, p0c):
    d = np.load(os.path.join(RUNS, f'{name}.npz'))
    names = list(d['names'])
    k = [names.index(e) for e in ENDS]
    dE = (d['exit_mean'][:, 5] - off['exit_mean'][:, 5]) * p0c / 1e6
    cum = dE[k]
    sec = np.diff(np.concatenate([[0.0], cum]))
    c, co = d['coords_end'], off['coords_end']
    sl = slices(c, p0c)
    sp = (sl[:, 2] - sl[:, 1]) < 1e-6
    return dict(sec=sec, total=cum[-1], rms=np.std((c[5] - co[5]) * p0c) / 1e6,
                e1=C.emit_n(c, p0c, 1) * 1e6, e3=C.emit_n(c, p0c, 3) * 1e6,
                sraw=sl[sp, 3].mean() * 1e6 if sp.any() else np.nan,
                scor=sl[sp, 4].mean() * 1e6 if sp.any() else np.nan,
                kicks=int(d['n_kicks']), t=float(d['runtime']) / 60)


def main(runs):
    off = np.load(os.path.join(RUNS, 'nocsr_1M.npz'))
    p0c = float(off['p0c'])
    co = off['coords_end']
    L = ['BC20, 1M beam; energies in MeV, emittances in um',
         f'{"run":36s} {"->Q2ER":>7s} {"B2RE":>7s} {"drift+wig":>9s} {"B1RE":>7s} {"total":>7s} {"rms dE":>7s} '
         f'{"enx 1st/3rd":>12s} {"spike slice raw/corr":>21s} {"kicks":>5s} {"min":>5s}']
    for r in runs:
        q = row(r, off, p0c)
        L.append(f'{r:36s} ' + ' '.join(f'{v:+7.3f}' for v in q['sec']) + f' {q["total"]:+7.3f} {q["rms"]:7.2f} '
                 f'{q["e1"]:6.2f}/{q["e3"]:5.2f} {q["sraw"]:10.1f}/{q["scor"]:5.1f}      {q["kicks"]:5d} {q["t"]:5.0f}')
    sl = slices(co, p0c)
    sp = (sl[:, 2] - sl[:, 1]) < 1e-6
    L.append(f'{"CSR off (nocsr_1M)":36s} {"":47s} {C.emit_n(co, p0c, 1)*1e6:6.2f}/{C.emit_n(co, p0c, 3)*1e6:5.2f} '
             f'{sl[sp, 3].mean()*1e6:10.1f}/{sl[sp, 4].mean()*1e6:5.1f}')
    print('\n'.join(L))
    open(os.path.join(HERE, 'compare_1M.txt'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main(sys.argv[1:] or DEFAULT)
