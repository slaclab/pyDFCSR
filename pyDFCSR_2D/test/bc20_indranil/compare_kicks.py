"""
Kick-schedule convergence of an Indranil run: <beam> (reference schedule) against <beam>_k2 (about
twice the kicks). Energy change per section and emittance from the .npz files; the projected wake of
the two runs compared at every reference kick (nearest k2 kick within 3 cm), as rms difference over
rms W, median by section.

    conda run -n pydfcsr python ../test/bc20_indranil/compare_kicks.py <beam>
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'facet_bc20'))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
import plot_bc20_wakes as P  # noqa: E402
import plot_indranil as PI  # noqa: E402
import compare_csr_runs as C  # noqa: E402

RUNS = os.path.join(HERE, '..', 'benchmark_results', 'bc20_indranil', 'runs')
P.RUNS = RUNS


def main(beam):
    off = np.load(os.path.join(RUNS, f'{beam}_off.npz'))
    p0c = float(off['p0c']); names = list(off['names']); k = [names.index(e) for e in PI.ENDS]
    print(f'{"run":10s} {"->q2er":>8s} {"b2re":>8s} {"drift":>8s} {"b1re":>8s} {"D32":>8s} {"total":>8s} {"enx 1st/3rd":>12s} {"kicks":>5s} {"min":>5s}')
    for run in (beam, f'{beam}_k2', f'{beam}_1d'):
        d = np.load(os.path.join(RUNS, f'{run}.npz'))
        dE = (d['exit_mean'][:, 5] - off['exit_mean'][:, 5]) * p0c / 1e6
        sec = np.diff(np.concatenate([[0.0], dE[k]]))
        c = d['coords_end']
        print(f'{run:10s} ' + ' '.join(f'{v:+8.3f}' for v in sec) + f' {dE[k][-1]:+8.3f} {C.emit_n(c, p0c, 1)*1e6:6.2f}/{C.emit_n(c, p0c, 3)*1e6:5.2f} {int(d["n_kicks"]):5d} {float(d["runtime"])/60:5.0f}')
    R1, R2 = P.load2d(beam), P.load2d(f'{beam}_k2')
    print(f'\nprojected W, {beam} vs {beam}_k2 at the nearest k2 kick (within 3 cm): rms(W1 - W2) / rms(W1), median [max] by section')
    for lo, hi, lab in PI.SECTIONS:
        vals = []
        for r in R1:
            if not (lo < r['s'] <= hi):
                continue
            j = P.nearest(R2, r['s'])
            if abs(R2[j]['s'] - r['s']) > 0.03:
                continue
            w1 = P.proj(r, r['W']); z = r['z']
            w2 = np.interp(z, R2[j]['z'], P.proj(R2[j], R2[j]['W']))
            vals.append(np.sqrt(np.mean((w1 - w2) ** 2)) / np.sqrt(np.mean(w1 ** 2)))
        if vals:
            print(f'   {lab:22s} {np.median(vals):.3f} [{np.max(vals):.3f}]  ({len(vals)} kicks compared)')


if __name__ == '__main__':
    main(sys.argv[1])
