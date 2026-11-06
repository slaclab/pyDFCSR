"""
Step 3 of 3: pyDFCSR with CSR off vs Bmad/Tao with every collective effect off, through BC20.

Runs CSR2D on example/input/facet_bc20_nocsr_config.yaml (mode: manual, so every element
boundary is a node), records the beam after every bmad-x call, and compares it at each element
exit with Tao's beam at the same s (bc20_export.npz). Writes
    test/benchmark_results/facet_bc20/tracking_vs_bmad.png
    test/facet_bc20/compare_tracking.txt          (table, also printed)

    cd pyDFCSR_2D/example && conda run -n pydfcsr python ../test/facet_bc20/compare_tracking.py
"""
import contextlib
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(ROOT, '..'))
from pyDFCSR_2D.CSR import CSR2D  # noqa: E402

OUT_DIR = os.path.join(ROOT, 'test', 'benchmark_results', 'facet_bc20')
NAMES = ('x', 'px', 'y', 'py', 'z', 'pz')


def run_pydfcsr():
    os.chdir(os.path.join(ROOT, 'example'))
    with contextlib.redirect_stdout(io.StringIO()):
        csr = CSR2D(input_file='input/facet_bc20_nocsr_config.yaml')
    rec = []
    track0 = csr.beam.track

    def track(element, step_size, update_step=True):
        track0(element, step_size, update_step)
        p = csr.beam.particle
        rec.append((csr.beam.position, np.array([p.x, p.px, p.y, p.py, p.z, p.pz])))
    csr.beam.track = track
    with contextlib.redirect_stdout(io.StringIO()):
        csr.run()
    return csr, rec


def main():
    X = np.load(os.path.join(HERE, 'bc20_export.npz'))
    elements = json.loads(str(X['elements']))
    s_tao = np.array([e['s'] for e in elements]) - elements[0]['s']
    tao = {'fine': X['coords_fine'], 'as_is': X['coords_as_is']}

    csr, rec = run_pydfcsr()
    ends = csr.lattice.distance
    keys = [k for k in csr.lattice.lattice_config if k != 'step_size']
    pos = np.array([r[0] for r in rec])

    rows = []
    for k, s_end in zip(keys, ends):
        i = int(np.argmin(np.abs(pos - s_end)))
        assert abs(pos[i] - s_end) < 1e-9, (k, pos[i], s_end)
        # last Tao element at this s: the zero-length ones after it are inert
        j = int(np.flatnonzero(np.abs(s_tao - s_end) < 1e-9)[-1])
        d = {m: rec[i][1] - tao[m][j][:6] for m in tao}
        rows.append((k, s_end, {m: np.abs(v).max(1) for m, v in d.items()},
                     {m: v.std(1) for m, v in d.items()}))

    rms = tao['fine'][-1][:6].std(1)
    lines = [f'pyDFCSR (CSR off) - Tao (collective effects off), max over {rec[-1][1].shape[1]} '
             f'particles, at each element exit',
             f'{"element":>16s} {"s [m]":>9s} ' + ' '.join(f'{n:>9s}' for n in NAMES)]
    for k, s, mx, _ in rows:
        lines.append(f'{k:>16s} {s:9.4f} ' + ' '.join(f'{v:9.1e}' for v in mx['fine']))
    fin = rows[-1]
    lines += ['', 'at ENDBC20:',
              '  beam rms                  ' + ' '.join(f'{v:9.2e}' for v in rms),
              '  max |pyDFCSR - Tao|       ' + ' '.join(f'{v:9.2e}' for v in fin[2]['fine']),
              '  rms (pyDFCSR - Tao)       ' + ' '.join(f'{v:9.2e}' for v in fin[3]['fine']),
              '  Tao num_steps=1 sextupoles vs converged (Tao - Tao):',
              '  max                       ' + ' '.join(
                  f'{v:9.2e}' for v in np.abs(tao['as_is'][-1][:6] - tao['fine'][-1][:6]).max(1)),
              '  rms                       ' + ' '.join(
                  f'{v:9.2e}' for v in (tao['as_is'][-1][:6] - tao['fine'][-1][:6]).std(1))]
    for m in ('fine', 'as_is'):
        p = rec[-1][1]
        t = tao[m][-1][:6]
        def emit(u, pu, d):
            c = np.cov([u, pu, d])
            # dispersion-corrected, as Twiss in pyDFCSR's twiss.py
            c2 = c[:2, :2] - np.outer(c[:2, 2], c[:2, 2]) / c[2, 2]
            return np.sqrt(np.linalg.det(c2))
        lines.append(f'  eps_x geometric [m] pyDFCSR {emit(p[0], p[1], p[5]):.6e}  '
                     f'Tao({m}) {emit(t[0], t[1], t[5]):.6e}; '
                     f'sigma_z pyDFCSR {p[4].std():.6e} Tao({m}) {t[4].std():.6e}')
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(HERE, 'compare_tracking.txt'), 'w') as f:
        f.write(txt + '\n')

    np.savez(os.path.join(HERE, 'compare_tracking.npz'),
             s=np.array([r[1] for r in rows]), names=np.array([r[0] for r in rows]),
             maxdiff=np.array([r[2]['fine'] for r in rows]),
             rmsdiff=np.array([r[3]['fine'] for r in rows]),
             maxdiff_as_is=np.array([r[2]['as_is'] for r in rows]),
             types=np.array([csr.lattice.lattice_config[k]['type'] for k in keys]),
             beam_rms=rms)


if __name__ == '__main__':
    main()
