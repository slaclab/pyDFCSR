"""
Kick-schedule accuracy in 1D (20k beam, peak 20/FWHM bins) against a kick at every snapshot
(dfcsr_1d_pk20_k05, 2062 kicks), at ENDBC20. As in progress_facet_bc20.md 9d:

    conda run -n pydfcsr python pyDFCSR_2D/test/facet_bc20/kick_count_check.py

  <dE>, rms dE    relative to the reference's (CSR on - CSR off) value
  enx 1st/3rd     emittance error relative to the reference's CSR-induced growth
  pz, x           rms per-particle difference relative to the rms CSR-induced change
CSR off is Tao's tracking (bc20_export.npz), which pyDFCSR matches to 1e-4 sigma.
Kick counts from the run logs. Writes kick_count_check.txt.
"""
import glob
import os
import re

import numpy as np

from compare_csr_runs import emit_n

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'benchmark_results', 'facet_bc20', 'runs')


def main():
    X = np.load(os.path.join(HERE, 'bc20_export.npz'))
    p0c = float(X['p0c'])
    off = X['coords_fine'][-1][:6]
    ref = np.load(os.path.join(RUNS, 'dfcsr_1d_pk20_k05.npz'))['coords'][-1]
    d_ref = ref - off
    e1r, e3r = emit_n(ref, p0c, 1), emit_n(ref, p0c, 3)
    e1o, e3o = emit_n(off, p0c, 1), emit_n(off, p0c, 3)
    rows = []
    for f in sorted(glob.glob(os.path.join(RUNS, 'dfcsr_1d_pk20_k*.npz'))):
        name = os.path.basename(f)[:-4]
        if name == 'dfcsr_1d_pk20_k05':
            continue
        c = np.load(f)['coords'][-1]
        m = re.search(r'kicks=(\d+)', open(os.path.join(RUNS, f'log_{name}.txt')).read())
        rows.append((int(m.group(1)) if m else -1, name,
                     (c[5].mean() - ref[5].mean()) / d_ref[5].mean(),
                     ((c[5] - off[5]).std() - d_ref[5].std()) / d_ref[5].std(),
                     (emit_n(c, p0c, 1) - e1r) / (e1r - e1o), (emit_n(c, p0c, 3) - e3r) / (e3r - e3o),
                     np.std(c[5] - ref[5]) / np.std(d_ref[5]), np.std(c[0] - ref[0]) / np.std(d_ref[0])))
    out = [f'{"kicks":>5s}  {"run":34s} {"<dE>":>7s} {"rms dE":>7s} {"enx 1st":>8s} {"enx 3rd":>8s} {"pz":>6s} {"x":>6s}']
    for r in sorted(rows):
        out.append(f'{r[0]:5d}  {r[1]:34s} ' + ' '.join(f'{100*v:+6.2f}%' if i < 4 else f'{100*v:5.2f}%'
                                                         for i, v in enumerate(r[2:])))
    print('\n'.join(out))
    open(os.path.join(HERE, 'kick_count_check.txt'), 'w').write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
