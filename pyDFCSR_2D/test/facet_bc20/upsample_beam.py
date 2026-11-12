"""
Upsample the 20 000-particle BC20 beam to N particles by nearest-neighbour interpolation in 6D.

    conda run -n pydfcsr python pyDFCSR_2D/test/facet_bc20/upsample_beam.py [N]      (default 1e6)

The beam is whitened (its full 6x6 covariance), and each original particle p gets N/20000 - 1
clones p + lam (q - p), q one of its 10 nearest neighbours (chosen at random), lam uniform in
[-0.5, 0.5]. Clones stay on the beam's own local correlations (chirp, dispersion, the folds the
sextupoles make downstream), and a convex-ish combination of neighbours adds almost no
uncorrelated spread -- Gaussian jitter large enough to decorrelate the clones would add slice
energy spread and so broaden the spike the beam is compressed into. Not physical: the real fine
structure below the 20 000-particle sampling is unknown; the originals are kept as clone 0.
Writes example/input/facet_bc20_beam_<N>.txt (same columns, same reference P0C).
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(HERE, '..', '..', 'example', 'input')


def upsample(b, n_out, k=10, seed=0):
    rng = np.random.default_rng(seed)
    n = b.shape[0]
    m = int(round(n_out / n))
    mu = b.mean(0)
    L = np.linalg.cholesky(np.cov(b.T))
    wv = np.linalg.solve(L, (b - mu).T).T                 # whitened
    _, nn = cKDTree(wv).query(wv, k=k + 1)
    nn = nn[:, 1:]
    out = [b]
    for _ in range(m - 1):
        q = nn[np.arange(n), rng.integers(0, k, n)]
        lam = rng.uniform(-0.5, 0.5, n)[:, None]
        out.append(b + lam * (b[q] - b))
    return np.concatenate(out)


def main(n_out=1_000_000):
    b = np.loadtxt(os.path.join(INPUT, 'facet_bc20_beam.txt'))
    u = upsample(b, n_out)
    f = os.path.join(INPUT, f'facet_bc20_beam_{u.shape[0]}.txt')
    np.savetxt(f, u)
    names = ('x', 'px', 'y', 'py', 'z', 'pz')
    print(f'{f}: {u.shape[0]} particles')
    print('       ' + ' '.join(f'{n:>11s}' for n in names))
    print('mean20k ' + ' '.join(f'{v:11.3e}' for v in b.mean(0)))
    print('mean1M  ' + ' '.join(f'{v:11.3e}' for v in u.mean(0)))
    print('rms20k  ' + ' '.join(f'{v:11.3e}' for v in b.std(0)))
    print('rms1M   ' + ' '.join(f'{v:11.3e}' for v in u.std(0)))
    # uncorrelated energy spread: pz after removing a cubic fit in z, in 40 z slices
    for lab, a in (('20k', b), ('1M', u)):
        z, d = a[:, 4], a[:, 5]
        r = d - np.polyval(np.polyfit(z, d, 3), z)
        print(f'{lab}: pz residual after a cubic fit in z: rms {r.std():.4e}')


if __name__ == '__main__':
    main(int(float(sys.argv[1])) if len(sys.argv) > 1 else 1_000_000)
