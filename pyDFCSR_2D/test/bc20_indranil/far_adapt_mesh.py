"""
Whole-mesh check of far_adapt_x on a pickled state: W and F meshes with the run's settings, with
far_adapt_x, and with 8x the far-region x' nodes as the reference; speckle (rms of F - 3x3 median
over max|F|), max |F - ref| and max |W - ref| relative to the horn values, and the wall time of each.

    conda run -n pydfcsr python ../test/bc20_indranil/far_adapt_mesh.py <run> <s_target>
"""
import os
import sys
import time

import numpy as np
from scipy.ndimage import median_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'facet_bc20'))
import dump_kick_state as D  # noqa: E402

D.RUNS = os.path.join(HERE, '..', 'benchmark_results', 'bc20_indranil', 'runs')


def mesh(sim):
    pos = sim.beam.position
    t0 = time.time()
    r = np.array([sim.get_CSR_wake(pos + z, x) for z, x in zip(sim.CSR_zmesh, sim.CSR_xmesh)])
    nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins
    return r[:, 0].reshape(nx, nz), r[:, 1].reshape(nx, nz), time.time() - t0


def speckle(F):
    return float(np.sqrt(np.mean((F - median_filter(F, size=(3, 3), mode='nearest')) ** 2)) / np.abs(F).max())


def main(run, s_target):
    sim = D.load_state(D.state_path(run, s_target))
    ip = sim.integration_params
    print(f'{run} at s = {sim.beam.position:.4f}')
    out = {}
    for name, fac in (('base', {}), ('far_adapt_x', dict(far_adapt_x=True)), ("x' x8 (reference)", dict(xbins=8 * ip.xbins))):
        keep = {k: getattr(ip, k) for k in fac}
        for k, v in fac.items():
            setattr(ip, k, v)
        W, F, dt = mesh(sim)
        for k, v in keep.items():
            setattr(ip, k, v)
        out[name] = (W, F, dt)
        print(f'  {name:20s}: {dt:5.0f} s; max|W| {np.abs(W).max():.3f}, max|F| {np.abs(F).max():.4f}, F speckle {speckle(F):.3f}', flush=True)
    Wr, Fr, _ = out["x' x8 (reference)"]
    horn = np.abs(Wr).max(0).argmax()
    fh, wh = np.abs(Fr[:, horn]).max(), np.abs(Wr[:, horn]).max()
    print(f'  reference horn column: max|F| {fh:.4f}, max|W| {wh:.3f}')
    for name in ('base', 'far_adapt_x'):
        W, F, _ = out[name]
        print(f'  {name:12s} vs reference: max|dF| {np.abs(F - Fr).max():.4f} ({np.abs(F - Fr).max()/fh*100:.0f}% of the horn F), rms dF {np.sqrt(np.mean((F-Fr)**2)):.4f}; '
              f'max|dW| {np.abs(W - Wr).max():.4f} ({np.abs(W - Wr).max()/wh*100:.1f}% of the horn W)')
    np.savez(os.path.join(D.RUNS, f'far_adapt_mesh_{run}_s{float(s_target):.2f}.npz'), **{f'W_{k}': v[0] for k, v in out.items()}, **{f'F_{k}': v[1] for k, v in out.items()},
             z=np.asarray(sim.CSR_zrange), xi=np.asarray(sim.CSR_xrange_transformed))


if __name__ == '__main__':
    main(*sys.argv[1:])
