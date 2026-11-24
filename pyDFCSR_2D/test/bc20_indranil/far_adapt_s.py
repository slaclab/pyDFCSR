"""
Density-adaptive far-region s' columns (CSR_integration.far_adapt) against the fixed 400-column
layout, whole wake mesh on pickled states: max |dW|, |dF| relative to the horn, wall time, realised
far column count.

    conda run -n pydfcsr python ../test/bc20_indranil/far_adapt_s.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'facet_bc20'))
import dump_kick_state as D  # noqa: E402

D.RUNS = os.path.join(HERE, '..', 'benchmark_results', 'bc20_indranil', 'runs')


def mesh(sim):
    pos = sim.beam.position
    t0 = time.time()
    far = []
    r = []
    for z, x in zip(sim.CSR_zmesh, sim.CSR_xmesh):
        r.append(sim.get_CSR_wake(pos + z, x))
        far.append(sim._last_region_nodes[0] if getattr(sim, '_last_region_nodes', None) else np.nan)
    r = np.array(r)
    nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins
    return r[:, 0].reshape(nx, nz), r[:, 1].reshape(nx, nz), time.time() - t0, float(np.nanmean(far))


def main():
    for run, st in (('loss', 37.74), ('loss', 48.73), ('gain', 0.5), ('loss', 0.148)):
        sim = D.load_state(D.state_path(run, st))
        ip = sim.integration_params
        ip.xbins = 100
        W0, F0, t0, c0 = mesh(sim)
        horn = np.abs(W0).max(0).argmax()
        wh, fh = np.abs(W0[:, horn]).max(), np.abs(F0[:, horn]).max()
        print(f'{run} s={sim.beam.position:.3f} (x\' 100): fixed far layout {t0:.0f} s, {c0:.0f} far columns; horn W {wh:.3f} F {fh:.4f}')
        for base in (50, 25):
            ip.far_adapt, ip.far_adapt_base = True, base
            W, F, t, c = mesh(sim)
            print(f'   far_adapt base {base:3d}: {t:.0f} s ({t/t0*100:.0f}%), {c:.0f} far columns; max|dW| {np.abs(W-W0).max()/wh*100:.2f}% of horn W (rms {np.sqrt(np.mean((W-W0)**2))/wh*100:.3f}%), max|dF| {np.abs(F-F0).max()/fh*100:.1f}% of horn F', flush=True)
        ip.far_adapt = False


if __name__ == '__main__':
    main()
