"""
Isolated spikes of the transverse force on the loss beam's wake mesh in B2RE / B1RE (single cells
0.1-1.8 MeV/m against a median ~0, in the tail columns behind the horns). On a pickled state
(dump_state.py loss <s> <s_start>) the F mesh is recomputed, the most isolated cells (|F - 3x3
median|) listed with their far / near / patch split, and recomputed under quadrature variants:
x' nodes x2/x4/x8, far columns x4, near_adapt eps/4, x' rounds x2, band margin x2, W1 only.

    conda run -n pydfcsr python ../test/bc20_indranil/f_speckle.py <run> <s_target> [n_cells] [--variants]
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
from near_field_check import wake_parts  # noqa: E402

D.RUNS = os.path.join(HERE, '..', 'benchmark_results', 'bc20_indranil', 'runs')
VARIANTS = {
    "x' x2": dict(xbins=lambda v: 2 * v), "x' x4": dict(xbins=lambda v: 4 * v), "x' x8": dict(xbins=lambda v: 8 * v),
    'far cols x4': dict(far_zbins=lambda v: 4 * v), 'near eps/4': dict(near_adapt_eps=lambda v: v / 4),
    "x' rounds x2": dict(near_adapt_x_rounds=lambda v: 2 * v, near_adapt_x_max=lambda v: 2 * v),
    'margin x2': dict(xi_band_margin=lambda v: 2 * v), 'W1 only': dict(transverse_terms='W1'),
    'far adapt_x': dict(far_adapt_x=True), "far adapt_x + x' x2": dict(far_adapt_x=True, xbins=lambda v: 2 * v),
}


def main(run, s_target, n_cells=6, *flags):
    n_cells = int(n_cells)
    sim = D.load_state(D.state_path(run, s_target))
    pos = sim.beam.position
    ip = sim.integration_params
    nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins
    z = np.asarray(sim.CSR_zrange); xi = np.asarray(sim.CSR_xrange_transformed); sx = sim.beam._sigma_x_transform
    print(f'{run} at s = {pos:.4f}: in bend {sim.inbend}, tau {sim._frame_tilt():+.3f}, sigma_z {sim.beam.sigma_z*1e6:.1f} um, sigma_xi {sx*1e6:.1f} um')
    t0 = time.time()
    r = np.array([sim.get_CSR_wake(pos + zz, xx) for zz, xx in zip(sim.CSR_zmesh, sim.CSR_xmesh)])
    W, F = r[:, 0].reshape(nx, nz), r[:, 1].reshape(nx, nz)
    print(f'  mesh in {time.time()-t0:.0f} s: max|W| {np.abs(W).max():.3g}, max|F| {np.abs(F).max():.3g}; F on the horn (max over rows at the |W| max column) {np.abs(F[:, np.abs(W).max(0).argmax()]).max():.3g}')
    med = median_filter(F, size=(3, 3), mode='nearest'); d = np.abs(F - med)
    flat = np.argsort(d.ravel())[::-1][:n_cells]
    cells = [(i // nz, i % nz) for i in flat]
    print(f'  most isolated F cells (row = xi, col = z): F, 3x3 median, W; far/near/prev/patch of F:')
    for (i, c) in cells:
        s, x = pos + sim.CSR_zmesh[i * nz + c], sim.CSR_xmesh[i * nz + c]
        p = wake_parts(sim, s, x)
        print(f'    row {i:2d} col {c:2d} (z {z[c]*1e6:+7.1f} um, xi {xi[i]/sx:+5.2f} sx): F {F[i, c]:+8.4f} med {med[i, c]:+8.4f} W {W[i, c]:+7.3f} | '
              f'far {p["far"][1]:+8.4f} near {p["near"][1]:+8.4f} prev {p["prev"][1]:+8.4f} patch {p["patch"][1]:+8.4f}  n_sp {p["n_sp"]}')
    np.savez(os.path.join(D.RUNS, f'f_speckle_{run}_s{float(s_target):.2f}.npz'), W=W, F=F, z=z, xi=xi, s=pos)
    if '--variants' in flags:
        keep = {}
        for fac in VARIANTS.values():
            for k in fac:
                keep[k] = getattr(ip, k)
        print('  variants at those cells (F):')
        print('    ' + f'{"base":14s}: ' + '  '.join(f'{F[i, c]:+9.4f}' for (i, c) in cells))
        for name, fac in VARIANTS.items():
            for k, v in keep.items():
                setattr(ip, k, v)
            for k, v in fac.items():
                setattr(ip, k, v(keep[k]) if callable(v) else v)
            t0 = time.time()
            vals = [sim.get_CSR_wake(pos + sim.CSR_zmesh[i * nz + c], sim.CSR_xmesh[i * nz + c])[1] for (i, c) in cells]
            print('    ' + f'{name:14s}: ' + '  '.join(f'{v:+9.4f}' for v in vals) + f'   ({time.time()-t0:.1f} s)', flush=True)
        for k, v in keep.items():
            setattr(ip, k, v)


if __name__ == '__main__':
    main(*sys.argv[1:])
