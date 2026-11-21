"""
Where do the outlier wake values of a pickled Indranil state come from?

    conda run -n pydfcsr python ../test/bc20_indranil/state_outliers.py <run> <s_target> [n_worst] [--variants]

Recomputes the whole wake mesh at the pickled kick (dump_state.py), lists the n_worst cells of largest
|W| with their (z, xi) and the charge-weighted projection, decomposes the wake at the worst cells into
the far / near / prev / patch regions (near_field_check.wake_parts), and with --variants recomputes those
cells under integration variants (near_adapt off, 4x far columns, 4x x' nodes, band margin x2,
band_jacobian_min 0.5, no tangent bands, transverse W1 only).
Writes state_outliers_<run>_s<s>.npz (W, F, axes) next to the pickle.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'facet_bc20'))
import dump_kick_state as D  # noqa: E402
from near_field_check import wake_parts  # noqa: E402

RUNS = os.path.join(HERE, '..', 'benchmark_results', 'bc20_indranil', 'runs')
D.RUNS = RUNS
VARIANTS = {
    'no_near_adapt': dict(near_adapt=False, near_adapt_x=False),
    'far4': dict(far_zbins=lambda v: 4 * v),
    'xbins4': dict(xbins=lambda v: 4 * v),
    'margin2': dict(xi_band_margin=lambda v: 2 * v),
    'jac05': dict(band_jacobian_min=0.5),
    'no_tangent': dict(tangent_bands=False),
    'W1': dict(transverse_terms='W1'),
}


def mesh_wake(sim):
    pos = sim.beam.position
    t0 = time.time()
    r = np.array([sim.get_CSR_wake(pos + z, x) for z, x in zip(sim.CSR_zmesh, sim.CSR_xmesh)])
    nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins
    print(f'  mesh of {r.shape[0]} points in {time.time()-t0:.0f} s')
    return r[:, 0].reshape(nx, nz), r[:, 1].reshape(nx, nz)


def main(run, s_target, n_worst=8, *flags):
    n_worst = int(n_worst)
    sim = D.load_state(D.state_path(run, s_target))
    pos = sim.beam.position
    ip = sim.integration_params
    nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins
    z = np.asarray(sim.CSR_zrange)
    xi = np.asarray(sim.CSR_xrange_transformed)
    tau = sim._frame_tilt()
    print(f'{run} state at s = {pos:.4f} m ({sim.lattice.current_element if hasattr(sim.lattice, "current_element") else ""}): '
          f'in bend {sim.inbend}, tau {tau:+.4f}, sigma_z {sim.beam.sigma_z*1e6:.2f} um, sigma_xi {sim.beam._sigma_x_transform*1e6:.1f} um, '
          f'formation length {sim.formation_length:.3f} m, history {len(sim.DF_tracker.DF_list) if hasattr(sim.DF_tracker, "DF_list") else "?"} snapshots')
    print(f'  mesh {nx} x {nz}: z {z[0]*1e6:+.2f} .. {z[-1]*1e6:+.2f} um (finest {np.diff(z).min()*1e6:.3f} um), xi {xi[0]*1e6:+.1f} .. {xi[-1]*1e6:+.1f} um')
    print('  integration: ' + ', '.join(f'{k}={getattr(ip, k)}' for k in ('xbins', 'zbins', 'far_zbins', 'near_adapt', 'near_adapt_x', 'xi_band_margin',
                                                                        'band_jacobian_min', 'tangent_bands', 'n_formation_length') if hasattr(ip, k)))
    W, F = mesh_wake(sim)
    np.savez(os.path.join(RUNS, f'state_outliers_{run}_s{float(s_target):.2f}.npz'), W=W, F=F, z=z, xi=xi, s=pos)
    med = np.median(np.abs(W))
    print(f'  |W| median {med:.3g}, 90% {np.percentile(np.abs(W), 90):.3g}, max {np.abs(W).max():.3g} MeV/m; |F| max {np.abs(F).max():.3g}')
    flat = np.argsort(np.abs(W).ravel())[::-1][:n_worst]
    worst = [(i // nz, i % nz) for i in flat]
    print(f'  worst {n_worst} cells (row = xi index, col = z index):')
    for (r, c) in worst:
        print(f'    row {r:2d} col {c:2d}: z {z[c]*1e6:+8.2f} um, xi {xi[r]*1e6:+8.1f} um: W {W[r, c]:+12.4g}  F {F[r, c]:+10.3g}; '
              f'row neighbours W {W[max(r-1,0), c]:+10.3g} / {W[min(r+1,nx-1), c]:+10.3g}, col neighbours {W[r, max(c-1,0)]:+10.3g} / {W[r, min(c+1,nz-1)]:+10.3g}')
    print('  decomposition at the worst 3 cells (far / near / prev / patch), W and F:')
    for (r, c) in worst[:3]:
        s, x = pos + sim.CSR_zmesh[r * nz + c], sim.CSR_xmesh[r * nz + c]
        p = wake_parts(sim, s, x)
        print(f'    row {r} col {c}: ' + '  '.join(f'{k} {p[k][0]:+10.4g}/{p[k][1]:+9.3g}' for k in ('far', 'near', 'prev', 'patch')) + f'  n_sp {p["n_sp"]}')
    if '--variants' in flags:
        keep = {}
        for fac in VARIANTS.values():
            for k in fac:
                keep[k] = getattr(ip, k)
        print('  variants at the worst 3 cells (W / F):')
        for name, fac in VARIANTS.items():
            for k, v in keep.items():
                setattr(ip, k, v)
            for k, v in fac.items():
                setattr(ip, k, v(keep[k]) if callable(v) else v)
            vals = []
            for (r, c) in worst[:3]:
                s, x = pos + sim.CSR_zmesh[r * nz + c], sim.CSR_xmesh[r * nz + c]
                vals.append(sim.get_CSR_wake(s, x))
            print(f'    {name:14s}: ' + '  '.join(f'{w:+10.4g}/{f:+9.3g}' for w, f in vals), flush=True)
        for k, v in keep.items():
            setattr(ip, k, v)


if __name__ == '__main__':
    main(*sys.argv[1:])
