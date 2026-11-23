"""
Why do the outermost xi rows of the wake mesh carry wakes several times the core's, of the
opposite sign (the rectangular 'frame' on the B1LE/B2LE maps of long bunches: Indranil gain,
Sergei's BC20 reference)? On a pickled state (dump_state.py <run> <s>) the wake is computed along
xi at one z column, on the mesh rows and beyond, with the regions separated (far / near / patch),
with the velocity terms off, with wider bands and finer quadrature.

    conda run -n pydfcsr python ../test/bc20_indranil/edge_rows.py <run> <s_target> [z_um]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'facet_bc20'))
import dump_kick_state as D  # noqa: E402
from near_field_check import wake_parts  # noqa: E402

D.RUNS = os.path.join(HERE, '..', 'benchmark_results', 'bc20_indranil', 'runs')
np.set_printoptions(linewidth=250, precision=4, suppress=True)


def parts(sim, s, x, ignore_vx=False):
    t = sim.beam.position
    bnds = sim._layout_bounds(s, x)
    (s1, s3), (_, s4) = bnds
    sim._region_node_counts(bnds)
    sps = [sim._far_region_nodes(s, x, t, s1, s3), sim._near_region_nodes(s, s3, s4, x=x, t=t)]
    radii = sim._near_patch_radii(s, s3, s4)
    tapers = [None, None if radii is None else (*radii, False)]
    out = {}
    for k, (name, spk, tp) in enumerate(zip(('far', 'near'), sps, tapers)):
        out[name] = sim._integrate_xi_region(s, x, t, spk, ignore_vx, tp, scale_bands=(k == 0), adapt_x=(k == 1))[0]
    pb = sim._prev_bend_reach_start()
    out['prev'] = (sim._integrate_xi_region(s, x, t, sim._prev_region_nodes(s, pb, s1), ignore_vx, None, scale_bands=True)[0]
                   if (pb is not None and pb < s1) else 0.0)
    out['patch'] = sim._integrate_near_patch(s, x, t, *radii, ignore_vx)[0] if radii is not None else 0.0
    return out


def main(run, s_target, z_um=None):
    sim = D.load_state(D.state_path(run, s_target))
    ip = sim.integration_params
    pos = sim.beam.position
    sx = sim.beam._sigma_x_transform
    zr = np.asarray(sim.CSR_zrange); xr = np.asarray(sim.CSR_xrange_transformed)
    nz = sim.CSR_params.zbins
    X = np.asarray(sim.CSR_xmesh).reshape(-1, nz)
    c = nz // 2 if z_um is None else int(np.argmin(np.abs(zr - float(z_um) * 1e-6)))
    off = float(np.median(X[:, c] - xr))
    xis = np.concatenate([xr, xr[0] - sx * np.array([0.5, 1.0]), xr[-1] + sx * np.array([0.5, 1.0])]); xis.sort()
    print(f'{run} at s = {pos:.4f}: sigma_z {sim.beam.sigma_z*1e6:.1f} um, sigma_xi {sx*1e6:.1f} um, tau {sim._frame_tilt():+.3f}; '
          f'column z = {zr[c]*1e6:+.1f} um; mesh xi rows {xr[0]/sx:+.2f}..{xr[-1]/sx:+.2f} sigma_xi')
    print('xi/sigma_xi      :', np.round(xis / sx, 2))
    s = pos + zr[c]
    tot, novx, per = [], [], {k: [] for k in ('far', 'near', 'prev', 'patch')}
    for xi in xis:
        x = xi + off
        tot.append(sim.get_CSR_wake(s, x)[0])
        p = parts(sim, s, x)
        for k in per:
            per[k].append(p[k])
        q = parts(sim, s, x, ignore_vx=True)
        novx.append(sum(q.values()))
    print('W total          :', np.array(tot))
    for k in per:
        print(f'  {k:6s}         :', np.array(per[k]))
    print('W, velocity off  :', np.array(novx))
    keep = {k: getattr(ip, k) for k in ('xi_band_margin', 'far_zbins', 'xbins', 'near_adapt', 'near_adapt_x', 'band_jacobian_min')}
    for lab, fac in (('margin x4', dict(xi_band_margin=4 * keep['xi_band_margin'])), ('far cols x4', dict(far_zbins=4 * keep['far_zbins'])),
                     ("x' nodes x4", dict(xbins=4 * keep['xbins'])), ('no near_adapt', dict(near_adapt=False, near_adapt_x=False)),
                     ('jacobian min 0', dict(band_jacobian_min=0.0))):
        for k, v in keep.items():
            setattr(ip, k, v)
        for k, v in fac.items():
            setattr(ip, k, v)
        print(f'{lab:17s}:', np.array([sim.get_CSR_wake(s, xi + off)[0] for xi in xis]), flush=True)
    for k, v in keep.items():
        setattr(ip, k, v)
    # band liveness for the outermost mesh row vs the central row, far region
    for lab, xi in (('centre row', xr[len(xr) // 2]), ('top row', xr[-1]), ('top + 1 sigma', xr[-1] + sx)):
        x = xi + off
        (s1, s3), (_, s4) = sim._layout_bounds(s, x)
        sp = sim._far_region_nodes(s, x, pos, s1, s3)
        sim._retarded_xi_bands(s, x, pos, sp)
        d = sim._branch_diag
        print(f'{lab:14s}: far region {sp.size} columns, live fraction per branch ' + ', '.join(f'{b["live"].mean():.2f}' for b in d['branches'])
              + f'; band half-width {np.nanmedian(d["branches"][0]["half"])*1e6:.0f} um, centre - x: {np.nanmedian(d["branches"][0]["centre"] - x)*1e6:+.0f} um')


if __name__ == '__main__':
    main(*sys.argv[1:])
