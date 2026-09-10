"""
What sets the longitudinal resolution requirement, and can per-region node
allocation replace a tilt-dependent zbins?

Section 6j found that CSR_integration: zbins = 400 carries ~0.5% error at shear 20
but 4.3% at shear 50 -- the requirement grows with tilt. A flat default is therefore
wrong, but "scale zbins with tilt" is only a prescription if we know what sets the
scale.

HYPOTHESIS. get_CSR_wake gives all three s' regions the same node COUNT, while their
lengths differ by ~100x. The binding constraint is the near region (s3, s4): it holds
the 1/|r-r'| singularity and the steep part of the integrand. Its length grows with
tilt as d ~ N sigma_x / |tan 2 alpha|, but the scale the integrand varies on is the
transverse support width sigma_xi, which is nearly tilt-independent. So the node
requirement should scale as sigma_x / sigma_xi -- the tilt amplification factor -- and
the invariant quantity should be the near-region cell measured in sigma_xi.

If that holds, the fix is not a tilt-dependent zbins but a per-region cell target:
allocate nodes so the near region gets cell <~ sigma_xi and let the far regions stay
coarse, since 1/|r-r'| has already damped them (section 6j measured the far edge s1 as
EXACTLY invariant). That would also be far cheaper than raising zbins globally.

Part 1 sweeps zbins at three shears and reports the converged cell in physical units,
in sigma_z, and in sigma_xi -- whichever is constant across shears is the controlling
scale. Part 2 tests the per-region allocation against the converged reference.
"""
import sys
import os
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyDFCSR_2D.CSR import CSR2D
from test_longitudinal_extent import (default_bounds, wake_from_bounds, z_scan,
                                      reldiff, STOP_S, DEP_BINS, N_PARTICLE,
                                      HIST_NFL, BASE, EXAMPLE_DIR)

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'long_res')

SHEARS = [2.0, 20.0, 50.0]
ZBINS_LADDER = [200, 400, 800, 1600, 3200]
REF_ZBINS = 3200


def write_inputs(shear):
    beam = {
        'n_particle': N_PARTICLE, 'species': 'electron',
        'px_dist': {'sigma_px': {'units': 'keV/c', 'value': 25.0}, 'type': 'gaussian'},
        'py_dist': {'sigma_py': {'units': 'keV/c', 'value': 25.0}, 'type': 'gaussian'},
        'pz_dist': {'avg_pz': {'units': 'GeV/c', 'value': 5},
                    'sigma_pz': {'units': 'MeV/c', 'value': 0.5}, 'type': 'gaussian'},
        'x_dist': {'sigma_x': {'units': 'um', 'value': 50}, 'type': 'gaussian'},
        'y_dist': {'sigma_y': {'units': 'um', 'value': 5}, 'type': 'gaussian'},
        'random': {'type': 'hammersley'},
        'start': {'tstart': {'units': 'sec', 'value': 0}, 'type': 'time'},
        'total_charge': {'units': 'nC', 'value': 1},
        'z_dist': {'avg_z': {'units': 'mm', 'value': 0},
                   'sigma_z': {'units': 'um', 'value': 50}, 'type': 'gaussian'},
    }
    if shear != 0.0:
        beam['transforms'] = {'s1': {'shear_coefficient':
                                     {'units': 'dimensionless', 'value': float(shear)},
                                     'type': 'shear z:x'}}
    with open(os.path.join(EXAMPLE_DIR, f'input/lr_beam_s{shear:g}.yaml'), 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)
    cfg = {
        'input_beam': {'style': 'distgen',
                       'distgen_input_file': f'input/lr_beam_s{shear:g}.yaml'},
        'input_lattice': {'lattice_input_file': 'input/dipole_lattice.yaml'},
        'particle_deposition': {'method': 'bspline_comoving',
                                'xbins': DEP_BINS, 'zbins': DEP_BINS,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': 1, 'velocity_threhold': 1000},
        'CSR_integration': {'n_formation_length': HIST_NFL,
                            'zbins': BASE['zbins'], 'xbins': BASE['xbins']},
        'CSR_computation': {'compute_CSR': 0, 'apply_CSR': 0, 'transverse_on': 1,
                            'xbins': 5, 'zbins': 40, 'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': f'lr{shear:g}', 'workdir': './output'},
    }
    p = f'input/lr_config_s{shear:g}.yaml'
    with open(os.path.join(EXAMPLE_DIR, p), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return p


def build(shear):
    csr = CSR2D(input_file=write_inputs(shear))
    csr.run(stop_time=STOP_S, debug=True)
    csr.get_CSR_mesh()
    ip = csr.integration_params
    ip.xi_bands = True
    for k, v in BASE.items():
        setattr(ip, k, v)
    return csr


def near_length(csr):
    """Length of the near region (s3, s4) at the mid-mesh observation point."""
    nz = csr.CSR_params.zbins
    ix = csr.CSR_params.xbins // 2
    k = ix*nz + nz//2
    s = csr.beam.position + csr.CSR_zmesh[k]
    bounds, _ = default_bounds(csr, s, csr.CSR_xmesh[k])
    a, b = bounds[-1]
    return abs(b - a)


def cell_alloc(csr, s, x, near_cell_sig_xi, far_nodes=200):
    """
    Per-region node allocation: the near region gets a cell of
    near_cell_sig_xi * sigma_xi, the two far regions get a fixed modest count.

    This is the candidate replacement for a flat zbins. It encodes section 6j's
    result that the far edge contributes nothing measurable, so spending equal nodes
    there is waste, and the hypothesis that the near region's requirement is set by
    sigma_xi rather than by its own length.
    """
    bounds, _ = default_bounds(csr, s, x)
    target = near_cell_sig_xi * csr.beam._sigma_x_transform
    a, b = bounds[-1]
    n_near = int(np.clip(round(abs(b - a)/target), 3, 20000))
    return bounds, [None, None, None], n_near, far_nodes


def wake_alloc(csr, s, x, near_cell_sig_xi, far_nodes=200):
    bounds, _, n_near, nf = cell_alloc(csr, s, x, near_cell_sig_xi, far_nodes)
    t = csr.beam.position
    sps = [np.linspace(*bounds[0], nf), np.linspace(*bounds[1], nf),
           np.linspace(*bounds[2], n_near)]
    radii = csr._near_patch_radii(s, bounds[-1][0], bounds[-1][1])
    tapers = [None, None, None if radii is None else (*radii, False)]
    parts = [csr._integrate_xi_region(s, x, t, spk, False, tp)
             for spk, tp in zip(sps, tapers)]
    if radii is not None:
        parts.append(csr._integrate_near_patch(s, x, t, *radii, False))
    return (sum(p[0] for p in parts), sum(p[1] for p in parts),
            2*nf + n_near)


def scan_alloc(csr, near_cell_sig_xi, far_nodes=200):
    nz = csr.CSR_params.zbins
    ix = csr.CSR_params.xbins // 2
    s0 = csr.beam.position
    dE = np.zeros(nz)
    tot = 0
    for j in range(nz):
        k = ix*nz + j
        dE[j], _, n = wake_alloc(csr, s0 + csr.CSR_zmesh[k], csr.CSR_xmesh[k],
                                 near_cell_sig_xi, far_nodes)
        tot += n
    return dE, tot/nz


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.chdir(EXAMPLE_DIR)
    lines = []

    def emit(t=''):
        print(t)
        lines.append(t)

    emit('=' * 86)
    emit('Longitudinal resolution: what sets it, and can per-region allocation fix it?')
    emit('=' * 86)
    emit('')

    store = {}
    for shear in SHEARS:
        csr = build(shear)
        b = csr.beam
        amp = b._sigma_x/b._sigma_x_transform
        L3 = near_length(csr)
        ip = csr.integration_params

        emit(f'--- shear z:x = {shear:g}   amp = {amp:.1f}x   '
             f'sigma_z = {b._sigma_z*1e6:.0f} um   sigma_xi = '
             f'{b._sigma_x_transform*1e6:.2f} um ---')
        emit(f'    near region (s3,s4) length = {L3*1e3:.2f} mm')
        emit(f"    {'zbins':>7} {'near cell':>11} {'/sigma_z':>9} {'/sigma_xi':>10} "
             f"{'rel L2 vs 3200':>15}")
        curves = {}
        for zb in ZBINS_LADDER:
            ip.zbins = zb
            curves[zb] = z_scan(csr, default_bounds)[0]
        ref = curves[REF_ZBINS]
        rows = []
        for zb in ZBINS_LADDER:
            cell = L3/(zb - 1)
            d = reldiff(curves[zb], ref)
            rows.append((zb, cell, d))
            emit(f"    {zb:>7} {cell*1e6:>9.2f}um {cell/b._sigma_z:>9.4f} "
                 f"{cell/b._sigma_x_transform:>10.3f} {d:>15.5f}")
        store[shear] = dict(amp=amp, L3=L3, rows=rows, csr=csr,
                            sig_xi=b._sigma_x_transform, sig_z=b._sigma_z,
                            ref=ref)
        emit('')

    # ---- Part 1 verdict: which scale is invariant? -------------------------
    emit('=' * 86)
    emit('PART 1: what is the controlling scale?')
    emit('=' * 86)
    emit('Interpolate the near-region cell at which each shear reaches 1% accuracy.')
    emit('')
    emit(f"  {'shear':>6} {'amp':>7} {'zbins@1%':>9} {'cell@1%':>10} "
         f"{'/sigma_z':>10} {'/sigma_xi':>10}")
    for shear, d in store.items():
        zb1 = cell1 = None
        for (zb, cell, dev) in d['rows']:
            if dev <= 0.01:
                zb1, cell1 = zb, cell
                break
        if zb1 is None:
            emit(f"  {shear:>6.0f} {d['amp']:>7.1f} {'>3200':>9} {'--':>10} "
                 f"{'--':>10} {'--':>10}")
        else:
            emit(f"  {shear:>6.0f} {d['amp']:>7.1f} {zb1:>9} {cell1*1e6:>8.2f}um "
                 f"{cell1/d['sig_z']:>10.4f} {cell1/d['sig_xi']:>10.3f}")
    emit('')
    emit('If the /sigma_xi column is roughly constant while /sigma_z is not, the')
    emit('requirement is set by the transverse support width, and the node count must')
    emit('scale as (near length)/sigma_xi ~ tilt amplification.')
    emit('')

    # ---- Part 2: per-region allocation ------------------------------------
    emit('=' * 86)
    emit('PART 2: per-region allocation vs a flat zbins')
    emit('=' * 86)
    emit('Near region gets cell = c * sigma_xi; far regions get 200 nodes each')
    emit('(section 6j: the far edge is exactly invariant, so equal nodes there is waste).')
    emit('Accuracy is measured against the SAME shear\'s zbins=3200 reference, and')
    emit('cost as the mean total column count per wake point -- flat zbins=3200 is 9600.')
    emit('')
    part2 = {}
    for shear, d in store.items():
        csr = d['csr']
        emit(f"--- shear {shear:g} (amp {d['amp']:.0f}x) ---")
        emit(f"  {'c (cell/sigma_xi)':>18} {'mean columns':>13} {'rel L2 vs 3200':>15}")
        rows = []
        for c in (2.0, 1.0, 0.5, 0.25):
            dE, cols = scan_alloc(csr, c)
            dev = reldiff(dE, d['ref'])
            rows.append((c, cols, dev))
            emit(f"  {c:>18.2f} {cols:>13.0f} {dev:>15.5f}")
        part2[shear] = rows
        emit('')

    emit('=' * 86)
    emit('SUMMARY')
    emit('=' * 86)
    emit(f"  {'shear':>6} {'flat zbins for 1%':>19} {'cols':>7} | "
         f"{'alloc c for 1%':>15} {'cols':>7} {'saving':>8}")
    for shear, d in store.items():
        zb1 = next((zb for zb, _, dev in d['rows'] if dev <= 0.01), None)
        flat_cols = 3*zb1 if zb1 else None
        best = None
        for c, cols, dev in part2[shear]:
            if dev <= 0.01:
                best = (c, cols)
                break
        if zb1 and best:
            emit(f"  {shear:>6.0f} {zb1:>19} {flat_cols:>7} | "
                 f"{best[0]:>15.2f} {best[1]:>7.0f} "
                 f"{flat_cols/best[1]:>7.1f}x")
        else:
            emit(f"  {shear:>6.0f} {str(zb1):>19} {str(flat_cols):>7} | "
                 f"{str(best):>15} {'--':>7} {'--':>8}")

    # ---- plots ------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axs[0]
    for shear, d in store.items():
        ax.loglog([c/d['sig_xi'] for _, c, _ in d['rows']],
                  [max(dev, 1e-6) for _, _, dev in d['rows']], 'o-',
                  label=f"shear {shear:g} (amp {d['amp']:.0f}x)")
    ax.axhline(0.01, color='k', ls=':', alpha=0.6, label='1%')
    ax.set_xlabel(r'near-region cell / $\sigma_\xi$')
    ax.set_ylabel('rel L2 vs zbins = 3200')
    ax.set_title('Collapse test: if the controlling scale is $\\sigma_\\xi$,\n'
                 'the three curves should overlie')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    ax = axs[1]
    for shear, d in store.items():
        ax.loglog([c/d['sig_z'] for _, c, _ in d['rows']],
                  [max(dev, 1e-6) for _, _, dev in d['rows']], 's--',
                  label=f"shear {shear:g}")
    ax.axhline(0.01, color='k', ls=':', alpha=0.6, label='1%')
    ax.set_xlabel(r'near-region cell / $\sigma_z$')
    ax.set_ylabel('rel L2 vs zbins = 3200')
    ax.set_title('Control: same data against $\\sigma_z$.\n'
                 'Should NOT collapse if the hypothesis holds')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
    fig.suptitle('What sets the longitudinal resolution requirement?', fontsize=12)
    plt.tight_layout()
    p = os.path.join(RESULT_DIR, 'longitudinal_resolution.png')
    plt.savefig(p, dpi=130)
    plt.close()
    emit('')
    emit(f'Plot saved: {p}')
    with open(os.path.join(RESULT_DIR, 'longitudinal_resolution_log.txt'), 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
