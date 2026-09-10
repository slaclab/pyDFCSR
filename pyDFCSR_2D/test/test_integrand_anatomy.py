"""
Step 3: dissect the CSR integrand at one noisy high-tilt wake-mesh point.

Step 2 showed the wake roughness does not fall when G is reduced ~5x, and that
the new method is 5-40x rougher than legacy at identical G. So the noise is not
ghosting. This script attributes it among the remaining candidates by looking at
the integrand itself over the (x', s') integration plane:

  (i)   out-of-bounds mask -- bilinear_single returns a hard 0.0 outside the
        +-xlim*sigma_xi grid, which in lab space is a narrow slanted band. A step
        discontinuity in the integrand destroys the quadrature.
  (ii)  index-truncation zone -- int() truncates toward zero, so a query with
        index in (-1, 0) passes the `x0 < 0` guard, gets x0 = 0 and a NEGATIVE
        fractional weight, i.e. silent linear EXTRAPOLATION rather than the
        intended zero. Same at the top edge.
  (iii) history-snapshot loci t_ret = t_k -- where the time blend switches
        bracketing pairs. Ghosting artifacts would show up as ripples here.
  (iv)  quadrature convergence -- recompute the same wake point with the
        integration grid refined 100 -> 200 -> 400. A C0 integrand with kinks on
        a cell lattice converges at 1st order and jitters; a smooth integrand
        converges at 2nd order.

It also scans the wake along z at fixed x for several integration resolutions.
If the point-to-point jitter shrinks as the integration grid is refined, the
jitter is quadrature error, not physics.
"""
import sys
import os
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pyDFCSR_2D.CSR as CSRmod
from pyDFCSR_2D.CSR import CSR2D

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'integrand_anatomy')
EXAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'example'))

SHEAR = 20.0
STOP_S = 0.60
DEP_BINS = 200
N_PARTICLE = 200000


def write_inputs():
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
        'transforms': {'s1': {'shear_coefficient': {'units': 'dimensionless',
                                                    'value': float(SHEAR)},
                              'type': 'shear z:x'}},
    }
    with open(os.path.join(EXAMPLE_DIR, 'input/anat_beam.yaml'), 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)

    config = {
        'input_beam': {'style': 'distgen',
                       'distgen_input_file': 'input/anat_beam.yaml'},
        'input_lattice': {'lattice_input_file': 'input/dipole_lattice.yaml'},
        'particle_deposition': {'method': 'bspline_fft',
                                'xbins': DEP_BINS, 'zbins': DEP_BINS,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': 3, 'velocity_threhold': 1000},
        'CSR_integration': {'n_formation_length': 1.5, 'zbins': 100, 'xbins': 100},
        'CSR_computation': {'compute_CSR': 1, 'apply_CSR': 0, 'transverse_on': 1,
                            'xbins': 5, 'zbins': 40, 'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': 'anat', 'workdir': './output'},
    }
    with open(os.path.join(EXAMPLE_DIR, 'input/anat_config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Spy on the interpolant to recover the query coordinates it was handed
# ---------------------------------------------------------------------------
_calls = []


def install_spy():
    orig = CSRmod.interpolate3D_transformed

    def spy(**kw):
        out = orig(**kw)
        if len(kw['xval']) > 1:
            _calls.append({'x': kw['xval'].copy(), 'z': kw['zval'].copy(),
                           't': kw['tval'].copy()})
        return out

    CSRmod.interpolate3D_transformed = spy
    return orig


def oob_diagnosis(tracker, xq, zq, tq):
    """
    Reproduce the index arithmetic of interpolate3D_transformed /
    bilinear_single to classify each query point.

    Returns (inside, oob, extrap) boolean arrays. `extrap` marks the band where
    int() truncation defeats the bounds guard and the kernel silently
    extrapolates with a negative or >1 weight.
    """
    n_t = tracker.data_density_interp.shape[0]
    n_xi = tracker.data_density_interp.shape[1]
    n_z = tracker.data_density_interp.shape[2]

    t_idx = (tq - tracker.min_x) / tracker.delta_x
    k = np.clip(t_idx.astype(int), 0, max(n_t - 2, 0))

    inside = np.ones(len(xq), dtype=bool)
    extrap = np.zeros(len(xq), dtype=bool)

    for kk_off in (0, 1):
        kk = np.clip(k + kk_off, 0, n_t - 1)
        p = tracker.poly_coeffs_interp[kk]                      # (n, deg+1)
        xi = xq - np.einsum('ij,ij->i', p,
                            np.vander(zq, p.shape[1]))
        xi_idx = (xi - tracker.min_xi_arr[kk]) / tracker.delta_xi_arr[kk]
        z_idx = (zq - tracker.min_z_arr[kk]) / tracker.delta_z_arr[kk]

        x0 = xi_idx.astype(int)      # truncation toward zero, as int() in numba
        y0 = z_idx.astype(int)
        ok = (x0 >= 0) & (y0 >= 0) & (x0 < n_xi - 1) & (y0 < n_z - 1)
        inside &= ok
        # guard passed but the true index is out of range -> extrapolation
        extrap |= ok & ((xi_idx < 0) | (z_idx < 0)
                        | (xi_idx > n_xi - 1) | (z_idx > n_z - 1))

    return inside, ~inside, extrap


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    write_inputs()
    os.chdir(EXAMPLE_DIR)

    orig_interp = install_spy()

    csr = CSR2D(input_file='input/anat_config.yaml')
    csr.run(stop_time=STOP_S)

    tracker = csr.DF_tracker
    s_obs = csr.beam.position
    print(f'\nStopped at s = {s_obs:.4f} m')
    print(f'history snapshots: {len(tracker.DF_log)}  '
          f't = [{tracker.min_x:.4f}, {tracker.max_x:.4f}]  dt = {tracker.delta_x:.4f}')
    print(f'deposition grid {tracker.data_density_interp.shape}, '
          f'poly_degree = {tracker.poly_coeffs_interp.shape[1] - 1}')

    lines = []

    def emit(s=''):
        print(s)
        lines.append(s)

    # ---- choose the noisiest wake point along z at mid-x -------------------
    csr.get_CSR_mesh()
    nx, nz = csr.CSR_params.xbins, csr.CSR_params.zbins
    ix = nx // 2
    zrange = csr.CSR_zrange
    xr = csr.CSR_xrange_transformed

    emit('=' * 78)
    emit('Step 3: integrand anatomy at a high-tilt wake point')
    emit('=' * 78)
    emit(f's_obs = {s_obs:.4f} m, shear z:x = {SHEAR}, '
         f'|slope| = {abs(csr.beam._slope[0]):.3f}')
    emit(f'sigma_x = {csr.beam._sigma_x*1e6:.2f} um, '
         f'sigma_z = {csr.beam._sigma_z*1e6:.2f} um')
    emit(f'history: {len(tracker.DF_log)} snapshots, dt = {tracker.delta_x:.4f} m')
    emit('')

    # ---- (0) geometry: does the integration mesh resolve the density? ------
    emit('--- (0) integration-band width vs deposition-grid width ---')
    emit('The integration bands are sized by sigma_x (lab frame, tilt INCLUDED).')
    emit('The deposition grid is sized by sigma_xi (tilt REMOVED). At high tilt')
    emit('these differ by the tilt amplification sigma_x/sigma_xi.')
    emit('')
    sig_x = csr.beam._sigma_x
    xi = csr.beam.x - np.polyval(csr.beam.slope, csr.beam.z)
    sig_xi = np.std(xi)
    dep_halfwidth = tracker.delta_xi_arr[-1] * (tracker.data_density_interp.shape[1] - 1) / 2
    emit(f'  sigma_x  (lab)        = {sig_x*1e6:>12.2f} um')
    emit(f'  sigma_xi (tilt removed)= {sig_xi*1e6:>12.2f} um')
    emit(f'  tilt amplification     = {sig_x/sig_xi:>12.1f} x')
    emit(f'  deposition grid half-width in xi = {dep_halfwidth*1e6:>10.2f} um '
         f'(= {dep_halfwidth/sig_xi:.1f} sigma_xi)')
    emit(f'  deposition cell in xi            = '
         f'{tracker.delta_xi_arr[-1]*1e6:>10.3f} um')
    emit('')
    emit(f"  {'region':<22} {'x-half-width':>14} {'cells across':>14} "
         f"{'cell/sigma_xi':>14} {'pts in density':>15}")
    for nm, halfw, nb in (('R1 far (20 sigma_x)', 20 * sig_x, 2 * 100),
                          ('R3 near (5 sigma_x)', 5 * sig_x, 100),
                          ('R near (10 sigma_x)', 10 * sig_x, 100)):
        cell = 2 * halfw / nb
        emit(f'  {nm:<22} {halfw*1e6:>13.1f}u {nb:>14} '
             f'{cell/sig_xi:>14.1f} {2*dep_halfwidth/cell:>15.2f}')
    emit('')
    emit('  The last column is how many integration nodes fall inside the density')
    emit('  support along x. Trapz needs many; << 1 means the integral is being')
    emit('  sampled by luck.')
    emit('')

    # ---- quadrature convergence: wake vs z at several integration grids ----
    emit('--- (iv) quadrature convergence of the wake along z ---')
    emit('If the point-to-point jitter is quadrature error it shrinks as the')
    emit('integration grid is refined. If it is the density it does not.')
    emit('')
    grids = [50, 100, 200, 400]
    scans = {}
    for g in grids:
        csr.integration_params.xbins = g
        csr.integration_params.zbins = g
        dE = np.zeros(nz)
        xk = np.zeros(nz)
        for j in range(nz):
            k = ix * nz + j
            dE[j], xk[j] = csr.get_CSR_wake(s_obs + csr.CSR_zmesh[k],
                                            csr.CSR_xmesh[k])
        scans[g] = (dE, xk)
        d2 = dE[2:] - 2 * dE[1:-1] + dE[:-2]
        r = np.linalg.norm(d2) / np.linalg.norm(dE)
        d2x = xk[2:] - 2 * xk[1:-1] + xk[:-2]
        rx = np.linalg.norm(d2x) / np.linalg.norm(xk)
        emit(f'  grid {g:>4}^2 : roughness(dE) = {r:.5f}   '
             f'roughness(x_kick) = {rx:.5f}')

    ref_dE = scans[400][0]
    emit('')
    emit('  Convergence toward the 400^2 answer:')
    for g in grids[:-1]:
        e = np.linalg.norm(scans[g][0] - ref_dE) / np.linalg.norm(ref_dE)
        ex = np.linalg.norm(scans[g][1] - scans[400][1]) / np.linalg.norm(scans[400][1])
        emit(f'    grid {g:>4}^2 : rel. diff dE = {e:.5f}   x_kick = {ex:.5f}')

    # ---- integrand map at the noisiest point ------------------------------
    csr.integration_params.xbins = 100
    csr.integration_params.zbins = 100
    dE100 = scans[100][0]
    d2 = np.abs(dE100[2:] - 2 * dE100[1:-1] + dE100[:-2])
    j_bad = int(np.argmax(d2)) + 1
    k_bad = ix * nz + j_bad
    s_q = s_obs + csr.CSR_zmesh[k_bad]
    x_q = csr.CSR_xmesh[k_bad]
    emit('')
    emit(f'--- noisiest wake point: x_transform = {xr[ix]*1e6:.2f} um, '
         f'z = {zrange[j_bad]*1e6:.2f} um ---')

    _calls.clear()
    out = csr.get_CSR_wake(s_q, x_q, debug=True)
    # non-chirp branch: xp_w, xp_n, sp1, sp2, sp3, then 6 integrand arrays
    chirp = len(out) > 11
    emit(f'chirp band active: {chirp}')

    if chirp:
        xp1, xp2, xp3, xp4, sp1, sp2, sp3 = out[:7]
        regions = [('R1 (far, wide x)', xp4, sp1, out[7]),
                   ('R2 (chirp band)', xp3, sp2, out[9]),
                   ('R3 (near, band a)', xp1, sp3, out[11]),
                   ('R4 (near, band b)', xp2, sp3, out[13])]
    else:
        xp_w, xp_n, sp1, sp2, sp3 = out[:5]
        regions = [('R1 (far, wide x)', xp_w, sp1, out[5]),
                   ('R2 (mid)', xp_n, sp2, out[7]),
                   ('R3 (near)', xp_n, sp3, out[9])]

    # spy calls: 5 field interpolations per region, in region order
    per_region = [_calls[i * 5] for i in range(len(regions))]

    emit('')
    emit('--- (i)/(ii) out-of-bounds and silent-extrapolation fractions ---')
    hdr = f"{'region':<20} {'n_pts':>8} {'OOB %':>8} {'extrap %':>9} {'|integrand|max':>15}"
    emit(hdr)
    emit('-' * len(hdr))
    diag = []
    for (name, xax, sax, integ), c in zip(regions, per_region):
        inside, oob, extrap = oob_diagnosis(tracker, c['x'], c['z'], c['t'])
        diag.append((name, xax, sax, integ, inside, oob, extrap, c))
        emit(f'{name:<20} {len(c["x"]):>8} {100*oob.mean():>8.2f} '
             f'{100*extrap.mean():>9.2f} {np.abs(integ).max():>15.4e}')

    # ---- plots -------------------------------------------------------------
    n_reg = len(regions)
    fig, axes = plt.subplots(3, n_reg, figsize=(5.2 * n_reg, 13))
    if n_reg == 1:
        axes = axes.reshape(3, 1)

    hist_t = np.array(list(tracker.time_log))

    for c_i, (name, xax, sax, integ, inside, oob, extrap, c) in enumerate(diag):
        shape = integ.shape
        ext = [sax[0], sax[-1], xax[0] * 1e6, xax[-1] * 1e6]

        # row 0: the integrand
        ax = axes[0, c_i]
        v = np.abs(integ).max() or 1.0
        im = ax.imshow(integ, origin='lower', extent=ext, aspect='auto',
                       cmap='RdBu_r', vmin=-v, vmax=v)
        plt.colorbar(im, ax=ax)
        ax.set_title(f'{name}\nCSR_integrand_z')
        ax.set_ylabel(r"x' ($\mu$m)")

        # overlay loci t_ret = t_k
        t_ret = c['t'].reshape(shape)
        for tk in hist_t:
            ax.contour(np.linspace(ext[0], ext[1], shape[1]),
                       np.linspace(ext[2], ext[3], shape[0]),
                       t_ret, levels=[tk], colors='k',
                       linewidths=0.6, alpha=0.55)

        # row 1: masks
        ax = axes[1, c_i]
        code = np.zeros(len(c['x']))
        code[oob] = 1.0
        code[extrap] = 2.0
        im = ax.imshow(code.reshape(shape), origin='lower', extent=ext,
                       aspect='auto', cmap='viridis', vmin=0, vmax=2)
        plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
        ax.set_title('0 = in bounds, 1 = hard zero, 2 = silent extrapolation')
        ax.set_ylabel(r"x' ($\mu$m)")

        # row 2: a 1D cut along x' through the row of peak |integrand|
        ax = axes[2, c_i]
        i_peak = int(np.unravel_index(np.abs(integ).argmax(), shape)[1])
        ax.plot(xax * 1e6, integ[:, i_peak], 'r-', lw=1.0)
        ob = oob.reshape(shape)[:, i_peak]
        ax.fill_between(xax * 1e6, integ[:, i_peak].min(), integ[:, i_peak].max(),
                        where=ob, color='k', alpha=0.12,
                        label='hard-zero region')
        ax.set_xlabel(r"x' ($\mu$m)")
        ax.set_ylabel('integrand')
        ax.set_title(f"cut at s' = {sax[i_peak]:.4f} m")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for ax in axes[0:2, :].ravel():
        ax.set_xlabel("s' (m)")

    fig.suptitle(f'Integrand anatomy, s = {s_obs:.3f} m, '
                 f'z = {zrange[j_bad]*1e6:.1f} um, |slope| = '
                 f'{abs(csr.beam._slope[0]):.2f}\n'
                 'black contours = history snapshot loci t_ret = t_k',
                 fontsize=12)
    plt.tight_layout()
    p1 = os.path.join(RESULT_DIR, 'integrand_map.png')
    plt.savefig(p1, dpi=120)
    plt.close()

    # quadrature convergence figure
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for g, c in zip(grids, ('#bbbbbb', '#7777ff', '#ff7777', 'k')):
        axes[0].plot(zrange * 1e6, scans[g][0], '-o', color=c, ms=3,
                     label=f'{g}$^2$')
        axes[1].plot(zrange * 1e6, scans[g][1], '-o', color=c, ms=3,
                     label=f'{g}$^2$')
    axes[0].set_xlabel(r'z ($\mu$m)')
    axes[0].set_ylabel('dE/dct (MeV/m)')
    axes[0].set_title('Longitudinal wake vs integration grid')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel(r'z ($\mu$m)')
    axes[1].set_ylabel('x_kick')
    axes[1].set_title('Transverse wake vs integration grid')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    rough = [np.linalg.norm(scans[g][0][2:] - 2 * scans[g][0][1:-1]
                            + scans[g][0][:-2]) / np.linalg.norm(scans[g][0])
             for g in grids]
    roughx = [np.linalg.norm(scans[g][1][2:] - 2 * scans[g][1][1:-1]
                             + scans[g][1][:-2]) / np.linalg.norm(scans[g][1])
              for g in grids]
    axes[2].loglog(grids, rough, 'ro-', label='dE/dct')
    axes[2].loglog(grids, roughx, 'bs-', label='x_kick')
    gg = np.array(grids, dtype=float)
    axes[2].loglog(gg, rough[0] * (gg / gg[0]) ** -1, 'k:', alpha=0.6,
                   label=r'$\propto N^{-1}$')
    axes[2].loglog(gg, rough[0] * (gg / gg[0]) ** -2, 'g:', alpha=0.6,
                   label=r'$\propto N^{-2}$')
    axes[2].set_xlabel('integration bins per dimension')
    axes[2].set_ylabel('wake roughness along z')
    axes[2].set_title('Is the jitter quadrature error?')
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3, which='both')

    fig.suptitle('Quadrature convergence of the wake at high tilt', fontsize=13)
    plt.tight_layout()
    p2 = os.path.join(RESULT_DIR, 'quadrature_convergence.png')
    plt.savefig(p2, dpi=130)
    plt.close()

    emit('')
    emit(f'Plot saved: {p1}')
    emit(f'Plot saved: {p2}')

    CSRmod.interpolate3D_transformed = orig_interp
    with open(os.path.join(RESULT_DIR, 'anatomy_log.txt'), 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
