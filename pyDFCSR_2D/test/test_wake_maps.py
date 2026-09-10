"""
What the wake and its integrand actually look like, across beam tilt.

Everything up to now has been scalar metrics -- roughness, relative L2, node counts.
Those are what you need to decide whether something is converged, but they hide what
the answer looks like. This produces the two pictures:

  1. The CSR integrand over the (x', s') integration plane, with the quadrature
     nodes the Step 6 locator actually chose drawn on top. This is where the two
     Eq 4.24 branches are directly visible, and where you can see by eye whether
     the located bands sit on the support or beside it.

  2. The longitudinal (dE/dct) and transverse (x_kick) wakes over the (x, z) plane.
     Note the CSR mesh is built on the TILT-REMOVED grid (get_CSR_mesh works in
     x_transform), so in physical (x, z) it is a sheared parallelogram, not a
     rectangle. Both views are plotted: physical (x, z) shows what the beam sees,
     the (x_transform, z) view shows the mesh as the code stores it.

Swept over shear z:x = 0, 2, 20, 50, i.e. tilt amplification sigma_x/sigma_xi from
about 1x to about 170x, so the low-tilt case (where the two branches merge and the
old bands are trustworthy) and the extreme case appear side by side.

Integration uses zbins = 400, the setting section 6d found is needed to reach the
curvature floor -- at 200 the maps carry visible quadrature noise.
"""
import sys
import os
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyDFCSR_2D.CSR import CSR2D

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'wake_maps')
EXAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'example'))

SHEARS = [0.0, 2.0, 20.0, 50.0]
STOP_S = 0.60
DEP_BINS = 200
N_PARTICLE = 200000

# wake-map resolution (observation points), and integration resolution
MESH_XBINS, MESH_ZBINS = 21, 51
INT_XBINS, INT_ZBINS = 200, 400

# dense background sampling of the integrand plane
BG_NX, BG_NS = 420, 360


def write_inputs(shear, tag):
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
    with open(os.path.join(EXAMPLE_DIR, f'input/wm_beam_{tag}.yaml'), 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)

    config = {
        'input_beam': {'style': 'distgen',
                       'distgen_input_file': f'input/wm_beam_{tag}.yaml'},
        'input_lattice': {'lattice_input_file': 'input/dipole_lattice.yaml'},
        'particle_deposition': {'method': 'bspline_comoving',
                                'xbins': DEP_BINS, 'zbins': DEP_BINS,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': 1, 'velocity_threhold': 1000},
        'CSR_integration': {'n_formation_length': 1.5,
                            'zbins': INT_ZBINS, 'xbins': INT_XBINS},
        'CSR_computation': {'compute_CSR': 1, 'apply_CSR': 0, 'transverse_on': 1,
                            'xbins': MESH_XBINS, 'zbins': MESH_ZBINS,
                            'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': f'wm_{tag}', 'workdir': './output'},
    }
    p = f'input/wm_config_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, p), 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return p


def integrand_map(csr, s, x, t):
    """
    Dense |integrand_z| over a window of the (x', s') plane that is guaranteed to
    contain both located branches, plus the nodes the locator chose.
    """
    sigma_z = csr.beam._sigma_z
    sp_lo, sp_hi = s - 30 * sigma_z, s + 5 * sigma_z
    sp_probe = np.linspace(sp_lo, sp_hi, 400)

    bands = csr._disjoint_bands(csr._retarded_xi_bands(s, x, t, sp_probe))
    los = np.concatenate([b[0] for b in bands])
    his = np.concatenate([b[1] for b in bands])
    lo, hi = np.nanmin(los), np.nanmax(his)
    pad = 0.35 * max(hi - lo, 1e-9)
    xlo, xhi = lo - pad, hi + pad

    xs = np.linspace(xlo, xhi, BG_NX)
    ss = np.linspace(sp_lo, sp_hi, BG_NS)
    XP, SP = np.meshgrid(xs, ss, indexing='ij')
    gz, gx = csr.get_CSR_integrand(s=s, t=t, x=x, xp=XP, sp=SP)

    # the nodes actually used, from the real region decomposition
    d = csr.get_CSR_wake(s, x, debug=True)
    nodes = [(xm, sm) for xm, sm in zip(d['xp_mesh'], d['sp_mesh'])]
    return XP, SP, gz, gx, nodes, (xlo, xhi, sp_lo, sp_hi)


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.chdir(EXAMPLE_DIR)
    lines = []

    def emit(s=''):
        print(s)
        lines.append(s)

    emit('=' * 84)
    emit('CSR integrand in the (x\', s\') plane and the 2D wakes, swept over beam tilt')
    emit('=' * 84)
    emit(f'method = bspline_comoving (poly_degree 1), two-branch bands, '
         f'integration {INT_XBINS} x {INT_ZBINS}')
    emit(f'wake mesh {MESH_XBINS} x {MESH_ZBINS} observation points')
    emit('')

    data = {}
    for shear in SHEARS:
        tag = f'sw{shear:g}'.replace('.', 'p')
        cfg = write_inputs(shear, tag)
        csr = CSR2D(input_file=cfg)
        csr.run(stop_time=STOP_S)
        csr.get_CSR_mesh()

        b = csr.beam
        t = b.position
        sig_xi = b._sigma_x_transform
        amp = b._sigma_x / sig_xi

        # observation point: mid-x, mid-z
        k = (MESH_XBINS // 2) * MESH_ZBINS + MESH_ZBINS // 2
        s_obs = t + csr.CSR_zmesh[k]
        x_obs = csr.CSR_xmesh[k]

        emit(f'--- shear z:x = {shear:g} ---')
        emit(f'    dx/dz = {b._slope[0]:+.4f}, sigma_x = {b._sigma_x*1e6:.1f} um, '
             f'sigma_xi = {sig_xi*1e6:.2f} um, amplification = {amp:.1f} x')

        XP, SP, gz, gx, nodes, win = integrand_map(csr, s_obs, x_obs, t)
        csr.calculate_2D_CSR()

        zz = csr.CSR_zmesh.reshape(MESH_XBINS, MESH_ZBINS)
        xx = csr.CSR_xmesh.reshape(MESH_XBINS, MESH_ZBINS)
        xt = xx - np.polyval(b.slope, zz)

        emit(f'    dE/dct  range [{csr.dE_dct.min():+.4f}, {csr.dE_dct.max():+.4f}] '
             f'MeV/m,  x_kick range [{csr.x_kick.min():+.4e}, '
             f'{csr.x_kick.max():+.4e}]')
        emit(f'    integrand window: x\' spans {(win[1]-win[0])*1e6:.1f} um '
             f'= {(win[1]-win[0])/sig_xi:.1f} sigma_xi, '
             f's\' spans {(win[3]-win[2])*1e3:.2f} mm')
        emit('')

        data[shear] = dict(XP=XP, SP=SP, gz=gz, gx=gx, nodes=nodes, win=win,
                           s_obs=s_obs, x_obs=x_obs, zz=zz, xx=xx, xt=xt,
                           dE=csr.dE_dct.copy(), xk=csr.x_kick.copy(),
                           slope=b._slope[0], amp=amp, sig_xi=sig_xi,
                           sig_z=b._sigma_z)

    # ---- Figure 1: integrand in the (x', s') plane --------------------------
    n = len(SHEARS)
    fig, axs = plt.subplots(2, n, figsize=(4.6 * n, 9.0))
    for j, shear in enumerate(SHEARS):
        d = data[shear]
        for row, (g, name) in enumerate(((d['gz'], 'longitudinal'),
                                         (d['gx'], 'transverse'))):
            ax = axs[row, j]
            a = np.abs(g)
            pos = a[a > 0]
            if pos.size:
                vmax = pos.max()
                vmin = max(vmax * 1e-6, pos.min())
                im = ax.pcolormesh((d['SP'] - d['s_obs']) * 1e3,
                                   (d['XP'] - d['x_obs']) * 1e6,
                                   np.maximum(a, vmin),
                                   norm=LogNorm(vmin=vmin, vmax=vmax),
                                   cmap='viridis', shading='auto')
                plt.colorbar(im, ax=ax, label=f'|integrand| ({name})')
            for xm, sm in d['nodes']:
                ax.plot((sm[::7, ::9] - d['s_obs']) * 1e3,
                        (xm[::7, ::9] - d['x_obs']) * 1e6,
                        '.', color='r', ms=0.6, alpha=0.5)
            ax.plot(0, 0, 'w+', ms=11, mew=2)
            ax.set_xlabel(r"$s' - s$ (mm)")
            ax.set_ylabel(r"$x' - x$ ($\mu$m)")
            if row == 0:
                ax.set_title(f"shear {shear:g},  amp {d['amp']:.0f}x\n"
                             f"dx/dz = {d['slope']:+.2f}", fontsize=10)
            else:
                ax.set_title(f'transverse integrand', fontsize=9)
    fig.suptitle('CSR integrand over the integration plane. Red dots: quadrature '
                 'nodes chosen by the two-branch locator (subsampled). '
                 'White + is the observation point, where the branches cross and '
                 'the kernel diverges.', fontsize=12)
    plt.tight_layout()
    p1 = os.path.join(RESULT_DIR, 'integrand_plane_vs_shear.png')
    plt.savefig(p1, dpi=125)
    plt.close()

    # ---- Figure 2: wakes over the (x, z) plane ------------------------------
    fig, axs = plt.subplots(2, n, figsize=(4.6 * n, 8.4))
    for j, shear in enumerate(SHEARS):
        d = data[shear]
        for row, (w, name, unit) in enumerate((
                (d['dE'], r'$dE/d(ct)$', 'MeV/m'),
                (d['xk'], r'transverse kick', 'a.u.'))):
            ax = axs[row, j]
            lim = np.abs(w).max()
            norm = (TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
                    if lim > 0 else None)
            im = ax.pcolormesh(d['zz'] * 1e6, d['xx'] * 1e6, w,
                               cmap='RdBu_r', norm=norm, shading='auto')
            plt.colorbar(im, ax=ax, label=f'{name} ({unit})')
            ax.set_xlabel(r'$z$ ($\mu$m)')
            ax.set_ylabel(r'$x$ ($\mu$m)')
            if row == 0:
                ax.set_title(f"shear {shear:g},  amp {d['amp']:.0f}x\n"
                             f"dx/dz = {d['slope']:+.2f}", fontsize=10)
    fig.suptitle('Longitudinal and transverse CSR wake over the physical (x, z) '
                 'plane. The mesh is built tilt-removed, so at high shear it is a '
                 'sheared parallelogram here.', fontsize=12)
    plt.tight_layout()
    p2 = os.path.join(RESULT_DIR, 'wake_xz_vs_shear.png')
    plt.savefig(p2, dpi=125)
    plt.close()

    # ---- Figure 3: same wakes on the tilt-removed mesh ----------------------
    fig, axs = plt.subplots(2, n, figsize=(4.6 * n, 8.4))
    for j, shear in enumerate(SHEARS):
        d = data[shear]
        for row, (w, name) in enumerate(((d['dE'], r'$dE/d(ct)$ (MeV/m)'),
                                         (d['xk'], 'transverse kick (a.u.)'))):
            ax = axs[row, j]
            lim = np.abs(w).max()
            norm = (TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
                    if lim > 0 else None)
            im = ax.pcolormesh(d['zz'] * 1e6, d['xt'] * 1e6, w,
                               cmap='RdBu_r', norm=norm, shading='auto')
            plt.colorbar(im, ax=ax, label=name)
            ax.set_xlabel(r'$z$ ($\mu$m)')
            ax.set_ylabel(r'$x - p(z)$ ($\mu$m)')
            if row == 0:
                ax.set_title(f"shear {shear:g},  amp {d['amp']:.0f}x", fontsize=10)
    fig.suptitle('The same wakes on the tilt-removed mesh the code actually uses. '
                 'Smoothness is easiest to judge here.', fontsize=12)
    plt.tight_layout()
    p3 = os.path.join(RESULT_DIR, 'wake_transformed_vs_shear.png')
    plt.savefig(p3, dpi=125)
    plt.close()

    emit(f'Plots saved:\n  {p1}\n  {p2}\n  {p3}')
    with open(os.path.join(RESULT_DIR, 'wake_maps_log.txt'), 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
