"""
CSR integrand and 2D wakes at several locations INSIDE the dipole, for shear 20 and 50.

Regenerates the §6f figures with the current code (per-column branch decision from §6p,
pole-free layout from §6o, per-region log-graded nodes from §6m), and adds two things
the earlier version did not show:

0. Both frame_blend modes are shown side by side: 'coeff' (the shipped default, which
   blends tau = tan(alpha) linearly) and 'orient' (§6r, orientation from the moment
   ratio). At s = 0.10 the near region contains the full-compression point, which is
   exactly where the two differ, so plotting only one would either hide the artefact or
   hide the fix. Colour scales are SHARED between the two modes in every panel pair --
   self-normalising each panel would make the comparison meaningless.

1. The s' window is the ACTUAL near region (s3, s4), not a +-30 sigma_z zoom about the
   observation point. At shear 20 the near region is ~100 mm while 30 sigma_z is ~1.5 mm,
   so the old window showed less than 2% of the domain that carries the singularity and
   could not show the chirp band's excursion at all.

2. The graded s' node spacing is reported, so the log grading is visible as a measured
   fact rather than a config value. Grading is ON by default (near_cell = 0.5,
   near_grade = 0.05) and applies to the NEAR region only; regions 1 and 2 stay uniform
   at far_zbins.

Uses the LONG upstream drift (1.0 m, step_size 0.05) deliberately. The published §6f
figures used a 0.1 m drift, so at 0.1 m into the dipole only three snapshots existed and
the integration reached back to the edge of the available history -- that confounds the
physics of the entrance transient with plain history truncation. With a 1.0 m drift the
reach is inside the history everywhere shown here.

Caveat on the x' window: it is sized from the LOCATED bands, so it cannot reveal support
that the locator missed entirely. test_integrand_plane.py answers that question with a
prediction-independent window; this script is for showing where the integrand and the
nodes are relative to each other.
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pyDFCSR_2D.CSR import CSR2D

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'example')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results',
                          'wake_evol_maps')

SHEARS = (0.0, 2.0, 5.0, 10.0, 20.0, 50.0)
MODES = ('coeff', 'orient')
S_DIPS = (0.10, 0.20, 0.40, 0.60, 0.80)     # metres into the dipole
DRIFT = 1.0
STEP = 0.05
N_PARTICLE = 200000
DEP_BINS = 200
INT_XBINS, INT_ZBINS = 200, 200
MESH_XBINS, MESH_ZBINS = 21, 51
BG_NX, BG_NS = 2000, 280


def write_inputs(shear, s_dip, mode):
    tag = f's{shear:g}_{s_dip:g}_{mode}'.replace('.', 'p')
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
                                     {'units': 'dimensionless',
                                      'value': float(shear)},
                                     'type': 'shear z:x'}}
    with open(os.path.join(EXAMPLE_DIR, f'input/wem_beam_{tag}.yaml'), 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)

    lat = {'step_size': STEP,
           'element_1': {'type': 'drift', 'L': DRIFT, 'nsep': 1},
           'element_2': {'type': 'dipole', 'L': 1.0, 'angle': 1.0,
                         'E1': 0, 'E2': 0, 'FINT': 0.0, 'FINTX': 0.5,
                         'HGAP': 0.1, 'HGAPX': 0.0, 'FRINGE_AT': 'both_ends',
                         'FRINGE_TYPE': 'linear_edge', 'TILT': 0.0, 'nsep': 1},
           'element_3': {'type': 'drift', 'L': 0.5, 'nsep': 1}}
    lp = f'input/wem_lat_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, lp), 'w') as f:
        yaml.dump(lat, f, default_flow_style=False, sort_keys=False)

    cfg = {
        'input_beam': {'style': 'distgen',
                       'distgen_input_file': f'input/wem_beam_{tag}.yaml'},
        'input_lattice': {'lattice_input_file': lp},
        'particle_deposition': {'method': 'bspline_comoving',
                                'xbins': DEP_BINS, 'zbins': DEP_BINS,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': 1, 'velocity_threhold': 1000,
                                'frame_blend': mode},
        'CSR_integration': {'n_formation_length': 1.5,
                            'zbins': INT_ZBINS, 'xbins': INT_XBINS},
        'CSR_computation': {'compute_CSR': 0, 'apply_CSR': 0, 'transverse_on': 1,
                            'xbins': MESH_XBINS, 'zbins': MESH_ZBINS,
                            'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': f'wem_{tag}', 'workdir': './output'},
    }
    p = f'input/wem_config_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, p), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return p


def integrand_near_region(csr, s, x, t):
    """
    |integrand| over the near region (s3, s4) in lab (x', s'), plus the located band
    edges and the graded s' nodes the quadrature actually uses.
    """
    bnds = csr._layout_bounds(s, x)
    s3, s4 = bnds[2]
    sp = np.linspace(s3, s4, BG_NS)

    bands = csr._disjoint_bands(csr._retarded_xi_bands(s, x, t, sp))
    los = np.concatenate([b[0] for b in bands])
    his = np.concatenate([b[1] for b in bands])
    lo, hi = np.nanmin(los), np.nanmax(his)
    pad = 0.25 * max(hi - lo, 4.0 * csr.beam._sigma_x_transform)
    xs = np.linspace(lo - pad, hi + pad, BG_NX)

    G = np.zeros((BG_NX, BG_NS))
    Gx = np.zeros((BG_NX, BG_NS))
    for j, spv in enumerate(sp):
        gz, gx = csr.get_CSR_integrand(s=s, t=t, x=x, xp=xs.reshape(-1, 1),
                                       sp=np.full((BG_NX, 1), spv))
        G[:, j] = np.abs(gz[:, 0])
        Gx[:, j] = np.abs(gx[:, 0])

    nodes = csr._near_region_nodes(s, s3, s4)
    return dict(xs=xs, sp=sp, G=G, Gx=Gx, bands=bands, nodes=nodes,
                s3=s3, s4=s4, s1=bnds[0][0])



def integrand_panel(ax, d, field, cmap, vmin, vmax):
    s, x = d['s_obs'], d['x_obs']
    im = ax.pcolormesh((d['sp'] - s) * 1e3, (d['xs'] - x) * 1e3,
                       np.maximum(d[field], vmin),
                       norm=LogNorm(vmin=vmin, vmax=vmax),
                       cmap=cmap, shading='auto', rasterized=True)
    for k, (blo, bhi) in enumerate(d['bands']):
        live = bhi > blo
        c = ('#39d0ff', '#7dff5a')[k % 2]
        for edge in (blo, bhi):
            ax.plot(np.where(live, (d['sp'] - s) * 1e3, np.nan),
                    np.where(live, (edge - x) * 1e3, np.nan), color=c, lw=0.8)
    y0 = ax.get_ylim()[0]
    ax.plot((d['nodes'] - s) * 1e3, np.full_like(d['nodes'], y0),
            '|', color='w', ms=4, mew=0.7, alpha=0.9)
    ax.plot(0, 0, 'w+', ms=9, mew=1.6)
    return im


def shared_log_limits(ds, field):
    """One LogNorm range for a set of panels, so the modes can be compared."""
    hi = max(d[field].max() for d in ds)
    pos = np.concatenate([d[field][d[field] > 0].ravel() for d in ds
                          if (d[field] > 0).any()]) if hi > 0 else np.array([1.0])
    lo = max(np.percentile(pos, 50), hi * 1e-7)
    return lo, max(hi, lo * 10)


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.chdir(EXAMPLE_DIR)
    lines = []

    def emit(t=''):
        print(t, flush=True)
        lines.append(t)

    emit('=' * 104)
    emit('CSR integrand and 2D wakes at several locations inside the dipole')
    emit('=' * 104)
    emit(f'method bspline_comoving, two-branch bands, polar patch; '
         f'frame_blend modes compared: {MODES}')
    emit(f'lattice: {DRIFT} m drift + 1 m dipole (1 rad), step_size {STEP} m; '
         f'wake mesh {MESH_XBINS} x {MESH_ZBINS}')
    emit('integration: far regions uniform at far_zbins, NEAR region log-graded '
         '(near_cell = 0.5 sigma_xi, near_grade = 0.05)')
    emit('')

    data = {}
    for shear in SHEARS:
        emit(f'--- shear z:x = {shear:g} ---')
        emit(f"  {'s_dip':>7} {'mode':>7} {'tau':>9} {'sig_z/um':>9} "
             f"{'sig_xi/um':>10} {'amp':>6} {'near len/mm':>12} {'nodes':>6} "
             f"{'floor/um':>9} {'ds max/um':>10} {'range':>7} {'unif.':>7} "
             f"{'d/sig_z':>8} {'d set by':>9} {'dE range (MeV/m)':>26}")
        for s_dip in S_DIPS:
            for mode in MODES:
                cfg = write_inputs(shear, s_dip, mode)
                csr = CSR2D(input_file=cfg)
                target = DRIFT + s_dip
                # run() stops after the first step whose end reaches stop_time and the
                # position accumulates, so ask half a step early to land on target.
                csr.run(stop_time=target - 0.5 * STEP, debug=True)
                csr.get_CSR_mesh()
                assert abs(csr.beam.position - target) < 1e-6, csr.beam.position

                b = csr.beam
                t = b.position
                k = (MESH_XBINS // 2) * MESH_ZBINS + MESH_ZBINS // 2
                s_obs = t + csr.CSR_zmesh[k]
                x_obs = csr.CSR_xmesh[k]

                d = integrand_near_region(csr, s_obs, x_obs, t)
                csr.calculate_2D_CSR()

                zz = csr.CSR_zmesh.reshape(MESH_XBINS, MESH_ZBINS)
                xx = csr.CSR_xmesh.reshape(MESH_XBINS, MESH_ZBINS)
                d.update(s_obs=s_obs, x_obs=x_obs, zz=zz, xx=xx,
                         xt=xx - np.polyval(b.slope, zz),
                         dE=csr.dE_dct.copy(), xk=csr.x_kick.copy(),
                         tau=b._slope[0], sig_z=b._sigma_z,
                         sig_xi=b._sigma_x_transform,
                         amp=b._sigma_x / b._sigma_x_transform)
                data[(shear, s_dip, mode)] = d

                ds = np.diff(d['nodes'])
                near_len = d['s4'] - d['s3']
                # The resolution the grading targets is the FLOOR, near_cell*sigma_xi.
                # ds.min() is not it: clipping the last cell to the region end leaves a
                # sliver (0.5 um against a 25 um floor at s_dip 0.10), harmless to
                # np.trapz but it would make the uniform-equivalent count look ~50x
                # better than the grading really is.
                floor = csr.integration_params.near_cell * d['sig_xi']
                n_unif = int(np.ceil(near_len / floor)) if floor > 0 else -1

                # Which of the three terms in _layout_bounds actually set d. The
                # history cap is applied LAST, so it can override the near_floor from
                # 6t; if that happens the region is limited by available history rather
                # than by geometry, which is an honest limitation but a different one,
                # and it must not pass unnoticed. Recomputed here from public
                # quantities -- duplicated formula, kept only as a diagnostic.
                ip = csr.integration_params
                s_mid = d['s_obs']
                d_act = s_mid - d['s3']
                tau_ = csr._frame_tilt()
                c2 = abs((1 - tau_ * tau_) / (1 + tau_ * tau_))
                s2_ = abs(2 * tau_ / (1 + tau_ * tau_))
                d_chirp = ((10 * b._sigma_x + d['x_obs'] - b._mean_x) * c2
                           / max(s2_, ip.branch_sin_min))
                d_floor = ip.near_floor * b._sigma_z
                d_cap = ip.n_formation_length * csr.formation_length
                src = min((abs(d_act - d_chirp), 'chirp'),
                          (abs(d_act - d_floor), 'floor'),
                          (abs(d_act - d_cap), 'CAP'))[1]
                emit(f'  {s_dip:>7.2f} {mode:>7} {d["tau"]:>9.3f} '
                     f'{d["sig_z"]*1e6:>9.2f} {d["sig_xi"]*1e6:>10.3f} '
                     f'{d["amp"]:>6.1f} {near_len*1e3:>12.3f} '
                     f'{len(d["nodes"]):>6} {floor*1e6:>9.2f} '
                     f'{ds.max()*1e6:>10.1f} {ds.max()/floor:>7.1f} '
                     f'{n_unif:>7} {d_act/b._sigma_z:>8.2f} {src:>9} '
                     f'[{d["dE"].min():+9.4f}, {d["dE"].max():+9.4f}]')
        emit('')

    emit("floor = near_cell*sigma_xi is the cell the grading targets near s' = s;")
    emit('"unif." is how many UNIFORM nodes that cell would need to cover the whole')
    emit('near region. The ratio of that to "nodes" is what the grading buys.')
    emit('"range" = ds max / floor is the dynamic range of cell size it spans.')
    emit('')

    # how much the two modes differ, per location
    emit('coeff vs orient, relative L2 over the whole wake mesh:')
    emit(f"  {'shear':>7} {'s_dip':>7} {'dE':>10} {'x_kick':>10}")
    for shear in SHEARS:
        for s_dip in S_DIPS:
            a = data[(shear, s_dip, 'coeff')]
            b_ = data[(shear, s_dip, 'orient')]
            def rl2(u, v):
                n = np.linalg.norm(v)
                return np.linalg.norm(u - v) / n if n > 0 else np.nan
            emit(f'  {shear:>7g} {s_dip:>7.2f} '
                 f'{rl2(a["dE"], b_["dE"]):>10.5f} '
                 f'{rl2(a["xk"], b_["xk"]):>10.5f}')
    emit('')

    # ---- figures: modes as rows, locations as columns ------------------------
    nm_ = len(MODES)
    ns = len(S_DIPS)
    for shear in SHEARS:
        for field, fname, cmap, what in (('G', 'longitudinal', 'inferno', 'integrand'),
                                         ('Gx', 'transverse', 'viridis', 'integrand')):
            fig, axs = plt.subplots(nm_, ns, figsize=(4.5 * ns, 4.3 * nm_),
                                    squeeze=False)
            for j, s_dip in enumerate(S_DIPS):
                ds = [data[(shear, s_dip, m)] for m in MODES]
                vmin, vmax = shared_log_limits(ds, field)
                for i, mode in enumerate(MODES):
                    d = data[(shear, s_dip, mode)]
                    ax = axs[i][j]
                    im = integrand_panel(ax, d, field, cmap, vmin, vmax)
                    plt.colorbar(im, ax=ax, pad=0.01,
                                 label=f'|integrand| ({fname})')
                    ax.set_xlabel(r"$s' - s$  (mm)")
                    ax.set_ylabel(r"$x' - x$  (mm)")
                    ttl = (f'{s_dip:.2f} m into dipole\n' if i == 0 else '')
                    ax.set_title(ttl + f"frame_blend = {mode}"
                                 + (fr",  $\tau$={d['tau']:+.2f}, "
                                    fr"amp {d['amp']:.0f}x" if i == 0 else ''),
                                 fontsize=9)
            fig.suptitle(
                f'{fname.capitalize()} CSR integrand over the NEAR region, '
                f'shear {shear:g}.  Rows: frame_blend mode (colour scale SHARED '
                f'down each column).  Cyan/green: located band edges.  '
                f"White ticks: log-graded $s'$ nodes.  White +: observation point.",
                fontsize=12)
            fig.tight_layout()
            p = os.path.join(RESULT_DIR,
                             f'integrand_{fname}_shear{shear:g}.png')
            fig.savefig(p, dpi=110, bbox_inches='tight')
            plt.close(fig)
            emit(f'Plot saved: {p}')

        for key, fname, unit in (('dE', 'longitudinal', 'MeV/m'),
                                 ('xk', 'transverse', 'rad/m')):
            fig, axs = plt.subplots(nm_, ns, figsize=(4.5 * ns, 4.1 * nm_),
                                    squeeze=False)
            for j, s_dip in enumerate(S_DIPS):
                # shared symmetric scale across the modes, or the before/after is a lie
                m = max(np.abs(data[(shear, s_dip, mo)][key]).max() for mo in MODES)
                for i, mode in enumerate(MODES):
                    d = data[(shear, s_dip, mode)]
                    W = d[key].reshape(MESH_XBINS, MESH_ZBINS)
                    ax = axs[i][j]
                    im = ax.pcolormesh(d['zz'] * 1e6, d['xt'] * 1e6, W,
                                       cmap='RdBu_r', vmin=-m, vmax=m,
                                       shading='auto')
                    plt.colorbar(im, ax=ax, pad=0.01, label=f'({unit})')
                    ax.set_xlabel(r'$z$  ($\mu$m)')
                    ax.set_ylabel(r'$x - p(z)$  ($\mu$m)')
                    ttl = (f'{s_dip:.2f} m into dipole\n' if i == 0 else '')
                    ax.set_title(ttl + f'frame_blend = {mode}\n'
                                 f'range [{W.min():+.3g}, {W.max():+.3g}]',
                                 fontsize=9)
            fig.suptitle(f'{fname.capitalize()} CSR wake on the tilt-removed mesh, '
                         f'shear {shear:g}.  Rows: frame_blend mode.  Colour scale is '
                         f'SHARED down each column so the two modes are directly '
                         f'comparable.', fontsize=12)
            fig.tight_layout()
            p = os.path.join(RESULT_DIR, f'wake_{fname}_shear{shear:g}.png')
            fig.savefig(p, dpi=110, bbox_inches='tight')
            plt.close(fig)
            emit(f'Plot saved: {p}')

    with open(os.path.join(RESULT_DIR, 'wake_evol_maps_log.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
