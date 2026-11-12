"""
Shared helpers for the cancellation study (progress_cancellation.md).

build()      a dipole case (dipole_lattice.yaml: drift 0.1, R = 1 m bend of 1 rad, drift 0.5)
             with a Gaussian beam of chosen sigma_x, sigma_z, shear, particle count and seed,
             tracked to s_stop with the history recorded (CSR not applied).
pieces()     the shipped quadrature of get_CSR_wake opened up node by node, for the
             longitudinal AND the transverse integrand, with the weights asserted to
             reproduce get_CSR_wake.
cancel()     C = sum|parts| / |sum parts|.

Inputs are written under benchmark_results/cancellation_effects/inputs with absolute
paths, so nothing is added to example/input.
"""
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyDFCSR_2D.CSR import CSR2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(HERE, 'benchmark_results', 'cancellation_effects')
INPUT_DIR = os.path.join(RESULT_DIR, 'inputs')
LATTICE = os.path.abspath(os.path.join(HERE, '..', 'example', 'input', 'dipole_lattice.yaml'))


def write_inputs(tag, sigma_x_um=50.0, sigma_z_um=50.0, shear=0.0, n_particle=200000,
                 seed=None, dep_bins=200, smoothing_sigma=3.0, int_bins=200,
                 n_formation_length=4.0, extra_dep=None):
    """seed None: Hammersley (quasi-random, as Step 12); an int: pseudo-random with that seed."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    beam = {
        'n_particle': int(n_particle), 'species': 'electron',
        'px_dist': {'sigma_px': {'units': 'keV/c', 'value': 25.0}, 'type': 'gaussian'},
        'py_dist': {'sigma_py': {'units': 'keV/c', 'value': 25.0}, 'type': 'gaussian'},
        'pz_dist': {'avg_pz': {'units': 'GeV/c', 'value': 5},
                    'sigma_pz': {'units': 'MeV/c', 'value': 0.5}, 'type': 'gaussian'},
        'x_dist': {'sigma_x': {'units': 'um', 'value': float(sigma_x_um)}, 'type': 'gaussian'},
        'y_dist': {'sigma_y': {'units': 'um', 'value': 5}, 'type': 'gaussian'},
        'random': ({'type': 'hammersley'} if seed is None
                   else {'type': 'pseudo', 'seed': int(seed)}),
        'start': {'tstart': {'units': 'sec', 'value': 0}, 'type': 'time'},
        'total_charge': {'units': 'nC', 'value': 1},
        'z_dist': {'avg_z': {'units': 'mm', 'value': 0},
                   'sigma_z': {'units': 'um', 'value': float(sigma_z_um)},
                   'type': 'gaussian'},
    }
    if shear:
        beam['transforms'] = {'s1': {'shear_coefficient':
                                     {'units': 'dimensionless', 'value': float(shear)},
                                     'type': 'shear z:x'}}
    bp = os.path.join(INPUT_DIR, f'beam_{tag}.yaml')
    with open(bp, 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)
    dep = {'method': 'bspline_comoving', 'xbins': int(dep_bins), 'zbins': int(dep_bins),
           'xlim': 5, 'zlim': 5, 'smoothing_sigma': float(smoothing_sigma),
           'poly_degree': 1, 'velocity_threhold': 1000}
    dep.update(extra_dep or {})
    config = {
        'input_beam': {'style': 'distgen', 'distgen_input_file': bp},
        'input_lattice': {'lattice_input_file': LATTICE},
        'particle_deposition': dep,
        'CSR_integration': dict(n_formation_length=float(n_formation_length),
                                xbins=int(int_bins), zbins=int(int_bins)),
        'CSR_computation': {'compute_CSR': 0, 'apply_CSR': 0, 'transverse_on': 1,
                            'xbins': 5, 'zbins': 40, 'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': 'cx', 'workdir': os.path.join(RESULT_DIR, 'output')},
    }
    cp = os.path.join(INPUT_DIR, f'config_{tag}.yaml')
    with open(cp, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return cp


def build(tag, s_stop=0.5, **kw):
    cfg = write_inputs(tag, **kw)
    csr = CSR2D(input_file=cfg)
    csr.run(stop_time=s_stop, debug=True)
    csr.get_CSR_mesh()
    return csr


def trapz_weights(x):
    x = np.asarray(x, float)
    w = np.zeros(x.size)
    d = np.diff(x)
    w[:-1] += 0.5 * d
    w[1:] += 0.5 * d
    return w


def region_layout(csr, s, x):
    """The s' regions, tapers and patch radii exactly as get_CSR_wake builds them."""
    t = csr.beam.position
    bnds = csr._layout_bounds(s, x)
    (s1, s3), (_, s4) = bnds
    sps = [csr._far_region_nodes(s, x, t, s1, s3), csr._near_region_nodes(s, s3, s4)]
    radii = csr._near_patch_radii(s, s3, s4)
    tapers = [None, None if radii is None else (*radii, False)]
    scale = [True, False]
    out = [dict(name=nm, sp=sp, taper=tp, scale_bands=sc)
           for nm, sp, tp, sc in zip(('far', 'near'), sps, tapers, scale)]
    pb = csr._prev_bend_reach_start()
    if pb is not None and pb < s1:
        out.append(dict(name='prev', sp=csr._prev_region_nodes(s, pb, s1), taper=None,
                        scale_bands=True))
    return out, radii


def band_meshes(csr, s, x, sp, scale_bands=True):
    """Per disjoint band: the (nx, ns) node mesh and its area weights (as _integrate_xi_region)."""
    t = csr.beam.position
    bands = csr._disjoint_bands(csr._retarded_xi_bands(s, x, t, sp, scale_bands=scale_bands))
    nx = csr.integration_params.xbins
    frac = np.linspace(0.0, 1.0, nx)
    wx = trapz_weights(np.arange(nx, dtype=float))
    ws = trapz_weights(sp)
    out = []
    for lo, hi in bands:
        width = hi - lo
        alive = width > 0.0
        if not alive.any():
            continue
        xp = lo[None, :] + frac[:, None] * width[None, :]
        spm = np.broadcast_to(sp[None, :], xp.shape).copy()
        dxp = np.where(alive, width / (nx - 1), 0.0)
        out.append(dict(xp=xp, sp=spm, w=wx[:, None] * (dxp * ws)[None, :], alive=alive,
                        dxp=dxp))
    return out


def patch_mesh(csr, s, x, radii):
    nr = csr.integration_params.near_patch_nr
    nphi = csr.integration_params.near_patch_nphi
    R1, R2 = radii
    dr = R2 / nr
    r = (np.arange(nr) + 0.5) * dr
    dphi = 2.0 * np.pi / nphi
    phi = np.arange(nphi) * dphi
    xp = x + r[:, None] * np.cos(phi)[None, :]
    spm = s + r[:, None] * np.sin(phi)[None, :]
    w = np.broadcast_to(r[:, None] * dr * dphi, xp.shape).copy()
    return dict(xp=xp, sp=spm, w=w, taper=(R1, R2, True))


def pieces(csr, s, x, ignore_vx=False):
    """
    Every node of the shipped quadrature with its signed contribution to dE (MeV/m) and to
    the transverse kick, tagged by region. Sums are asserted against get_CSR_wake.
    """
    t = csr.beam.position
    regions, radii = region_layout(csr, s, x)
    out = []
    for reg in regions:
        for b in band_meshes(csr, s, x, reg['sp'], reg['scale_bands']):
            gz, gx = csr.get_CSR_integrand(s=s, t=t, x=x, xp=b['xp'], sp=b['sp'],
                                           ignore_vx=ignore_vx, taper=reg['taper'])
            out.append(dict(region=reg['name'], z=-csr.CSR_scaling * b['w'] * gz,
                            x=csr.CSR_scaling * b['w'] * gx, **b))
    if radii is not None:
        p = patch_mesh(csr, s, x, radii)
        gz, gx = csr.get_CSR_integrand(s=s, t=t, x=x, xp=p['xp'], sp=p['sp'],
                                       ignore_vx=ignore_vx, taper=p['taper'])
        out.append(dict(region='patch', z=-csr.CSR_scaling * p['w'] * gz,
                        x=csr.CSR_scaling * p['w'] * gx, **p))
    return out


def check_pieces(csr, s, x, ps, tol=1e-9):
    ref = csr.get_CSR_wake(s, x)
    got = (sum(float(p['z'].sum()) for p in ps), sum(float(p['x'].sum()) for p in ps))
    rel = [abs(g - r) / max(abs(r), 1e-300) for g, r in zip(got, ref)]
    if max(rel) > tol:
        raise AssertionError(f'mirrored quadrature disagrees with get_CSR_wake: '
                             f'{got} vs {ref} (rel {rel})')
    return ref, rel


def cancel(parts):
    parts = np.asarray(parts, float).ravel()
    tot = parts.sum()
    mag = np.abs(parts).sum()
    return (mag / abs(tot) if tot != 0 else np.inf), tot, mag


def z_cut(csr, ix=None, which=None):
    """(dE, x_kick) along the wake mesh's z cut at one x row; returns z/sigma_z too."""
    nz = csr.CSR_params.zbins
    ix = csr.CSR_params.xbins // 2 if ix is None else ix
    s0 = csr.beam.position
    out = []
    for j in range(nz):
        k = ix * nz + j
        out.append(csr.get_CSR_wake(s0 + csr.CSR_zmesh[k], csr.CSR_xmesh[k]))
    out = np.asarray(out)
    zz = np.array([csr.CSR_zmesh[ix * nz + j] for j in range(nz)]) / csr.beam._sigma_z
    return zz, out[:, 0], out[:, 1]
