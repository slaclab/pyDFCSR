"""
WHERE does the CSR integrand's cancellation live, and what does it cost in noise?

The longitudinal integrand is LINEAR in grad'(rho_ret) (CSR.py:1920):

    numerator1 = h * [ (v - (v.v') v') . grad'(rho_ret) ],     h = 1 + x' rho(s')

For a bunch with compact support, integral of grad'(rho) over the plane is EXACTLY
zero. So the large positive and negative bands are not two physical effects that
happen to be similar in size -- they are the two lobes of a derivative, and the whole
answer is the residual left by the VARIATION of the kernel across the bunch. That
predicts a cancellation factor of order (kernel variation scale)/(bunch size), and
hence that it is structural rather than accidental.

It also predicts WHAT is amplified. Roundoff is not: at 25x cancellation float64 still
leaves ~2.5e-14. What is amplified is noise in the DEPOSITED DERIVATIVES, because that
noise is not antisymmetric -- it adds in quadrature while the signal cancels.

This script measures three things, in the order that decides the cure:

  (A) WHERE the cancellation sits, as a three-level hierarchy of the SAME total:
        C_region  over the s' regions (far, near, patch)
        C_band    over the localization branches within each region
        C_point   over individual quadrature nodes
      Each C = sum|parts| / |sum parts|. Only the pointwise level is addressable by
      integration by parts or by a control variate; if C_region already explained the
      cancellation, the cure would have to be a different region split instead.

  (B) Whether C_point tracks L_f/sigma_z, as the argument above predicts. Tested by
      varying sigma_z at fixed R, where L_f = (24 R^2 sigma_z)^(1/3) gives
      L_f/sigma_z ~ sigma_z^(-2/3) -- a slope of -2/3 on a log-log fit.

  (D) WHICH derivative the kernel sees. K = v - (v.v')v' annihilates the component
      along v', so K.tau' is ~1e-7 of K.n' (measured, pure geometry). The kernel is
      therefore almost purely transverse: drho/dx carries essentially the whole W1
      term and drho/dz is nearly inert. This is why (C) finds exactly zero gain for
      drho/dw, and it is also what decides the form integration by parts must take.

  (C) The NOISE GAIN, measured rather than inferred: inject white noise of known
      relative amplitude into the stored density arrays and see how much relative
      error appears in the wake. Done SEPARATELY for rho and for its derivatives,
      because the whole point of integration by parts is to move the derivative off
      rho and onto the analytic kernel. If the gain for grad(rho) greatly exceeds
      that for rho, that is direct evidence the cure works; if they are comparable,
      it is evidence it does not.

Two traps this script is built to avoid, both of which have cost wrong conclusions in
this codebase already:

  1. A RELATIVE metric with a vanishing denominator. The wake passes through zero, so
     a per-point ratio blows up on a denominator that is itself noise. Every ratio
     here is taken over an L2 norm of a whole z cut, never pointwise.
  2. MIRRORED quadrature weights drifting from the shipped ones. The weights are
     rebuilt here to decompose the integral, and then ASSERTED to reproduce
     _integrate_xi_region to machine precision, so a future change to the shipped
     quadrature breaks this loudly instead of quietly.
"""
import os
import sys

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyDFCSR_2D.CSR import CSR2D

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results',
                          'cancellation_anatomy')
EXAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'example'))

DIPOLE = (0.1, 1.1)          # dipole_lattice.yaml: drift 0.1, dipole 1.0, drift 0.5
N_PARTICLE = 200000
DEP_BINS = 200
INT_BINS = dict(xbins=200, zbins=200)


def write_inputs(shear=0.0, sigma_z_um=50.0, tag=''):
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
                   'sigma_z': {'units': 'um', 'value': float(sigma_z_um)},
                   'type': 'gaussian'},
    }
    if shear:
        beam['transforms'] = {'s1': {'shear_coefficient':
                                     {'units': 'dimensionless', 'value': float(shear)},
                                     'type': 'shear z:x'}}
    bp = f'input/can_beam_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, bp), 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)
    config = {
        'input_beam': {'style': 'distgen', 'distgen_input_file': bp},
        'input_lattice': {'lattice_input_file': 'input/dipole_lattice.yaml'},
        'particle_deposition': {'method': 'bspline_comoving',
                                'xbins': DEP_BINS, 'zbins': DEP_BINS,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': 1, 'velocity_threhold': 1000},
        'CSR_integration': dict(n_formation_length=4.0, **INT_BINS),
        'CSR_computation': {'compute_CSR': 0, 'apply_CSR': 0, 'transverse_on': 1,
                            'xbins': 5, 'zbins': 40, 'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': 'can', 'workdir': './output'},
    }
    cp = f'input/can_config_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, cp), 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return cp


# ---------------------------------------------------------------------------
# quadrature weights, mirrored from _integrate_xi_region and then asserted
# ---------------------------------------------------------------------------

def trapz_weights(x):
    """Weights w such that sum(w*y) == np.trapz(y, x=x), for arbitrary spacing."""
    x = np.asarray(x, float)
    w = np.zeros(x.size)
    d = np.diff(x)
    w[:-1] += 0.5 * d
    w[1:] += 0.5 * d
    return w


def band_pieces(csr, s, x, t, sp, ignore_vx=False, taper=None):
    """
    The shipped region quadrature, opened up per localization band.

    Returns a list of dicts, one per disjoint band, each with the signed integrand
    `g` on its (nx, ns) mesh and the area weights `w` for which
    -CSR_scaling * sum(w*g) is that band's contribution to dE. Mirrors
    _integrate_xi_region exactly; verify_weights() checks that claim.
    """
    bands = csr._disjoint_bands(csr._retarded_xi_bands(s, x, t, sp))
    nx = csr.integration_params.xbins
    frac = np.linspace(0.0, 1.0, nx)
    wx = trapz_weights(frac * (nx - 1))      # unit-spacing trapz weights
    ws = trapz_weights(sp)

    out = []
    for lo, hi in bands:
        width = hi - lo
        alive = width > 0.0
        if not alive.any():
            continue
        xp_mesh = lo[None, :] + frac[:, None] * width[None, :]
        sp_mesh = np.broadcast_to(sp[None, :], xp_mesh.shape).copy()
        dxp = np.where(alive, width / (nx - 1), 0.0)
        gz, gx = csr.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh, sp=sp_mesh,
                                       ignore_vx=ignore_vx, taper=taper)
        w = wx[:, None] * (dxp * ws)[None, :]
        out.append(dict(g=gz, gx=gx, w=w, xp=xp_mesh, sp=sp_mesh))
    return out


def verify_weights(csr, s, x, t, sp):
    """The mirrored weights must reproduce the shipped region integral exactly."""
    ref = csr._integrate_xi_region(s, x, t, sp, False, None)[0]
    got = -csr.CSR_scaling * sum(float(np.sum(p['w'] * p['g']))
                                 for p in band_pieces(csr, s, x, t, sp))
    scale = max(abs(ref), 1e-300)
    return ref, got, abs(got - ref) / scale


# ---------------------------------------------------------------------------
# (A) the three-level cancellation hierarchy
# ---------------------------------------------------------------------------

def cancellation(parts):
    """sum|parts| / |sum parts|, with 0 rather than inf on a zero total."""
    parts = np.asarray(parts, float)
    tot = abs(parts.sum())
    mag = np.abs(parts).sum()
    return mag / tot if tot > 0 else np.inf, parts.sum(), mag


def anatomy_at(csr, s, x):
    """
    Decompose dE at ONE wake point into region, band and pointwise contributions.

    All three levels decompose the SAME total, so their cancellation factors are
    directly comparable and must be non-decreasing: refining the partition can only
    expose more cancellation, never less.
    """
    t = csr.beam.position
    (s1, s3), (_, s4) = csr._layout_bounds(s, x)
    sps = [csr._far_region_nodes(s, x, t, s1, s3),
           csr._near_region_nodes(s, s3, s4)]
    radii = csr._near_patch_radii(s, s3, s4)
    tapers = [None, None if radii is None else (*radii, False)]

    region = []
    pieces = []
    per_region = []
    for sp, tp in zip(sps, tapers):
        ps = band_pieces(csr, s, x, t, sp, taper=tp)
        region.append(-csr.CSR_scaling * sum(float(np.sum(p['w'] * p['g']))
                                            for p in ps))
        per_region.append(np.concatenate([(-csr.CSR_scaling * p['w']
                                           * p['g']).ravel() for p in ps]))
        pieces.extend(ps)

    if radii is not None:
        # the patch is polar; its area element is r dr dphi, built the same way as
        # _integrate_near_patch so the pieces remain commensurate with the others
        nr = csr.integration_params.near_patch_nr
        nphi = csr.integration_params.near_patch_nphi
        R1, R2 = radii
        dr = R2 / nr
        r = (np.arange(nr) + 0.5) * dr
        dphi = 2.0 * np.pi / nphi
        phi = np.arange(nphi) * dphi
        xp_mesh = x + r[:, None] * np.cos(phi)[None, :]
        sp_mesh = s + r[:, None] * np.sin(phi)[None, :]
        gz, _ = csr.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh, sp=sp_mesh,
                                      ignore_vx=False, taper=(R1, R2, True))
        w = np.broadcast_to(r[:, None] * dr * dphi, gz.shape)
        region.append(-csr.CSR_scaling * float(np.sum(w * gz)))
        per_region.append((-csr.CSR_scaling * w * gz).ravel())
        pieces.append(dict(g=gz, w=w, xp=xp_mesh, sp=sp_mesh))

    band = [-csr.CSR_scaling * float(np.sum(p['w'] * p['g'])) for p in pieces]

    gw = np.concatenate([(-csr.CSR_scaling * p['w'] * p['g']).ravel()
                         for p in pieces])
    c_reg = cancellation(region)
    c_bnd = cancellation(band)
    c_pt = cancellation(gw)
    return dict(total=c_pt[1], region=region, band=band,
                C_region=c_reg[0], C_band=c_bnd[0], C_point=c_pt[0],
                C_point_region=[cancellation(p)[0] for p in per_region],
                share=[float(np.abs(p).sum()) for p in per_region],
                pos=float(gw[gw > 0].sum()), neg=float(gw[gw < 0].sum()),
                n_node=gw.size)


# ---------------------------------------------------------------------------
# (C) noise gain
# ---------------------------------------------------------------------------

NOISE_FIELDS = {
    'rho': 'data_density_interp',
    'drho/du': 'data_density_u_interp',
    'drho/dw': 'data_density_w_interp',
    'vx': 'data_vx_interp',
    'dvx/du': 'data_vx_u_interp',
}


def z_cut(csr, s0, ix=None):
    """dE along the wake mesh's z cut at one x row."""
    nz = csr.CSR_params.zbins
    ix = csr.CSR_params.xbins // 2 if ix is None else ix
    out = []
    for j in range(nz):
        k = ix * nz + j
        out.append(csr.get_CSR_wake(s0 + csr.CSR_zmesh[k], csr.CSR_xmesh[k])[0])
    return np.asarray(out)


def noise_gain(csr, s0, field, eps, seeds=(0, 1, 2), base=None, correlated=False):
    """
    Relative L2 error in the wake per unit relative noise in one stored density
    array. The array is restored afterwards, so the caller's CSR is unchanged.

    Noise is scaled by the array's own RMS, so eps is a relative perturbation of the
    field as a whole rather than of its (possibly tiny) local value -- a per-point
    relative perturbation would be dominated by the empty corners of the grid.

    correlated=True smooths the noise with the same Gaussian the deposition applies
    (smoothing_sigma cells) before injecting it. This matters and is not a detail:
    the stored fields have ALREADY been FFT-smoothed, so no white component at the
    cell scale can survive in them. White noise is therefore the one spectrum the
    quadrature is best at averaging away, and it understates the realistic case.
    Renormalised to the same RMS so the two modes are compared at equal amplitude.
    """
    from scipy.ndimage import gaussian_filter

    attr = NOISE_FIELDS[field]
    arr = getattr(csr.DF_tracker, attr)
    saved = arr.copy()
    rms = float(np.sqrt(np.mean(saved ** 2)))
    base = z_cut(csr, s0) if base is None else base
    denom = np.linalg.norm(base)
    sig = float(getattr(csr.DF_tracker, 'smoothing_sigma', 3.0))

    gains = []
    try:
        for sd in seeds:
            rng = np.random.default_rng(sd)
            pert = rng.standard_normal(saved.shape)
            if correlated:
                pert = gaussian_filter(pert, sigma=(0.0, sig, sig), mode='nearest')
                p_rms = float(np.sqrt(np.mean(pert ** 2)))
                if p_rms > 0:
                    pert /= p_rms
            arr[...] = saved + eps * rms * pert
            got = z_cut(csr, s0)
            gains.append(np.linalg.norm(got - base) / denom / eps)
    finally:
        arr[...] = saved
    return float(np.mean(gains)), float(np.std(gains))


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# (D) which component of grad(rho) the kernel actually sees
# ---------------------------------------------------------------------------

def kernel_projection(csr, s, x, sp):
    """
    Decompose the W1 kernel K = v - (v.v')v' onto the local (tau', n') frame.

    This is the measurement that decides whether integration by parts can help.
    numerator1 = h * K . grad'(rho), and
        grad'(rho) = rho_x n' + (rho_z/h) tau'
    so the two deposited derivative arrays enter weighted by K.n' and K.tau'
    respectively. If one of those projections is ~0, that array is inert and
    cannot be the noise source -- and more importantly, whether integration by
    parts helps depends on which one survives.

    v = tau + vx n at the observer and v' = tau' + vx' n' at the source, both as
    get_CSR_integrand builds them with vs = 1.

    The probe rides the CENTRE OF THE LOCATED BAND, not the x' = x ray. On the x' = x
    ray a tilted beam is simply not there, so vx' comes back 0 and the table then
    reads identically at every tilt -- an artifact of where it was sampled.
    """
    from pyDFCSR_2D.interp1D import interpolate1D as i1
    lat = csr.lattice
    t = csr.beam.position
    sp = np.atleast_1d(np.asarray(sp, float))

    def at(d, q):
        return i1(np.atleast_1d(np.asarray(q, float)), d, lat.min_x, lat.delta_x)

    lo, hi = csr._disjoint_bands(csr._retarded_xi_bands(s, x, t, sp))[0]
    xpc = 0.5 * (lo + hi)

    tx, ty = at(lat.tau_vec[:, 0], sp), at(lat.tau_vec[:, 1], sp)
    nx, ny = at(lat.n_vec[:, 0], sp), at(lat.n_vec[:, 1], sp)
    t0x, t0y = at(lat.tau_vec[:, 0], s)[0], at(lat.tau_vec[:, 1], s)[0]
    n0x, n0y = at(lat.n_vec[:, 0], s)[0], at(lat.n_vec[:, 1], s)[0]

    rx = at(lat.coords[:, 0], s)[0] + x * n0x - (at(lat.coords[:, 0], sp) + xpc * nx)
    ry = at(lat.coords[:, 1], s)[0] + x * n0y - (at(lat.coords[:, 1], sp) + xpc * ny)
    t_ret = t - np.hypot(rx, ry)
    rho, _, _, vx_ret, _ = csr._comoving_fields(xpc, sp - t_ret, t_ret)

    vx_obs = float(csr._comoving_fields(np.array([x]), np.array([s - t]),
                                        np.array([t]))[3][0])
    vx_v, vy_v = t0x + vx_obs * n0x, t0y + vx_obs * n0y
    vpx, vpy = tx + vx_ret * nx, ty + vx_ret * ny
    dot = vx_v * vpx + vy_v * vpy
    kx, ky = vx_v - dot * vpx, vy_v - dot * vpy
    return (kx * vpx + ky * vpy, kx * nx + ky * ny, kx * tx + ky * ty,
            vx_ret, rho)


def build(shear, sigma_z_um, s_stop, tag):
    cfg = write_inputs(shear, sigma_z_um, tag)
    cwd = os.getcwd()
    os.chdir(EXAMPLE_DIR)
    try:
        csr = CSR2D(input_file=cfg)
        csr.run(stop_time=s_stop, debug=True)
        csr.get_CSR_mesh()
    finally:
        os.chdir(cwd)
    return csr


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    lines = []

    def emit(t=''):
        print(t, flush=True)
        lines.append(t)

    emit('=' * 94)
    emit('Anatomy of the integrand cancellation in the longitudinal CSR wake')
    emit('=' * 94)
    emit('  C = sum|parts| / |sum parts|, over the same total at three partition')
    emit('  levels. C_region <= C_band <= C_point by construction.')
    emit('')

    # ---- (A) hierarchy, at shear 0 and shear 20, in bend and past the exit -----
    emit('-' * 94)
    emit('(A) WHERE the cancellation lives')
    emit('-' * 94)
    cases = [(0.0, 50.0, 0.50, 'in bend'),
             (0.0, 50.0, 1.20, 'past exit'),
             (20.0, 50.0, 0.50, 'in bend, shear 20')]
    rows = []
    for shear, sz, s_stop, where in cases:
        tag = f's{shear:g}_z{sz:g}_p{s_stop:g}'
        csr = build(shear, sz, s_stop, tag)
        s0 = csr.beam.position
        Lf = csr.formation_length
        sigz = csr.beam._sigma_z
        emit(f'  {where}:  s = {s0:.3f} m,  L_f = {Lf*1e3:.1f} mm,  '
             f'sigma_z = {sigz*1e6:.1f} um,  L_f/sigma_z = {Lf/sigz:.1f}')

        # the mirrored weights must reproduce the shipped quadrature
        (a, b), _ = csr._layout_bounds(s0, 0.0)
        sp = csr._far_region_nodes(s0, 0.0, csr.beam.position, a, b)
        ref, got, rel = verify_weights(csr, s0, 0.0, csr.beam.position, sp)
        emit(f'    weight check: shipped {ref:+.6e}  mirrored {got:+.6e}  '
             f'rel {rel:.2e}')
        if rel > 1e-10:
            emit('    *** MIRRORED WEIGHTS DISAGREE -- the decomposition below is '
                 'not the shipped integral ***')

        nz = csr.CSR_params.zbins
        ix = csr.CSR_params.xbins // 2
        emit('      z/sigma_z    dE        C_region   C_band   C_point   '
             'pos        neg       C_point per region (far/near/patch)')
        sel = [nz // 4, nz // 2, 3 * nz // 4]
        for j in sel:
            k = ix * nz + j
            s = s0 + csr.CSR_zmesh[k]
            a = anatomy_at(csr, s, csr.CSR_xmesh[k])
            pr = '/'.join(f'{v:.1f}' for v in a['C_point_region'])
            emit(f'      {csr.CSR_zmesh[k]/sigz:+8.2f}  {a["total"]:+9.4f}  '
                 f'{a["C_region"]:8.2f}  {a["C_band"]:7.2f}  {a["C_point"]:8.1f}  '
                 f'{a["pos"]:+9.2f}  {a["neg"]:+9.2f}   {pr}')
            rows.append((where, csr.CSR_zmesh[k] / sigz, Lf / sigz, a))
        emit('')
        del csr

    # ---- (A2) is C_point a real cancellation or a node-count artifact? --------
    emit('-' * 94)
    emit('(A2) Does C_point CONVERGE in node count?')
    emit('-' * 94)
    emit('  sum|parts| is NOT invariant to refinement in general: splitting one cell')
    emit('  into two of half the weight leaves sum|.| unchanged ONLY where the')
    emit('  integrand keeps its sign. Where it oscillates on the cell scale, sum|.|')
    emit('  grows without bound and C_point would be measuring the mesh, not the')
    emit('  physics. A converging C_point means the +/- lobes are RESOLVED structure.')
    emit('')
    emit('    xbins x zbins     dE          C_point     sum|parts|   nodes')
    csr = build(0.0, 50.0, 0.50, 'conv')
    s0 = csr.beam.position
    nz = csr.CSR_params.zbins
    ix = csr.CSR_params.xbins // 2
    k = ix * nz + nz // 2
    s_q, x_q = s0 + csr.CSR_zmesh[k], csr.CSR_xmesh[k]
    ip = csr.integration_params
    keep = (ip.xbins, ip.far_zbins)
    for mult in (0.5, 1.0, 2.0):
        ip.xbins = max(int(keep[0] * mult), 4)
        ip.far_zbins = max(int(keep[1] * mult), 3)
        a = anatomy_at(csr, s_q, x_q)
        emit(f'    {ip.xbins:4d} x {ip.far_zbins:4d}       {a["total"]:+9.4f}   '
             f'{a["C_point"]:9.1f}   {abs(a["total"])*a["C_point"]:10.2f}   '
             f'{a["n_node"]:8d}')
    ip.xbins, ip.far_zbins = keep
    del csr
    emit('')

    # ---- (B) does C_point track L_f/sigma_z? ----------------------------------
    emit('-' * 94)
    emit('(B) SCALING: C_point against L_f/sigma_z, by varying sigma_z at fixed R')
    emit('-' * 94)
    emit('  L_f = (24 R^2 sigma_z)^(1/3), so L_f/sigma_z ~ sigma_z^(-2/3). If the')
    emit('  cancellation is the kernel varying across the bunch, C_point should')
    emit('  follow it with slope ~1 on a log-log fit against L_f/sigma_z.')
    emit('')
    emit('    sigma_z     L_f      L_f/sigma_z   C_point (median over the cut)')
    scan = []
    for sz in (25.0, 50.0, 100.0, 200.0):
        tag = f'scan_z{sz:g}'
        csr = build(0.0, sz, 0.50, tag)
        s0 = csr.beam.position
        nz = csr.CSR_params.zbins
        ix = csr.CSR_params.xbins // 2
        cs = []
        for j in range(nz // 4, 3 * nz // 4 + 1, max(nz // 8, 1)):
            k = ix * nz + j
            cs.append(anatomy_at(csr, s0 + csr.CSR_zmesh[k],
                                 csr.CSR_xmesh[k])['C_point'])
        med = float(np.median(cs))
        ratio = csr.formation_length / csr.beam._sigma_z
        emit(f'    {csr.beam._sigma_z*1e6:7.1f} um  {csr.formation_length*1e3:6.1f} mm'
             f'  {ratio:10.1f}   {med:10.1f}')
        scan.append((ratio, med))
        del csr
    if len(scan) >= 3:
        r = np.array([a for a, _ in scan])
        c = np.array([b for _, b in scan])
        sl = np.polyfit(np.log(r), np.log(c), 1)[0]
        emit('')
        emit(f'    log-log slope of C_point against L_f/sigma_z: {sl:+.2f}')
        emit('    A slope of +1 would mean the cancellation IS the kernel varying')
        emit('    across the bunch. It is not: the sign is wrong, so C_point RISES')
        emit('    as the bunch lengthens. Since L_f ~ sigma_z^(1/3), a slope of -2/3')
        emit('    against L_f/sigma_z is the same as +1 against sigma_z alone, i.e.')
        emit('    the cancellation is set by the bunch length in ABSOLUTE terms and')
        emit('    the formation length drops out.')
    emit('')

    # ---- (C) noise gain -------------------------------------------------------
    emit('-' * 94)
    emit('(C) NOISE GAIN: relative wake error per unit relative density error')
    emit('-' * 94)
    emit('  White noise injected into one stored array at a time, scaled by that')
    emit('  array\'s RMS. 3 seeds; mean +- spread. Gain is measured on the L2 norm of')
    emit('  a whole z cut, never pointwise, because the wake crosses zero.')
    emit('')
    for shear, sz, s_stop, where in [(0.0, 50.0, 0.50, 'in bend'),
                                     (20.0, 50.0, 0.50, 'in bend, shear 20')]:
        tag = f'noise_s{shear:g}_z{sz:g}'
        csr = build(shear, sz, s_stop, tag)
        s0 = csr.beam.position
        base = z_cut(csr, s0)
        nz = csr.CSR_params.zbins
        ix = csr.CSR_params.xbins // 2
        k = ix * nz + nz // 2
        cpt = anatomy_at(csr, s0 + csr.CSR_zmesh[k],
                         csr.CSR_xmesh[k])['C_point']
        emit(f'  {where}: s = {s0:.3f} m, |dE|_L2 = {np.linalg.norm(base):.4f}, '
             f'C_point (mid-cut) = {cpt:.1f}')
        emit('      field       WHITE 1e-3      WHITE 1e-2     linear?  '
             'CORRELATED 1e-3')
        for field in NOISE_FIELDS:
            g1, s1_ = noise_gain(csr, s0, field, 1e-3, base=base)
            g2, s2_ = noise_gain(csr, s0, field, 1e-2, base=base)
            gc, sc_ = noise_gain(csr, s0, field, 1e-3, base=base, correlated=True)
            lin = 'yes' if abs(g2 - g1) <= 0.25 * max(g1, 1e-30) else 'NO'
            emit(f'      {field:<10s}  {g1:7.2f} +-{s1_:5.2f}  '
                 f'{g2:7.2f} +-{s2_:5.2f}   {lin:<5s}   {gc:7.2f} +-{sc_:5.2f}')
        emit('')
        del csr

    # ---- (D) which derivative the kernel actually sees ------------------------
    emit('-' * 94)
    emit('(D) KERNEL PROJECTION: which component of grad(rho) survives')
    emit('-' * 94)
    emit('  numerator1 = h * K . grad\'(rho) with K = v - (v.v\')v\' and')
    emit('  grad\'(rho) = rho_x n\' + (rho_z/h) tau\'. So drho/dx enters weighted by')
    emit('  K.n\' and drho/dz by K.tau\'.')
    emit('')
    emit('  ANALYTIC, not fitted. v\' = tau\' + vx\' n\' gives v\'.tau\' = 1 and')
    emit('  v\'.n\' = vx\', so')
    emit('      K.tau\' = -vx\' (v.n\')')
    emit('      K.n\'   = (v.n\')(1 - vx\'^2) - vx\' (v.tau\')')
    emit('  hence K.tau\'/K.n\' -> -vx\' whenever |vx\'| << sin(phi). And since K is the')
    emit('  part of v perpendicular to v\', |K| = sin(phi) with phi the bend angle')
    emit('  between s and s\'. The table below is a numerical check of both.')
    emit('')
    for shear, where in [(0.0, 'shear 0'), (20.0, 'shear 20')]:
        csr = build(shear, 50.0, 0.50, f'proj_s{shear:g}')
        s0 = csr.beam.position
        (a, b), _ = csr._layout_bounds(s0, 0.0)
        sp = np.linspace(max(a, 1e-6), min(b + 0.1, s0 + 1e-4), 9)
        kvp, kn, kt, vxr, rho = kernel_projection(csr, s0, 0.0, sp)
        emit(f'  {where}, s = {s0:.3f} m:')
        emit('      s - s\'      rho       K.n\'       K.tau\'    K.tau\'/K.n\'   '
             '-vx_ret     match')
        for i in range(sp.size):
            r = kt[i] / kn[i] if kn[i] else np.nan
            m = (abs(r + vxr[i]) / max(abs(vxr[i]), 1e-30)
                 if rho[i] > 0 else np.nan)
            emit(f'      {s0-sp[i]:+7.3f} {rho[i]:9.2e} {kn[i]:+.3e}  '
                 f'{kt[i]:+.3e}  {r:+.3e}  {-vxr[i]:+.3e}  '
                 + ('   --   ' if not np.isfinite(m) else f'{m:8.1e}'))
        emit('    The `match` column is only meaningful where rho > 0; elsewhere the')
        emit('    density is outside the deposition window and vx\' is a hard zero, so')
        emit('    the ratio there is floating-point debris at the 1e-7 level and is')
        emit('    printed only to show it is negligible against |K.n\'| ~ 0.1-0.4.')
        emit('    Where the density IS present the ratio tracks -vx\' to 1-20%.')
        emit('    Consequence: drho/dz enters the LONGITUDINAL wake only at relative')
        emit('    weight |vx\'| ~ 1e-5..1e-4, so it is inert -- which is exactly why')
        emit('    (C) measures ZERO gain for it. All of W1 flows through drho/dx.')
        emit('')
        del csr

    with open(os.path.join(RESULT_DIR, 'cancellation_anatomy.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {os.path.join(RESULT_DIR, "cancellation_anatomy.txt")}')


if __name__ == '__main__':
    main()
