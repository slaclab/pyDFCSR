"""
The CSR integrand over the (x', s') integration plane, in band-relative coordinates.

A first attempt plotted the integrand on a plain linear (x', s') window. That cannot
work at high tilt: the chirp ridge is about sigma_xi wide (12 um) but sweeps across
113 mm of x' over the s' range, a 9000:1 dynamic range, so on any single linear axis
the ridge is thinner than one pixel.

The fix is to plot each branch in its OWN band coordinate,

    v = (x' - centre_k(s')) / half_k(s')

so that v = 0 is the branch and |v| <= 1 is exactly the band the quadrature covers.
This straightens each ridge and makes the question that matters directly visible:
is the integrand support inside the band the locator chose? If yes, all the colour
sits within |v| <= 1. If the locator were misplacing a branch, the colour would sit
off-centre or outside.

Also reports the quantitative version of the same question -- the fraction of
integrand magnitude that falls inside |v| <= 1 -- per s' region and per branch, which
is the number the picture is a picture of.

Reuses the beam/lattice of test_wake_maps.py but with compute_CSR turned OFF during
tracking. That matters a lot: test_wake_maps leaves compute_CSR = 1 with a 21x51 wake
mesh, so csr.run() computes the full 2D wake at every one of the 7 lattice steps --
about 7500 wake points -- before the script does any of its own work. This script only
needs the deposition history, so all of that is waste, and turning it off takes the
per-shear cost from ~8 minutes to seconds.
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import yaml

from pyDFCSR_2D.CSR import CSR2D
from test_wake_maps import (SHEARS, STOP_S, MESH_XBINS, MESH_ZBINS,
                            write_inputs, EXAMPLE_DIR)


def config_no_csr(shear, tag):
    """test_wake_maps' inputs, with the wake computation disabled during tracking."""
    p = write_inputs(shear, tag)
    full = os.path.join(EXAMPLE_DIR, p)
    with open(full) as f:
        cfg = yaml.safe_load(f)
    cfg['CSR_computation']['compute_CSR'] = 0
    p2 = p.replace('wm_config_', 'ip_config_')
    with open(os.path.join(EXAMPLE_DIR, p2), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return p2

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'wake_maps')

NV, NS = 320, 300          # band-relative and longitudinal sampling
VMAX = 3.0                 # plot |v| out to 3 band half-widths


def region_bounds(csr, s):
    """The same three s' regions get_CSR_wake builds."""
    sigma_z = csr.beam._sigma_z
    tan_theta = csr.beam._slope[0]
    if abs(tan_theta) <= 1:
        s2, s3, s4 = s - 500 * sigma_z, s - 20 * sigma_z, s + 5 * sigma_z
    else:
        s4 = s + 3 * sigma_z
        d = 10 * csr.beam._sigma_x / abs(2 * tan_theta / (1 - tan_theta ** 2))
        s3 = max(0.0, s - d)
        s2 = s3 - 200 * sigma_z
    s1 = max(0.0, s2 - csr.integration_params.n_formation_length
             * csr.formation_length)
    return [('sp1', s1, s2), ('sp2', s2, s3), ('sp3', s3, s4)]


def branch_slice(csr, s, x, t, sp, k):
    """
    |integrand| on the mesh xp = centre_k(s') + v * half_k(s'), plus the fraction
    of magnitude that lands inside the band (|v| <= 1).
    """
    csr._retarded_xi_bands(s, x, t, sp)
    br = csr._branch_diag['branches'][k]
    centre, half = br['centre'], br['half']

    v = np.linspace(-VMAX, VMAX, NV)
    XP = centre[None, :] + v[:, None] * half[None, :]
    SP = np.broadcast_to(sp[None, :], XP.shape).copy()
    gz, gx = csr.get_CSR_integrand(s=s, t=t, x=x, xp=XP, sp=SP)

    a = np.abs(gz)
    tot = a.sum()
    inside = a[np.abs(v) <= 1.0].sum()
    return v, a, np.abs(gx), (inside / tot if tot > 0 else np.nan), br['live']


def union_coverage(csr, s, x, t, sp, n_wide=1500, span=40.0):
    """
    Fraction of total |integrand| magnitude that falls inside the UNION of the
    located bands, over a window `span` band half-widths wide.

    The per-branch "inside" number cannot answer the question that matters: if one
    branch's band is off-centre, the other band may still cover the support. Only
    the union tells you whether the located domain misses real integrand. This is
    the quantitative form of Step 6's axis A, measured directly on the integrand
    instead of inferred from the wake's response to a wider domain.
    """
    bands = csr._disjoint_bands(csr._retarded_xi_bands(s, x, t, sp))
    br = csr._branch_diag['branches']
    # Centre the window on whichever branch is live more often. Centring blindly on
    # branch 0 puts the window nowhere near the support in regions where branch 0 is
    # entirely dead, which reads as an empty window rather than as good coverage.
    k0 = max(range(len(br)), key=lambda i: br[i]['live'].mean())
    half = br[k0]['half']
    centre = br[k0]['centre']

    u = np.linspace(-span, span, n_wide)
    XP = centre[None, :] + u[:, None] * half[None, :]
    SP = np.broadcast_to(sp[None, :], XP.shape).copy()
    gz, _ = csr.get_CSR_integrand(s=s, t=t, x=x, xp=XP, sp=SP)
    a = np.abs(gz)

    covered = np.zeros(XP.shape, dtype=bool)
    for lo, hi in bands:
        covered |= (XP >= lo[None, :]) & (XP <= hi[None, :]) & (hi > lo)[None, :]

    tot = a.sum()
    return (a[covered].sum() / tot if tot > 0 else np.nan), tot


def lab_geometry(csr, s, x, t, sp, n_scan=24000, pad=8.0):
    """
    The support of the integrand in LAB (x', s') coordinates, plus both predicted
    branch curves.

    The band-relative view co-moves with each branch, so a correctly tracked chirp
    band is drawn flat and its characteristic diagonal is subtracted away. This is
    the view where the thesis 4.4.2 picture should appear literally: a narrow band
    running parallel to the s' axis near x' = x, and a chirp band leaving it at
    angle 2*alpha, with the chirp band's excursion GROWING with tilt.

    Support is measured, not modelled: for each s' column scan x' and record the
    first and last nonzero sample. n_scan has to be large because the ridge is about
    sigma_xi wide inside a window set by the chirp excursion.
    """
    csr._retarded_xi_bands(s, x, t, sp)
    br = csr._branch_diag['branches']
    c0, c1 = br[0]['centre'], br[1]['centre']

    # The scan window must NOT be derived from the predicted centres. Anchoring it to
    # the prediction makes it structurally unable to reveal support the prediction
    # missed -- which is the one thing this figure is for. Use the same fixed,
    # prediction-independent window as test_localization_branches: -14 sigma_x to
    # +6 sigma_x about the observation point.
    sig_x = csr.beam._sigma_x
    xs = np.linspace(x - 14 * sig_x, x + 6 * sig_x, n_scan)

    seg_s, seg_lo, seg_hi = [], [], []
    for spv in sp:
        gz, gx = csr.get_CSR_integrand(s=s, t=t, x=x, xp=xs.reshape(-1, 1),
                                       sp=np.full((n_scan, 1), spv))
        nz = (gz[:, 0] != 0.0) | (gx[:, 0] != 0.0)
        if not nz.any():
            continue
        idx = np.where(nz)[0]
        for q in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            if len(q) < 2:
                continue
            seg_s.append(spv)
            seg_lo.append(xs[q[0]])
            seg_hi.append(xs[q[-1]])
    return (np.array(seg_s), np.array(seg_lo), np.array(seg_hi),
            c0, c1, br[0]['live'], br[1]['live'], sp)


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.chdir(EXAMPLE_DIR)
    lines = []

    def emit(s=''):
        print(s)
        lines.append(s)

    emit('=' * 86)
    emit("CSR integrand in band-relative coordinates v = (x' - centre_k) / half_k")
    emit('=' * 86)
    emit('v = 0 is the located branch; |v| <= 1 is the band the quadrature covers.')
    emit('"inside" is the fraction of total |integrand| magnitude at |v| <= 1, over')
    emit('the sampled strip |v| <= 3. It should be close to 1 for a live branch.')
    emit('')

    data = {}
    for shear in SHEARS:
        tag = f'sw{shear:g}'.replace('.', 'p')
        cfg = config_no_csr(shear, tag)
        csr = CSR2D(input_file=cfg)
        # debug=True is REQUIRED here, not cosmetic. CSR.py:311 gates the whole
        # density-history build on `debug or compute_CSR`, so compute_CSR = 0 alone
        # silently leaves the DF_tracker without a history -- get_CSR_integrand then
        # returns near-zeros and every diagnostic below looks plausible and is wrong.
        # debug=True builds the history; compute_CSR = 0 still skips the wake (:334).
        csr.run(stop_time=STOP_S, debug=True)
        csr.get_CSR_mesh()

        b = csr.beam
        t = b.position
        k = (MESH_XBINS // 2) * MESH_ZBINS + MESH_ZBINS // 2
        s_obs = t + csr.CSR_zmesh[k]
        x_obs = csr.CSR_xmesh[k]

        emit(f'--- shear z:x = {shear:g}   dx/dz = {b._slope[0]:+.4f}   '
             f'amp = {b._sigma_x/b._sigma_x_transform:.1f}x ---')
        emit(f"  {'region':>6} {'branch':>7} {'live %':>8} {'inside |v|<=1':>15}"
             f" | {'UNION coverage':>15}")

        per_region = {}
        for name, a, c in region_bounds(csr, s_obs):
            sp = np.linspace(a, c, NS)
            cov, tot = union_coverage(csr, s_obs, x_obs, t, sp)
            for kk in (0, 1):
                v, gz, gx, frac, live = branch_slice(csr, s_obs, x_obs, t, sp, kk)
                tail = f" | {cov:>15.5f}" if kk == 0 else ''
                emit(f"  {name:>6} {kk:>7} {live.mean()*100:>7.1f}% "
                     f"{frac:>15.4f}{tail}")
                per_region[(name, kk)] = (sp, v, gz, gx, frac, live)
            per_region[(name, 'cov')] = cov
        emit('')
        emit('  UNION coverage = fraction of |integrand| over a 40-half-width window')
        emit('  that lies inside the union of the located bands. This is the number')
        emit('  that matters: a low per-branch "inside" is harmless if the other band')
        emit('  covers it, but low UNION coverage means real support is being missed.')
        emit('')

        # Branch 0 is the narrow band -- it runs parallel to the s' axis over the
        # whole retained history, so the near region alone badly understates its
        # extent. Plot it out to one formation length, which is the scale that sets
        # how much history contributes at all.
        sp3 = per_region[('sp3', 0)][0]
        L_f = csr.formation_length
        sp_wide = np.linspace(max(0.0, s_obs - L_f), sp3[-1], NS)
        wide0 = (sp_wide,) + branch_slice(csr, s_obs, x_obs, t, sp_wide, 0)
        emit(f'  formation length L_f = {L_f*1e3:.1f} mm  '
             f'(branch 0 panel spans s\'-s from {(sp_wide[0]-s_obs)*1e3:.1f} mm)')
        data[shear] = dict(near=[wide0, per_region[('sp3', 1)]],
                           L_f=L_f,
                           s_obs=s_obs, x_obs=x_obs, slope=b._slope[0],
                           amp=b._sigma_x / b._sigma_x_transform,
                           lab=lab_geometry(csr, s_obs, x_obs, t,
                                            np.linspace(sp3[0], sp3[-1], 140)))
        tau = b._slope[0]
        emit(f'  tan(2 alpha) = 2 tau / (1 - tau^2) = '
             f'{2*tau/(1-tau**2):+.4f}  -> chirp band angle '
             f'{np.degrees(np.arctan(2*tau/(1-tau**2))):+.2f} deg from the s\' axis')
        emit('')

    # ---- figure -------------------------------------------------------------
    n = len(SHEARS)
    fig, axs = plt.subplots(2, n, figsize=(4.7 * n, 8.6))
    for j, shear in enumerate(SHEARS):
        d = data[shear]
        for kk in (0, 1):
            sp, v, gz, gx, frac, live = d['near'][kk]
            ax = axs[kk, j]
            # branch 0 spans a formation length, branch 1 only the near region
            scale, unit = ((1e3, 'mm') if kk == 0 else (1e6, r'$\mu$m'))
            pos = gz[gz > 0]
            if pos.size:
                vmax = np.percentile(pos, 99.9)
                vmin = max(vmax * 1e-5, pos.min())
                im = ax.pcolormesh((sp - d['s_obs']) * scale, v,
                                   np.clip(gz, vmin, vmax),
                                   norm=LogNorm(vmin=vmin, vmax=vmax),
                                   cmap='magma', shading='auto')
                plt.colorbar(im, ax=ax, label='|integrand$_z$|')
            ax.axhline(+1, color='c', lw=1.2, ls='--')
            ax.axhline(-1, color='c', lw=1.2, ls='--', label='band edge')
            ax.axhline(0, color='w', lw=0.7, alpha=0.5)
            ax.set_xlabel(rf"$s' - s$ ({unit})")
            ax.set_ylabel(r"$v = (x' - c_k)/\mathrm{half}_k$")
            span = ('to $-L_f$ = ' f"{-d['L_f']*1e3:.0f} mm" if kk == 0
                    else 'near region (sp3)')
            ax.set_title(f"shear {shear:g} (amp {d['amp']:.0f}x), branch {kk}, "
                         f"{span}\nlive {live.mean()*100:.0f}%, "
                         f"inside band {frac:.3f}", fontsize=9)
            ax.legend(fontsize=7, loc='upper right')
    fig.suptitle("CSR integrand in each branch's own band coordinate, near region "
                 "(sp3). v = 0 is the located branch, dashed lines are the band "
                 "edges. Support inside |v| < 1 means the locator is right.",
                 fontsize=12)
    plt.tight_layout()
    p = os.path.join(RESULT_DIR, 'integrand_band_relative.png')
    plt.savefig(p, dpi=125)
    plt.close()
    emit(f'Plot saved: {p}')

    # ---- figure 2: LAB-frame geometry, the thesis 4.4.2 picture -------------
    fig, axs = plt.subplots(1, n, figsize=(4.9 * n, 4.9))
    for j, shear in enumerate(SHEARS):
        d = data[shear]
        ss, slo, shi, c0, c1, l0, l1, sp_lab = d['lab']
        ax = np.atleast_1d(axs)[j]
        ds = (sp_lab - d['s_obs']) * 1e3

        if len(ss):
            ax.vlines((ss - d['s_obs']) * 1e3, (slo - d['x_obs']) * 1e6,
                      (shi - d['x_obs']) * 1e6, color='0.55', lw=2.2,
                      label='measured support')
        # Only where the branch is LIVE. On a dead column the fixed point has no
        # root and wanders (tens of mm), which is parked out of the integration but
        # is still the raw value in _branch_diag -- plotting it unmasked swamps the
        # axis and hides the real support entirely.
        ax.plot(ds, np.where(l0, (c0 - d['x_obs']) * 1e6, np.nan), 'r-', lw=1.6,
                label='Eq 4.24 branch 0 (live)')
        ax.plot(ds, np.where(l1, (c1 - d['x_obs']) * 1e6, np.nan), 'b--', lw=1.6,
                label='Eq 4.24 branch 1 (live)')
        ax.plot(0, 0, 'k+', ms=12, mew=2, label='observation point')
        ax.axhline(0, color='k', lw=0.5, alpha=0.4)

        tau = d['slope']
        ax.set_title(f"shear {shear:g}, amp {d['amp']:.0f}x\n"
                     r"$\tan 2\alpha$ = " f"{2*tau/(1-tau**2):+.2f} "
                     f"({np.degrees(np.arctan(2*tau/(1-tau**2))):+.0f}"
                     r"$^\circ$)", fontsize=10)
        ax.set_xlabel(r"$s' - s$ (mm)")
        ax.set_ylabel(r"$x' - x$ ($\mu$m)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle('Integrand support in LAB (x\', s\') coordinates, near region. '
                 'This is where the thesis 4.4.2 picture appears literally: a narrow '
                 'band near x\' = x and a chirp band leaving it at angle 2 alpha, '
                 'whose excursion grows with tilt.', fontsize=11)
    plt.tight_layout()
    p2 = os.path.join(RESULT_DIR, 'integrand_lab_geometry.png')
    plt.savefig(p2, dpi=125)
    plt.close()
    emit(f'Plot saved: {p2}')

    with open(os.path.join(RESULT_DIR, 'integrand_plane_log.txt'), 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
