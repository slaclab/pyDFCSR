"""
Acceptance test for the time interpolation of the density-function history.

Ground truth
------------
A Gaussian beam under pure linear shear:

    rho(x, z, t) = 1/(2 pi s_xi s_z) * exp(-xi^2/(2 s_xi^2) - z^2/(2 s_z^2)),
    xi = x - p(t) * z,     p(t) linear in t.

Snapshots are stored at t_k. The interpolant is queried at the midpoint between
two snapshots and compared against the TRUE density at that time -- i.e. a
Gaussian sheared by p(t_mid). Because p(t) is linear in t, any scheme that
interpolates the *frame* in time is exact here, so the measured error is purely
the error of the time-interpolation scheme.

Controlling parameter
---------------------
    G = |p_{k+1} - p_k| * sigma_z / sigma_xi

Blending two snapshots in the LAB frame returns a superposition of two
shear-displaced copies of the beam, separated in x by G*sigma_xi. It is
therefore O(1) wrong once G >~ 1, no matter how well each individual snapshot
is resolved.

Note on the pre-existing test: test_interp_tilted_gaussians.py compares against
(1-a)*rho(slope_k) + a*rho(slope_k+1), which IS the lab-frame blend. That
reference cannot detect this error -- it measures the interpolant against the
approximation under test.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyDFCSR_2D.interp3D import interpolate3D, interpolate3D_transformed

try:
    from pyDFCSR_2D.interp3D import interpolate3D_comoving_fields
    HAVE_COMOVING = True
except ImportError:
    HAVE_COMOVING = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'ghosting')
os.makedirs(RESULT_DIR, exist_ok=True)

SIGMA_XI = 50e-6
SIGMA_Z = 50e-6
XLIM = 5
NBINS = 200          # deposition bins per dimension, matches production configs
NBINS_LEGACY = 200   # legacy shares one lab grid at the same bin count


# ----------------------------------------------------------------------------
# Analytic truth
# ----------------------------------------------------------------------------
def rho_true(x, z, slope):
    xi = x - slope * z
    amp = 1.0 / (2 * np.pi * SIGMA_XI * SIGMA_Z)
    return amp * np.exp(-xi**2 / (2 * SIGMA_XI**2) - z**2 / (2 * SIGMA_Z**2))


def drho_dx_true(x, z, slope):
    """d rho / d x at fixed z  =  d rho / d xi."""
    xi = x - slope * z
    return -xi / SIGMA_XI**2 * rho_true(x, z, slope)


def drho_dz_true(x, z, slope):
    """d rho / d z at fixed x  =  d rho/dz|_xi - slope * d rho/d xi."""
    xi = x - slope * z
    r = rho_true(x, z, slope)
    return -z / SIGMA_Z**2 * r - slope * (-xi / SIGMA_XI**2 * r)


# ----------------------------------------------------------------------------
# Snapshot stacks, built by sampling the analytic density (no deposition, so
# the only error sources are grid resolution and time interpolation)
# ----------------------------------------------------------------------------
def build_xi_frame_stack(slopes):
    """Per-snapshot xi-frame grids, as DF_tracker_smooth stores them."""
    n_t = len(slopes)
    dens = np.zeros((n_t, NBINS, NBINS))
    dens_x = np.zeros((n_t, NBINS, NBINS))
    dens_z = np.zeros((n_t, NBINS, NBINS))
    min_xi = np.zeros(n_t)
    d_xi = np.zeros(n_t)
    min_z = np.zeros(n_t)
    d_z = np.zeros(n_t)
    polys = np.zeros((n_t, 2))

    xi_grid = np.linspace(-XLIM * SIGMA_XI, XLIM * SIGMA_XI, NBINS)
    z_grid = np.linspace(-XLIM * SIGMA_Z, XLIM * SIGMA_Z, NBINS)
    XI, Z = np.meshgrid(xi_grid, z_grid, indexing='ij')
    amp = 1.0 / (2 * np.pi * SIGMA_XI * SIGMA_Z)
    r = amp * np.exp(-XI**2 / (2 * SIGMA_XI**2) - Z**2 / (2 * SIGMA_Z**2))
    r_dxi = -XI / SIGMA_XI**2 * r
    r_dz_xi = -Z / SIGMA_Z**2 * r

    for k, slope in enumerate(slopes):
        dens[k] = r
        dens_x[k] = r_dxi
        # chain rule baked in with THIS snapshot's slope, as deposit_smooth.py does
        dens_z[k] = r_dz_xi - slope * r_dxi
        min_xi[k] = xi_grid[0]
        d_xi[k] = xi_grid[1] - xi_grid[0]
        min_z[k] = z_grid[0]
        d_z[k] = z_grid[1] - z_grid[0]
        polys[k] = [slope, 0.0]

    return dens, dens_x, dens_z, min_xi, d_xi, min_z, d_z, polys


def build_lab_frame_stack(slopes):
    """One shared lab grid spanning the full tilted extent, as DF_tracker does."""
    n_t = len(slopes)
    max_slope = max(abs(s) for s in slopes)
    x_extent = XLIM * SIGMA_XI + XLIM * SIGMA_Z * max_slope

    x_grid = np.linspace(-x_extent, x_extent, NBINS_LEGACY)
    z_grid = np.linspace(-XLIM * SIGMA_Z, XLIM * SIGMA_Z, NBINS_LEGACY)
    X, Z = np.meshgrid(x_grid, z_grid, indexing='ij')

    dens = np.zeros((n_t, NBINS_LEGACY, NBINS_LEGACY))
    dens_x = np.zeros((n_t, NBINS_LEGACY, NBINS_LEGACY))
    dens_z = np.zeros((n_t, NBINS_LEGACY, NBINS_LEGACY))
    for k, slope in enumerate(slopes):
        dens[k] = rho_true(X, Z, slope)
        dens_x[k] = drho_dx_true(X, Z, slope)
        dens_z[k] = drho_dz_true(X, Z, slope)

    return dens, dens_x, dens_z, x_grid, z_grid


def build_comoving_stack(slopes):
    """
    Per-snapshot NORMALIZED grids, as DF_tracker_comoving stores them.

    u = (xi - xi_bar)/sigma_xi, w = (z - z_bar)/sigma_z, with the same grid for
    every snapshot. Here xi_bar = z_bar = 0 and the sigmas are constant, so the
    stored shape rho_hat is literally identical at every snapshot and only the
    frame (the slope) changes with time -- which is the whole point: a scheme
    that interpolates the frame is then exact.

    rho_hat is normalized so that the integral over du dw is 1, matching the
    tracker's `density /= npart * delta_u * delta_w`.
    """
    n_t = len(slopes)

    # bin centres, matching histogram_bspline_2d / DF_tracker_comoving
    delta_u = 2.0 * XLIM / NBINS
    delta_w = 2.0 * XLIM / NBINS
    u = -XLIM + (np.arange(NBINS) + 0.5) * delta_u
    w = -XLIM + (np.arange(NBINS) + 0.5) * delta_w
    U, W = np.meshgrid(u, w, indexing='ij')

    rho_hat = np.exp(-0.5 * (U**2 + W**2)) / (2.0 * np.pi)
    rho_hat_u = -U * rho_hat
    rho_hat_w = -W * rho_hat

    dens = np.repeat(rho_hat[None], n_t, axis=0)
    dens_u = np.repeat(rho_hat_u[None], n_t, axis=0)
    dens_w = np.repeat(rho_hat_w[None], n_t, axis=0)
    zeros = np.zeros_like(dens)

    polys = np.array([[sl, 0.0] for sl in slopes])
    xi_bar = np.zeros(n_t)
    z_bar = np.zeros(n_t)
    s_xi = np.full(n_t, SIGMA_XI)
    s_z = np.full(n_t, SIGMA_Z)

    return dict(data_rho=dens, data_rho_u=dens_u, data_rho_w=dens_w,
                data_vx=zeros, data_vx_u=zeros,
                poly_coeffs=polys, xi_bar_arr=xi_bar, sigma_xi_arr=s_xi,
                z_bar_arr=z_bar, sigma_z_arr=s_z,
                u_start=-float(XLIM), delta_u=delta_u,
                w_start=-float(XLIM), delta_w=delta_w)


def query_points(slope_mid, n=60, extent=3.0):
    """
    Query on a grid laid out in the TRUE beam frame at the query time, so every
    point sits where the beam actually is. Avoids masking heuristics.
    """
    xi_1d = np.linspace(-extent * SIGMA_XI, extent * SIGMA_XI, n)
    z_1d = np.linspace(-extent * SIGMA_Z, extent * SIGMA_Z, n)
    XI, Z = np.meshgrid(xi_1d, z_1d, indexing='ij')
    z = Z.ravel()
    x = XI.ravel() + slope_mid * z
    return x, z


def rel_err(approx, exact):
    norm = np.linalg.norm(exact)
    return np.linalg.norm(approx - exact) / norm if norm > 0 else 0.0


# ----------------------------------------------------------------------------
# One measurement at a given shear increment
# ----------------------------------------------------------------------------
def measure(ds, delta_t=0.05):
    """
    slopes = [0, ds, 2ds]  ->  p(t) linear in t.
    Query at the midpoint of the first interval; truth is slope = ds/2.
    Returns dict of relative errors.
    """
    slopes = [0.0, ds, 2.0 * ds]
    times = np.arange(3) * delta_t
    t_mid = 0.5 * delta_t
    slope_mid = 0.5 * ds

    x_q, z_q = query_points(slope_mid)
    t_q = np.full(len(x_q), t_mid)

    ex_rho = rho_true(x_q, z_q, slope_mid)
    ex_dx = drho_dx_true(x_q, z_q, slope_mid)
    ex_dz = drho_dz_true(x_q, z_q, slope_mid)

    # the reference the old test used: the lab-frame blend itself
    blend_rho = 0.5 * (rho_true(x_q, z_q, slopes[0]) + rho_true(x_q, z_q, slopes[1]))

    out = {'ds': ds, 'G': abs(ds) * SIGMA_Z / SIGMA_XI}

    # --- current method: per-snapshot xi frame, lab-frame blend ---
    dn, dxn, dzn, mxi, dxi, mz, dz_, polys = build_xi_frame_stack(slopes)
    kw = dict(poly_coeffs=polys, min_xi_arr=mxi, min_z_arr=mz, min_t=times[0],
              delta_xi_arr=dxi, delta_z_arr=dz_, delta_t=delta_t)
    new_rho = interpolate3D_transformed(x_q, z_q, t_q, dn, **kw)
    new_dx = interpolate3D_transformed(x_q, z_q, t_q, dxn, **kw)
    new_dz = interpolate3D_transformed(x_q, z_q, t_q, dzn, **kw)
    out['new_rho'] = rel_err(new_rho, ex_rho)
    out['new_dx'] = rel_err(new_dx, ex_dx)
    out['new_dz'] = rel_err(new_dz, ex_dz)

    # --- legacy: shared lab grid, lab-frame blend ---
    dl, dxl, dzl, xg, zg = build_lab_frame_stack(slopes)
    kwl = dict(min_x=times[0], min_y=xg[0], min_z=zg[0],
               delta_x=delta_t, delta_y=xg[1] - xg[0], delta_z=zg[1] - zg[0])
    leg_rho = interpolate3D(t_q, x_q, z_q, dl, **kwl)
    leg_dx = interpolate3D(t_q, x_q, z_q, dxl, **kwl)
    leg_dz = interpolate3D(t_q, x_q, z_q, dzl, **kwl)
    out['leg_rho'] = rel_err(leg_rho, ex_rho)
    out['leg_dx'] = rel_err(leg_dx, ex_dx)
    out['leg_dz'] = rel_err(leg_dz, ex_dz)

    # --- new: co-moving frame interpolation ---
    if HAVE_COMOVING:
        cm = build_comoving_stack(slopes)
        c_rho, c_dx, c_dz, _, _ = interpolate3D_comoving_fields(
            x_q, z_q, t_q,
            cm['data_rho'], cm['data_rho_u'], cm['data_rho_w'],
            cm['data_vx'], cm['data_vx_u'],
            cm['poly_coeffs'], cm['xi_bar_arr'], cm['sigma_xi_arr'],
            cm['z_bar_arr'], cm['sigma_z_arr'],
            cm['u_start'], cm['delta_u'], cm['w_start'], cm['delta_w'],
            times[0], delta_t)
        out['cmv_rho'] = rel_err(c_rho, ex_rho)
        out['cmv_dx'] = rel_err(c_dx, ex_dx)
        out['cmv_dz'] = rel_err(c_dz, ex_dz)
    else:
        out['cmv_rho'] = out['cmv_dx'] = out['cmv_dz'] = np.nan

    # --- how wrong the old test's own reference is ---
    out['blend_ref_rho'] = rel_err(blend_rho, ex_rho)

    # transverse resolution actually achieved, in units of sigma_xi
    out['dx_new_over_sxi'] = (2 * XLIM * SIGMA_XI / NBINS) / SIGMA_XI
    out['dx_leg_over_sxi'] = ((xg[1] - xg[0])) / SIGMA_XI

    return out


def cut_plot(ds, delta_t=0.05):
    """1D lab-frame cut at z = +2 sigma_z showing the ghost double image."""
    slopes = [0.0, ds, 2.0 * ds]
    times = np.arange(3) * delta_t
    t_mid = 0.5 * delta_t
    slope_mid = 0.5 * ds

    z0 = 2.0 * SIGMA_Z
    # span both ghost images plus margin
    half = (abs(ds) * z0) + 4 * SIGMA_XI
    x_1d = np.linspace(slope_mid * z0 - half, slope_mid * z0 + half, 800)
    z_1d = np.full_like(x_1d, z0)
    t_1d = np.full_like(x_1d, t_mid)

    ex = rho_true(x_1d, z_1d, slope_mid)

    dn, dxn, dzn, mxi, dxi, mz, dz_, polys = build_xi_frame_stack(slopes)
    new = interpolate3D_transformed(
        x_1d, z_1d, t_1d, dn, poly_coeffs=polys, min_xi_arr=mxi, min_z_arr=mz,
        min_t=times[0], delta_xi_arr=dxi, delta_z_arr=dz_, delta_t=delta_t)

    dl, dxl, dzl, xg, zg = build_lab_frame_stack(slopes)
    leg = interpolate3D(t_1d, x_1d, z_1d, dl, min_x=times[0], min_y=xg[0],
                        min_z=zg[0], delta_x=delta_t, delta_y=xg[1] - xg[0],
                        delta_z=zg[1] - zg[0])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_1d * 1e6, ex, 'k-', lw=2.5, label='truth (sheared by $p_\\alpha$)')
    ax.plot(x_1d * 1e6, new, 'r-', lw=1.5,
            label="current 'bspline_fft' (lab-frame blend)")
    ax.plot(x_1d * 1e6, leg, 'b--', lw=1.5, label='legacy (lab grid, blurred)')
    for s, c in ((slopes[0], 'r'), (slopes[1], 'r')):
        ax.axvline(s * z0 * 1e6, color=c, ls=':', alpha=0.5)
    G = abs(ds) * SIGMA_Z / SIGMA_XI
    ax.set_xlabel('x ($\\mu$m)  at  z = +2$\\sigma_z$')
    ax.set_ylabel(r'$\rho$')
    ax.set_title(f'Ghosting: slope {slopes[0]:.1f} $\\to$ {slopes[1]:.1f}, '
                 f'queried at midpoint.  G = {G:.1f}\n'
                 'dotted lines = where each snapshot places the beam')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, f'ghost_cut_G{G:.0f}.png')
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def run_test():
    print("=" * 78)
    print("Time-interpolation error vs ghosting parameter G = |dp'| sigma_z / sigma_xi")
    print("=" * 78)
    print(f"sigma_xi = {SIGMA_XI*1e6:.0f} um, sigma_z = {SIGMA_Z*1e6:.0f} um, "
          f"{NBINS} bins, xlim = {XLIM}")
    print("Reference is the TRUE density at the query time (a Gaussian sheared")
    print("by the time-interpolated slope), NOT the blend of the two snapshots.\n")

    ds_list = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    rows = [measure(ds) for ds in ds_list]

    hdr = (f"{'G':>7} | {'new rho':>9} {'new dx':>9} {'new dz':>9} | "
           f"{'leg rho':>9} {'leg dx':>9} {'leg dz':>9} | "
           f"{'CMV rho':>9} {'CMV dx':>9} {'CMV dz':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['G']:>7.2f} | {r['new_rho']:>9.4f} {r['new_dx']:>9.4f} "
              f"{r['new_dz']:>9.4f} | {r['leg_rho']:>9.4f} {r['leg_dx']:>9.4f} "
              f"{r['leg_dz']:>9.4f} | {r['cmv_rho']:>9.6f} {r['cmv_dx']:>9.6f} "
              f"{r['cmv_dz']:>9.6f}")
    print("\nCMV = co-moving frame interpolation. p(t) is linear in t and the sigmas")
    print("are constant here, so any scheme that interpolates the FRAME is exact and")
    print("the CMV columns should stay flat at the B-spline grid error, independent")
    print("of G. The 'new' and 'leg' columns are the two lab-frame blends.")

    print(f"\nTransverse cell size, current method: "
          f"{rows[0]['dx_new_over_sxi']:.3f} sigma_xi (independent of tilt)")
    print("Legacy cell size grows with tilt (last column) -- that blur is what")
    print("merges the ghost images and makes the legacy wakes look smooth.\n")

    print("Error of the reference used by test_interp_tilted_gaussians.py")
    print("(the lab-frame blend), measured against the true density:")
    for r in rows:
        print(f"  G = {r['G']:>6.2f}   blend-vs-truth = {r['blend_ref_rho']:.4f}")

    # ---- summary plot ----
    G = [r['G'] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, (key_n, key_l, lab) in zip(axes, [
            ('new_rho', 'leg_rho', r'$\rho$'),
            ('new_dx', 'leg_dx', r'$\partial\rho/\partial x$'),
            ('new_dz', 'leg_dz', r'$\partial\rho/\partial z$')]):
        ax.loglog(G, [r[key_n] for r in rows], 'ro-', label="current 'bspline_fft'")
        ax.loglog(G, [r[key_l] for r in rows], 'bs--', label='legacy')
        key_c = 'cmv_' + key_n.split('_', 1)[1]
        if np.isfinite([r[key_c] for r in rows]).any():
            ax.loglog(G, [r[key_c] for r in rows], 'g^-', lw=2,
                      label="new 'bspline_comoving'")
        ax.axvline(1.0, color='k', ls=':', alpha=0.6)
        ax.text(1.05, 0.02, 'G = 1', rotation=90, fontsize=9, alpha=0.7)
        ax.set_xlabel(r"$G = |\Delta p^{\prime}|\,\sigma_z/\sigma_\xi$")
        ax.set_title(lab)
        ax.grid(alpha=0.3, which='both')
    axes[0].set_ylabel('relative L2 error vs truth')
    axes[0].legend()
    fig.suptitle('Density-history time interpolation error vs ghosting parameter\n'
                 'reference = true density at the query time', fontsize=12)
    plt.tight_layout()
    p = os.path.join(RESULT_DIR, 'error_vs_G.png')
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"\nPlot saved: {p}")

    for ds in (2.0, 10.0):
        print(f"Plot saved: {cut_plot(ds)}")


if __name__ == '__main__':
    run_test()
