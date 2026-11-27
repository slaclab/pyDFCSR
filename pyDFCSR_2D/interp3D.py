import math

import numpy as np
from numba import jit

from .lookup import lookup_hybrid


@jit(nopython=True, cache=True)
def eval_poly(coeffs, z):
    """Evaluate polynomial using Horner's method. coeffs in numpy polyfit order (highest degree first)."""
    result = coeffs[0]
    for j in range(1, len(coeffs)):
        result = result * z + coeffs[j]
    return result


@jit(nopython=True, cache=True)
def curv_g(c2, c3, w, wl):
    """
    Curvature shear of the normalized frame, u' = u - g(w):
        g(w) = c2 (w^2 - 1) + c3 w^3                    for |w| <= wl
    continued LINEARLY beyond |w| = wl (value and slope matched), so g' has no jump
    and the fit is not extrapolated as a cubic into the empty tails.
    Returns (g, dg/dw).
    """
    if w > wl:
        g0 = c2 * (wl * wl - 1.0) + c3 * wl * wl * wl
        d0 = 2.0 * c2 * wl + 3.0 * c3 * wl * wl
        return g0 + d0 * (w - wl), d0
    if w < -wl:
        g0 = c2 * (wl * wl - 1.0) - c3 * wl * wl * wl
        d0 = -2.0 * c2 * wl + 3.0 * c3 * wl * wl
        return g0 + d0 * (w + wl), d0
    return c2 * (w * w - 1.0) + c3 * w * w * w, 2.0 * c2 * w + 3.0 * c3 * w * w


@jit(nopython=True, cache=True)
def eval_poly_deriv(coeffs, z):
    """Evaluate derivative of polynomial. coeffs in numpy polyfit order (highest degree first)."""
    n = len(coeffs)
    if n <= 1:
        return 0.0
    # derivative coeffs: for degree-d term with coeff c, derivative is d*c * z^(d-1)
    # coeffs[0] is degree n-1, coeffs[1] is degree n-2, etc.
    result = coeffs[0] * (n - 1)
    for j in range(1, n - 1):
        result = result * z + coeffs[j] * (n - 1 - j)
    return result


# ---------------------------------------------------------------------------
# Co-moving (Lagrangian) interpolation of the density history.
#
# A lab-frame blend (the removed bspline_fft path) evaluates each snapshot
# in its OWN tilted frame and then averages the two results. That is a
# superposition of two shear-displaced copies of the beam, so for a beam whose
# tilt rate changes by dp' between snapshots it returns two peaks where there is
# one, separated by G*sigma_xi with G = |dp'|*sigma_z/sigma_xi. It is O(1) wrong
# once G >~ 1, which a chicane reaches easily.
#
# Here the FRAME is interpolated in time instead of the field. Every snapshot
# stores its shape on the same normalized grid u = (xi - xi_bar)/sigma_xi,
# w = (z - z_bar)/sigma_z, so index (i, j) refers to the same material point of
# the beam at every snapshot. Blending at fixed (u, w) is then exact for any
# affine evolution of the beam -- shear plus scaling -- which is what linear
# optics does between history steps.
#
# Note the normalized grid is shared in INDEX space only. Each snapshot's
# physical cell is still delta_xi_k = 2*xlim*sigma_xi_k/nbins, so no transverse
# resolution is given up.
# ---------------------------------------------------------------------------


@jit(nopython=True, cache=True)
def cubic_bspline_w(x):
    """Cubic B-spline basis B3(x), support [-2, 2]. Matches deposit_smooth."""
    ax = abs(x)
    if ax >= 2.0:
        return 0.0
    elif ax >= 1.0:
        return (2.0 - ax) ** 3 / 6.0
    else:
        return (4.0 - 6.0 * ax * ax + 3.0 * ax * ax * ax) / 6.0


@jit(nopython=True, cache=True)
def bspline_eval_single(data2d, u_cell, w_cell, n_u, n_w):
    """
    Evaluate a 2D field with the same cubic B-spline kernel used to deposit it.

    C2 continuous, so the integrand it feeds has no derivative jumps on the cell
    lattice -- unlike bilinear_single, whose first derivative jumps at every cell
    boundary and degrades the outer trapezoid rule to 1st order.

    u_cell, w_cell are positions in BIN-CENTRE units, i.e.
        u_cell = (u - u_start)/delta_u - 0.5
    which is exactly the convention histogram_bspline_2d deposits with.

    Out-of-range stencil entries are treated as zero, again as the deposition
    does, so the result tapers smoothly to 0 outside the grid instead of
    stepping. math.floor (not int()) because int() truncates toward zero, which
    silently turns indices in (-1, 0) into extrapolation with negative weights.
    """
    i0 = int(math.floor(u_cell)) - 1
    j0 = int(math.floor(w_cell)) - 1

    if i0 + 3 < 0 or i0 >= n_u or j0 + 3 < 0 or j0 >= n_w:
        return 0.0

    out = 0.0
    for di in range(4):
        i = i0 + di
        if i < 0 or i >= n_u:
            continue
        wi = cubic_bspline_w(u_cell - i)
        if wi == 0.0:
            continue
        for dj in range(4):
            j = j0 + dj
            if j < 0 or j >= n_w:
                continue
            out += data2d[i, j] * wi * cubic_bspline_w(w_cell - j)
    return out


def _interpolate3D_comoving_fields_impl(xval, zval, tval,
                                  data_rho, data_rho_u, data_rho_w,
                                  data_vx, data_vx_u,
                                  poly_coeffs, xi_bar_arr, sigma_xi_arr,
                                  z_bar_arr, sigma_z_arr,
                                  u_start, delta_u, w_start, delta_w,
                                  min_t, delta_t,
                                  blend_code, var_z_arr, cov_arr, var_x_arr,
                                  x_bar_arr,
                                  t_arr, bucket, bucket_inv_h, bucket_t0, bucket_M,
                                  is_uniform, curv_c2, curv_c3, curv_wl):
    """
    Interpolate the density history in a co-moving affine frame, plus an optional
    curvature shear u' = u - g(w) (curv_c2/curv_c3 per snapshot, blended linearly;
    all zero = the affine frame exactly).

    All five fields the CSR integrand needs are returned from ONE pass, because
    they share the frame construction and the stencil indices. get_CSR_integrand
    previously made five separate interpolate3D_transformed calls, each redoing
    that work.

    Returns (rho, rho_x, rho_z, vx, vx_x) with
        rho    = density in the lab frame
        rho_x  = d rho / dx at fixed z
        rho_z  = d rho / dz at fixed x   (chain rule applied with the BLENDED p')
        vx     = transverse velocity
        vx_x   = d vx / dx

    Frame blending, selected by blend_code (see DF_tracker_comoving.BLEND_CODES):

      0 'coeff'   the polynomial and the means are linear in alpha; the sigmas are
                  blended log-linearly so they stay positive and so that a beam
                  growing exponentially (which is what a drift does to a diverging
                  beam) is tracked smoothly rather than with a kink.
      1 'moment'  (var_z, cov, var_x) and the centroid are linear in alpha, and
                  tau, sigma_z, sigma_xi are derived from them. tau = tan(alpha) is
                  the wrong chart across a longitudinal waist -- see the frame_blend
                  note in DF_tracker_comoving.configure_params.
      2 'orient'  orientation from the moment ratio tau = cov/var_z, widths
                  log-linear as in 'coeff'. Fixes the waist without paying 'moment's
                  sigma_xi inflation.

    CSR._comoving_frame_at must mirror whichever branch is active: a band located
    with a different blend than the interpolant uses will not sit where the density
    is.
    """
    n = len(xval)
    rho = np.zeros(n)
    rho_x = np.zeros(n)
    rho_z = np.zeros(n)
    vx = np.zeros(n)
    vx_x = np.zeros(n)

    n_t = data_rho.shape[0]
    n_u = data_rho.shape[1]
    n_w = data_rho.shape[2]

    for i in range(n):
        # Bracketing index and blend weight. The is_uniform branch inside lookup_hybrid is the
        # historical expression character for character, so a uniform history -- every run under
        # step_control mode 'legacy' -- reproduces previous results bit-for-bit. A locally refined
        # schedule takes the bucket table instead, at a measured +1.84% of wake-mesh time.
        k, a = lookup_hybrid(tval[i], t_arr, bucket, bucket_inv_h, bucket_t0, bucket_M,
                             n_t, is_uniform, min_t, delta_t)
        b = 1.0 - a

        z = zval[i]

        # --- blend the frame, not the field ---
        if blend_code == 0:
            # eval_poly is linear in the coefficients, so blending its value is the
            # same as blending the coefficients and evaluating.
            p_a = b * eval_poly(poly_coeffs[k], z) + a * eval_poly(poly_coeffs[k + 1], z)
            dp_a = b * eval_poly_deriv(poly_coeffs[k], z) + a * eval_poly_deriv(poly_coeffs[k + 1], z)

            xi_bar = b * xi_bar_arr[k] + a * xi_bar_arr[k + 1]
            z_bar = b * z_bar_arr[k] + a * z_bar_arr[k + 1]
            s_xi = math.exp(b * math.log(sigma_xi_arr[k]) + a * math.log(sigma_xi_arr[k + 1]))
            s_z = math.exp(b * math.log(sigma_z_arr[k]) + a * math.log(sigma_z_arr[k + 1]))
        else:
            var_z = b * var_z_arr[k] + a * var_z_arr[k + 1]
            cov = b * cov_arr[k] + a * cov_arr[k + 1]
            z_bar = b * z_bar_arr[k] + a * z_bar_arr[k + 1]
            x_bar = b * x_bar_arr[k] + a * x_bar_arr[k + 1]

            # Orientation from the moment RATIO. This is the part that survives a
            # longitudinal waist, where tau = tan(alpha) is the wrong chart.
            tau = cov / var_z
            # Centroid-pivoted axis. xi_bar is folded into p_a, so the shared code
            # below needs no branch.
            dp_a = tau
            p_a = tau * (z - z_bar) + x_bar
            xi_bar = 0.0

            if blend_code == 1:
                # 'moment': widths from the blended covariance as well.
                var_x = b * var_x_arr[k] + a * var_x_arr[k + 1]
                s_z = math.sqrt(var_z)
                # PSD of the blend guarantees this is >= 0 in exact arithmetic; the
                # clamp is only against round-off at the degenerate point.
                var_xi = var_x - cov * cov / var_z
                if var_xi < 0.0:
                    var_xi = 0.0
                s_xi = math.sqrt(var_xi)
            else:
                # 'orient': widths log-linear, as in 'coeff'.
                #
                # Blending the full covariance INFLATES sigma_xi badly, because a
                # convex mix of two thin ellipses at different tilts is fatter than
                # either -- by a(1-a)(dtau)^2 var_z. Measured at shear 20 in the body
                # of the dipole, mid-interval sigma_xi comes out 8x-10x too large
                # (20.3 um -> 196.9 um at tau: -9.92 -> -6.58), and the wake then
                # fails to converge under step refinement. The widths do not need the
                # moment chart -- only the orientation does -- so take just that.
                s_xi = math.exp(b * math.log(sigma_xi_arr[k]) + a * math.log(sigma_xi_arr[k + 1]))
                s_z = math.exp(b * math.log(sigma_z_arr[k]) + a * math.log(sigma_z_arr[k + 1]))

        # --- map the query point into the shared normalized frame ---
        xi = xval[i] - p_a
        u = (xi - xi_bar) / s_xi
        w = (z - z_bar) / s_z
        # curvature shear: the coefficients are blended linearly like the centroids;
        # unit Jacobian, so densities need no rescaling
        c2 = b * curv_c2[k] + a * curv_c2[k + 1]
        c3 = b * curv_c3[k] + a * curv_c3[k + 1]
        g_w, dg_w = curv_g(c2, c3, w, curv_wl)
        u = u - g_w

        u_cell = (u - u_start) / delta_u - 0.5
        w_cell = (w - w_start) / delta_w - 0.5

        # --- evaluate both snapshots at the SAME (u, w), then blend ---
        #
        # This used to be ten bspline_eval_single calls (5 fields x snapshots k, k+1). All
        # ten share the same (u_cell, w_cell), hence the same 4x4 stencil and the same eight
        # basis weights -- but cubic_bspline_w(w_cell - j) sat INSIDE the dj loop, so every
        # call recomputed the four w-weights sixteen times, and LLVM cannot common that up
        # across ten separate calls over ten different arrays. It came to ~200 weight
        # evaluations per query where only 8 distinct values exist. Profiling put this
        # kernel at 67% of wake-mesh time, so this was the whole hot spot.
        #
        # Weights are now computed once and all ten accumulators filled in one pass.
        #
        # BIT-IDENTICAL, deliberately: each accumulator receives the same terms in the same
        # order as before, and the multiply keeps the original left-to-right association
        # (d * wi) * ww -- the algebraically equal d * (wi * ww) would change the last bits.
        # cubic_bspline_w is pure, so hoisting it cannot change a value.
        i0 = int(math.floor(u_cell)) - 1
        j0 = int(math.floor(w_cell)) - 1

        r0 = 0.0
        r1 = 0.0
        ru0 = 0.0
        ru1 = 0.0
        rw0 = 0.0
        rw1 = 0.0
        v0 = 0.0
        v1 = 0.0
        vu0 = 0.0
        vu1 = 0.0

        # the same whole-stencil-misses-the-grid early out each of the ten calls performed
        if not (i0 + 3 < 0 or i0 >= n_u or j0 + 3 < 0 or j0 >= n_w):
            for di in range(4):
                ii = i0 + di
                if ii < 0 or ii >= n_u:
                    continue
                wi = cubic_bspline_w(u_cell - ii)
                if wi == 0.0:
                    continue
                for dj in range(4):
                    jj = j0 + dj
                    if jj < 0 or jj >= n_w:
                        continue
                    ww = cubic_bspline_w(w_cell - jj)
                    r0 += (data_rho[k, ii, jj] * wi) * ww
                    r1 += (data_rho[k + 1, ii, jj] * wi) * ww
                    ru0 += (data_rho_u[k, ii, jj] * wi) * ww
                    ru1 += (data_rho_u[k + 1, ii, jj] * wi) * ww
                    rw0 += (data_rho_w[k, ii, jj] * wi) * ww
                    rw1 += (data_rho_w[k + 1, ii, jj] * wi) * ww
                    v0 += (data_vx[k, ii, jj] * wi) * ww
                    v1 += (data_vx[k + 1, ii, jj] * wi) * ww
                    vu0 += (data_vx_u[k, ii, jj] * wi) * ww
                    vu1 += (data_vx_u[k + 1, ii, jj] * wi) * ww

        rho_h = b * r0 + a * r1
        rho_hu = b * ru0 + a * ru1
        rho_hw = b * rw0 + a * rw1
        vx_h = b * v0 + a * v1
        vx_hu = b * vu0 + a * vu1

        # --- back to physical units ---
        # the shear xi = x - p(z) has unit Jacobian, so the area element is
        # only the sigma scaling: dx dz = s_xi * s_z du dw
        jac = s_xi * s_z
        rho[i] = rho_h / jac
        rho_x[i] = rho_hu / (s_xi * jac)
        # d rho/dz|_x = d rho/dz|_xi - p'(z) d rho/dxi, with the BLENDED p', and the
        # curvature shear adds d u'/dz = -g'(w)/s_z
        rho_z[i] = (rho_hw / s_z - (dp_a / s_xi + dg_w / s_z) * rho_hu) / jac
        vx[i] = vx_h
        vx_x[i] = vx_hu / s_xi

    return rho, rho_x, rho_z, vx, vx_x


# serial and thread-parallel compilations of the same source; prange is a plain range in the
# serial one, so it is the historical kernel bit for bit. CSR2D picks the parallel one only for
# large point sets when CSR_integration.threads > 1 (the fork/join cost dominates small calls).
interpolate3D_comoving_fields = jit(nopython=True, cache=True)(_interpolate3D_comoving_fields_impl)

def _interpolate3D_comoving_ibp_impl(xval, zval, tval,
                                  data_rho, data_rho_u, data_rho_w,
                                  data_vx, data_vx_u,
                                  poly_coeffs, xi_bar_arr, sigma_xi_arr,
                                  z_bar_arr, sigma_z_arr,
                                  u_start, delta_u, w_start, delta_w,
                                  min_t, delta_t,
                                  blend_code, var_z_arr, cov_arr, var_x_arr,
                                  x_bar_arr,
                                  t_arr, bucket, bucket_inv_h, bucket_t0, bucket_M,
                                  is_uniform, curv_c2, curv_c3, curv_wl, want_fields):
    """
    interpolate3D_comoving_fields, plus the two frame-motion coefficients the integrated-by-parts
    longitudinal integrand needs (CSR2D._integrate_xi_region with CSR_integration.ibp;
    progress_cancellation.md §2). The five fields are computed by the same statements in the
    same order, so they are bit-identical to interpolate3D_comoving_fields.

    Along an s' column the retarded point moves as dz/dx' = R_x, dt/dx' = -R_x, and

        d rho_ret / dx' = rho_x * (1 - A R_x) + B R_x            (exact, for this interpolant)

    with, for rho = F(u, w, t) / (s_xi s_z) and the blended frame u(x, z, t), w(z, t):

        A = -s_xi (u_z - u_t)
        B = [F_w (w_z - w_t) - F_t] / (s_xi s_z) + rho (d ln(s_xi s_z)/dt)

    u_t, w_t and F_t are the time derivatives of the piecewise-linear blend between the two
    bracketing snapshots (0 outside the history, where the lookup clamps). A is the beam axis'
    slope against the column (the J = 1 - A R_x of the by-parts form) and B R_x the slice-to-slice
    change along the column at fixed u: the part of d rho_ret/dx' that has no +- lobes.

    want_fields = False skips the B-spline stencil and returns only A (fields and B are 0): the
    by-parts form differentiates J in x' by finite differences, which needs A alone.

    Returns (rho, rho_x, rho_z, vx, vx_x, A, B).
    """
    n = len(xval)
    rho = np.zeros(n)
    rho_x = np.zeros(n)
    rho_z = np.zeros(n)
    vx = np.zeros(n)
    vx_x = np.zeros(n)
    A_out = np.zeros(n)
    B_out = np.zeros(n)

    n_t = data_rho.shape[0]
    n_u = data_rho.shape[1]
    n_w = data_rho.shape[2]

    for i in range(n):
        # Bracketing index and blend weight. The is_uniform branch inside lookup_hybrid is the
        # historical expression character for character, so a uniform history -- every run under
        # step_control mode 'legacy' -- reproduces previous results bit-for-bit. A locally refined
        # schedule takes the bucket table instead, at a measured +1.84% of wake-mesh time.
        k, a = lookup_hybrid(tval[i], t_arr, bucket, bucket_inv_h, bucket_t0, bucket_M,
                             n_t, is_uniform, min_t, delta_t)
        b = 1.0 - a

        z = zval[i]
        var_z = 1.0
        cov = 0.0
        tau = 0.0

        # --- blend the frame, not the field ---
        if blend_code == 0:
            # eval_poly is linear in the coefficients, so blending its value is the
            # same as blending the coefficients and evaluating.
            p_a = b * eval_poly(poly_coeffs[k], z) + a * eval_poly(poly_coeffs[k + 1], z)
            dp_a = b * eval_poly_deriv(poly_coeffs[k], z) + a * eval_poly_deriv(poly_coeffs[k + 1], z)

            xi_bar = b * xi_bar_arr[k] + a * xi_bar_arr[k + 1]
            z_bar = b * z_bar_arr[k] + a * z_bar_arr[k + 1]
            s_xi = math.exp(b * math.log(sigma_xi_arr[k]) + a * math.log(sigma_xi_arr[k + 1]))
            s_z = math.exp(b * math.log(sigma_z_arr[k]) + a * math.log(sigma_z_arr[k + 1]))
        else:
            var_z = b * var_z_arr[k] + a * var_z_arr[k + 1]
            cov = b * cov_arr[k] + a * cov_arr[k + 1]
            z_bar = b * z_bar_arr[k] + a * z_bar_arr[k + 1]
            x_bar = b * x_bar_arr[k] + a * x_bar_arr[k + 1]

            # Orientation from the moment RATIO. This is the part that survives a
            # longitudinal waist, where tau = tan(alpha) is the wrong chart.
            tau = cov / var_z
            # Centroid-pivoted axis. xi_bar is folded into p_a, so the shared code
            # below needs no branch.
            dp_a = tau
            p_a = tau * (z - z_bar) + x_bar
            xi_bar = 0.0

            if blend_code == 1:
                # 'moment': widths from the blended covariance as well.
                var_x = b * var_x_arr[k] + a * var_x_arr[k + 1]
                s_z = math.sqrt(var_z)
                # PSD of the blend guarantees this is >= 0 in exact arithmetic; the
                # clamp is only against round-off at the degenerate point.
                var_xi = var_x - cov * cov / var_z
                if var_xi < 0.0:
                    var_xi = 0.0
                s_xi = math.sqrt(var_xi)
            else:
                # 'orient': widths log-linear, as in 'coeff'.
                #
                # Blending the full covariance INFLATES sigma_xi badly, because a
                # convex mix of two thin ellipses at different tilts is fatter than
                # either -- by a(1-a)(dtau)^2 var_z. Measured at shear 20 in the body
                # of the dipole, mid-interval sigma_xi comes out 8x-10x too large
                # (20.3 um -> 196.9 um at tau: -9.92 -> -6.58), and the wake then
                # fails to converge under step refinement. The widths do not need the
                # moment chart -- only the orientation does -- so take just that.
                s_xi = math.exp(b * math.log(sigma_xi_arr[k]) + a * math.log(sigma_xi_arr[k + 1]))
                s_z = math.exp(b * math.log(sigma_z_arr[k]) + a * math.log(sigma_z_arr[k + 1]))

        # --- map the query point into the shared normalized frame ---
        xi = xval[i] - p_a
        u = (xi - xi_bar) / s_xi
        w = (z - z_bar) / s_z
        # curvature shear: the coefficients are blended linearly like the centroids;
        # unit Jacobian, so densities need no rescaling
        c2 = b * curv_c2[k] + a * curv_c2[k + 1]
        c3 = b * curv_c3[k] + a * curv_c3[k + 1]
        g_w, dg_w = curv_g(c2, c3, w, curv_wl)
        u = u - g_w

        # --- time derivatives of the blend (the by-parts coefficients A, B) ---
        q = tval[i]
        if is_uniform:
            dt_k = delta_t
            t_lo = min_t
            t_hi = min_t + (n_t - 1) * delta_t
        else:
            dt_k = t_arr[k + 1] - t_arr[k]
            t_lo = t_arr[0]
            t_hi = t_arr[n_t - 1]
        dadt = 0.0
        if q > t_lo and q < t_hi and dt_k > 0.0:
            dadt = 1.0 / dt_k
        if blend_code == 0:
            p_t = (eval_poly(poly_coeffs[k + 1], z) - eval_poly(poly_coeffs[k], z)) * dadt
            xib_t = (xi_bar_arr[k + 1] - xi_bar_arr[k]) * dadt
            sxi_t = s_xi * (math.log(sigma_xi_arr[k + 1]) - math.log(sigma_xi_arr[k])) * dadt
            sz_t = s_z * (math.log(sigma_z_arr[k + 1]) - math.log(sigma_z_arr[k])) * dadt
        else:
            vz_d = var_z_arr[k + 1] - var_z_arr[k]
            cv_d = cov_arr[k + 1] - cov_arr[k]
            tau_t = (cv_d * var_z - cov * vz_d) / (var_z * var_z) * dadt
            zb_d = (z_bar_arr[k + 1] - z_bar_arr[k]) * dadt
            p_t = tau_t * (z - z_bar) - tau * zb_d + (x_bar_arr[k + 1] - x_bar_arr[k]) * dadt
            xib_t = 0.0
            if blend_code == 1:
                sz_t = 0.5 * vz_d * dadt / s_z
                vxi_t = ((var_x_arr[k + 1] - var_x_arr[k]) - 2.0 * cov * cv_d / var_z
                         + cov * cov * vz_d / (var_z * var_z)) * dadt
                sxi_t = 0.5 * vxi_t / s_xi if s_xi > 0.0 else 0.0
            else:
                sxi_t = s_xi * (math.log(sigma_xi_arr[k + 1]) - math.log(sigma_xi_arr[k])) * dadt
                sz_t = s_z * (math.log(sigma_z_arr[k + 1]) - math.log(sigma_z_arr[k])) * dadt
        zb_t = (z_bar_arr[k + 1] - z_bar_arr[k]) * dadt
        # curvature: d g / d c2, d c3 at fixed w, with curv_g's linear continuation
        wl = curv_wl
        if w > wl:
            gc2 = (wl * wl - 1.0) + 2.0 * wl * (w - wl)
            gc3 = wl * wl * wl + 3.0 * wl * wl * (w - wl)
        elif w < -wl:
            gc2 = (wl * wl - 1.0) - 2.0 * wl * (w + wl)
            gc3 = -wl * wl * wl + 3.0 * wl * wl * (w + wl)
        else:
            gc2 = w * w - 1.0
            gc3 = w * w * w
        g_t = (gc2 * (curv_c2[k + 1] - curv_c2[k]) + gc3 * (curv_c3[k + 1] - curv_c3[k])) * dadt
        w_z = 1.0 / s_z
        w_t = -zb_t / s_z - w * sz_t / s_z
        u_z = -dp_a / s_xi - dg_w * w_z
        # (u + g_w) is the u before the curvature shear, (x - p - xi_bar) / s_xi
        u_t = -(p_t + xib_t) / s_xi - (u + g_w) * sxi_t / s_xi - dg_w * w_t - g_t
        A_out[i] = -s_xi * (u_z - u_t)
        if not want_fields:
            continue

        u_cell = (u - u_start) / delta_u - 0.5
        w_cell = (w - w_start) / delta_w - 0.5

        # --- evaluate both snapshots at the SAME (u, w), then blend ---
        #
        # This used to be ten bspline_eval_single calls (5 fields x snapshots k, k+1). All
        # ten share the same (u_cell, w_cell), hence the same 4x4 stencil and the same eight
        # basis weights -- but cubic_bspline_w(w_cell - j) sat INSIDE the dj loop, so every
        # call recomputed the four w-weights sixteen times, and LLVM cannot common that up
        # across ten separate calls over ten different arrays. It came to ~200 weight
        # evaluations per query where only 8 distinct values exist. Profiling put this
        # kernel at 67% of wake-mesh time, so this was the whole hot spot.
        #
        # Weights are now computed once and all ten accumulators filled in one pass.
        #
        # BIT-IDENTICAL, deliberately: each accumulator receives the same terms in the same
        # order as before, and the multiply keeps the original left-to-right association
        # (d * wi) * ww -- the algebraically equal d * (wi * ww) would change the last bits.
        # cubic_bspline_w is pure, so hoisting it cannot change a value.
        i0 = int(math.floor(u_cell)) - 1
        j0 = int(math.floor(w_cell)) - 1

        r0 = 0.0
        r1 = 0.0
        ru0 = 0.0
        ru1 = 0.0
        rw0 = 0.0
        rw1 = 0.0
        v0 = 0.0
        v1 = 0.0
        vu0 = 0.0
        vu1 = 0.0

        # the same whole-stencil-misses-the-grid early out each of the ten calls performed
        if not (i0 + 3 < 0 or i0 >= n_u or j0 + 3 < 0 or j0 >= n_w):
            for di in range(4):
                ii = i0 + di
                if ii < 0 or ii >= n_u:
                    continue
                wi = cubic_bspline_w(u_cell - ii)
                if wi == 0.0:
                    continue
                for dj in range(4):
                    jj = j0 + dj
                    if jj < 0 or jj >= n_w:
                        continue
                    ww = cubic_bspline_w(w_cell - jj)
                    r0 += (data_rho[k, ii, jj] * wi) * ww
                    r1 += (data_rho[k + 1, ii, jj] * wi) * ww
                    ru0 += (data_rho_u[k, ii, jj] * wi) * ww
                    ru1 += (data_rho_u[k + 1, ii, jj] * wi) * ww
                    rw0 += (data_rho_w[k, ii, jj] * wi) * ww
                    rw1 += (data_rho_w[k + 1, ii, jj] * wi) * ww
                    v0 += (data_vx[k, ii, jj] * wi) * ww
                    v1 += (data_vx[k + 1, ii, jj] * wi) * ww
                    vu0 += (data_vx_u[k, ii, jj] * wi) * ww
                    vu1 += (data_vx_u[k + 1, ii, jj] * wi) * ww

        rho_h = b * r0 + a * r1
        rho_hu = b * ru0 + a * ru1
        rho_hw = b * rw0 + a * rw1
        vx_h = b * v0 + a * v1
        vx_hu = b * vu0 + a * vu1

        # --- back to physical units ---
        # the shear xi = x - p(z) has unit Jacobian, so the area element is
        # only the sigma scaling: dx dz = s_xi * s_z du dw
        jac = s_xi * s_z
        rho[i] = rho_h / jac
        rho_x[i] = rho_hu / (s_xi * jac)
        # d rho/dz|_x = d rho/dz|_xi - p'(z) d rho/dxi, with the BLENDED p', and the
        # curvature shear adds d u'/dz = -g'(w)/s_z
        rho_z[i] = (rho_hw / s_z - (dp_a / s_xi + dg_w / s_z) * rho_hu) / jac
        vx[i] = vx_h
        vx_x[i] = vx_hu / s_xi
        F_t = (r1 - r0) * dadt
        B_out[i] = (rho_hw * (w_z - w_t) - F_t) / jac + rho[i] * (sxi_t / s_xi + sz_t / s_z)

    return rho, rho_x, rho_z, vx, vx_x, A_out, B_out

# serial and thread-parallel compilations of the same source; prange is a plain range in the
# serial one, so it is the historical kernel bit for bit. CSR2D picks the parallel one only for
# large point sets when CSR_integration.threads > 1 (the fork/join cost dominates small calls).
interpolate3D_comoving_ibp = jit(nopython=True, cache=True)(_interpolate3D_comoving_ibp_impl)

