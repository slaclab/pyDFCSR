import math

import numpy as np
from numba import jit
from numba.experimental import jitclass
from numba import double
spec = [
    ('min_x', double),
    ('min_y', double), # a simple scalar field
    ('min_z', double),
    ('max_x', double),
    ('max_y', double),
    ('max_z', double),
    ('delta_x', double),
    ('delta_y', double),
    ('delta_z', double),
    ('data', double[:, :, :]),          # an array field
]

@jit(nopython = True,  cache = True)
def interpolate3D(xval, yval, zval, data, min_x, min_y, min_z,  delta_x, delta_y, delta_z):
    result = np.zeros(len(xval))
    x_size, y_size, z_size = data.shape[0], data.shape[1], data.shape[2]
    xval = (xval - min_x) / delta_x
    yval = (yval - min_y) / delta_y
    zval = (zval - min_z) / delta_z
    for i in range(len(xval)):
        x = xval[i]
        y = yval[i]
        z = zval[i]

        x0 = int(x)
        if x0 == x_size - 1:
            x1 = x0
        else:
            x1 = x0 + 1

        y0 = int(y)
        if y0 == y_size - 1:
            y1 = y0
        else:
            y1 = y0 + 1

        z0 = int(z)
        if z0 == z_size - 1:
            z1 = z0
        else:
            z1 = z0 + 1

        xd = x - x0
        yd = y - y0
        zd = z - z0

        if x0 >= 0 and y0 >= 0 and z0 >= 0 and x1 < x_size and y1 < y_size and z1 < z_size:
            c00 = data[x0, y0, z0] * (1 - xd) + data[x1, y0, z0] * xd
            c01 = data[x0, y0, z1] * (1 - xd) + data[x1, y0, z1] * xd
            c10 = data[x0, y1, z0] * (1 - xd) + data[x1, y1, z0] * xd
            c11 = data[x0, y1, z1] * (1 - xd) + data[x1, y1, z1] * xd

            c0 = c00 * (1 - yd) + c10 * yd
            c1 = c01 * (1 - yd) + c11 * yd

            result[i] = c0 * (1 - zd) + c1 * zd

        else:
            result[i] = 0.0

    return result

@jit(nopython = True,  cache = True)
def interpolate_3d_vectorized(data, x, y, z, min_x, min_y, min_z,  delta_x, delta_y, delta_z):
    """
    Perform linear interpolation for multiple points (x, y, z) within a 3D space defined by 'data'.
    Extrapolated values outside the dataset boundaries are set to zero.

    Parameters:
        data (numpy.ndarray): The 3D numpy array containing data values.
        x (numpy.ndarray): The x-coordinates of the interpolation points.
        y (numpy.ndarray): The y-coordinates of the interpolation points.
        z (numpy.ndarray): The z-coordinates of the interpolation points.

    Returns:
        numpy.ndarray: The interpolated values or zero if the points are outside the data boundaries.
    """
    #nx, ny, nz = data.shape
    #interpolated_values = np.zeros(x.shape)

    nx, ny, nz = data.shape
    output = np.zeros_like(x)

    # Clamp coordinates to valid index ranges
    x = np.clip(x, 0, nx - 1)
    y = np.clip(y, 0, ny - 1)
    z = np.clip(z, 0, nz - 1)

    # Convert floating indices to integer indices
    ix = np.floor(x).astype(np.int32)
    iy = np.floor(y).astype(np.int32)
    iz = np.floor(z).astype(np.int32)

    # Calculate linear indices for the corners of the interpolation cube
    i000 = ix + iy * nx + iz * nx * ny
    i001 = ix + iy * nx + (iz + 1) * nx * ny
    i010 = ix + (iy + 1) * nx + iz * nx * ny
    i011 = ix + (iy + 1) * nx + (iz + 1) * nx * ny
    i100 = (ix + 1) + iy * nx + iz * nx * ny
    i101 = (ix + 1) + iy * nx + (iz + 1) * nx * ny
    i110 = (ix + 1) + (iy + 1) * nx + iz * nx * ny
    i111 = (ix + 1) + (iy + 1) * nx + (iz + 1) * nx * ny

    # Flatten the data to use linear indexing
    data_flat = data.ravel()

    # Retrieve values using linear indices
    v000 = data_flat[i000]
    v001 = data_flat[i001]
    v010 = data_flat[i010]
    v011 = data_flat[i011]
    v100 = data_flat[i100]
    v101 = data_flat[i101]
    v110 = data_flat[i110]
    v111 = data_flat[i111]

    # Fractional parts for interpolation
    xd = x - np.floor(x)
    yd = y - np.floor(y)
    zd = z - np.floor(z)

    # Interpolate along z-axis
    c00 = v000 * (1 - zd) + v001 * zd
    c10 = v010 * (1 - zd) + v011 * zd
    c01 = v100 * (1 - zd) + v101 * zd
    c11 = v110 * (1 - zd) + v111 * zd

    # Interpolate along y-axis
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    # Final interpolation along x-axis
    output = c0 * (1 - xd) + c1 * xd

    return output



@jit(nopython=True, cache=True)
def eval_poly(coeffs, z):
    """Evaluate polynomial using Horner's method. coeffs in numpy polyfit order (highest degree first)."""
    result = coeffs[0]
    for j in range(1, len(coeffs)):
        result = result * z + coeffs[j]
    return result


@jit(nopython=True, cache=True)
def bilinear_single(data2d, xi_idx, z_idx, n_xi, n_z):
    """Bilinear interpolation on a single 2D slice. Returns 0 if out of bounds."""
    x0 = int(xi_idx)
    y0 = int(z_idx)

    if x0 < 0 or y0 < 0 or x0 >= n_xi - 1 or y0 >= n_z - 1:
        return 0.0

    x1 = x0 + 1
    y1 = y0 + 1
    xd = xi_idx - x0
    yd = z_idx - y0

    c00 = data2d[x0, y0]
    c10 = data2d[x1, y0]
    c01 = data2d[x0, y1]
    c11 = data2d[x1, y1]

    c0 = c00 * (1 - xd) + c10 * xd
    c1 = c01 * (1 - xd) + c11 * xd

    return c0 * (1 - yd) + c1 * yd


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


@jit(nopython=True, cache=True)
def get_poly_deriv_blended(zval, tval, poly_coeffs, min_t, delta_t):
    """
    Compute the time-blended polynomial derivative poly'(z) at each query point.
    Uses the same time-blending logic as interpolate3D_transformed.

    Returns poly'(z) for each query point, blended between bracketing timesteps.
    This is needed for the chain rule: d_rho/dz_lab = d_rho/dz_xi - d_rho/dxi * poly'(z)
    """
    result = np.zeros(len(zval))
    n_t = poly_coeffs.shape[0]

    for i in range(len(zval)):
        t_idx = (tval[i] - min_t) / delta_t
        k = int(t_idx)

        if k < 0:
            k = 0
        if k >= n_t - 1:
            k = n_t - 2

        alpha = t_idx - k
        if alpha < 0.0:
            alpha = 0.0
        if alpha > 1.0:
            alpha = 1.0

        dp_k = eval_poly_deriv(poly_coeffs[k], zval[i])
        dp_k1 = eval_poly_deriv(poly_coeffs[k + 1], zval[i])

        result[i] = (1.0 - alpha) * dp_k + alpha * dp_k1

    return result


@jit(nopython=True, cache=True)
def interpolate3D_transformed(xval, zval, tval, data, poly_coeffs,
                               min_xi_arr, min_z_arr, min_t,
                               delta_xi_arr, delta_z_arr, delta_t):
    """
    Interpolation with per-timestep polynomial coordinate transforms and
    per-timestep grid extents.

    For each query point (x, z, t):
    1. Find bracketing timesteps k, k+1
    2. Transform x -> xi using poly_coeffs[k] and poly_coeffs[k+1]
    3. Bilinear interp at each timestep using that timestep's grid metadata
    4. Linear blend in t

    Parameters:
        xval: x coordinates of query points (physical frame)
        zval: z coordinates of query points
        tval: t coordinates of query points
        data: 3D array (n_t, n_xi, n_z) — density in transformed frame per timestep
        poly_coeffs: 2D array (n_t, poly_degree+1) — polynomial coefficients per timestep
        min_xi_arr: 1D array (n_t,) — per-timestep xi grid origin
        min_z_arr: 1D array (n_t,) — per-timestep z grid origin
        min_t: scalar — time grid origin
        delta_xi_arr: 1D array (n_t,) — per-timestep xi grid spacing
        delta_z_arr: 1D array (n_t,) — per-timestep z grid spacing
        delta_t: scalar — time grid spacing
    """
    result = np.zeros(len(xval))
    n_t = data.shape[0]
    n_xi = data.shape[1]
    n_z = data.shape[2]

    for i in range(len(xval)):
        # Find t index
        t_idx = (tval[i] - min_t) / delta_t
        k = int(t_idx)

        if k < 0:
            k = 0
        if k >= n_t - 1:
            k = n_t - 2

        alpha = t_idx - k
        if alpha < 0.0:
            alpha = 0.0
        if alpha > 1.0:
            alpha = 1.0

        # Transform x to xi in timestep k's frame
        poly_k = poly_coeffs[k]
        xi_k = xval[i] - eval_poly(poly_k, zval[i])

        # Transform x to xi in timestep k+1's frame
        poly_k1 = poly_coeffs[k + 1]
        xi_k1 = xval[i] - eval_poly(poly_k1, zval[i])

        # Bilinear interp at timestep k (using k's grid)
        xi_idx_k = (xi_k - min_xi_arr[k]) / delta_xi_arr[k]
        z_idx_k = (zval[i] - min_z_arr[k]) / delta_z_arr[k]
        val_k = bilinear_single(data[k], xi_idx_k, z_idx_k, n_xi, n_z)

        # Bilinear interp at timestep k+1 (using k+1's grid)
        xi_idx_k1 = (xi_k1 - min_xi_arr[k + 1]) / delta_xi_arr[k + 1]
        z_idx_k1 = (zval[i] - min_z_arr[k + 1]) / delta_z_arr[k + 1]
        val_k1 = bilinear_single(data[k + 1], xi_idx_k1, z_idx_k1, n_xi, n_z)

        # Blend in time
        result[i] = (1.0 - alpha) * val_k + alpha * val_k1

    return result


@jit(nopython=True, cache=True)
def interpolate3D_transformed_with_derivs(xval, zval, tval, data, poly_coeffs,
                                           min_xi_arr, min_z_arr, min_t,
                                           delta_xi_arr, delta_z_arr, delta_t):
    """
    Interpolate density AND compute spatial derivatives on-the-fly via
    finite differences on the stored density grid.

    Returns (rho, drho_dx, drho_dz) where:
      - drho_dx = ∂ρ/∂x = ∂ρ/∂ξ (since ∂ξ/∂x = 1)
      - drho_dz = ∂ρ/∂z|_x (lab frame, chain rule applied)
    """
    n_pts = len(xval)
    rho = np.zeros(n_pts)
    drho_dx = np.zeros(n_pts)
    drho_dz = np.zeros(n_pts)

    n_t = data.shape[0]
    n_xi = data.shape[1]
    n_z = data.shape[2]

    for i in range(n_pts):
        t_idx = (tval[i] - min_t) / delta_t
        k = int(t_idx)
        if k < 0:
            k = 0
        if k >= n_t - 1:
            k = n_t - 2
        alpha = t_idx - k
        if alpha < 0.0:
            alpha = 0.0
        if alpha > 1.0:
            alpha = 1.0

        val_blend = 0.0
        dxi_blend = 0.0
        dz_blend = 0.0

        for kk_idx in range(2):
            kk = k + kk_idx
            w = alpha if kk_idx == 1 else (1.0 - alpha)

            poly_kk = poly_coeffs[kk]
            xi = xval[i] - eval_poly(poly_kk, zval[i])
            xi_idx = (xi - min_xi_arr[kk]) / delta_xi_arr[kk]
            z_idx = (zval[i] - min_z_arr[kk]) / delta_z_arr[kk]

            val = bilinear_single(data[kk], xi_idx, z_idx, n_xi, n_z)

            val_xp = bilinear_single(data[kk], xi_idx + 1.0, z_idx, n_xi, n_z)
            val_xm = bilinear_single(data[kk], xi_idx - 1.0, z_idx, n_xi, n_z)
            dval_dxi = (val_xp - val_xm) / (2.0 * delta_xi_arr[kk])

            val_zp = bilinear_single(data[kk], xi_idx, z_idx + 1.0, n_xi, n_z)
            val_zm = bilinear_single(data[kk], xi_idx, z_idx - 1.0, n_xi, n_z)
            dval_dz_xi = (val_zp - val_zm) / (2.0 * delta_z_arr[kk])

            poly_prime = eval_poly_deriv(poly_kk, zval[i])
            dval_dz_lab = dval_dz_xi - poly_prime * dval_dxi

            val_blend += w * val
            dxi_blend += w * dval_dxi
            dz_blend += w * dval_dz_lab

        rho[i] = val_blend
        drho_dx[i] = dxi_blend
        drho_dz[i] = dz_blend

    return rho, drho_dx, drho_dz


@jitclass(spec)
class TrilinearInterpolator:
    def __init__(self, data, x, y, z):
        self.data = data
        self.min_x, self.max_x = x[0], x[-1]
        self.min_y, self.max_y = y[0], y[-1]
        self.min_z, self.max_z = z[0], z[-1]
        self.delta_x = (self.max_x - self.min_x) / (x.shape[0] - 1)
        self.delta_y = (self.max_y - self.min_y) / (y.shape[0] - 1)
        self.delta_z = (self.max_z - self.min_z) / (z.shape[0] - 1)

    def interp(self, xval, yval, zval):
        return interpolate3D(xval, yval, zval, self.data,
                           self.min_x, self.min_y, self.min_z,
                           self.delta_x, self.delta_y, self.delta_z)



class TrilinearInterpolator_vec:
    def __init__(self, data, x, y, z):
        self.data = data
        self.min_x, self.max_x = x[0], x[-1]
        self.min_y, self.max_y = y[0], y[-1]
        self.min_z, self.max_z = z[0], z[-1]
        self.delta_x = (self.max_x - self.min_x) / (x.shape[0] - 1)
        self.delta_y = (self.max_y - self.min_y) / (y.shape[0] - 1)
        self.delta_z = (self.max_z - self.min_z) / (z.shape[0] - 1)

    def interp(self, xval, yval, zval):
        return interpolate3D_vec(xval, yval, zval, self.data,
                           self.min_x, self.min_y, self.min_z,
                           self.delta_x, self.delta_y, self.delta_z)

# ---------------------------------------------------------------------------
# Co-moving (Lagrangian) interpolation of the density history.
#
# The lab-frame blend used by interpolate3D_transformed evaluates each snapshot
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
# physical cell is still delta_xi_k = 2*xlim*sigma_xi_k/nbins, exactly as
# 'bspline_fft' does today, so no transverse resolution is given up.
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


@jit(nopython=True, cache=True)
def interpolate3D_comoving_fields(xval, zval, tval,
                                  data_rho, data_rho_u, data_rho_w,
                                  data_vx, data_vx_u,
                                  poly_coeffs, xi_bar_arr, sigma_xi_arr,
                                  z_bar_arr, sigma_z_arr,
                                  u_start, delta_u, w_start, delta_w,
                                  min_t, delta_t):
    """
    Interpolate the density history in a co-moving affine frame.

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

    Frame blending: the polynomial and the means are linear in alpha; the sigmas
    are blended log-linearly so they stay positive and so that a beam growing
    exponentially (which is what a drift does to a diverging beam) is tracked
    smoothly rather than with a kink.
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
        t_idx = (tval[i] - min_t) / delta_t
        k = int(math.floor(t_idx))
        if k < 0:
            k = 0
        if k >= n_t - 1:
            k = n_t - 2
        a = t_idx - k
        if a < 0.0:
            a = 0.0
        if a > 1.0:
            a = 1.0
        b = 1.0 - a

        z = zval[i]

        # --- blend the frame, not the field ---
        # eval_poly is linear in the coefficients, so blending its value is the
        # same as blending the coefficients and evaluating.
        p_a = b * eval_poly(poly_coeffs[k], z) + a * eval_poly(poly_coeffs[k + 1], z)
        dp_a = b * eval_poly_deriv(poly_coeffs[k], z) + a * eval_poly_deriv(poly_coeffs[k + 1], z)

        xi_bar = b * xi_bar_arr[k] + a * xi_bar_arr[k + 1]
        z_bar = b * z_bar_arr[k] + a * z_bar_arr[k + 1]
        s_xi = math.exp(b * math.log(sigma_xi_arr[k]) + a * math.log(sigma_xi_arr[k + 1]))
        s_z = math.exp(b * math.log(sigma_z_arr[k]) + a * math.log(sigma_z_arr[k + 1]))

        # --- map the query point into the shared normalized frame ---
        xi = xval[i] - p_a
        u = (xi - xi_bar) / s_xi
        w = (z - z_bar) / s_z

        u_cell = (u - u_start) / delta_u - 0.5
        w_cell = (w - w_start) / delta_w - 0.5

        # --- evaluate both snapshots at the SAME (u, w), then blend ---
        rho_h = (b * bspline_eval_single(data_rho[k], u_cell, w_cell, n_u, n_w)
                 + a * bspline_eval_single(data_rho[k + 1], u_cell, w_cell, n_u, n_w))
        rho_hu = (b * bspline_eval_single(data_rho_u[k], u_cell, w_cell, n_u, n_w)
                  + a * bspline_eval_single(data_rho_u[k + 1], u_cell, w_cell, n_u, n_w))
        rho_hw = (b * bspline_eval_single(data_rho_w[k], u_cell, w_cell, n_u, n_w)
                  + a * bspline_eval_single(data_rho_w[k + 1], u_cell, w_cell, n_u, n_w))
        vx_h = (b * bspline_eval_single(data_vx[k], u_cell, w_cell, n_u, n_w)
                + a * bspline_eval_single(data_vx[k + 1], u_cell, w_cell, n_u, n_w))
        vx_hu = (b * bspline_eval_single(data_vx_u[k], u_cell, w_cell, n_u, n_w)
                 + a * bspline_eval_single(data_vx_u[k + 1], u_cell, w_cell, n_u, n_w))

        # --- back to physical units ---
        # the shear xi = x - p(z) has unit Jacobian, so the area element is
        # only the sigma scaling: dx dz = s_xi * s_z du dw
        jac = s_xi * s_z
        rho[i] = rho_h / jac
        rho_x[i] = rho_hu / (s_xi * jac)
        # d rho/dz|_x = d rho/dz|_xi - p'(z) d rho/dxi, with the BLENDED p'
        rho_z[i] = (rho_hw / s_z - dp_a * rho_hu / s_xi) / jac
        vx[i] = vx_h
        vx_x[i] = vx_hu / s_xi

    return rho, rho_x, rho_z, vx, vx_x


@jit(nopython=True, cache=True)
def interpolate3D_comoving(xval, zval, tval, data,
                           poly_coeffs, xi_bar_arr, sigma_xi_arr,
                           z_bar_arr, sigma_z_arr,
                           u_start, delta_u, w_start, delta_w,
                           min_t, delta_t):
    """
    Co-moving interpolation of a single normalized field, WITHOUT any Jacobian
    scaling. Used by the acceptance test (test_ghosting.py) to compare the frame
    interpolation on its own; production code should call
    interpolate3D_comoving_fields, which returns physical quantities.
    """
    n = len(xval)
    out = np.zeros(n)
    n_t = data.shape[0]
    n_u = data.shape[1]
    n_w = data.shape[2]

    for i in range(n):
        t_idx = (tval[i] - min_t) / delta_t
        k = int(math.floor(t_idx))
        if k < 0:
            k = 0
        if k >= n_t - 1:
            k = n_t - 2
        a = t_idx - k
        if a < 0.0:
            a = 0.0
        if a > 1.0:
            a = 1.0
        b = 1.0 - a

        z = zval[i]
        p_a = b * eval_poly(poly_coeffs[k], z) + a * eval_poly(poly_coeffs[k + 1], z)
        xi_bar = b * xi_bar_arr[k] + a * xi_bar_arr[k + 1]
        z_bar = b * z_bar_arr[k] + a * z_bar_arr[k + 1]
        s_xi = math.exp(b * math.log(sigma_xi_arr[k]) + a * math.log(sigma_xi_arr[k + 1]))
        s_z = math.exp(b * math.log(sigma_z_arr[k]) + a * math.log(sigma_z_arr[k + 1]))

        u_cell = ((xval[i] - p_a - xi_bar) / s_xi - u_start) / delta_u - 0.5
        w_cell = ((z - z_bar) / s_z - w_start) / delta_w - 0.5

        out[i] = (b * bspline_eval_single(data[k], u_cell, w_cell, n_u, n_w)
                  + a * bspline_eval_single(data[k + 1], u_cell, w_cell, n_u, n_w))
    return out
