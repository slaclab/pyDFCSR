from numba import jit
import math
import numpy as np
from collections import deque


@jit(nopython=True)
def cubic_bspline(x):
    """Evaluate the cubic B-spline basis function B3(x) with support [-2, 2]."""
    ax = abs(x)
    if ax >= 2.0:
        return 0.0
    elif ax >= 1.0:
        return (2.0 - ax) ** 3 / 6.0
    else:
        return (4.0 - 6.0 * ax * ax + 3.0 * ax * ax * ax) / 6.0


@jit(nopython=True)
def histogram_bspline_2d(q1, q2, w, nbins_1, bins_start_1, bins_end_1,
                          nbins_2, bins_start_2, bins_end_2):
    """
    2D histogram using cubic B-spline weighting.
    Each particle contributes to 4x4 = 16 nearby bins.

    Parameters:
        q1, q2: particle coordinates (1D arrays)
        w: weights (1D array)
        nbins_1, nbins_2: number of bins in each dimension
        bins_start_1, bins_end_1: range in dimension 1
        bins_start_2, bins_end_2: range in dimension 2

    Returns:
        2D array of shape (nbins_1, nbins_2)
    """
    bin_spacing_1 = (bins_end_1 - bins_start_1) / nbins_1
    bin_spacing_2 = (bins_end_2 - bins_start_2) / nbins_2
    inv_spacing_1 = 1.0 / bin_spacing_1
    inv_spacing_2 = 1.0 / bin_spacing_2
    n_ptcl = len(w)

    hist_data = np.zeros((nbins_1, nbins_2), dtype=np.float64)

    for i in range(n_ptcl):
        # Position in grid units (centered on bin centers)
        q1_cell = (q1[i] - bins_start_1) * inv_spacing_1 - 0.5
        q2_cell = (q2[i] - bins_start_2) * inv_spacing_2 - 0.5

        # Cubic B-spline has support [-2, 2], so affects 4 bins
        i1_start = int(math.floor(q1_cell)) - 1
        i2_start = int(math.floor(q2_cell)) - 1

        for di in range(4):
            i1 = i1_start + di
            if i1 < 0 or i1 >= nbins_1:
                continue
            w1 = cubic_bspline(q1_cell - i1)

            for dj in range(4):
                i2 = i2_start + dj
                if i2 < 0 or i2 >= nbins_2:
                    continue
                w2 = cubic_bspline(q2_cell - i2)

                hist_data[i1, i2] += w[i] * w1 * w2

    return hist_data


def smooth_and_differentiate(field, dx, dz, sigma_smooth):
    """
    Apply FFT Gaussian smoothing and compute spectral derivatives.
    Uses zero-padding to avoid Gibbs ringing from periodicity assumption.

    Parameters:
        field: 2D array (nxi, nz)
        dx: grid spacing in xi direction
        dz: grid spacing in z direction
        sigma_smooth: smoothing kernel width in grid spacings

    Returns:
        field_smooth: smoothed field
        dfield_dx: derivative w.r.t. xi (spectral)
        dfield_dz: derivative w.r.t. z (spectral)
    """
    nxi, nz = field.shape

    # Zero-pad to avoid periodicity artifacts (Gibbs ringing)
    pad_xi = nxi // 2
    pad_z = nz // 2
    field_padded = np.pad(field, ((pad_xi, pad_xi), (pad_z, pad_z)), mode='constant', constant_values=0)
    nxi_p, nz_p = field_padded.shape

    # FFT frequencies for padded array
    kxi = np.fft.fftfreq(nxi_p, d=dx) * 2 * np.pi
    kz = np.fft.fftfreq(nz_p, d=dz) * 2 * np.pi
    KXI, KZ = np.meshgrid(kxi, kz, indexing='ij')

    # Gaussian smoothing kernel in Fourier space
    sigma_x_phys = sigma_smooth * dx
    sigma_z_phys = sigma_smooth * dz
    gaussian_filter = np.exp(-0.5 * (KXI ** 2 * sigma_x_phys ** 2 + KZ ** 2 * sigma_z_phys ** 2))

    # FFT of padded field
    field_fft = np.fft.fft2(field_padded)

    # Smoothed field (crop back to original size)
    field_smooth_padded = np.real(np.fft.ifft2(field_fft * gaussian_filter))
    field_smooth = field_smooth_padded[pad_xi:pad_xi + nxi, pad_z:pad_z + nz]

    # Spectral derivatives (crop back)
    dfield_dx_padded = np.real(np.fft.ifft2(field_fft * gaussian_filter * (1j * KXI)))
    dfield_dx = dfield_dx_padded[pad_xi:pad_xi + nxi, pad_z:pad_z + nz]

    dfield_dz_padded = np.real(np.fft.ifft2(field_fft * gaussian_filter * (1j * KZ)))
    dfield_dz = dfield_dz_padded[pad_xi:pad_xi + nxi, pad_z:pad_z + nz]

    return field_smooth, dfield_dx, dfield_dz


class DF_tracker_smooth:
    """
    Improved density function tracker using cubic B-spline deposition,
    polynomial tilt removal, FFT Gaussian smoothing, and spectral derivatives.

    Drop-in replacement for DF_tracker with same external interface.
    """

    def __init__(self, input_dic={}):
        self.configure_params(**input_dic)

        # Current DF state
        self.sigma_xi = None
        self.sigma_z = None
        self.density = None
        self.density_x = None
        self.density_z = None
        self.vx = None
        self.vx_x = None
        self.xi_grids = None
        self.z_grids = None
        self.poly_coeffs = None
        self.start_time = 0.0
        self.t = 0.0

        # DF history log
        self.slope_log = deque([])
        self.DF_log = deque([])
        self.sigma_xi_log = deque([])
        self.sigma_z_log = deque([])
        self.time_log = deque([])
        self.poly_coeffs_log = deque([])

        # Interpolant state (built from DF_log directly)
        self.end_time = 0.0

    def configure_params(self, method='bspline_fft', xbins=128, zbins=128,
                         xlim=5, zlim=5, smoothing_sigma=3.0, poly_degree=3,
                         velocity_threhold=5, upper_limit=None,
                         filter_order=0, filter_window=0):
        self.method = method
        self.xbins = xbins
        self.zbins = zbins
        self.xlim = xlim
        self.zlim = zlim
        self.smoothing_sigma = smoothing_sigma
        self.poly_degree = poly_degree
        self.velocity_threhold = velocity_threhold
        self.upper_limit = upper_limit

    def get_DF(self, x, z, px, t):
        """
        Compute density functions and derivatives using B-spline + FFT pipeline.

        Steps:
        1. Polynomial tilt removal
        2. Cubic B-spline deposition in (xi, z)
        3. FFT Gaussian smoothing + spectral derivatives
        """
        # Step 1: Polynomial tilt removal
        poly_coeffs = np.polyfit(z, x, deg=self.poly_degree)
        xi = x - np.polyval(poly_coeffs, z)

        sigma_xi = np.std(xi)
        sigma_z = np.std(z)
        self.sigma_xi = sigma_xi
        self.sigma_z = sigma_z
        self.xmean = np.mean(xi)
        self.zmean = np.mean(z)

        # Step 2: Define grid in (xi, z) space
        xi_start = self.xmean - self.xlim * sigma_xi
        xi_end = self.xmean + self.xlim * sigma_xi
        z_start = self.zmean - self.zlim * sigma_z
        z_end = self.zmean + self.zlim * sigma_z

        xi_grids = np.linspace(xi_start, xi_end, self.xbins)
        z_grids = np.linspace(z_start, z_end, self.zbins)
        dx = xi_grids[1] - xi_grids[0]
        dz = z_grids[1] - z_grids[0]

        # Step 3: Cubic B-spline deposition
        density = histogram_bspline_2d(
            q1=xi, q2=z, w=np.ones(xi.shape),
            nbins_1=self.xbins, bins_start_1=xi_start, bins_end_1=xi_end,
            nbins_2=self.zbins, bins_start_2=z_start, bins_end_2=z_end
        )

        vx_weighted = histogram_bspline_2d(
            q1=xi, q2=z, w=px,
            nbins_1=self.xbins, bins_start_1=xi_start, bins_end_1=xi_end,
            nbins_2=self.zbins, bins_start_2=z_start, bins_end_2=z_end
        )

        # Normalize density
        dsum = np.trapz(np.trapz(density, xi_grids, axis=0), z_grids)
        if dsum > 0:
            density /= dsum
            vx_weighted /= dsum

        # Step 4: FFT smoothing + spectral derivatives
        density_smooth, density_dxi, density_dz = smooth_and_differentiate(
            density, dx, dz, self.smoothing_sigma
        )

        # Smooth ρ·vx with the same kernel, then divide by smooth(ρ)
        rho_vx_smooth, _, _ = smooth_and_differentiate(
            vx_weighted, dx, dz, self.smoothing_sigma
        )
        epsilon = np.max(density_smooth) / self.velocity_threhold
        vx_smooth = rho_vx_smooth / (density_smooth + epsilon)

        # Differentiate vx with full smoothing sigma to suppress boundary noise
        _, vx_dxi, _ = smooth_and_differentiate(
            vx_smooth, dx, dz, self.smoothing_sigma
        )
        # Taper ∂vx/∂ξ to zero where density is negligible
        density_mask = (density_smooth / (density_smooth + epsilon)) ** 3
        vx_dxi *= density_mask

        # Chain rule correction for ∂ρ/∂z:
        # ξ = x - p(z), so ∂ξ/∂z|_x = -p'(z)
        # ∂ρ/∂z|_x = ∂ρ/∂z|_ξ + ∂ρ/∂ξ · ∂ξ/∂z|_x = ∂ρ/∂z|_ξ - p'(z) · ∂ρ/∂ξ
        poly_deriv = np.polyder(poly_coeffs)
        p_prime_z = np.polyval(poly_deriv, z_grids)  # p'(z) at each z grid point
        density_dz_phys = density_dz - p_prime_z[np.newaxis, :] * density_dxi

        # Store results
        self.xi_grids = xi_grids
        self.z_grids = z_grids
        self.x_grids = xi_grids  # alias for interface compatibility
        self.density = density_smooth
        self.density_x = density_dxi
        self.density_z = density_dz_phys
        self.vx = vx_smooth
        self.vx_x = vx_dxi
        self.poly_coeffs = poly_coeffs
        self.t = t

    def append_DF(self):
        """Append current DF to the history log."""
        self.DF_log.append((
            self.xi_grids, self.z_grids,
            self.density, self.vx,
            self.density_x, self.density_z, self.vx_x,
            self.poly_coeffs
        ))
        self.time_log.append(self.t)
        self.sigma_xi_log.append(self.sigma_xi)
        self.sigma_z_log.append(self.sigma_z)
        self.poly_coeffs_log.append(self.poly_coeffs)
        self.end_time = self.t

    def pop_left_DF(self, new_start_time):
        """Pop history of DFs until new_start_time."""
        while self.start_time < new_start_time:
            self.DF_log.popleft()
            self.time_log.popleft()
            self.sigma_xi_log.popleft()
            self.sigma_z_log.popleft()
            self.poly_coeffs_log.popleft()
            self.start_time = self.time_log[0]

    def append_interpolant(self, formation_length, n_formation_length):
        """Truncate history and build interpolant. No re-gridding needed."""
        start_point = np.amax(a=(0, self.end_time - n_formation_length * formation_length))
        self.pop_left_DF(new_start_time=start_point)
        self.build_interpolant()

    def build_interpolant(self):
        """
        Build the 3D interpolation arrays by stacking native per-timestep grids.
        Each timestep keeps its own grid extent — no shared grid, no re-interpolation.
        """
        n_t = len(self.DF_log)
        times = list(self.time_log)

        # Time grid metadata
        self.min_x = times[0]
        self.max_x = times[-1]
        self.delta_x = (self.max_x - self.min_x) / (n_t - 1) if n_t > 1 else 1.0

        # Stack native 2D fields into 3D arrays (all same shape: xbins × zbins)
        self.data_density_interp = np.array([entry[2] for entry in self.DF_log])
        self.data_density_x_interp = np.array([entry[4] for entry in self.DF_log])
        self.data_density_z_interp = np.array([entry[5] for entry in self.DF_log])
        self.data_vx_interp = np.array([entry[3] for entry in self.DF_log])
        self.data_vx_x_interp = np.array([entry[6] for entry in self.DF_log])

        # Per-timestep grid metadata arrays
        self.min_xi_arr = np.array([entry[0][0] for entry in self.DF_log])
        self.delta_xi_arr = np.array([entry[0][1] - entry[0][0] for entry in self.DF_log])
        self.min_z_arr = np.array([entry[1][0] for entry in self.DF_log])
        self.delta_z_arr = np.array([entry[1][1] - entry[1][0] for entry in self.DF_log])

        # Polynomial coefficients per timestep: shape (n_t, poly_degree+1)
        self.poly_coeffs_interp = np.array([entry[7] for entry in self.DF_log])
