from numba import jit
import math
import numpy as np
from collections import deque

from .lookup import build_bucket, empty_bucket, is_uniform_times


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


def _curv_g_vec(c2, c3, w, wl):
    """Vectorised interp3D.curv_g (value only)."""
    w = np.asarray(w, dtype=np.float64)
    wc = np.clip(w, -wl, wl)
    g = c2 * (wc * wc - 1.0) + c3 * wc ** 3
    d = 2.0 * c2 * wc + 3.0 * c3 * wc * wc
    return g + d * (w - wc)


def smooth_and_differentiate(field, dx, dz, sigma_smooth, sigma_smooth_z=None):
    """
    Apply FFT Gaussian smoothing and compute spectral derivatives.
    Uses zero-padding to avoid Gibbs ringing from periodicity assumption.

    Parameters:
        field: 2D array (nxi, nz)
        dx: grid spacing in xi direction
        dz: grid spacing in z direction
        sigma_smooth: smoothing kernel width in grid spacings
        sigma_smooth_z: the same along z, if it differs (None: sigma_smooth)

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
    sigma_z_phys = (sigma_smooth if sigma_smooth_z is None else sigma_smooth_z) * dz
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


def velocity_mask(density_smooth, epsilon):
    """
    (rho / (rho + eps))^3 with rho clamped at zero: the taper of d vx/dx where the density is
    negligible. The FFT Gaussian smoothing can leave the density slightly negative near a sharp
    edge (see smoothing_min_bins); without the clamp rho + eps passes through zero there and the
    mask (and vx = rho vx / (rho + eps), which uses the same clamp) blow up. Bit-identical to the
    unclamped form wherever the smoothed density is non-negative.
    """
    rho = np.maximum(density_smooth, 0.0)
    return (rho / (rho + epsilon)) ** 3


def smooth_levels(field, dx, dz, sigma_x, sigma_z_levels, weights, want=(True, True, True)):
    """
    Spatially varying z smoothing: the field smoothed at every z width in sigma_z_levels (x width
    sigma_x for all), sharing one forward FFT, blended cell by cell with weights
    (n_levels, nx, nz) that sum to 1. The derivatives are those of the BLEND (spectral, on the
    blended field), so the returned field and its derivatives are consistent with each other --
    blending the levels' own derivatives instead leaves them off by sum_k f_k grad(w_k), and the
    CSR integrand, whose terms cancel against each other near the observer, turns that into a
    spurious field. Returns (smooth, d/dx, d/dz), None where `want` is False.
    """
    nxi, nz = field.shape
    pad_xi, pad_z = nxi // 2, nz // 2
    fp = np.pad(field, ((pad_xi, pad_xi), (pad_z, pad_z)), mode='constant', constant_values=0)
    F = np.fft.fft2(fp)
    kxi = np.fft.fftfreq(fp.shape[0], d=dx) * 2 * np.pi
    kz = np.fft.fftfreq(fp.shape[1], d=dz) * 2 * np.pi
    KXI, KZ = np.meshgrid(kxi, kz, indexing='ij')
    gx = np.exp(-0.5 * KXI ** 2 * (sigma_x * dx) ** 2)
    crop = lambda a: np.real(a)[pad_xi:pad_xi + nxi, pad_z:pad_z + nz]
    blend = np.zeros_like(field)
    for k, sz in enumerate(sigma_z_levels):
        wk = weights[k]
        if np.any(wk):
            blend += wk * crop(np.fft.ifft2(F * gx * np.exp(-0.5 * KZ ** 2 * (sz * dz) ** 2)))
    out = [blend if want[0] else None, None, None]
    if want[1] or want[2]:
        B = np.fft.fft2(np.pad(blend, ((pad_xi, pad_xi), (pad_z, pad_z)), mode='constant'))
        if want[1]:
            out[1] = crop(np.fft.ifft2(B * (1j * KXI)))
        if want[2]:
            out[2] = crop(np.fft.ifft2(B * (1j * KZ)))
    return tuple(out)


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
        vx_smooth = rho_vx_smooth / (np.maximum(density_smooth, 0.0) + epsilon)

        # Differentiate vx with full smoothing sigma to suppress boundary noise
        _, vx_dxi, _ = smooth_and_differentiate(
            vx_smooth, dx, dz, self.smoothing_sigma
        )
        # Taper ∂vx/∂ξ to zero where density is negligible
        density_mask = velocity_mask(density_smooth, epsilon)
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

    def append_interpolant(self, formation_length, n_formation_length,
                           min_start_time=None):
        """Truncate history and build interpolant. No re-gridding needed."""
        start_point = np.amax(a=(0, self.end_time - n_formation_length * formation_length))
        if min_start_time is not None:
            start_point = min(start_point, float(min_start_time))
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

        # The band construction in CSR.py is shared with the co-moving path and reads these, so
        # every tracker must provide them. Added after a regression: only DF_tracker_comoving had
        # them, which broke the bspline_fft and legacy paths at import time (§11c did not run
        # test_interp_bounds).
        self.t_arr = np.asarray(times, dtype=np.float64)
        self.is_uniform = is_uniform_times(self.t_arr)
        self.bucket, self.bucket_inv_h, self.bucket_t0, self.bucket_M = empty_bucket()
        self.bucket_worst = 0
        self.min_xi_arr = np.array([entry[0][0] for entry in self.DF_log])
        self.delta_xi_arr = np.array([entry[0][1] - entry[0][0] for entry in self.DF_log])
        self.min_z_arr = np.array([entry[1][0] for entry in self.DF_log])
        self.delta_z_arr = np.array([entry[1][1] - entry[1][0] for entry in self.DF_log])

        # Polynomial coefficients per timestep: shape (n_t, poly_degree+1)
        self.poly_coeffs_interp = np.array([entry[7] for entry in self.DF_log])


class DF_tracker_comoving:
    """
    Density tracker for the co-moving (Lagrangian) history interpolant.

    Difference from DF_tracker_smooth
    ---------------------------------
    DF_tracker_smooth stores each snapshot on its own PHYSICAL xi grid, and the
    interpolant then evaluates each snapshot in its own tilted frame and averages
    the results. That lab-frame blend is a superposition of two shear-displaced
    copies of the beam, and is O(1) wrong once
    G = |dp'| sigma_z / sigma_xi >~ 1.

    Here each snapshot is stored on a NORMALIZED grid,

        u = (xi - xi_bar)/sigma_xi ,   xi = x - p(z)
        w = (z  - z_bar )/sigma_z

    which is the same for every snapshot, so index (i, j) refers to the same
    material point of the beam at all times. Blending at fixed (u, w) is then
    exact for any affine evolution -- shear plus scaling -- i.e. exact for linear
    optics between history steps.

    The shared grid is shared in INDEX space only. Snapshot k's physical cell is
    still delta_xi_k = 2*xlim*sigma_xi_k/xbins, identical to DF_tracker_smooth,
    so no transverse resolution is traded away for the shared labelling.

    Also note the z-derivative is stored in the NORMALIZED frame. The chain rule
    d rho/dz|_x = d rho/dz|_xi - p'(z) d rho/dxi is applied at query time with the
    time-blended p', which is the consistent thing to do and which the old path
    could not do because it baked p'_k in per snapshot.
    """

    def __init__(self, input_dic={}):
        self.configure_params(**input_dic)

        self.sigma_xi = None
        self.sigma_z = None
        self.start_time = 0.0
        self.t = 0.0
        self.end_time = 0.0

        self.DF_log = deque([])
        self.time_log = deque([])
        self.sigma_xi_log = deque([])
        self.sigma_z_log = deque([])
        self.poly_coeffs_log = deque([])
        self.clipped_log = deque([])

        # ---- ring buffer for the five large field arrays -------------------------
        # The deque only changes by one append and a few poplefts per step, but
        # build_interpolant used to rebuild the whole stack with
        # np.array([e[k] for e in DF_log]) -- O(N) copying for an O(1) change, every
        # step. Since N is bounded by the history window n_fl*L_f/step_size, total
        # cost went as (1/step_size)^2, and allocating a fresh 164 MB set of arrays
        # every step drove peak RSS to 8.75 GB at 200^2, which is what blocked the
        # fine stepping 6x showed is needed at a waist.
        #
        # Storage is preallocated and written one slice at a time. The live region is
        # kept CONTIGUOUS (compacted when the cursor reaches the end) rather than
        # wrapping modularly, so the interpolant is a zero-copy VIEW and
        # interpolate3D_comoving_fields needs no change at all.
        self._ring = None            # (NFIELD, cap, xbins, zbins)
        self._ring_head = 0          # one past the newest live slice
        self._ring_tail = 0          # index of the oldest live slice
        self._ring_cap0 = 64         # grows by doubling; L_f is not known this early

        # The normalized grid, fixed once and shared by every snapshot.
        # Registration matches histogram_bspline_2d: nodes sit at BIN CENTRES,
        # spacing (end - start)/nbins, not the (n-1) spacing np.linspace gives.
        self.u_start = -float(self.xlim)
        self.u_end = float(self.xlim)
        self.w_start = -float(self.zlim)
        self.w_end = float(self.zlim)
        self.delta_u = (self.u_end - self.u_start) / self.xbins
        self.delta_w = (self.w_end - self.w_start) / self.zbins
        self.u_grids = self.u_start + (np.arange(self.xbins) + 0.5) * self.delta_u
        self.w_grids = self.w_start + (np.arange(self.zbins) + 0.5) * self.delta_w

    # frame_blend codes, shared with the numba kernel, which cannot take strings
    BLEND_COEFF = 0
    BLEND_MOMENT = 1
    BLEND_ORIENT = 2
    BLEND_CODES = {'coeff': BLEND_COEFF, 'moment': BLEND_MOMENT,
                   'orient': BLEND_ORIENT}

    def configure_params(self, method='bspline_comoving', xbins=128, zbins=128,
                         xlim=5, zlim=5, smoothing_sigma=3.0, poly_degree=1,
                         velocity_threhold=5, upper_limit=None,
                         filter_order=0, filter_window=0, fit_zlim=3.0,
                         clip_warn=1e-3, frame_blend='orient', vx_model='affine',
                         curvature=True, curv_zlim=4.0, smoothing_sigma_z=None,
                         frame='rms', frame_quantile=0.005, frame_margin=0.1,
                         smoothing_adaptive=False, smoothing_min_particles=100,
                         smoothing_sigma_z_max=16.0, peak_halfwidth=3.0, peak_taper=2.0,
                         peak_rate=0.2, peak_shift=0.25, smoothing_res_fwhm=None,
                         smoothing_min_bins=1.0,
                         u_taper=0.5, edge_warn=1e-4):
        self.method = method
        self.xbins = xbins
        self.zbins = zbins
        self.xlim = xlim
        self.zlim = zlim
        self.smoothing_sigma = smoothing_sigma
        # z smoothing in bins, if it should differ from x (None: smoothing_sigma). With
        # frame 'quantile' and many z bins, z can be smoothed over ~1 bin while x keeps 3.
        self.smoothing_sigma_z = None if smoothing_sigma_z is None else float(smoothing_sigma_z)
        # frames 'quantile'/'peak': cap the z smoothing in physical length, so that the effective
        # z resolution (cell * sqrt(sz^2 + 2/3)) stays <= smoothing_res_fwhm * FWHM of the highest
        # current peak (floor smoothing_min_bins). The smoothing is in bins and the bins follow the window,
        # so without a cap a smoothing chosen for the long bunch (BC20 entrance, 2 um bins) is
        # a few bins of a 0.1 um grid once the window has shrunk onto a spike -- harmless -- but
        # one chosen to remove the long bunch's fine structure (6.5 bins = 14 um there) is 0.6-1 um
        # on the spike, comparable to its rms. None: no cap.
        self.smoothing_res_fwhm = None if smoothing_res_fwhm is None else float(smoothing_res_fwhm)
        # the cap never takes the z smoothing below this many bins. smooth_and_differentiate is a
        # Gaussian applied in Fourier space; below ~1 bin it is not band-limited (at 0.5 bins the
        # response at Nyquist is 0.29), so its real-space kernel rings and the smoothed density of a
        # sharp edge goes NEGATIVE by up to ~1e-4 of the peak -- enough to reach the velocity floor
        # epsilon (1.6e-4 at velocity_threhold 1000) and send vx = rho vx / (rho + eps) and its mask
        # to infinity (Indranil's loss beam: 7654 MeV/m in W2 at the second kick, 1e20 at the end,
        # progress_facet_bc20.md 11b). The old floor was 0.5.
        self.smoothing_min_bins = float(smoothing_min_bins)
        # Which normalized frame the fixed (u, w) grid is laid in, per snapshot:
        # 'rms'       w = (z - mean)/std over all particles (the original frame)
        # 'quantile'  the grid spans [z_q, z_(1-q)] widened by frame_margin*span on each side,
        #             q = frame_quantile; particles are weighted by T(z), 1 on [z_q, z_(1-q)] and
        #             a C2 smootherstep to 0 across the margin, so the far tails are dropped as
        #             a source smoothly instead of being clipped. A short or non-Gaussian bunch
        #             (BC20: full span 208 um against +-5 rms = 515 um) gets bins fitted to
        #             where its charge is, and quantiles are continuous in the particle
        #             coordinates, so the frame does not jitter or jump between horns.
        # 'peak'      the quantile window, narrowed to the highest current peak: full weight over
        #             centre +- peak_halfwidth * FWHM, a taper over peak_taper * FWHM, never wider
        #             than the quantile window. A long bunch gets exactly the quantile frame; once a
        #             spike forms, the low-current pedestal behind it is dropped as a source (its
        #             CSR is small; CSR2D reports its 1D share per kick and warns above
        #             CSR_computation.pedestal_warn, which also catches a second peak outside the
        #             window). Centre and FWHM from peakfind.peak_center_fwhm, rate-limited between
        #             snapshots (|d ln H| <= peak_rate, |d centre| <= peak_shift * H), because the
        #             history lookup blends neighbouring snapshots' frames.
        if frame not in ('rms', 'quantile', 'peak'):
            raise ValueError(f"frame must be 'rms', 'quantile' or 'peak', got {frame!r}")
        self.peak_halfwidth = float(peak_halfwidth)
        self.peak_taper = float(peak_taper)
        self.peak_rate = float(peak_rate)
        self.peak_shift = float(peak_shift)
        self._peak_prev = None       # (centre, half-width) of the previous snapshot, frame peak
        if not 0.0 < float(frame_quantile) < 0.5:
            raise ValueError(f'frame_quantile must be in (0, 0.5), got {frame_quantile}')
        if not float(frame_margin) > 0.0:
            raise ValueError(f'frame_margin must be > 0, got {frame_margin}')
        self.frame = frame
        # frame 'quantile' only: density-adaptive z smoothing. Each cell's z smoothing width is
        # the smallest of smoothing_sigma_z * 2^k (up to smoothing_sigma_z_max bins) whose kernel
        # holds at least smoothing_min_particles particles of a smooth pilot density, blended
        # between neighbouring widths in log2. Dense spikes keep the fine width, the sparse
        # pedestal and a long early bunch get wide ones, so the bins can stay fine everywhere
        # without their shot noise (smooth_levels).
        self.smoothing_adaptive = bool(smoothing_adaptive)
        self.smoothing_min_particles = float(smoothing_min_particles)
        self.smoothing_sigma_z_max = float(smoothing_sigma_z_max)
        if self.smoothing_adaptive and frame == 'rms':
            raise ValueError("smoothing_adaptive needs frame 'quantile' or 'peak'")
        self.frame_quantile = float(frame_quantile)
        self.frame_margin = float(frame_margin)
        self.window = None           # (lo, hi, margin_lo, margin_hi) of the newest snapshot
        self.f_window = 1.0          # sum(T)/N of the newest snapshot
        self.res_z = None            # effective z resolution of the newest snapshot [m]
        self.res_log = []            # frame quantile: (t, fwhm of the highest peak, res_z, f_window)
        # deg 1 by default: a higher-degree fit is extrapolated to +-zlim*sigma_z
        # where there are no particles, and p(z), p'(z) blow up in the z tails.
        # Degree 1 also matches the transform the wake mesh and the integration
        # bands use, and makes the localization branches of thesis Eq 4.24 exact.
        self.poly_degree = poly_degree
        self.velocity_threhold = velocity_threhold
        self.upper_limit = upper_limit
        # restrict the tilt fit to the core, so tail particles cannot lever it
        self.fit_zlim = fit_zlim
        self.clip_warn = clip_warn
        # Transverse edge taper (progress_cancellation.md §4c-d). The deposit used to end in a
        # hard cut at u = +-xlim: particles beyond were dropped and the interpolant is zero past
        # the grid, so wherever the beam reaches the edge rho has a step there that the stored
        # d rho/du does not contain. The longitudinal integrand is G * d rho/dx, whose +- lobes
        # cancel by a factor C = 60-1000 (C ~ 188 sigma_z^(2/3)/sigma_xi um), and an error in
        # d rho/dx with a nonzero integral across the beam is amplified by ~C: clipping 0.3% of
        # the charge moved the wake by 3%, against 0.19% when the cut is consistent.
        # So each particle's deposit weight is multiplied by T(u): 1 for |u| <= e - u_taper,
        # a C2 smootherstep to 0 at |u| = e, with e = xlim - 3 smoothing_sigma delta_u so the
        # smoothed field is down to ~1e-2 of the ramp's density at the grid edge. The charge
        # removed is then removed consistently (the stored derivatives are those of the tapered
        # field), and counts as clipped in the clip_warn diagnostic. A z cut does not need this:
        # T(z) rho has d/dx = T(z) d rho/dx, still cancelling; frames quantile/peak taper z
        # for the physical step in the current. u_taper = 0: the hard edge, as before.
        self.u_taper = float(u_taper)
        # warn (once per run) if the smoothed density at the u edge exceeds edge_warn * peak
        self.edge_warn = float(edge_warn)
        self.edge_u = 0.0            # u-edge density / peak of the newest snapshot
        self.edge_u_max = 0.0        # its maximum over the run
        self._edge_warned = False

        # Which parametrisation of the frame is interpolated BETWEEN snapshots.
        # Deposition is identical either way; only the blend differs.
        #
        # 'coeff'  blends tau = tan(alpha) linearly. tan is the wrong chart across a
        #          longitudinal waist: the beam's orientation goes +87 deg -> 90 deg
        #          (vertical) -> -87 deg, and no continuous path in tau can pass
        #          through 90 deg -- it must pass through 0, i.e. HORIZONTAL. Measured
        #          at a shear-20 waist, the blended frame then spends 2.75% of the near
        #          region at |alpha| < 45 deg, orientations the beam never occupies,
        #          and |tan 2a| = 2 tau/(1 - tau^2) reaches 285 against endpoint values
        #          of 0.10 because tau crosses the poles at +-1 twice.
        # 'moment' blends (var_z, cov, var_x) linearly and derives
        #          tau = cov/var_z, sigma_z = sqrt(var_z),
        #          sigma_xi = sqrt(var_x - cov^2/var_z).
        #          Same measurement: 0.03% and |tan 2a| <= 1.234. But it pays for that
        #          with a width defect: a convex mix of two thin ellipses at different
        #          tilts is FATTER than either, by a(1-a)(dtau)^2 var_z, so mid-interval
        #          sigma_xi comes out 8x-10x too large at ordinary high tilt and the
        #          wake stops converging under step refinement. Kept because it is the
        #          honest full-covariance option and the defect is measured, not
        #          assumed.
        # 'orient' takes the orientation from the moment ratio and the widths
        #          log-linearly, which is where each chart is well conditioned. Only
        #          tau needs the moment chart; sigma_xi and sigma_z do not.
        #
        # Default stays 'coeff' so every recorded baseline reproduces byte-identically.
        if frame_blend not in self.BLEND_CODES:
            raise ValueError(f'frame_blend must be one of '
                             f'{sorted(self.BLEND_CODES)}, got {frame_blend!r}')
        if frame_blend != 'coeff' and poly_degree != 1:
            raise ValueError(f"frame_blend='{frame_blend}' needs poly_degree = 1 "
                             f'(the moment triple describes a straight axis), '
                             f'got poly_degree = {poly_degree}')
        self.frame_blend = frame_blend
        self.frame_blend_code = self.BLEND_CODES[frame_blend]

        # Control variate for the velocity field.
        # 'none'   deposit px itself: v_x = smooth(rho px) / (smooth(rho) + eps).
        # 'affine' fit px ~ b x + g(z) (g cubic in w) over the core, deposit only the
        #          residual, and add b x + g(z) back analytically (b to vx_u exactly).
        # Why: smoothing a ratio is biased by ~ sigma_k^2 grad(ln rho) . grad(px). On a
        # strongly tilted beam px varies fastest ALONG the beam (x ~ tau z, so even
        # px = b x has d px/dw = b tau sigma_z), and when the beam is also curved in
        # (u, w) -- the straight tilt fit leaves a banana -- d^2 ln rho / du dw changes
        # along the bunch, so the bias in d vx/du acquires a z-trend. Measured in the
        # chicane D3 (tau = -49): the code's d vx/dx ran -0.03 .. -0.42 /m tail to head
        # against a true -0.26 .. -0.30, and that trend drove the spurious x-structure of
        # the far-drift wake. With 'affine' the error is <= 0.02 /m in every slice.
        # Default 'affine' since 2026-11-04 (progress.md). It also removes the 6% low
        # bias of the FODO W2 test, which was the smoothing shrinking d vx/dx, so
        # the W2 ratio there moved 0.1327..0.1356 -> 0.1415..0.1418. Results recorded
        # before that date were computed with 'none'; set it explicitly to reproduce.
        if vx_model not in ('none', 'affine'):
            raise ValueError(f"vx_model must be 'none' or 'affine', got {vx_model!r}")
        self.vx_model = vx_model
        # Curvature shear of the normalized frame: u' = u - g(w), g = c2 (w^2 - 1) +
        # c3 w^3 fitted to the slice centroids of the core (|w| < fit_zlim) and continued
        # linearly beyond (interp3D.curv_g). A straight tilt fit on a curved beam leaves a
        # banana in (u, w) whose tails leave the +-xlim grid: 0.28% of the charge in the
        # chicane's B2, a false edge in rho near |w| = 3, and +-3 sigma_z spikes in the
        # wake. Default on since 2026-11-05 (curv_zlim 4); the localization axis stays the
        # straight affine one and the bands are shifted onto the ridge (CSR2D).
        self.curvature = bool(curvature)
        # fit window of the curvature shear in |w|, and where g turns linear; default
        # fit_zlim. Wider than the tilt fit because the tails it must follow are exactly
        # what a +-3 window leaves out (B3 exit: cubic misses the ridge by 1.5 sxi at 3.25).
        self.curv_zlim = float(fit_zlim if curv_zlim is None else curv_zlim)

    def u_flat(self):
        """|u| up to which the edge taper keeps the full deposit weight (xlim if off)."""
        if self.u_taper <= 0.0:
            return self.u_end
        return self.u_end - 3.0 * self.smoothing_sigma * self.delta_u - self.u_taper

    def u_weight(self, u):
        """Deposit weight T(u) of the transverse edge taper (see configure_params)."""
        if self.u_taper <= 0.0:
            return np.ones_like(u)
        from .peakfind import window_weight
        flat = self.u_flat()
        if flat <= 0.0:
            raise ValueError(f'u_taper = {self.u_taper} does not fit inside xlim = {self.xlim} '
                             f'(edge pad {3.0 * self.smoothing_sigma * self.delta_u:.2f} sigma_xi)')
        return window_weight(u, -flat, flat, self.u_taper)

    def _check_edge(self, density_smooth, t):
        """Smoothed density at the u edges relative to its peak; warn once per run."""
        pk = float(np.max(density_smooth))
        e = 0.0
        if pk > 0.0:
            e = float(max(np.abs(density_smooth[0]).max(), np.abs(density_smooth[-1]).max())) / pk
        self.edge_u = e
        self.edge_u_max = max(self.edge_u_max, e)
        if e > self.edge_warn and not self._edge_warned:
            self._edge_warned = True
            print(f'  [DF_tracker_comoving] warning: the density at the u = +-{self.xlim} grid '
                  f'edge is {e:.1e} of its peak at t = {t:.4f}. The longitudinal integrand '
                  f'amplifies such an edge by its cancellation factor; widen xlim or set '
                  f'u_taper > 0 (further snapshots are not reported; see edge_u_max).')

    def get_DF(self, x, z, px, t):
        if self.frame in ('quantile', 'peak'):
            return self._get_DF_quantile(x, z, px, t)
        zmean = np.mean(z)
        zstd = np.std(z)

        # --- tilt removal, fitted on the core only ---
        core = np.abs(z - zmean) < self.fit_zlim * zstd
        if core.sum() < 10 * (self.poly_degree + 1):
            core = np.ones(len(z), dtype=bool)
        poly_coeffs = np.polyfit(z[core], x[core], deg=self.poly_degree)
        xi = x - np.polyval(poly_coeffs, z)

        xi_bar = np.mean(xi)
        z_bar = zmean
        sigma_xi = np.std(xi)
        sigma_z = zstd

        # --- normalized coordinates ---
        u = (xi - xi_bar) / sigma_xi
        w = (z - z_bar) / sigma_z

        # --- optional curvature shear, fitted in (u, w) on the core ---
        c2 = c3 = 0.0
        if self.curvature:
            cw = np.abs(w) < self.curv_zlim
            wc, uc = w[cw], u[cw]
            A = np.column_stack([wc * wc - 1.0, wc ** 3, np.ones(wc.size), wc])
            coef = np.linalg.lstsq(A, uc, rcond=None)[0]
            c2, c3 = float(coef[0]), float(coef[1])
            g = _curv_g_vec(c2, c3, w, self.curv_zlim)
            u = u - g
            # sigma_xi stays the affine (all-particle) width: the grid and the bands are
            # sized by it, and keeping it makes curvature=True a pure reshuffle of u
        self.curv_c2, self.curv_c3 = c2, c3

        npart = len(u)
        Tu = self.u_weight(u)
        density = histogram_bspline_2d(
            q1=u, q2=w, w=Tu,
            nbins_1=self.xbins, bins_start_1=self.u_start, bins_end_1=self.u_end,
            nbins_2=self.zbins, bins_start_2=self.w_start, bins_end_2=self.w_end)
        if self.vx_model == 'affine':
            # px ~ c0 + b_u u + g(w), g = g1 w + g2 w^2 + g3 w^3, fitted on the core.
            # In (u, w), b x = b sigma_xi u + (a function of w), so the u-coefficient
            # carries the slope and the w-polynomial absorbs the rest of b x.
            wc = w[core]
            A = np.column_stack([np.ones(wc.size), u[core], wc, wc ** 2, wc ** 3])
            vx_coef = np.linalg.lstsq(A, px[core], rcond=None)[0]
        else:
            vx_coef = np.zeros(5)
        # the cubic is held flat beyond the fit window, so it is not extrapolated
        # into the empty tails of the +-zlim grid
        wq = np.clip(w, -self.fit_zlim, self.fit_zlim)
        vx_fit = (vx_coef[0] + vx_coef[1] * u + vx_coef[2] * wq
                  + vx_coef[3] * wq ** 2 + vx_coef[4] * wq ** 3)
        vx_weighted = histogram_bspline_2d(
            q1=u, q2=w, w=Tu * (px - vx_fit if self.vx_model == 'affine' else px),
            nbins_1=self.xbins, bins_start_1=self.u_start, bins_end_1=self.u_end,
            nbins_2=self.zbins, bins_start_2=self.w_start, bins_end_2=self.w_end)

        # --- exact, grid-independent normalization ---
        # The old path used trapz of the deposited field and divided by it, which
        # renormalizes clipped charge back to 1 and lets the clipped fraction
        # (different every snapshot) leak in as temporal jitter. Dividing by the
        # known particle count and cell area instead makes the clipping visible.
        norm = npart * self.delta_u * self.delta_w
        density /= norm
        vx_weighted /= norm
        captured = density.sum() * self.delta_u * self.delta_w
        self.clipped = 1.0 - captured
        if self.clipped > self.clip_warn:
            print(f'  [DF_tracker_comoving] warning: {100*self.clipped:.2f}% of the '
                  f'charge fell outside the +-{self.xlim}/{self.zlim} sigma grid or in its '
                  f'u edge taper at t = {t:.4f}')

        # --- smoothing and derivatives, in the NORMALIZED frame ---
        density_smooth, density_du, density_dw = smooth_and_differentiate(
            density, self.delta_u, self.delta_w, self.smoothing_sigma, self.smoothing_sigma_z)

        rho_vx_smooth, _, _ = smooth_and_differentiate(
            vx_weighted, self.delta_u, self.delta_w, self.smoothing_sigma, self.smoothing_sigma_z)

        # snapshot-independent velocity floor: scale by the peak of a normalized
        # Gaussian on this grid, a fixed number, instead of this snapshot's own
        # peak density. Otherwise the floor and the mask below breathe with the
        # beam and inject temporal inconsistency into the W2 / div(v) term.
        rho_hat_ref = 1.0 / (2.0 * np.pi)
        epsilon = rho_hat_ref / self.velocity_threhold
        vx_smooth = rho_vx_smooth / (np.maximum(density_smooth, 0.0) + epsilon)

        _, vx_du, _ = smooth_and_differentiate(
            vx_smooth, self.delta_u, self.delta_w, self.smoothing_sigma, self.smoothing_sigma_z)
        vx_du *= velocity_mask(density_smooth, epsilon)
        self._check_edge(density_smooth, t)

        if self.vx_model == 'affine':
            # the fitted part is exact and noise-free, so it is added without the
            # floor taper; only the deposited residual is suppressed in the tails
            wg = np.clip(self.w_grids, -self.fit_zlim, self.fit_zlim)[None, :]
            vx_smooth = vx_smooth + (vx_coef[0] + vx_coef[1] * self.u_grids[:, None]
                                     + vx_coef[2] * wg + vx_coef[3] * wg ** 2
                                     + vx_coef[4] * wg ** 3)
            vx_du = vx_du + vx_coef[1]
        self.vx_coef = vx_coef

        self.density = density_smooth
        self.density_u = density_du
        self.density_w = density_dw          # normalized frame, NO chain rule baked in
        self.vx = vx_smooth
        self.vx_u = vx_du

        self.poly_coeffs = poly_coeffs
        self.xi_bar = xi_bar
        self.z_bar = z_bar
        self.sigma_xi = sigma_xi
        self.sigma_z = sigma_z
        self.t = t

        # physical cell sizes, for diagnostics and for locating the lab-frame
        # support of this snapshot
        self.delta_xi_phys = self.delta_u * sigma_xi
        self.delta_z_phys = self.delta_w * sigma_z

    def window_weight(self, z):
        """T(z) of the newest snapshot's quantile window; 1 everywhere in frame 'rms'."""
        if self.window is None:
            return np.ones_like(np.asarray(z, dtype=float))
        from .peakfind import window_weight
        lo, hi, m_lo, m_hi = self.window
        return window_weight(z, lo, hi, m_lo, m_hi)

    def _adaptive_weights(self, density, npart, s0):
        """
        Level widths and per-cell blend weights for smoothing_adaptive. The required z width at a
        cell, in bins, so that a Gaussian kernel of x width smoothing_sigma holds
        smoothing_min_particles particles of the pilot density rho_p (normalised: integral over
        (u, w) is f_window, so a cell holds npart * rho_p * du * dw particles):
            sigma_z = N_min / (npart * rho_p * 2 pi * sigma_x * du * dw),
        clipped to [s0, smoothing_sigma_z_max] and blended linearly in log2 between the levels
        s0 * 2^k. Returns (levels, weights, width used at the density peak).
        """
        from scipy.ndimage import gaussian_filter
        sx = self.smoothing_sigma
        smax = max(self.smoothing_sigma_z_max, s0)
        nlev = int(np.ceil(np.log2(smax / s0))) + 1
        levels = s0 * 2.0 ** np.arange(nlev)
        cell = npart * self.delta_u * self.delta_w

        def weights(pilot):
            need = self.smoothing_min_particles / (np.maximum(pilot, 1e-300) * cell * 2 * np.pi * sx)
            lev = np.clip(np.log2(np.clip(need, s0, levels[-1]) / s0), 0.0, nlev - 1)
            # the blend weights must vary slowly, or grad(w) puts structure into the field
            lev = np.clip(gaussian_filter(lev, (sx, 2.0 * s0), mode='nearest'), 0.0, nlev - 1)
            lo = np.floor(lev).astype(int)
            frac = lev - lo
            W = np.zeros((nlev,) + density.shape)
            for k in range(nlev):
                W[k] += np.where(lo == k, 1.0 - frac, 0.0)
                W[k] += np.where(lo + 1 == k, frac, 0.0)
            return W, lev

        # two passes: the widest level is a noise-free but blurred pilot (it would understate a
        # spike's density and over-smooth it), so the pilot is then re-estimated with the
        # adaptive smoothing it implies
        pilot = smooth_and_differentiate(density, self.delta_u, self.delta_w, sx, levels[-1])[0]
        W, lev = weights(pilot)
        pilot = smooth_levels(density, self.delta_u, self.delta_w, sx, levels, W,
                              want=(True, False, False))[0]
        W, lev = weights(pilot)
        i_pk = np.unravel_index(np.argmax(pilot), pilot.shape)
        return levels, W, float(s0 * 2.0 ** lev[i_pk])

    def _peak_window(self, z, lo_q, hi_q, m_q):
        """
        Frame 'peak': (lo, hi, margin_lo, margin_hi) of the window around the highest current
        peak, intersected with the quantile window [lo_q, hi_q] (margin m_q). A side cut by the
        peak window is tapered over peak_taper * FWHM, a side cut by the quantile keeps m_q.
        """
        from .peakfind import peak_center_fwhm
        c, fw = peak_center_fwhm(z, prev=self._peak_prev)
        H = self.peak_halfwidth * fw
        if self._peak_prev is not None:
            cp, Hp = self._peak_prev
            H = float(np.clip(H, Hp * np.exp(-self.peak_rate), Hp * np.exp(self.peak_rate)))
            c = float(np.clip(c, cp - self.peak_shift * Hp, cp + self.peak_shift * Hp))
        self._peak_prev = (c, H)
        self._peak_fw = H / self.peak_halfwidth      # the rate-limited FWHM, for smoothing_res_fwhm
        tap = self.peak_taper * H / self.peak_halfwidth
        lo, m_lo = (c - H, tap) if c - H > lo_q else (lo_q, m_q)
        hi, m_hi = (c + H, tap) if c + H < hi_q else (hi_q, m_q)
        return lo, hi, m_lo, m_hi

    def _get_DF_quantile(self, x, z, px, t):
        """
        get_DF in frame 'quantile' (see configure_params). The same steps as the rms path,
        with every fit and the deposit weighted by T(z), so the dropped tails do not lever
        the frame either. The density is still normalised by the TOTAL particle count:
        it integrates to f_window = sum(T)/N, the dropped charge is simply absent.
        """
        from .peakfind import peak_fwhm, window_weight
        npart = len(z)
        lo, hi = np.quantile(z, [self.frame_quantile, 1.0 - self.frame_quantile])
        margin = self.frame_margin * (hi - lo)
        if self.frame == 'peak':
            lo, hi, m_lo, m_hi = self._peak_window(z, lo, hi, margin)
            T = window_weight(z, lo, hi, m_lo, m_hi)
            # frame: centred on the grid span [lo - m_lo, hi + m_hi], which the +-zlim grid spans
            z_bar = 0.5 * ((lo - m_lo) + (hi + m_hi))
            sigma_z = 0.5 * ((hi + m_hi) - (lo - m_lo)) / float(self.zlim)
        else:
            m_lo = m_hi = margin
            T = window_weight(z, lo, hi, margin)
            # frame: centred on the window, scaled so the fixed +-zlim grid spans it exactly
            z_bar = 0.5 * (lo + hi)
            sigma_z = (0.5 * (hi - lo) + margin) / float(self.zlim)
        sel = T > 0.0
        Ts = T[sel]
        sw = np.sqrt(Ts)

        # tilt, fitted with weights T over the window
        poly_coeffs = np.polyfit(z[sel], x[sel], deg=self.poly_degree, w=sw)
        xi = x - np.polyval(poly_coeffs, z)
        xi_bar = np.average(xi[sel], weights=Ts)
        sigma_xi = np.sqrt(np.average((xi[sel] - xi_bar) ** 2, weights=Ts))

        u = (xi - xi_bar) / sigma_xi
        w = (z - z_bar) / sigma_z
        # the vx cubic and its clip use the window span [lo, hi], where T = 1
        w_lo, w_hi = (lo - z_bar) / sigma_z, (hi - z_bar) / sigma_z

        c2 = c3 = 0.0
        if self.curvature:
            cw = sel & (np.abs(w) < self.curv_zlim)
            wc, uc, sc = w[cw], u[cw], np.sqrt(T[cw])
            A = np.column_stack([wc * wc - 1.0, wc ** 3, np.ones(wc.size), wc])
            coef = np.linalg.lstsq(A * sc[:, None], uc * sc, rcond=None)[0]
            c2, c3 = float(coef[0]), float(coef[1])
            u = u - _curv_g_vec(c2, c3, w, self.curv_zlim)
        self.curv_c2, self.curv_c3 = c2, c3

        Tu = self.u_weight(u)
        density = histogram_bspline_2d(
            q1=u, q2=w, w=T * Tu,
            nbins_1=self.xbins, bins_start_1=self.u_start, bins_end_1=self.u_end,
            nbins_2=self.zbins, bins_start_2=self.w_start, bins_end_2=self.w_end)
        wq = np.clip(w, w_lo, w_hi)
        if self.vx_model == 'affine':
            wc = wq[sel]
            A = np.column_stack([np.ones(wc.size), u[sel], wc, wc ** 2, wc ** 3])
            vx_coef = np.linalg.lstsq(A * sw[:, None], px[sel] * sw, rcond=None)[0]
        else:
            vx_coef = np.zeros(5)
        vx_fit = (vx_coef[0] + vx_coef[1] * u + vx_coef[2] * wq
                  + vx_coef[3] * wq ** 2 + vx_coef[4] * wq ** 3)
        vx_weighted = histogram_bspline_2d(
            q1=u, q2=w, w=T * Tu * (px - vx_fit if self.vx_model == 'affine' else px),
            nbins_1=self.xbins, bins_start_1=self.u_start, bins_end_1=self.u_end,
            nbins_2=self.zbins, bins_start_2=self.w_start, bins_end_2=self.w_end)

        norm = npart * self.delta_u * self.delta_w
        density /= norm
        vx_weighted /= norm
        f_window = float(T.sum()) / npart
        captured = density.sum() * self.delta_u * self.delta_w
        # clipped = charge INSIDE the window that missed the grid or fell in the u edge taper
        # (should be ~0: T vanishes at the grid edge); the window's own taper is reported as
        # f_window, not as clipping
        self.clipped = 1.0 - captured / f_window
        if self.clipped > self.clip_warn:
            print(f'  [DF_tracker_comoving] warning: {100*self.clipped:.2f}% of the windowed '
                  f'charge fell outside the +-{self.xlim} sigma_xi grid or in its u edge '
                  f'taper at t = {t:.4f}')

        sz = self.smoothing_sigma if self.smoothing_sigma_z is None else self.smoothing_sigma_z
        fw_pk = peak_fwhm(z[sel])
        if self.smoothing_res_fwhm:
            fw = self._peak_fw if self.frame == 'peak' else fw_pk
            r = self.smoothing_res_fwhm * fw / (self.delta_w * sigma_z)
            sz = min(sz, max(np.sqrt(max(r * r - 2.0 / 3.0, 0.0)), self.smoothing_min_bins))
        self.sz_used = sz
        # the floor scales with the windowed charge, so eps / peak density stays as in the
        # rms frame; f_window is smooth in time, unlike a per-snapshot peak
        epsilon = f_window * (1.0 / (2.0 * np.pi)) / self.velocity_threhold
        if self.smoothing_adaptive:
            levels, W, sz_peak = self._adaptive_weights(density, npart, sz)
            sx = self.smoothing_sigma
            density_smooth, density_du, density_dw = smooth_levels(
                density, self.delta_u, self.delta_w, sx, levels, W)
            rho_vx_smooth = smooth_levels(vx_weighted, self.delta_u, self.delta_w, sx, levels, W,
                                          want=(True, False, False))[0]
            vx_smooth = rho_vx_smooth / (np.maximum(density_smooth, 0.0) + epsilon)
            vx_du = smooth_levels(vx_smooth, self.delta_u, self.delta_w, sx, levels, W,
                                  want=(False, True, False))[1]
        else:
            sz_peak = sz
            density_smooth, density_du, density_dw = smooth_and_differentiate(
                density, self.delta_u, self.delta_w, self.smoothing_sigma, sz)
            rho_vx_smooth, _, _ = smooth_and_differentiate(
                vx_weighted, self.delta_u, self.delta_w, self.smoothing_sigma, sz)
            vx_smooth = rho_vx_smooth / (np.maximum(density_smooth, 0.0) + epsilon)
            _, vx_du, _ = smooth_and_differentiate(
                vx_smooth, self.delta_u, self.delta_w, self.smoothing_sigma, sz)
        vx_du *= velocity_mask(density_smooth, epsilon)
        self._check_edge(density_smooth, t)
        if self.vx_model == 'affine':
            wg = np.clip(self.w_grids, w_lo, w_hi)[None, :]
            vx_smooth = vx_smooth + (vx_coef[0] + vx_coef[1] * self.u_grids[:, None]
                                     + vx_coef[2] * wg + vx_coef[3] * wg ** 2
                                     + vx_coef[4] * wg ** 3)
            vx_du = vx_du + vx_coef[1]
        self.vx_coef = vx_coef

        self.density = density_smooth
        self.density_u = density_du
        self.density_w = density_dw
        self.vx = vx_smooth
        self.vx_u = vx_du
        self.poly_coeffs = poly_coeffs
        self.xi_bar = xi_bar
        self.z_bar = z_bar
        self.sigma_xi = sigma_xi
        self.sigma_z = sigma_z
        self.t = t
        self.delta_xi_phys = self.delta_u * sigma_xi
        self.delta_z_phys = self.delta_w * sigma_z

        self.window = (float(lo), float(hi), float(m_lo), float(m_hi))
        self.f_window = f_window
        # effective z resolution: B-spline deposit + B-spline evaluation (variance 2/3 cell^2)
        # and the Gaussian smoothing (sz cells; with smoothing_adaptive, the width used at the
        # density peak, i.e. the finest the run resolves)
        self.res_z = self.delta_z_phys * np.sqrt(sz_peak * sz_peak + 2.0 / 3.0)
        self.res_log.append((t, fw_pk, self.res_z, f_window))

    NFIELD = 5      # density, density_u, density_w, vx, vx_u

    def _ring_push(self, fields):
        """
        Write one snapshot's five field arrays into the ring. O(1) amortised.

        Compaction (not modular wrapping) keeps the live region contiguous, which is
        what lets build_interpolant hand out a view instead of a copy. The memmove
        costs O(n) but runs only once per (cap - n) pushes.
        """
        nx, nz = fields[0].shape
        if self._ring is None:
            self._ring = np.empty((self.NFIELD, self._ring_cap0, nx, nz))
            self._ring_head = self._ring_tail = 0
        assert self._ring.shape[2:] == (nx, nz), (
            f'snapshot grid changed from {self._ring.shape[2:]} to {(nx, nz)}; the '
            f'co-moving normalized grid is supposed to be fixed for all time')

        cap = self._ring.shape[1]
        n = self._ring_head - self._ring_tail
        # Shrink is tested on EVERY push, not only when the cursor reaches the end.
        # The history grows deep during the pre-first-bend drift, where
        # formation_length is an accumulated drift length and nothing is truncated,
        # then collapses once a bend sets L_f from sigma_z. Checking only at
        # compaction meant the cursor needed cap - n further pushes to notice, which
        # is more than a short run has: the buffer sat at 1.64 GB for 251 live slices
        # needing 0.40 GB. After shrinking, cap = 2n, so the condition is immediately
        # false again and this cannot thrash.
        target = None
        if self._ring_head == cap:
            target = 2 * cap if 2 * n > cap else cap
        elif cap > self._ring_cap0 and 4 * n < cap:
            target = max(self._ring_cap0, 2 * n)

        if target is not None:
            if target != cap:
                resized = np.empty((self.NFIELD, target, nx, nz))
                resized[:, :n] = self._ring[:, self._ring_tail:self._ring_head]
                self._ring = resized
            else:
                self._ring[:, :n] = self._ring[:, self._ring_tail:self._ring_head]
            self._ring_tail, self._ring_head = 0, n

        for k in range(self.NFIELD):
            self._ring[k, self._ring_head] = fields[k]
        self._ring_head += 1

    def append_DF(self):
        self._ring_push((self.density, self.density_u, self.density_w,
                         self.vx, self.vx_u))
        # the small per-snapshot scalars stay in a deque: N * ~6 floats is free, and
        # keeping them here avoids duplicating the popleft bookkeeping
        self.DF_log.append((self.poly_coeffs, self.xi_bar, self.sigma_xi,
                            self.z_bar, self.sigma_z, self.curv_c2, self.curv_c3))
        self.time_log.append(self.t)
        self.sigma_xi_log.append(self.sigma_xi)
        self.sigma_z_log.append(self.sigma_z)
        self.poly_coeffs_log.append(self.poly_coeffs)
        self.clipped_log.append(self.clipped)
        self.end_time = self.t

    def thin_previous(self, tol, max_gap=np.inf):
        """
        Drop the second-newest snapshot P if the newest N and the one before P (Q)
        reproduce it by linear interpolation in t within `tol`, and N - Q <= max_gap.

        Error measure, the max of
          - relative L2 error of the density and of vx (normalized grid),
          - the frame: |d ln sigma_xi|, |d ln sigma_z|, |d tau| sigma_z/sigma_xi,
            |d xi_bar|/sigma_xi, |d z_bar|/sigma_z (each a shift in units of the width).
        O(1): P sits at head-2 of the ring, so it is overwritten by N.
        Returns (dropped, error).
        """
        n = len(self.time_log)
        if n < 3:
            return False, np.inf
        tQ, tP, tN = self.time_log[-3], self.time_log[-2], self.time_log[-1]
        if tN - tQ > max_gap:
            return False, np.inf
        a = (tP - tQ) / (tN - tQ)
        h = self._ring_head
        err = 0.0
        for k in (0, 3):                                   # density, vx
            Fq, Fp, Fn = self._ring[k, h - 3], self._ring[k, h - 2], self._ring[k, h - 1]
            nrm = np.linalg.norm(Fp)
            if nrm > 0:
                err = max(err, np.linalg.norm(Fp - ((1 - a) * Fq + a * Fn)) / nrm)
        (pq, xq, sxq, zq, szq, c2q, c3q), (pp, xp, sxp, zp, szp, c2p, c3p), (pn, xn, sxn, zn, szn, c2n, c3n) = \
            self.DF_log[-3], self.DF_log[-2], self.DF_log[-1]
        lin = lambda q, nn: (1 - a) * q + a * nn
        err = max(err,
                  abs(np.log(sxp) - lin(np.log(sxq), np.log(sxn))),
                  abs(np.log(szp) - lin(np.log(szq), np.log(szn))),
                  abs(pp[0] - lin(pq[0], pn[0])) * szp / sxp,
                  abs(xp - lin(xq, xn)) / sxp,
                  abs(zp - lin(zq, zn)) / szp,
                  abs(c2p - lin(c2q, c2n)), abs(c3p - lin(c3q, c3n)))
        if err >= tol:
            return False, err
        self._ring[:, h - 2] = self._ring[:, h - 1]
        self._ring_head -= 1
        for dq in (self.DF_log, self.time_log, self.sigma_xi_log, self.sigma_z_log,
                   self.poly_coeffs_log, self.clipped_log):
            last = dq.pop(); dq.pop(); dq.append(last)
        return True, err

    def pop_left_DF(self, new_start_time):
        while self.start_time < new_start_time:
            self.DF_log.popleft()
            self._ring_tail += 1        # must stay in lockstep with DF_log
            self.time_log.popleft()
            self.sigma_xi_log.popleft()
            self.sigma_z_log.popleft()
            self.poly_coeffs_log.popleft()
            self.clipped_log.popleft()
            self.start_time = self.time_log[0]

    def append_interpolant(self, formation_length, n_formation_length,
                           min_start_time=None):
        start_point = np.amax(a=(0, self.end_time - n_formation_length * formation_length))
        if min_start_time is not None:
            start_point = min(start_point, float(min_start_time))
        self.pop_left_DF(new_start_time=start_point)
        self.build_interpolant()

    def build_interpolant(self):
        """
        Stack the history. Because the normalized grid never changes there is no
        per-timestep grid metadata and no re-gridding: this permanently retires
        the 'reinterpolate all history when sigma changes by 2x' path.
        """
        n_t = len(self.DF_log)
        times = list(self.time_log)

        self.min_x = times[0]
        self.max_x = times[-1]
        self.delta_x = (self.max_x - self.min_x) / (n_t - 1) if n_t > 1 else 1.0

        # --- retain the ACTUAL snapshot times and build the bracketing table -----------------
        # These used to be thrown away: only times[0], times[-1] and n_t survived, so the
        # interpolators reconstructed the index arithmetically and could only ever work on a
        # uniform grid. A locally refined schedule needs the real times.
        self.t_arr = np.asarray(times, dtype=np.float64)
        assert n_t < 2 or np.all(np.diff(self.t_arr) > 0.0), (
            'snapshot times must be strictly increasing; a repeated time divides by zero in the '
            'non-uniform interpolation weight')
        self.is_uniform = is_uniform_times(self.t_arr)
        if self.is_uniform:
            # keep the uniform fast path exactly as it was -- bit-for-bit
            self.bucket, self.bucket_inv_h, self.bucket_t0, self.bucket_M = empty_bucket()
            self.bucket_worst = 0
        else:
            (self.bucket, self.bucket_inv_h, self.bucket_t0,
             self.bucket_M, self.bucket_worst) = build_bucket(self.t_arr)

        # Zero-copy views into the ring. A slice along the outermost axis of a
        # C-contiguous array is itself C-contiguous, so Numba sees exactly the same
        # array layout it saw when these were freshly stacked every step.
        lo, hi = self._ring_tail, self._ring_head
        assert hi - lo == n_t, (f'ring holds {hi - lo} slices but DF_log has {n_t}; '
                                f'the popleft bookkeeping is out of lockstep')
        self.data_density_interp = self._ring[0, lo:hi]
        self.data_density_u_interp = self._ring[1, lo:hi]
        self.data_density_w_interp = self._ring[2, lo:hi]
        self.data_vx_interp = self._ring[3, lo:hi]
        self.data_vx_u_interp = self._ring[4, lo:hi]

        self.poly_coeffs_interp = np.array([e[0] for e in self.DF_log])
        self.xi_bar_arr = np.array([e[1] for e in self.DF_log])
        self.sigma_xi_arr = np.array([e[2] for e in self.DF_log])
        self.z_bar_arr = np.array([e[3] for e in self.DF_log])
        self.sigma_z_arr = np.array([e[4] for e in self.DF_log])
        self.curv_c2_arr = np.array([e[5] for e in self.DF_log], dtype=np.float64)
        self.curv_c3_arr = np.array([e[6] for e in self.DF_log], dtype=np.float64)
        self._build_moment_arrays()

    def _build_moment_arrays(self):
        """
        The frame re-expressed as a centroid and a second-moment triple, for
        frame_blend = 'moment'.

        DERIVED from the stored frame rather than measured at deposition, so that the
        node values are reproduced EXACTLY: tau = cov/var_z, sqrt(var_z) = sigma_z and
        sqrt(var_x - cov^2/var_z) = sigma_xi all hold identically at a snapshot. That
        exactness is the property that matters -- it guarantees no discontinuity is
        introduced AT snapshot times, whichever blend is selected.
        (These are therefore an effective triple that reproduces the frame, not the
        beam's true moments: sigma_xi is the all-particle spread while the tilt is fit
        on the core, so a directly measured np.cov would not satisfy the identities.)
        """
        if self.poly_degree != 1:
            # frame_blend != 'coeff' is rejected for poly_degree != 1, so these are
            # never read. They must still be float arrays, not None, or numba cannot
            # type the kernel signature.
            z = np.zeros_like(self.sigma_z_arr)
            self.var_z_arr, self.cov_arr, self.var_x_arr = z, z.copy(), z.copy()
            self.x_bar_arr = z.copy()
            return
        tau = self.poly_coeffs_interp[:, 0]
        self.var_z_arr = self.sigma_z_arr ** 2
        self.cov_arr = tau * self.var_z_arr
        self.var_x_arr = self.sigma_xi_arr ** 2 + tau ** 2 * self.var_z_arr
        # Pivot at the centroid. The first moments are linear in the compression
        # parameter and blend exactly; pivoting there also keeps the lever arm
        # (z - z_bar) small, so a residual tau error costs the least displacement.
        self.x_bar_arr = (tau * self.z_bar_arr + self.poly_coeffs_interp[:, -1]
                          + self.xi_bar_arr)

    # ------------------------------------------------------------------
    # Lab-frame support of a snapshot, used by CSR.get_CSR_wake to place the
    # transverse integration nodes on the density rather than on a sigma_x-wide
    # rectangle. Mirrors the metadata DF_tracker_smooth exposes directly.
    # ------------------------------------------------------------------
    def xi_extent(self, k):
        """(xi_lo, xi_hi) of snapshot k's grid, in the tilt-removed frame."""
        return (self.xi_bar_arr[k] + self.u_start * self.sigma_xi_arr[k],
                self.xi_bar_arr[k] + self.u_end * self.sigma_xi_arr[k])

    def z_extent(self, k):
        """(z_lo, z_hi) of snapshot k's grid."""
        return (self.z_bar_arr[k] + self.w_start * self.sigma_z_arr[k],
                self.z_bar_arr[k] + self.w_end * self.sigma_z_arr[k])
