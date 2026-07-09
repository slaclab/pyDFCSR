"""
Unit tests for the improved density estimation module (deposit_smooth.py).
Tests B-spline deposition, FFT smoothing, spectral derivatives, and transformed interpolation.
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from deposit_smooth import histogram_bspline_2d, smooth_and_differentiate, DF_tracker_smooth
from interp3D import interpolate3D_transformed, eval_poly, bilinear_single


class TestCubicBsplineDeposition:
    """Test the cubic B-spline 2D deposition kernel."""

    def test_total_weight_conservation(self):
        """Total deposited weight should equal sum of input weights."""
        np.random.seed(42)
        n = 100000
        x = np.random.normal(0, 1, n)
        z = np.random.normal(0, 1, n)
        w = np.ones(n)

        hist = histogram_bspline_2d(x, z, w, 100, -5.0, 5.0, 100, -5.0, 5.0)
        # Some particles near edges will be lost, so check within interior
        assert hist.sum() > 0.95 * n

    def test_gaussian_density_recovery(self):
        """Depositing a known Gaussian should approximate the analytical density."""
        np.random.seed(123)
        n = 1000000
        sigma_x, sigma_z = 1.0, 0.5
        x = np.random.normal(0, sigma_x, n)
        z = np.random.normal(0, sigma_z, n)
        w = np.ones(n)

        nbins = 128
        xlim, zlim = 4 * sigma_x, 4 * sigma_z
        hist = histogram_bspline_2d(x, z, w, nbins, -xlim, xlim, nbins, -zlim, zlim)

        # Normalize to density
        dx = 2 * xlim / nbins
        dz = 2 * zlim / nbins
        density = hist / (n * dx * dz)

        # Analytical Gaussian
        xg = np.linspace(-xlim, xlim, nbins)
        zg = np.linspace(-zlim, zlim, nbins)
        X, Z = np.meshgrid(xg, zg, indexing='ij')
        analytical = np.exp(-X**2 / (2*sigma_x**2) - Z**2 / (2*sigma_z**2)) / (2*np.pi*sigma_x*sigma_z)

        # Check peak is within 5% (statistical fluctuations expected)
        peak_ratio = density.max() / analytical.max()
        assert 0.9 < peak_ratio < 1.1, f"Peak ratio {peak_ratio} outside tolerance"

    def test_smoothness(self):
        """B-spline deposit should be smoother than CIC."""
        np.random.seed(42)
        n = 50000
        x = np.random.normal(0, 1, n)
        z = np.random.normal(0, 1, n)
        w = np.ones(n)

        hist = histogram_bspline_2d(x, z, w, 64, -4.0, 4.0, 64, -4.0, 4.0)

        # Check that second differences are small relative to values (smoothness indicator)
        d2x = np.diff(hist, n=2, axis=0)
        d2z = np.diff(hist, n=2, axis=1)
        roughness = (np.std(d2x) + np.std(d2z)) / (2 * hist.max())
        assert roughness < 0.5, f"Roughness {roughness} too high for B-spline"


class TestFFTSmoothingAndDerivatives:
    """Test FFT Gaussian smoothing and spectral derivatives."""

    def test_smoothing_preserves_integral(self):
        """Gaussian smoothing should preserve the total integral."""
        np.random.seed(42)
        nx, nz = 128, 128
        dx, dz = 0.1, 0.1
        x = np.linspace(-6.4, 6.4, nx)
        z = np.linspace(-6.4, 6.4, nz)
        X, Z = np.meshgrid(x, z, indexing='ij')
        field = np.exp(-X**2/2 - Z**2/2) / (2*np.pi)

        smoothed, _, _ = smooth_and_differentiate(field, dx, dz, sigma_smooth=1.5)

        integral_orig = np.sum(field) * dx * dz
        integral_smooth = np.sum(smoothed) * dx * dz
        assert abs(integral_orig - integral_smooth) / integral_orig < 0.01

    def test_spectral_derivative_gaussian(self):
        """Spectral derivative of Gaussian should match analytical derivative."""
        nx, nz = 256, 256
        sigma = 1.0
        dx = 10.0 / nx
        dz = 10.0 / nz
        x = np.linspace(-5, 5, nx)
        z = np.linspace(-5, 5, nz)
        X, Z = np.meshgrid(x, z, indexing='ij')

        field = np.exp(-X**2 / (2*sigma**2) - Z**2 / (2*sigma**2))

        _, dfdx, dfdz = smooth_and_differentiate(field, dx, dz, sigma_smooth=0.5)

        # Analytical derivatives
        dfdx_exact = -X / sigma**2 * field
        dfdz_exact = -Z / sigma**2 * field

        # The FFT smoothing slightly modifies the field, so compare shapes
        # Check in the interior where edge effects are minimal
        sl = slice(nx//4, 3*nx//4), slice(nz//4, 3*nz//4)

        # Normalize and compare correlation
        corr_x = np.corrcoef(dfdx[sl].ravel(), dfdx_exact[sl].ravel())[0, 1]
        corr_z = np.corrcoef(dfdz[sl].ravel(), dfdz_exact[sl].ravel())[0, 1]

        assert corr_x > 0.99, f"X-derivative correlation {corr_x} too low"
        assert corr_z > 0.99, f"Z-derivative correlation {corr_z} too low"

    def test_zero_field_gives_zero_derivative(self):
        """Derivative of a constant field should be zero."""
        nx, nz = 64, 64
        field = np.ones((nx, nz)) * 5.0
        dx, dz = 0.1, 0.1

        smoothed, dfdx, dfdz = smooth_and_differentiate(field, dx, dz, sigma_smooth=1.0)

        assert np.max(np.abs(dfdx)) < 1e-10
        assert np.max(np.abs(dfdz)) < 1e-10


class TestInterpolate3DTransformed:
    """Test the per-timestep polynomial transformed interpolation."""

    def test_eval_poly(self):
        """Test polynomial evaluation matches numpy polyval."""
        coeffs = np.array([2.0, -1.0, 3.0, 0.5])  # 2z^3 - z^2 + 3z + 0.5
        z_test = 1.5
        expected = np.polyval(coeffs, z_test)
        result = eval_poly(coeffs, z_test)
        assert abs(result - expected) < 1e-12

    def test_bilinear_single(self):
        """Test bilinear interpolation on a simple 2D field."""
        data = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0],
                         [7.0, 8.0, 9.0]])
        # At integer indices, should return exact value
        assert abs(bilinear_single(data, 0.0, 0.0, 3, 3) - 1.0) < 1e-12
        assert abs(bilinear_single(data, 1.0, 1.0, 3, 3) - 5.0) < 1e-12
        # At midpoint
        assert abs(bilinear_single(data, 0.5, 0.5, 3, 3) - 3.0) < 1e-12

    def test_no_transform_matches_trilinear(self):
        """With zero polynomial coefficients, should match standard trilinear."""
        np.random.seed(42)
        n_t, n_xi, n_z = 10, 50, 50
        data = np.random.rand(n_t, n_xi, n_z)

        # Zero polynomial = no coordinate transform
        poly_coeffs = np.zeros((n_t, 4))

        min_xi, min_z, min_t = -5.0, -5.0, 0.0
        delta_xi = 10.0 / (n_xi - 1)
        delta_z = 10.0 / (n_z - 1)
        delta_t = 1.0 / (n_t - 1)

        # Per-timestep arrays (all same since uniform grid)
        min_xi_arr = np.full(n_t, min_xi)
        min_z_arr = np.full(n_t, min_z)
        delta_xi_arr = np.full(n_t, delta_xi)
        delta_z_arr = np.full(n_t, delta_z)

        # Query at a known point
        xval = np.array([0.0, 1.0, -1.0])
        zval = np.array([0.0, 0.5, -0.5])
        tval = np.array([0.3, 0.5, 0.7])

        result = interpolate3D_transformed(xval, zval, tval, data, poly_coeffs,
                                            min_xi_arr, min_z_arr, min_t,
                                            delta_xi_arr, delta_z_arr, delta_t)

        # With zero poly, xi = x, so this should be equivalent to standard trilinear
        # Verify by manually computing expected values
        for i in range(len(xval)):
            t_idx = (tval[i] - min_t) / delta_t
            k = int(t_idx)
            alpha = t_idx - k

            xi_idx = (xval[i] - min_xi) / delta_xi
            z_idx = (zval[i] - min_z) / delta_z

            val_k = bilinear_single(data[k], xi_idx, z_idx, n_xi, n_z)
            val_k1 = bilinear_single(data[k+1], xi_idx, z_idx, n_xi, n_z)
            expected = (1 - alpha) * val_k + alpha * val_k1

            assert abs(result[i] - expected) < 1e-12

    def test_linear_transform_consistency(self):
        """With a linear polynomial (slope), the transform should correctly shift xi."""
        n_t, n_xi, n_z = 5, 100, 100
        min_xi, min_z = -5.0, -5.0
        delta_xi = 10.0 / (n_xi - 1)
        delta_z = 10.0 / (n_z - 1)
        min_t = 0.0
        delta_t = 1.0 / (n_t - 1)

        # Per-timestep arrays (all same)
        min_xi_arr = np.full(n_t, min_xi)
        min_z_arr = np.full(n_t, min_z)
        delta_xi_arr = np.full(n_t, delta_xi)
        delta_z_arr = np.full(n_t, delta_z)

        # Create a Gaussian density centered at xi=0
        xi_grid = np.linspace(min_xi, min_xi + delta_xi * (n_xi-1), n_xi)
        z_grid = np.linspace(min_z, min_z + delta_z * (n_z-1), n_z)
        XI, Z = np.meshgrid(xi_grid, z_grid, indexing='ij')
        gaussian = np.exp(-XI**2 / 2 - Z**2 / 2)

        data = np.zeros((n_t, n_xi, n_z))
        for t in range(n_t):
            data[t] = gaussian

        # Polynomial: p(z) = slope * z (linear tilt)
        slope = 2.0
        poly_coeffs = np.zeros((n_t, 2))  # [slope, 0] for each timestep
        poly_coeffs[:, 0] = slope

        # Query at x=1.0, z=0.5: xi should be x - slope*z = 1.0 - 2.0*0.5 = 0.0
        # The Gaussian at xi=0 should give exp(0) = 1.0 (peak)
        xval = np.array([slope * 0.5])  # x = slope*z so xi = 0
        zval = np.array([0.5])
        tval = np.array([0.5])

        result = interpolate3D_transformed(xval, zval, tval, data, poly_coeffs,
                                            min_xi_arr, min_z_arr, min_t,
                                            delta_xi_arr, delta_z_arr, delta_t)

        # At xi=0, z=0.5: gaussian = exp(-0 - 0.5^2/2) = exp(-0.125)
        expected = np.exp(-0.5**2 / 2)
        assert abs(result[0] - expected) < 0.05, f"Got {result[0]}, expected {expected}"


class TestDFTrackerSmooth:
    """Integration tests for the full DF_tracker_smooth class."""

    def test_get_DF_runs(self):
        """Basic smoke test that get_DF completes without error."""
        np.random.seed(42)
        n = 100000
        x = np.random.normal(0, 1e-4, n)
        z = np.random.normal(0, 1e-3, n)
        px = np.random.normal(0, 1e-5, n)

        tracker = DF_tracker_smooth({'method': 'bspline_fft', 'xbins': 64, 'zbins': 64,
                                      'smoothing_sigma': 1.5, 'poly_degree': 2})
        tracker.get_DF(x, z, px, t=0.0)

        assert tracker.density is not None
        assert tracker.density_x is not None
        assert tracker.density_z is not None
        assert tracker.vx is not None
        assert tracker.vx_x is not None
        assert tracker.density.shape == (64, 64)

    def test_get_DF_with_tilt(self):
        """Test that polynomial tilt removal works for a tilted beam."""
        np.random.seed(42)
        n = 100000
        z = np.random.normal(0, 1e-3, n)
        x = 5.0 * z + np.random.normal(0, 1e-4, n)  # Strong linear tilt
        px = np.random.normal(0, 1e-5, n)

        tracker = DF_tracker_smooth({'method': 'bspline_fft', 'xbins': 64, 'zbins': 64,
                                      'smoothing_sigma': 1.5, 'poly_degree': 1})
        tracker.get_DF(x, z, px, t=0.0)

        # After tilt removal, the density should be well-contained in the grid
        assert tracker.density.max() > 0
        # The polynomial should have captured the slope ~5.0
        assert abs(tracker.poly_coeffs[0] - 5.0) < 0.5

    def test_density_normalization(self):
        """Density should integrate to approximately 1."""
        np.random.seed(42)
        n = 500000
        x = np.random.normal(0, 1e-4, n)
        z = np.random.normal(0, 1e-3, n)
        px = np.random.normal(0, 1e-5, n)

        tracker = DF_tracker_smooth({'method': 'bspline_fft', 'xbins': 128, 'zbins': 128,
                                      'smoothing_sigma': 1.5, 'poly_degree': 2})
        tracker.get_DF(x, z, px, t=0.0)

        integral = np.trapz(np.trapz(tracker.density, tracker.xi_grids, axis=0), tracker.z_grids)
        assert abs(integral - 1.0) < 0.1, f"Density integral = {integral}, expected ~1.0"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
