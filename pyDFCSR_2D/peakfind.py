"""
Small helpers on the longitudinal profile, shared by the 1D and 2D codes. A leaf module (numpy
and scipy only), so csr1d (which imports CSR) and CSR/deposit_smooth can both import it.

  peak_fwhm      FWHM of the highest current peak. 1D: CSR_1D.bin_mode 'peak' sets the bin
                 width from it. 2D: only a resolution WARNING, never the frame -- with a double
                 horn the "highest" peak can switch between horns from step to step.
  window_weight  the taper T(z) of the 2D quantile/peak frames: 1 on [lo, hi], C2 smootherstep
                 to 0 across a margin on each side (the two sides may differ).
  peak_center_fwhm  centre (centroid-refined) and FWHM of the highest current peak, for the
                 2D peak frame.
  equidistribute nodes on [a, b] with equal integral of a monitor M between neighbours, M
                 graded so neighbouring spacings differ by at most max_ratio. The 2D wake mesh
                 in mesh_mode 'charge' uses it with M = uniform floor + smoothed current, so
                 every horn or spike gets nodes in proportion to its charge.
"""
import numpy as np


def peak_fwhm(z):
    """
    FWHM of the highest peak of a 1000-bin histogram of the central 99.8% of z, smoothed over
    2 bins (at least 2 bins wide).
    """
    from scipy.ndimage import gaussian_filter1d
    z = np.asarray(z, dtype=np.float64)
    lo, hi = np.quantile(z, [0.001, 0.999])
    h, e = np.histogram(z, bins=1000, range=(lo, hi))
    h = gaussian_filter1d(h.astype(np.float64), 2.0)
    k = int(np.argmax(h))
    above = h >= 0.5 * h[k]
    a = k
    while a > 0 and above[a - 1]:
        a -= 1
    b = k
    while b < h.size - 1 and above[b + 1]:
        b += 1
    return max(b - a + 1, 2) * (e[1] - e[0])


def _smootherstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def window_weight(z, lo, hi, margin, margin_hi=None):
    """
    T(z): 1 on [lo, hi], C2 smootherstep down to 0 at lo - margin and hi + margin_hi
    (margin_hi defaults to margin).
    """
    z = np.asarray(z, dtype=np.float64)
    m_hi = margin if margin_hi is None else margin_hi
    if margin <= 0 or m_hi <= 0:
        return ((z >= lo - max(margin, 0)) & (z <= hi + max(m_hi, 0))).astype(np.float64)
    return _smootherstep((z - (lo - margin)) / margin) * _smootherstep(((hi + m_hi) - z) / m_hi)


def peak_center_fwhm(z, n_hist=2000, n_shift=3, prev=None, switch=1.3):
    """
    Centre and FWHM of the highest current peak, for the 2D peak frame. The histogram arg-max
    (n_hist bins over the central 99.8%, smoothed over 2 bins) only starts the search: the FWHM
    uses linearly interpolated half-maximum crossings, and the centre is then refined as the
    centroid of the particles within +-FWHM/2 (n_shift passes), which is continuous in the particle
    coordinates and not pulled from bin to bin by sub-peaks of similar height.

    prev = (centre, half-width) of the peak followed so far: the search is then restricted to
    within 2 half-widths of it, and moves to another peak only if that one is `switch` times
    higher. Without this, two horns of equal height make the arg-max flip between them from
    snapshot to snapshot.
    """
    from scipy.ndimage import gaussian_filter1d
    z = np.asarray(z, dtype=np.float64)
    lo, hi = np.quantile(z, [0.001, 0.999])
    h, e = np.histogram(z, bins=n_hist, range=(lo, hi))
    h = gaussian_filter1d(h.astype(np.float64), 2.0)
    c = 0.5 * (e[1:] + e[:-1])
    k = int(np.argmax(h))
    if prev is not None:
        near = np.flatnonzero(np.abs(c - prev[0]) <= 2.0 * prev[1])
        if near.size:
            kl = int(near[np.argmax(h[near])])
            if h[k] < switch * h[kl]:
                k = kl
    half = 0.5 * h[k]
    a = k
    while a > 0 and h[a - 1] >= half:
        a -= 1
    b = k
    while b < h.size - 1 and h[b + 1] >= half:
        b += 1
    za = c[a] if a == 0 else np.interp(half, [h[a - 1], h[a]], [c[a - 1], c[a]])
    zb = c[b] if b == h.size - 1 else np.interp(half, [h[b + 1], h[b]], [c[b + 1], c[b]])
    fwhm = max(zb - za, 2.0 * (e[1] - e[0]))
    zc = c[k]
    for _ in range(n_shift):
        m = np.abs(z - zc) < 0.5 * fwhm
        if m.sum() < 10:
            break
        zc = float(z[m].mean())
    return zc, float(fwhm)


def grade_monitor(g, m, n, max_ratio):
    """
    Raise the monitor m (sampled on the uniform grid g) until the equidistributed spacing
    h ~ C/m (C = int m / (n - 1)) changes by at most a factor max_ratio from one node to the
    next. Done in log m: d(ln m)/dz >= -ln(max_ratio) * m / C, by forward and backward sweeps.
    Raising m only adds nodes, so the result is never coarser than the ungraded monitor.
    """
    m = np.asarray(m, dtype=np.float64).copy()
    dg = g[1] - g[0]
    lr = np.log(max_ratio)
    for _ in range(4):                       # C depends on m; a few passes converge
        C = np.trapz(m, g) / max(n - 1, 1)
        lm = np.log(m)
        for i in range(1, m.size):           # left to right: limit the decay
            lm[i] = max(lm[i], lm[i - 1] - lr * dg * m[i - 1] / C)
        for i in range(m.size - 2, -1, -1):  # right to left
            lm[i] = max(lm[i], lm[i + 1] - lr * dg * m[i + 1] / C)
        m_new = np.exp(lm)
        if np.allclose(m_new, m, rtol=1e-6):
            m = m_new
            break
        m = m_new
    return m


def equidistribute(a, b, g, m, n, max_ratio=1.5):
    """
    n nodes from a to b (both included) with equal integral of the monitor m between
    neighbours; m is sampled on the uniform grid g covering [a, b]. Strictly increasing.
    """
    if n < 2 or not b > a:
        raise ValueError(f'equidistribute: need n >= 2 and b > a, got n={n}, [{a}, {b}]')
    m = grade_monitor(g, np.maximum(m, 1e-300), n, max_ratio)
    F = np.concatenate([[0.0], np.cumsum(0.5 * (m[1:] + m[:-1]) * np.diff(g))])
    nodes = np.interp(np.linspace(0.0, F[-1], n), F, g)
    nodes[0], nodes[-1] = a, b
    if not np.all(np.diff(nodes) > 0):
        raise RuntimeError('equidistribute: nodes are not strictly increasing')
    return nodes


def charge_nodes(q, n, uniform_frac=0.3, smooth=None, pad=0.02, max_ratio=1.5, nfine=4000):
    """
    Mesh axis for coordinates q (one per particle): n nodes over [min q, max q] widened by
    pad*span, equidistributing M = uniform_frac/L + (1 - uniform_frac) * lambda_s, where
    lambda_s is the normalised histogram of q smoothed by a Gaussian of rms `smooth`
    (default 2 fine cells). Every particle lies inside the axis.
    """
    from scipy.ndimage import gaussian_filter1d
    q = np.asarray(q, dtype=np.float64)
    q0, q1 = q.min(), q.max()
    span = q1 - q0
    if not span > 0:
        raise ValueError('charge_nodes: all coordinates equal')
    a, b = q0 - pad * span, q1 + pad * span
    edges = np.linspace(a, b, nfine + 1)
    g = 0.5 * (edges[1:] + edges[:-1])
    h, _ = np.histogram(q, bins=edges)
    dg = edges[1] - edges[0]
    sig = 2.0 if smooth is None else max(smooth / dg, 0.0)
    lam = gaussian_filter1d(h.astype(np.float64), sig, mode='constant') if sig > 0 else h.astype(float)
    lam = lam / (lam.sum() * dg)
    L = b - a
    m = uniform_frac / L + (1.0 - uniform_frac) * lam
    g = np.concatenate([[a], g, [b]])
    m = np.concatenate([[m[0]], m, [m[-1]]])
    return equidistribute(a, b, g, m, n, max_ratio)
