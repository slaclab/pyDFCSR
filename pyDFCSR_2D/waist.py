"""
Predict longitudinal waists from linear optics, BEFORE tracking starts.

Why this exists. The co-moving interpolant reconstructs the retarded density by blending
the stored frame between snapshots. Across a longitudinal waist that blend fails, and it
fails silently: the bracketing snapshots carry almost no signal about the dip between
them (progress 6n measured sigma_z = 50.00 um and 50.09 um at the endpoints of an
interval whose true minimum is 2.51 um). Refining the s' quadrature does not help --
that alternative was tested and refuted in 6v, 289x more nodes changing the wake 3.4%
and leaving its roughness untouched. Only sampling helps, and the user cannot know to
refine step_size because nothing in the output says a waist was crossed.

So: propagate the beam's second moments through the lattice with linear transport before
any particle is tracked, find the minima of var_z(s), and say which of them step_size
cannot resolve.

Deliberately a WARNING, not a correction. Two options were rejected:

  * silently refining step_size -- changes the kick cadence and cost without being asked
  * falling back to the nearest snapshot at a waist -- 6n's plan notes this manufactures
    the very symptom being diagnosed: a step of the full inter-snapshot frame change at
    fixed alpha, i.e. hundreds of band half-widths, once per snapshot interval, inside
    the near region, and it puts a jump inside the _retarded_xi_bands fixed point.

Accuracy caveat. This is LINEAR optics on the design lattice: no CSR, no space charge, no
fringe-field detail beyond what r_gen6 models, and r_gen6's (x, x', y, y', z, dp/p)
is identified with Bmad-X's (x, px, y, py, z, pz), which agrees only to first order. It
is a diagnostic that says "look here", not a prediction of the wake.
"""
import numpy as np

from .r_gen6 import r_gen6


def sigma_from_coords(x, px, y, py, z, pz):
    """6x6 second-moment matrix of a particle distribution, centroids removed."""
    return np.cov(np.vstack((x, px, y, py, z, pz)))


def propagate_var_z(lattice_config, sigma0, n_sub=2000):
    """
    var_z(s) through the lattice under linear transport.

    Returns (s, var_z, rho) sampled n_sub times per element. Each element is sliced and
    the slice matrices accumulated, so a waist INSIDE an element is seen -- which is the
    whole point, since that is where they occur.

    rho is the slice curvature (1/R), carried so the caller can report the waist's
    location relative to the bends and estimate its width.
    """
    M = np.eye(6)
    s_all, vz_all, rho_all = [0.0], [float(sigma0[4, 4])], [0.0]
    s0 = 0.0

    for key in list(lattice_config.keys())[1:]:
        el = lattice_config[key]
        L = float(el['L'])
        if L <= 0:
            continue
        etype = el.get('type', 'drift')
        angle = float(el.get('angle', 0.0)) if etype == 'dipole' else 0.0
        k1 = float(el.get('k1', 0.0)) if etype in ('quad', 'quadrupole') else 0.0
        rho = angle / L if L else 0.0

        dL = L / n_sub
        dang = angle / n_sub
        # E1/E2 are deliberately not applied: pole-face rotation is a thin x-x' kick,
        # and var_z is driven by R51/R52/R56, so it does not enter at this order. Adding
        # it would only imply a precision this scan does not have.
        R_slice = r_gen6(L=dL, angle=dang, k1=k1)
        for _ in range(n_sub):
            M = R_slice @ M
            sig = M @ sigma0 @ M.T
            s0 += dL
            s_all.append(s0)
            vz_all.append(float(sig[4, 4]))
            rho_all.append(rho)

    return np.array(s_all), np.array(vz_all), np.array(rho_all)


def find_waists(s, var_z, rho, sigma_x_of_s=None, rel_depth=0.5):
    """
    Interior local minima of var_z(s).

    rel_depth filters out numerical wiggles: a minimum is kept only if var_z rises by
    at least this fraction of its value on BOTH sides before the next turning point.
    Without it, a monotone-ish var_z with round-off ripple produces dozens of "waists".
    """
    sz = np.sqrt(np.maximum(var_z, 0.0))
    out = []
    for i in range(1, len(sz) - 1):
        if not (sz[i] <= sz[i - 1] and sz[i] <= sz[i + 1]):
            continue
        if sz[i] <= 0:
            continue
        # rise on each side, up to the array ends
        left = sz[:i].max() if i > 0 else sz[i]
        right = sz[i + 1:].max() if i + 1 < len(sz) else sz[i]
        if (left - sz[i]) < rel_depth * sz[i] or (right - sz[i]) < rel_depth * sz[i]:
            continue
        # keep only the deepest point of a flat-ish basin: skip if a neighbour is lower
        out.append(dict(index=i, s=float(s[i]), sigma_z=float(sz[i]),
                        rho=float(rho[i])))
    # collapse adjacent duplicates from a flat minimum
    dedup = []
    for w in out:
        if dedup and abs(w['s'] - dedup[-1]['s']) < 1e-9:
            continue
        if dedup and w['sigma_z'] >= dedup[-1]['sigma_z'] and \
                abs(w['index'] - dedup[-1]['index']) <= 2:
            continue
        dedup.append(w)
    return dedup


def waist_width(s, var_z, i):
    """
    Half-width in s over which sigma_z grows by sqrt(2) from the minimum.

    This is the scale the frame blend has to resolve. Near a waist
    sigma_z(s)^2 ~ sigma_min^2 + (d sigma_z/ds)^2 (s - s_w)^2, so the sqrt(2) point is
    one "Rayleigh range" of the longitudinal focus. Measured from the propagated curve
    rather than from a formula, so it needs no assumption about what drives the
    compression.

    The crossing is INTERPOLATED, not snapped to a grid node. Without that the width is
    floor-limited by the scan spacing: at n_sub = 200 over a 1 m dipole the grid is 5 mm,
    and both the shear-20 and shear-50 waists reported exactly 5.0000 mm when their true
    widths are 2.5 mm and 0.40 mm. That error is in the dangerous direction -- it makes a
    narrow waist look wider, so the warning understates the problem.
    """
    sz = np.sqrt(np.maximum(var_z, 0.0))
    target = np.sqrt(2.0) * sz[i]

    def cross(idx_iter):
        prev = i
        for j in idx_iter:
            if sz[j] >= target:
                # linear interpolation of sigma_z between the bracketing nodes
                f = ((target - sz[prev]) / (sz[j] - sz[prev])
                     if sz[j] != sz[prev] else 0.0)
                return abs(s[prev] + f * (s[j] - s[prev]) - s[i])
            prev = j
        return np.inf

    w = min(cross(range(i + 1, len(sz))), cross(range(i - 1, -1, -1)))
    return float(w) if np.isfinite(w) else np.inf


def scan_waists(lattice_config, sigma0, step_size, min_steps=2.0, n_sub=2000):
    """
    Full report: every predicted waist, its width, and whether step_size resolves it.

    min_steps is how many tracking steps must span the waist half-width for the frame
    blend to have a chance. CALIBRATED, in progress 6x, against wake error at a shear-20
    waist of width 2.39 mm:

        steps across   0.05    0.19    0.80    2.39    3.99
        wake rel L2    0.790   0.440   0.017   0.007   0 (reference)

    so the error collapses between 0.2 and 0.8 steps across and is already ~1% at 0.8.
    2.0 puts the threshold just past the knee: it still warns at 0.8 steps (1.7% error,
    worth a word) and stays quiet at 2.4 (0.7%). The first version used 4.0, which called
    a converged run unresolved -- a false alarm, and 6t is a reminder that a warning about
    a problem that is not happening costs real credibility.

    Calibrated on ONE waist, so it is a defensible default rather than a law; the report
    always prints `steps across` so a reader can apply their own threshold.
    """
    s, vz, rho = propagate_var_z(lattice_config, sigma0, n_sub=n_sub)
    waists = find_waists(s, vz, rho)
    for w in waists:
        w['width'] = waist_width(s, vz, w['index'])
        w['steps_across'] = w['width'] / step_size if step_size > 0 else np.inf
        w['resolved'] = w['steps_across'] >= min_steps
        w['sigma_z_start'] = float(np.sqrt(max(vz[0], 0.0)))
        w['compression'] = (w['sigma_z_start'] / w['sigma_z']
                            if w['sigma_z'] > 0 else np.inf)
    return dict(s=s, var_z=vz, waists=waists, step_size=step_size,
                min_steps=min_steps)


def format_report(report, max_lines=8):
    """Human-readable warning block, or '' when there is nothing to say."""
    ws = report['waists']
    if not ws:
        return ''
    bad = [w for w in ws if not w['resolved']]
    lines = []
    lines.append('  [waist scan] linear optics predicts %d longitudinal waist(s) '
                 'in this lattice' % len(ws))
    lines.append('  %10s %12s %12s %12s %11s %9s'
                 % ('s (m)', 'sigma_z (um)', 'compress', 'width (mm)',
                    'steps across', 'resolved'))
    for w in ws[:max_lines]:
        lines.append('  %10.4f %12.3f %12.1fx %12.4f %11.2f %9s'
                     % (w['s'], w['sigma_z'] * 1e6, w['compression'],
                        w['width'] * 1e3, w['steps_across'],
                        'yes' if w['resolved'] else 'NO'))
    if len(ws) > max_lines:
        lines.append('  ... %d more' % (len(ws) - max_lines))
    if bad:
        lines.append('  WARNING: %d waist(s) are NOT resolved by step_size = %g m.'
                     % (len(bad), report['step_size']))
        need = min(w['width'] / report['min_steps'] for w in bad)
        lines.append('  The co-moving frame blend cannot reconstruct the retarded '
                     'density across an')
        lines.append('  unresolved waist, and the endpoints carry almost no signal '
                     'that one was crossed,')
        lines.append('  so the wake near it will be wrong without looking wrong. '
                     'step_size <= %.4g m' % need)
        lines.append('  would put %g steps across the narrowest one.'
                     % report['min_steps'])
    return '\n'.join(lines)
