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
        # 'K1' is what the lattice YAMLs and get_bmadx_element use (CSR.py:243); 'k1' was
        # the only spelling read here, so every quad looked like a DRIFT to the linear-optics
        # scan and auto sized its steps from the wrong optics. Accept both.
        k1 = (float(el.get('K1', el.get('k1', 0.0)))
              if etype in ('quad', 'quadrupole') else 0.0)
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



def propagate_frame(lattice_config, sigma0, n_sub=400):
    """
    The full co-moving frame along the lattice: sigma_z, sigma_x, tau and sigma_xi.

    propagate_var_z returns only var_z, which is blind to the transverse frame. In a DRIFT
    sigma_z is exactly constant while sigma_xi can be doing anything, because var_x grows
    quadratically in s and var_z does not -- so a step criterion built on sigma_z alone will
    step straight over a transverse waist. The interpolant blends all of
    (tau, sigma_z, sigma_xi, centroids), so the schedule has to see all of them.

    tau = cov_zx/var_z and sigma_xi = sqrt(var_x - cov^2/var_z), the same definitions the
    deposition uses, so the schedule and the interpolant are talking about the same frame.
    """
    M = np.eye(6)
    keys = [k for k in lattice_config if k != 'step_size']
    s_all, sz, sx, tau, sxi, rho_all = [], [], [], [], [], []

    def push(s_pos, g, rho):
        vz, vx, cov = g[4, 4], g[0, 0], g[4, 0]
        s_all.append(s_pos)
        sz.append(np.sqrt(max(vz, 0.0)))
        sx.append(np.sqrt(max(vx, 0.0)))
        tau.append(cov / vz if vz > 0 else 0.0)
        sxi.append(np.sqrt(max(vx - (cov ** 2) / vz, 0.0)) if vz > 0 else 0.0)
        rho_all.append(rho)

    push(0.0, sigma0, 0.0)
    s0 = 0.0
    for key in keys:
        el = lattice_config[key]
        L = float(el['L'])
        if L <= 0:
            continue
        etype = el.get('type', 'drift')
        angle = float(el.get('angle', 0.0)) if etype == 'dipole' else 0.0
        # 'K1' is what the lattice YAMLs and get_bmadx_element use (CSR.py:243); 'k1' was
        # the only spelling read here, so every quad looked like a DRIFT to the linear-optics
        # scan and auto sized its steps from the wrong optics. Accept both.
        k1 = (float(el.get('K1', el.get('k1', 0.0)))
              if etype in ('quad', 'quadrupole') else 0.0)
        rho = angle / L if L else 0.0
        R_slice = r_gen6(L=L / n_sub, angle=angle / n_sub, k1=k1)
        for _ in range(n_sub):
            M = R_slice @ M
            s0 += L / n_sub
            push(s0, M @ sigma0 @ M.T, rho)

    return (np.array(s_all), np.array(sz), np.array(sx), np.array(tau),
            np.array(sxi), np.array(rho_all))


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



def formation_length_profile(lattice_config, s, sz, rho):
    """
    L_f(s), mirroring CSR2D's three regimes exactly.

    Shared by `retention_schedule` and the `auto` step scheduler, because a disagreement between
    the retention floor and the step schedule about where the formation length is long would be
    silent and confusing. The three regimes must match CSR.py or both are wrong:

      before any bend       formation_length accumulates ELEMENT lengths at element entry
                            (CSR.py:336), so it is constant within an element
      inside a bend         (24 R^2 * 5 sigma_z)^(1/3)                 (CSR.py:157, 175)
      drift after a bend    the same expression with the LAST bend's R (CSR.py:179)

    Note the factor 5 on sigma_z; dropping it understates L_f by 5^(1/3) = 1.71.
    """
    keys = [k for k in lattice_config if k != 'step_size']
    lengths = np.array([float(lattice_config[k]['L']) for k in keys])
    edges = np.cumsum(lengths)
    is_bend = np.array([lattice_config[k].get('type') == 'dipole'
                        and float(lattice_config[k].get('angle', 0.0)) != 0.0 for k in keys])
    angles = np.array([float(lattice_config[k].get('angle', 1.0)) or 1.0 for k in keys])
    R_el = np.where(is_bend, np.abs(lengths / angles), np.nan)
    first_bend = int(np.argmax(is_bend)) if is_bend.any() else len(keys)
    predrift = np.cumsum(lengths)

    out = np.empty_like(np.asarray(s, dtype=float))
    R_rec = np.nan
    for i, si in enumerate(np.asarray(s, dtype=float)):
        j = min(int(np.searchsorted(edges, si, side='left')), len(keys) - 1)
        if is_bend[j]:
            R_rec = R_el[j]
        if j < first_bend and np.isnan(R_rec):
            out[i] = predrift[j]
        else:
            out[i] = (24.0 * R_rec ** 2 * 5.0 * max(float(sz[i]), 1e-30)) ** (1.0 / 3.0)
    return out


class RetentionPlan:
    """
    How far back the density history must reach at every point in the lattice,
    precomputed from linear optics before tracking starts.

    The problem this solves. The history is truncated to
    [end_time - n_fl*L_f, end_time] by pop_left_DF, which only ever POPS. With
    L_f = (24 R^2 * 5 sigma_z)^(1/3) that is safe exactly as long as sigma_z decreases
    monotonically, because then the required window shrinks monotonically too -- the
    assumption the original design was built on. At a waist sigma_z decreases and then
    INCREASES, L_f collapses and recovers with it, and the required start time moves
    back to where snapshots have already been discarded. Measured at shear 20: the
    history was cut to start at 0.2996 at the waist, and five steps later the
    integration wanted 0.1773, a 0.123 m shortfall that interp3D silently absorbed by
    clamping to the oldest surviving snapshot.

    The fix. The quantity that must be retained at s is not start_point(s) but

        floor(s) = min over all s' >= s of start_point(s')

    the oldest time any FUTURE step will still ask for. That is a suffix minimum over
    the whole lattice, so it needs sigma_z(s) in advance -- which propagate_var_z
    already supplies. floor is non-decreasing in s by construction, so retaining back
    to it is both correct (nothing needed later is discarded) and minimal (nothing is
    kept that will not be used).

    Accuracy. This is linear optics: no CSR, no space charge. L_f ~ sigma_z^(1/3) is
    forgiving -- a factor 2 error in sigma_z is only 26% in L_f -- and `safety` widens
    L_f further. It is still a prediction, so CSR2D also checks the ACTUAL requirement
    against what was retained at every step and reports any shortfall; this must not
    become a silent assumption.
    """

    def __init__(self, s, L_f, start_point, floor, n_formation_length, safety):
        self.s = s
        self.L_f = L_f
        self.start_point = start_point
        self.floor = floor
        self.n_formation_length = n_formation_length
        self.safety = safety
        self.max_depth = float(np.max(s - floor))

    def floor_at(self, s_now):
        """
        Retention floor at s_now, taken at the grid point at or below it.

        floor is non-decreasing, so using the left node returns a value <= floor(s_now)
        and therefore retains slightly MORE than strictly necessary. Interpolating
        would be tighter and could round the wrong way.
        """
        i = int(np.searchsorted(self.s, s_now, side='right')) - 1
        if i < 0:
            return 0.0
        return float(self.floor[min(i, len(self.floor) - 1)])

    def max_snapshots(self, step_size, pad=8):
        return int(np.ceil(self.max_depth / step_size)) + pad


def retention_schedule(lattice_config, sigma0, n_formation_length, safety=1.25,
                       n_sub=2000):
    """
    Build a RetentionPlan by mirroring CSR2D's own formation-length regimes.

    Three regimes, and they must match CSR.py exactly or the floor is wrong:
      * before any bend  -- formation_length accumulates ELEMENT lengths at element
                            entry (CSR.py:336), so it is constant within an element
      * inside a bend    -- (24 R^2 * 5 sigma_z)^(1/3)          (CSR.py:157, 175)
      * drift after a bend -- same expression with the LAST bend's R (CSR.py:179)
    Note the factor 5 on sigma_z; dropping it would understate L_f by 5^(1/3) = 1.71.
    """
    s, vz, rho = propagate_var_z(lattice_config, sigma0, n_sub=n_sub)
    sz = np.sqrt(np.maximum(vz, 0.0))

    keys = list(lattice_config.keys())[1:]
    lengths = np.array([float(lattice_config[k]['L']) for k in keys])
    edges = np.cumsum(lengths)
    is_bend = np.array([lattice_config[k].get('type') == 'dipole'
                        and float(lattice_config[k].get('angle', 0.0)) != 0.0
                        for k in keys])
    # R per element, and the cumulative pre-first-bend drift length at each element
    R_el = np.where(is_bend,
                    np.abs(lengths / np.where(is_bend, np.array(
                        [float(lattice_config[k].get('angle', 1.0)) for k in keys]),
                        1.0)),
                    np.nan)
    first_bend = int(np.argmax(is_bend)) if is_bend.any() else len(keys)
    predrift = np.cumsum(lengths)

    L_f = formation_length_profile(lattice_config, s, sz, rho)

    start_point = np.maximum(0.0, s - n_formation_length * safety * L_f)
    floor = np.minimum.accumulate(start_point[::-1])[::-1]   # suffix minimum
    return RetentionPlan(s, L_f, start_point, floor, n_formation_length, safety)


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
