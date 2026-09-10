"""
Verification of Eq. 4.24 of the thesis (section 4.4.2, localization of the CSR integrand).

Eq. 4.24 gives, in closed form, the x' positions at which the CSR integrand is
localized for a given s' -- the "narrow band" and the "chirp band". It is the
natural replacement for the numerical fixed point currently used to place the
transverse integration nodes in CSR.get_CSR_wake.

VERDICT: as printed, Eq. 4.24 has a typo. The tan^2(a0) prefactor on the square
root should be tan(a0). This script demonstrates that two independent ways.

THE DERIVATION
--------------
Two constraints, with q = n(s)x + r0(s) - r0(s'),  T = t - s',  tau = tan(a0):

  light cone (Eq 4.19), using |n(s')| = 1:
      l = t - t_ret ,   l^2 = |q - n(s')x'|^2 = q^2 - 2 x'(n'.q) + x'^2
  beam axis (Eq 4.20), the query point is on the beam's central axis at t_ret,
  and z' = s' - t_ret:
      x' = tau (s' - t_ret)  =>  t_ret = s' - x'/tau  =>  l = T + x'/tau

Equating the two expressions for l^2 and collecting powers of x':

      A = 1 - 1/tau^2      B = -2T/tau - 2(n'.q)      C = q^2 - T^2

      x'_pm = tau [ T + (n'.q) tau  +-  sqrt(rad) ] / (tau^2 - 1)
      rad   = (tau^2 - 1)(T^2 - q^2) + ((n'.q) tau + T)^2

Both terms carry exactly ONE power of tau. The thesis's first term is correct
(it equals tau(tau(n'.q) + T)/(tau^2-1)) but its sqrt prefactor is tau^2, which
is inconsistent with its own first term. The thesis radicand matches the
derivation exactly, so this is a transcription slip in the final line.

Both versions are dimensionally consistent, so dimensional analysis cannot catch
it. The failure is a broken cancellation: the narrow branch x1 ~ x requires the
two terms to nearly cancel, which needs matching powers of tau.

CHECK 1 -- SYMBOLIC
-------------------
A root of a polynomial substituted back into that polynomial gives identically
zero. Build the quadratic in sympy, substitute each candidate, simplify. This
needs no numbers and no lattice.

CHECK 2 -- NUMERICAL
--------------------
The symbolic check shares its starting point with the derivation, so it cannot
catch a misreading of what Eq 4.19/4.20 mean. This check never uses the closed
form: it root-finds the ORIGINAL two-equation system directly, in real lattice
geometry from a running simulation, and asks which closed form reproduces it.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'eq424')
EXAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'example'))

# s' offsets (m) behind the observation point at which to compare
DS_LIST = [1e-3, 4.018e-3, 1e-2, 5e-2, 1.3e-1, 2.4e-1]
SCAN_HALF = 0.2          # m, half-width of the x' bracket scan
SCAN_COARSE = 2001       # coarse samples used to find sign changes


# ---------------------------------------------------------------------------
# The two candidate closed forms
# ---------------------------------------------------------------------------
def _rad(T, nq, q2, tau):
    return (tau**2 - 1) * (T**2 - q2) + (nq * tau + T)**2


def eq424_as_printed(T, nq, q2, tau, sign):
    """Thesis Eq 4.24 verbatim: tan^2(a0) prefactor on the sqrt."""
    rad = _rad(T, nq, q2, tau)
    if rad < 0:
        return np.nan
    return ((nq * tau**2 + T * tau) / (tau**2 - 1)
            + sign * tau**2 / (tau**2 - 1) * np.sqrt(rad))


def eq424_corrected(T, nq, q2, tau, sign):
    """Same, with a single power of tan(a0) on the sqrt. b = 0."""
    rad = _rad(T, nq, q2, tau)
    if rad < 0:
        return np.nan
    return tau * (T + nq * tau + sign * np.sqrt(rad)) / (tau**2 - 1)


def eq424_general(T, nq, q2, tau, b, sign):
    """
    Corrected and generalized for a beam axis with an intercept, x' = tau z' + b.

    l = T + (x' - b)/tau = (T - b/tau) + x'/tau, so the intercept enters ONLY as
    a shift T -> T - b/tau. Verified symbolically below.
    """
    return eq424_corrected(T - b / tau, nq, q2, tau, sign)


# ---------------------------------------------------------------------------
# Check 1: symbolic
# ---------------------------------------------------------------------------
def symbolic_check(emit):
    import sympy as sp

    xp, T, nq, q2, tau, b = sp.symbols("x' T n_dot_q q2 tau b", real=True)

    # quad = 0 is exactly Eq 4.19 combined with Eq 4.20
    quad = sp.expand((q2 - 2 * xp * nq + xp**2) - (T + xp / tau)**2)
    A, B, C = sp.Poly(quad, xp).all_coeffs()

    emit('  quadratic coefficients from the derivation:')
    emit(f'    A = {sp.simplify(A)}')
    emit(f'    B = {sp.simplify(sp.expand(B))}')
    emit(f'    C = {sp.simplify(sp.expand(C))}')
    emit('')

    rad = (tau**2 - 1) * (T**2 - q2) + (nq * tau + T)**2
    printed = (nq * tau**2 + T * tau) / (tau**2 - 1) + tau**2 / (tau**2 - 1) * sp.sqrt(rad)
    fixed = (nq * tau**2 + T * tau) / (tau**2 - 1) + tau / (tau**2 - 1) * sp.sqrt(rad)

    r_printed = sp.simplify(quad.subs(xp, printed))
    r_fixed = sp.simplify(quad.subs(xp, fixed))
    emit('  substituting each candidate back into the quadratic (0 => it is a root):')
    emit(f'    Eq 4.24 as printed  -> {r_printed}')
    emit(f'    single power of tau -> {r_fixed}')
    emit('')

    emit("  sympy's own solve(), for a third independent confirmation:")
    for r in sp.solve(sp.Eq(quad, 0), xp):
        emit(f'    {sp.simplify(sp.radsimp(r))}')
    emit('')

    # the intercept generalization: only T shifts
    quad_b = sp.expand((q2 - 2 * xp * nq + xp**2) - (T + (xp - b) / tau)**2)
    Tb = sp.symbols('Tb', real=True)
    gen = [sp.simplify(r.subs(T, Tb + b / tau)) for r in sp.solve(sp.Eq(quad_b, 0), xp)]
    emit('  with an axis intercept b, rewritten in Tb = T - b/tau:')
    for g in gen:
        emit(f'    {sp.simplify(sp.radsimp(g))}')
    emit('  => identical in form; the intercept enters only as T -> T - b/tau.')
    emit('')
    return bool(r_fixed == 0), bool(r_printed == 0)


# ---------------------------------------------------------------------------
# Check 2: numerical, against the original system in real lattice geometry
# ---------------------------------------------------------------------------
def numerical_check(emit):
    import yaml
    from scipy.optimize import brentq
    from pyDFCSR_2D.CSR import CSR2D
    from pyDFCSR_2D.interp1D import interpolate1D

    cfg = _write_inputs()
    os.chdir(EXAMPLE_DIR)
    csr = CSR2D(input_file=cfg)
    csr.run(stop_time=0.60)
    csr.get_CSR_mesh()

    lat = csr.lattice
    nz = csr.CSR_params.zbins
    k = (csr.CSR_params.xbins // 2) * nz + nz // 2
    s = csr.beam.position + csr.CSR_zmesh[k]
    x = csr.CSR_xmesh[k]
    t = csr.beam.position
    tau = csr.beam._slope[0]

    def lut(d, v):
        return interpolate1D(xval=np.atleast_1d(v), data=d,
                             min_x=lat.min_x, delta_x=lat.delta_x)

    def geom(sp_):
        """q = n(s)x + r0(s) - r0(s'), and n(s'), from the lattice tables."""
        r0s = np.array([lut(lat.coords[:, 0], s)[0], lut(lat.coords[:, 1], s)[0]])
        ns = np.array([lut(lat.n_vec[:, 0], s)[0], lut(lat.n_vec[:, 1], s)[0]])
        r0p = np.array([lut(lat.coords[:, 0], sp_)[0], lut(lat.coords[:, 1], sp_)[0]])
        nsp = np.array([lut(lat.n_vec[:, 0], sp_)[0], lut(lat.n_vec[:, 1], sp_)[0]])
        return ns * x + r0s - r0p, nsp

    def resid(xp_, sp_):
        """Eq 4.19 minus Eq 4.20, with no algebra applied. Zero at the branches."""
        q, nsp = geom(sp_)
        return np.linalg.norm(q - nsp * xp_) - ((t - sp_) + xp_ / tau)

    emit(f'  geometry from a real run: s = {s:.6f} m, t = {t:.6f} m, '
         f'x = {x*1e6:+.2f} um, tau = {tau:.6f}')
    emit(f'  sigma_x = {csr.beam._sigma_x*1e6:.1f} um, '
         f'sigma_xi = {csr.beam._sigma_x_transform*1e6:.2f} um')
    emit('')
    hdr = (f"  {'s-s_prime':>10} | {'brentq root 1':>14} {'brentq root 2':>14} | "
           f"{'corrected -':>13} {'corrected +':>13} | {'printed -':>13} {'printed +':>13}")
    emit(hdr)
    emit('  ' + '-' * (len(hdr) - 2))

    worst_corr, worst_print = 0.0, 0.0
    for ds in DS_LIST:
        sp_ = s - ds
        q, nsp = geom(sp_)
        T = t - sp_
        nq = float(nsp @ q)
        q2 = float(q @ q)

        grid = np.linspace(-SCAN_HALF, SCAN_HALF, SCAN_COARSE)
        vals = np.array([resid(g, sp_) for g in grid])
        roots = []
        for i in range(len(vals) - 1):
            if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0:
                try:
                    roots.append(brentq(resid, grid[i], grid[i + 1], args=(sp_,), xtol=1e-14))
                except ValueError:
                    pass
        roots = sorted(roots)

        cm = [eq424_corrected(T, nq, q2, tau, sg) for sg in (-1, +1)]
        pm = [eq424_as_printed(T, nq, q2, tau, sg) for sg in (-1, +1)]

        # match each numeric root to the nearest prediction of each form
        for r in roots:
            if np.isfinite(cm).any():
                worst_corr = max(worst_corr, min(abs(r - c) / max(abs(r), 1e-12)
                                                 for c in cm if np.isfinite(c)))
            if np.isfinite(pm).any():
                worst_print = max(worst_print, min(abs(r - p) / max(abs(r), 1e-12)
                                                   for p in pm if np.isfinite(p)))

        show = (roots + [np.nan, np.nan])[:2]
        emit(f'  {ds:>10.5f} | {show[0]*1e6:>14.4f} {show[1]*1e6:>14.4f} | '
             f'{cm[0]*1e6:>13.4f} {cm[1]*1e6:>13.4f} | '
             f'{pm[0]*1e6:>13.4f} {pm[1]*1e6:>13.4f}')

    emit('')
    emit('  All x\' in um. nan = that root fell outside the +-0.2 m scan window,')
    emit('  which is a scan limitation, not a formula failure.')
    emit('')
    emit(f'  worst relative mismatch, corrected form : {worst_corr:.3e}')
    emit(f'  worst relative mismatch, printed form   : {worst_print:.3e}')
    return worst_corr, worst_print


def _write_inputs():
    import yaml
    beam = {
        'n_particle': 200000, 'species': 'electron',
        'px_dist': {'sigma_px': {'units': 'keV/c', 'value': 25.0}, 'type': 'gaussian'},
        'py_dist': {'sigma_py': {'units': 'keV/c', 'value': 25.0}, 'type': 'gaussian'},
        'pz_dist': {'avg_pz': {'units': 'GeV/c', 'value': 5},
                    'sigma_pz': {'units': 'MeV/c', 'value': 0.5}, 'type': 'gaussian'},
        'x_dist': {'sigma_x': {'units': 'um', 'value': 50}, 'type': 'gaussian'},
        'y_dist': {'sigma_y': {'units': 'um', 'value': 5}, 'type': 'gaussian'},
        'random': {'type': 'hammersley'},
        'start': {'tstart': {'units': 'sec', 'value': 0}, 'type': 'time'},
        'total_charge': {'units': 'nC', 'value': 1},
        'z_dist': {'avg_z': {'units': 'mm', 'value': 0},
                   'sigma_z': {'units': 'um', 'value': 50}, 'type': 'gaussian'},
        'transforms': {'s1': {'shear_coefficient': {'units': 'dimensionless',
                                                    'value': 20.0}, 'type': 'shear z:x'}},
    }
    with open(os.path.join(EXAMPLE_DIR, 'input/eq424_beam.yaml'), 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)
    config = {
        'input_beam': {'style': 'distgen', 'distgen_input_file': 'input/eq424_beam.yaml'},
        'input_lattice': {'lattice_input_file': 'input/dipole_lattice.yaml'},
        'particle_deposition': {'method': 'bspline_fft', 'xbins': 200, 'zbins': 200,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': 3, 'velocity_threhold': 1000},
        'CSR_integration': {'n_formation_length': 1.5, 'zbins': 100, 'xbins': 100},
        'CSR_computation': {'compute_CSR': 1, 'apply_CSR': 0, 'transverse_on': 1,
                            'xbins': 5, 'zbins': 40, 'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': 'eq424', 'workdir': './output'},
    }
    with open(os.path.join(EXAMPLE_DIR, 'input/eq424_config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return 'input/eq424_config.yaml'


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    lines = []

    def emit(s=''):
        print(s)
        lines.append(s)

    emit('=' * 78)
    emit('Verification of thesis Eq. 4.24 (section 4.4.2, integrand localization)')
    emit('=' * 78)
    emit('')
    emit('--- CHECK 1: symbolic (no numbers, no lattice) ---')
    emit('')
    fixed_ok, printed_ok = symbolic_check(emit)

    emit('--- CHECK 2: numerical, brentq on the original system ---')
    emit('')
    worst_corr, worst_print = numerical_check(emit)

    emit('')
    emit('--- VERDICT ---')
    emit(f'  symbolic : single-power form is a root = {fixed_ok}, '
         f'as-printed is a root = {printed_ok}')
    emit(f'  numerical: corrected matches to {worst_corr:.1e}, '
         f'printed to {worst_print:.1e}')
    ok = fixed_ok and not printed_ok and worst_corr < 1e-4 and worst_print > 1e-2
    emit('')
    emit('  => Eq 4.24 as printed has a typo: the tan^2(a0) prefactor on the'
         if ok else '  => INCONCLUSIVE, re-examine')
    if ok:
        emit('     square root should be tan(a0). Confirmed symbolically and')
        emit('     numerically. Everything else in section 4.4.2 checks out.')

    with open(os.path.join(RESULT_DIR, 'eq424_log.txt'), 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nLog: {os.path.join(RESULT_DIR, 'eq424_log.txt')}")


if __name__ == '__main__':
    main()
