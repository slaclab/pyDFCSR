"""
Validate the pre-tracking waist predictor against waists measured by actually tracking.

The predictor (pyDFCSR_2D/waist.py) propagates the beam's second moments with linear
transport before any particle moves. That is only useful if it lands on the waists the
full tracker produces, so this test compares it against §6u's measured values, which came
from tracking and depositing at five locations per shear:

    shear   measured waist (m into the dipole)
        0   none
        2   ~0.40-0.50   (6u sampled 0.40 and 0.60; sigma_z minimum 22.54 um at 0.40)
        5   ~0.20        (sigma_z 10.00 um)
       10   ~0.10        (sigma_z  5.02 um)
       20   ~0.05        (6q, sigma_z  2.51 um from the snapshot log)
       50   ~0.02        (upstream of 6u's first sample)

There is also an exact analytic expectation for this lattice. In a bend R51 = -sin(theta),
so with u = sin(theta),

    var_z(u) = var_z0 - 2 u cov_zx + u^2 var_x     ->  minimised at  u = cov_zx / var_x

hence s_waist = R arcsin(cov_zx / var_x). Both the measurement and this value are checked,
because agreeing with only one would leave the sign conventions unverified: r_gen6 uses
(x, x', y, y', z, dp/p) and Bmad-X uses (x, px, y, py, z, pz), and a sign slip in R51
would move the waist or delete it.

Note the crude form u = 1/tau0 is WRONG at low shear and was the first version of this
test. It assumes the x-z correlation is perfect (var_x = tau0^2 var_z0); at shear 2 the
uncorrelated sigma_x = 50 um is half the sheared 100 um, so r = 0.894 and the true
minimum sits at u = 0.400 rather than 0.500 -- 0.41 m instead of 0.52 m. The predictor
was right and the test was wrong.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pyDFCSR_2D.waist import scan_waists, sigma_from_coords, format_report

R_BEND = 1.0
SIGMA_X0 = 50e-6
SIGMA_Z0 = 50e-6
SIGMA_PX0 = 5e-6          # 25 keV/c at 5 GeV
SIGMA_PZ0 = 1e-4          # 0.5 MeV/c at 5 GeV
DRIFT = 1.0
STEP = 0.05

# shear -> measured waist position in metres INTO the dipole, or None
# shear 2's entry is 0.41 rather than a round number: 6u sampled only 0.40 and 0.60,
# and its 0.40 sample measured sigma_z = 22.54 um against the scan's 22.57 um at
# 0.41 m, so 0.41 is where the tracked minimum actually is.
MEASURED = {0.0: None, 2.0: 0.41, 5.0: 0.20, 10.0: 0.10, 20.0: 0.05, 50.0: 0.02}


def lattice(step_size=STEP, drift=DRIFT):
    return {'step_size': step_size,
            'element_1': {'type': 'drift', 'L': drift, 'nsep': 1},
            'element_2': {'type': 'dipole', 'L': 1.0, 'angle': 1.0, 'nsep': 1},
            'element_3': {'type': 'drift', 'L': 0.5, 'nsep': 1}}


def sigma0_for(shear, n=200000, seed=0):
    """
    Initial 6x6 moments for the sheared Gaussian used throughout this work.

    Built from a sampled distribution rather than written down analytically, so the
    shear transform is applied the same way distgen applies it (x += shear * z) and the
    test cannot agree with the predictor by sharing an algebraic mistake.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, SIGMA_Z0, n)
    x = rng.normal(0.0, SIGMA_X0, n)
    px = rng.normal(0.0, SIGMA_PX0, n)
    pz = rng.normal(0.0, SIGMA_PZ0, n)
    y = rng.normal(0.0, 5e-6, n)
    py = rng.normal(0.0, SIGMA_PX0, n)
    if shear:
        x = x + shear * z
    return sigma_from_coords(x, px, y, py, z, pz)


@pytest.mark.parametrize('shear', sorted(MEASURED))
def test_predicted_waist_matches_measurement(shear):
    rep = scan_waists(lattice(), sigma0_for(shear), STEP)
    inside = [w for w in rep['waists'] if DRIFT <= w['s'] <= DRIFT + 1.0]
    expect = MEASURED[shear]

    if expect is None:
        assert not inside, (f'shear {shear:g}: predicted a waist at '
                            f'{[round(w["s"]-DRIFT, 4) for w in inside]} m into the '
                            f'dipole, but tracking shows none')
        return

    assert inside, f'shear {shear:g}: no waist predicted, tracking shows one at {expect} m'
    got = min(inside, key=lambda w: abs((w['s'] - DRIFT) - expect))
    s_dip = got['s'] - DRIFT

    # analytic value for this lattice, checked independently of the measurement
    sig = sigma0_for(shear)
    u = sig[4, 0] / sig[0, 0]          # cov_zx / var_x
    s_analytic = R_BEND * np.arcsin(np.clip(u, -1.0, 1.0))
    assert abs(s_dip - s_analytic) < 0.005, (
        f'shear {shear:g}: predicted {s_dip:.4f} m, analytic '
        f'R*arcsin(cov_zx/var_x) = {s_analytic:.4f} m')
    # 6u sampled the dipole every 0.2 m, so the measurement localises the waist only
    # to about that; require agreement within one sampling interval
    assert abs(s_dip - expect) < 0.11, (
        f'shear {shear:g}: predicted {s_dip:.4f} m, measured ~{expect} m')


def test_unresolved_waists_are_flagged():
    """The high-shear waists must come back NOT resolved at the shipped step_size."""
    for shear in (10.0, 20.0, 50.0):
        rep = scan_waists(lattice(), sigma0_for(shear), STEP)
        inside = [w for w in rep['waists'] if DRIFT <= w['s'] <= DRIFT + 1.0]
        assert inside, f'shear {shear:g}: expected a waist'
        w = inside[0]
        assert not w['resolved'], (
            f'shear {shear:g}: waist width {w["width"]*1e3:.3f} mm reported as resolved '
            f'by step_size {STEP} m ({w["steps_across"]:.2f} steps across)')
        assert format_report(rep), 'a report with an unresolved waist must not be empty'


def test_fine_step_resolves_and_silences():
    """
    A step_size small enough must clear the warning. Guards against a report that always
    complains, which would be as useless as one that never does.
    """
    shear = 10.0
    rep = scan_waists(lattice(), sigma0_for(shear), STEP)
    w = [x for x in rep['waists'] if DRIFT <= x['s'] <= DRIFT + 1.0][0]
    fine = w['width'] / rep['min_steps'] / 2.0
    rep2 = scan_waists(lattice(step_size=fine), sigma0_for(shear), fine)
    inside = [x for x in rep2['waists'] if DRIFT <= x['s'] <= DRIFT + 1.0]
    assert inside and all(x['resolved'] for x in inside), (
        f'step_size {fine:g} m should resolve a {w["width"]*1e3:.3f} mm waist')
    assert 'WARNING' not in format_report(rep2)


def test_zero_shear_report_is_silent():
    assert format_report(scan_waists(lattice(), sigma0_for(0.0), STEP)) == '' or \
        'WARNING' not in format_report(scan_waists(lattice(), sigma0_for(0.0), STEP))


if __name__ == '__main__':
    for sh in sorted(MEASURED):
        rep = scan_waists(lattice(), sigma0_for(sh), STEP)
        ins = [w for w in rep['waists'] if DRIFT <= w['s'] <= DRIFT + 1.0]
        pred = f'{ins[0]["s"]-DRIFT:.4f}' if ins else 'none'
        if sh:
            _sg = sigma0_for(sh)
            ana = f'{R_BEND * np.arcsin(np.clip(_sg[4, 0] / _sg[0, 0], -1, 1)):.4f}'
        else:
            ana = 'none'
        print(f'shear {sh:>5g}:  predicted {pred:>8}  analytic {ana:>8}  '
              f'measured {str(MEASURED[sh]):>6}')
        if ins:
            w = ins[0]
            print(f'              sigma_z {w["sigma_z"]*1e6:8.3f} um  '
                  f'compress {w["compression"]:6.1f}x  width {w["width"]*1e3:7.4f} mm  '
                  f'steps {w["steps_across"]:6.2f}  resolved {w["resolved"]}')
    print()
    print(format_report(scan_waists(lattice(), sigma0_for(20.0), STEP)))
    sys.exit(pytest.main([__file__, '-q']))
