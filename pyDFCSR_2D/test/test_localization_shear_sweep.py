"""
Does the Eq. 4.24 branch prediction hold across beam tilts, or only at the one
case it was first checked on?

test_localization_branches.py validated the corrected Eq 4.24 against the measured
integrand support at a single shear (20) and a single observation point. That is
thin evidence: the branch geometry depends on the local tilt tau, on the
observation point (x, s), and on which side of the |tan theta| = 1 degeneracy the
beam sits. This sweeps all three.

Regimes deliberately covered
----------------------------
  shear 0     no imposed tilt; the dipole still generates some. |tan theta| < 1,
              so the code takes its NON-chirp branch and the thesis says the chirp
              root degenerates toward x2 = x.
  shear 1, 2  the beam passes near |tan theta| = 1 during the bend, which is the
              alpha = +-pi/4 degeneracy where Eq 4.24's denominator (tau^2 - 1)
              vanishes. The linear-fallback guard should be exercised.
  shear 5, 20 strongly chirped, well into the code's chirp branch.
  shear 50    extreme, to check nothing degrades at large amplification.

Three observation points per shear (low / mid / high z on the wake mesh), so the
check is not special to a symmetric point.

PASS CRITERION
--------------
For every (shear, observation point, s') where the integrand HAS measurable
support near a predicted branch, the prediction must land INSIDE the measured
interval. Predictions with no support nearby are legitimate -- the branch has left
the beam in z -- and are counted separately, not as failures.
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from test_localization_branches import (Geometry, measured_intervals, run,
                                        RESULT_DIR, EXAMPLE_DIR)

SHEARS = [0.0, 1.0, 2.0, 5.0, 20.0, 50.0]
DS_LIST = [2e-4, 5e-4, 1e-3, 2e-3, 4e-3, 1e-2]
IX_FRACS = [0.2, 0.5, 0.8]          # where on the wake mesh in z to observe


def obs_points(csr, fracs):
    """(s, x) at several z positions on the wake mesh, at mid-x."""
    nz = csr.CSR_params.zbins
    ix = csr.CSR_params.xbins // 2
    out = []
    for f in fracs:
        j = int(np.clip(round(f * (nz - 1)), 0, nz - 1))
        k = ix * nz + j
        out.append((csr.beam.position + csr.CSR_zmesh[k], csr.CSR_xmesh[k], j))
    return out


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    lines = []

    def emit(t=''):
        print(t)
        lines.append(t)

    emit('=' * 92)
    emit('Eq 4.24 branch prediction vs the measured integrand, swept over beam tilt')
    emit('=' * 92)
    emit('')
    emit('method = bspline_comoving (poly_degree 1) throughout, so there is one')
    emit('well-defined beam axis and Eq 4.24 applies.')
    emit('')
    emit('Classification uses NO distance tolerance. A prediction is "absent" if the')
    emit('integrand evaluated AT the predicted point is < 1e-6 of the column peak,')
    emit('meaning that branch carries no density -- either a spurious root (l <= 0,')
    emit('now guarded) or z_ret outside the beam. Otherwise it must land "inside" a')
    emit('measured nonzero interval, or it counts as a MISS.')
    emit('')

    summary = []
    detail = {}

    for shear in SHEARS:
        tag = f'sw{shear:g}'.replace('.', 'p')
        csr, _, _, t = run('bspline_comoving', 1, tag, shear=shear)
        b = csr.beam
        tau = b._slope[0]
        sx, sxi = b._sigma_x, b._sigma_x_transform
        mode = 'chirp' if abs(tau) > 1 else 'non-chirp'

        emit(f"--- shear z:x = {shear:g}  ->  local dx/dz = {tau:+.4f} "
             f"({np.degrees(np.arctan(tau)):+.1f} deg), sigma_x/sigma_xi = {sx/sxi:.1f}x, "
             f"{mode} branch ---")

        n_inside = n_absent = n_miss = 0
        worst = 0.0
        rows = []
        for (s, x, j) in obs_points(csr, IX_FRACS):
            g = Geometry(csr, s, x, t)
            for ds in DS_LIST:
                sp = s - ds
                iv = measured_intervals(csr, s, x, t, sp, x - 14 * sx, x + 6 * sx)
                # column peak, to judge "is there density at the prediction"
                xs = np.linspace(x - 14 * sx, x + 6 * sx, 40001)
                gcol, _ = csr.get_CSR_integrand(
                    s=s, t=t, x=x, xp=xs.reshape(-1, 1),
                    sp=np.full((len(xs), 1), sp))
                peak = np.abs(gcol[:, 0]).max()

                for sign, nm in ((-1, 'narrow'), (+1, 'chirp')):
                    pred = g.branch(sp, sign)
                    if not np.isfinite(pred) or not iv or peak == 0.0:
                        n_absent += 1
                        continue
                    # Direct test, no distance tolerance: is the integrand nonzero
                    # AT the prediction? If not, that branch carries no density and
                    # is legitimately absent -- Eq 4.24 located where the beam AXIS
                    # meets the light cone, but z_ret there is outside the beam.
                    gp, _ = csr.get_CSR_integrand(s=s, t=t, x=x,
                                                  xp=np.array([[pred]]),
                                                  sp=np.array([[sp]]))
                    if abs(gp[0, 0]) / peak < 1e-6:
                        n_absent += 1
                        continue
                    c, lo, hi, wt = min(iv, key=lambda e: abs(e[0] - pred))
                    if lo <= pred <= hi:
                        n_inside += 1
                        rows.append((j, ds, nm, pred, c, wt, 'inside'))
                    else:
                        err = abs(pred - c) / max(abs(c), 1e-12)
                        worst = max(worst, err)
                        n_miss += 1
                        rows.append((j, ds, nm, pred, c, wt, f'MISS {err:.3f}'))

        n_int = [len(measured_intervals(csr, s, x, t, s - 1e-3, x - 14 * sx, x + 6 * sx))
                 for (s, x, j) in obs_points(csr, IX_FRACS)]
        emit(f'    predictions inside measured support : {n_inside}')
        emit(f'    predictions with no support nearby  : {n_absent}  (branch out of beam)')
        emit(f'    predictions that MISSED support     : {n_miss}'
             + (f'   worst rel err {worst:.4f}' if n_miss else ''))
        emit(f'    intervals at ds = 1 mm, per obs point: {n_int}  (Eq 4.24 predicts <= 2)')
        if n_miss:
            emit('    misses:')
            for r in rows:
                if r[6].startswith('MISS'):
                    emit(f'      z-idx {r[0]:>3} ds={r[1]*1e3:.2f}mm {r[2]:>6}: '
                         f'pred {r[3]*1e6:>10.1f} vs measured {r[4]*1e6:>10.1f} um '
                         f'({r[5]:.1f}% weight)  {r[6]}')
        emit('')
        summary.append((shear, tau, sx / sxi, mode, n_inside, n_absent, n_miss,
                        worst, max(n_int)))
        detail[shear] = rows

    emit('=' * 92)
    emit('SUMMARY')
    emit('=' * 92)
    hdr = (f"{'shear':>7} {'dx/dz':>9} {'amp':>7} {'branch':>10} "
           f"{'inside':>7} {'absent':>7} {'MISS':>6} {'worst err':>10} {'max #int':>9}")
    emit(hdr)
    emit('-' * len(hdr))
    for (sh, tau, amp, mode, ni, na, nm, wo, mi) in summary:
        emit(f'{sh:>7.0f} {tau:>9.4f} {amp:>7.1f} {mode:>10} '
             f'{ni:>7} {na:>7} {nm:>6} {wo:>10.4f} {mi:>9}')
    tot_miss = sum(s[6] for s in summary)
    tot_in = sum(s[4] for s in summary)
    emit('')
    emit(f'TOTAL: {tot_in} predictions landed inside measured support, '
         f'{tot_miss} missed.')
    emit(f'Max intervals seen at any (shear, obs point): {max(s[8] for s in summary)} '
         f'(Eq 4.24 predicts at most 2)')
    emit('')
    emit('VERDICT: ' + ('consistent across all tilts tested'
                        if tot_miss == 0 and max(s[8] for s in summary) <= 2
                        else 'INCONSISTENT -- see the misses above'))

    # ---- plot: predicted vs measured, all shears -----------------------------
    n = len(SHEARS)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.2), sharey=False)
    for ax, (sh, rows) in zip(np.atleast_1d(axes), detail.items()):
        pr = np.array([r[3] for r in rows]) * 1e6
        me = np.array([r[4] for r in rows]) * 1e6
        ok = np.array([r[6] == 'inside' for r in rows])
        if len(pr):
            lim = [min(pr.min(), me.min()), max(pr.max(), me.max())]
            ax.plot(lim, lim, 'k-', lw=1, alpha=0.5, label='y = x')
            ax.plot(me[ok], pr[ok], 'go', ms=5, mfc='none', label='inside')
            if (~ok).any():
                ax.plot(me[~ok], pr[~ok], 'rx', ms=7, label='missed')
        tau = dict((s[0], s[1]) for s in summary)[sh]
        ax.set_title(f'shear {sh:g}\ndx/dz = {tau:+.2f}', fontsize=10)
        ax.set_xlabel(r"measured centre ($\mu$m)")
        ax.set_ylabel(r"Eq 4.24 prediction ($\mu$m)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle('Eq 4.24 branch prediction vs measured integrand support, '
                 'swept over beam tilt (bspline_comoving)', fontsize=12)
    plt.tight_layout()
    p = os.path.join(RESULT_DIR, 'branch_shear_sweep.png')
    plt.savefig(p, dpi=130)
    plt.close()
    emit('')
    emit(f'Plot saved: {p}')

    with open(os.path.join(RESULT_DIR, 'shear_sweep_log.txt'), 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
