"""
The transient wake inside each of the four chicane dipoles.

Replots from the maps `test_chicane_auto.py` captured, so this costs nothing and does not repeat the
7-minute run. Every map here is a wake that was actually applied to the beam.

What this is for: §11h split the bend transient into an ENTRANCE scale (the overtaking length
`L_0 = (24 R^2 5 sigma_z)^(1/3)`) and an EXIT scale (`R phi_m / 2`, Stupakov & Emma Eq. 10). This
chicane is the case where that distinction bites hardest, because for these weak bends the entrance
transient is LONGER THAN THE MAGNET:

    bend   sigma_z um   L_entrance m   L_entrance / L_bend   kicks inside
    B1         200.00         1.3705                  2.74              4
    B2         201.41         1.3737                  2.75             18
    B3         110.27         1.1238                  2.25             18
    B4          19.37         0.6293                  1.26              8

R = 10.356 m and L_bend = 0.5002 m, so every dipole is 1.3-2.7 formation lengths SHORT of steady
state. The steady-state 1D wake formula does not apply anywhere in this chicane -- the beam is in the
entrance transient for the entire magnet, every time. That is the regime §11h's Fig-3 remark was
about, and it is why the exit scale had to be fixed separately.

Two figures:
  * dE/ds against z, one panel per dipole, one curve per kick, coloured by position through the
    magnet. Shows the transient BUILDING, which a single x-z map cannot.
  * the peak and the roughness against distance into each magnet, on a common axis scaled by
    L_entrance, so the four bends can be compared despite differing by 10x in sigma_z.
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'chicane_auto')

# chicane_lattice.yaml: R = L/angle = 0.5002/0.0483
R_BEND, PHI, L_BEND = 0.5002 / 0.0483, 0.0483, 0.5002
L_EXIT = 0.5 * R_BEND * PHI
BENDS = [('B1', 0.1000, 0.6002), ('B2', 5.6060, 6.1062),
         ('B3', 7.1062, 7.6064), ('B4', 12.6122, 13.1124)]

# (bend, exit face s, end of the downstream drift). Lengths in units of L_exit = 0.2501 m:
#   B1 -> 20.0,  B2 -> 4.0,  B3 -> 20.0,  B4 -> 0.8
# B1 and B3 are the useful ones; B4's drift is shorter than one L_exit so its decay cannot be seen.
EXITS = [('B1', 0.6002, 5.6060), ('B2', 6.1062, 7.1062),
         ('B3', 7.6064, 12.6122), ('B4', 13.1124, 13.3124)]


def roughness(w):
    """Second-difference norm along z relative to amplitude; see §11q."""
    w = np.asarray(w, float)
    amp = np.abs(w).max()
    if amp <= 0:
        return 0.0
    d2 = w[2:] - 2.0 * w[1:-1] + w[:-2]
    return float(np.linalg.norm(d2) / amp)


def main():
    lines = []

    def emit(t=''):
        print(t, flush=True)
        lines.append(t)

    maps = list(np.load(os.path.join(RESULT_DIR, 'chicane_auto_maps.npz'),
                        allow_pickle=True)['maps'])
    ms = np.array([m['s'] for m in maps])

    emit('=' * 96)
    emit('Transient wake inside each chicane dipole (replotted from the captured maps)')
    emit('=' * 96)
    emit(f'R = {R_BEND:.3f} m, phi = {PHI} rad, L_bend = {L_BEND} m, L_exit = {L_EXIT:.4f} m')
    emit('')
    emit('  the entrance transient is LONGER than the magnet in every dipole:')
    emit(f"    {'bend':>5} {'sigma_z um':>11} {'L_entrance':>11} {'L_ent/L_bend':>13} "
         f"{'kicks in':>9} {'peak dE':>9}")
    groups = {}
    for name, a, b in BENDS:
        sel = [m for m in maps if a - 1e-9 <= m['s'] <= b + 1e-9]
        groups[name] = (a, b, sel)
        sz = sel[0]['sigma_z']
        Lf = (24.0 * R_BEND ** 2 * 5.0 * sz) ** (1.0 / 3.0)
        pk = max(np.abs(m['dE']).max() for m in sel)
        emit(f'    {name:>5} {sz * 1e6:>11.2f} {Lf:>11.4f} {Lf / L_BEND:>13.2f} '
             f'{len(sel):>9} {pk:>9.3f}')
    emit('')

    # ---- dE(z) per dipole, one curve per kick -------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for c, (name, a, b) in enumerate(BENDS):
        _, _, sel = groups[name]
        sz0 = sel[0]['sigma_z']
        Lf = (24.0 * R_BEND ** 2 * 5.0 * sz0) ** (1.0 / 3.0)
        cols = cm.viridis(np.linspace(0, 0.92, len(sel)))
        ax = axes[0, c]
        for m, col in zip(sel, cols):
            mid = m['dE'][m['dE'].shape[0] // 2, :]
            zc = m['zz'][m['zz'].shape[0] // 2, :] * 1e3
            ax.plot(zc, mid, color=col, lw=1.1)
        ax.set_title(f'{name}   s = {a:.3f} to {b:.3f} m\n'
                     f'{len(sel)} kicks, L_ent/L_bend = {Lf / L_BEND:.2f}', fontsize=9)
        ax.set_xlabel('z  [mm]', fontsize=8)
        ax.set_ylabel('dE/ds  [MeV/m]  (mid-x row)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        ax.text(0.03, 0.03, 'dark = entrance\nbright = exit', transform=ax.transAxes,
                fontsize=6.5, va='bottom',
                bbox=dict(fc='white', ec='0.6', alpha=0.85, pad=1.4))

        # the same, normalised to each curve's own peak: shows whether the SHAPE changes
        ax = axes[1, c]
        for m, col in zip(sel, cols):
            mid = m['dE'][m['dE'].shape[0] // 2, :]
            pk = np.abs(mid).max()
            if pk > 0:
                ax.plot(m['zz'][m['zz'].shape[0] // 2, :] * 1e3, mid / pk, color=col, lw=1.1)
        ax.set_xlabel('z  [mm]', fontsize=8)
        ax.set_ylabel('dE/ds, normalised to own peak', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle('transient wake through each chicane dipole: TOP absolute, BOTTOM shape only '
                 '(colour = position through the magnet)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(os.path.join(RESULT_DIR, 'chicane_transients.png'), dpi=130)
    plt.close(fig)
    emit('  wrote chicane_transients.png')

    # ---- peak and roughness vs distance into the magnet, scaled by L_entrance ------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    emit('')
    emit('  growth of the wake through each magnet (peak |dE| at entrance vs exit)')
    emit(f"    {'bend':>5} {'at entrance':>12} {'at exit':>10} {'ratio':>8} "
         f"{'d/L_ent at exit':>16}")
    for name, a, b in BENDS:
        _, _, sel = groups[name]
        sz0 = sel[0]['sigma_z']
        Lf = (24.0 * R_BEND ** 2 * 5.0 * sz0) ** (1.0 / 3.0)
        d = np.array([m['s'] - a for m in sel])
        pk = np.array([np.abs(m['dE']).max() for m in sel])
        rg = np.array([roughness(m['dE'][m['dE'].shape[0] // 2, :]) for m in sel])
        axes[0].plot(d / Lf, pk, 'o-', ms=3.5, lw=1.2, label=name)
        axes[1].plot(d / Lf, rg, 'o-', ms=3.5, lw=1.2, label=name)
        emit(f'    {name:>5} {pk[0]:>12.4f} {pk[-1]:>10.4f} '
             f'{pk[-1] / pk[0] if pk[0] > 0 else float("nan"):>8.1f} {d[-1] / Lf:>16.3f}')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('peak |dE/ds|  [MeV/m]')
    axes[0].set_title('wake growth through the magnet')
    axes[1].axhline(1.0, color='C3', ls='--', lw=1, label='roughness 1.0')
    axes[1].set_ylabel('relative second-difference norm')
    axes[1].set_title('wake smoothness through the magnet')
    for ax in axes:
        ax.set_xlabel('distance into the magnet / L_entrance')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which='both')
    fig.suptitle('every dipole exits while still inside its own entrance transient '
                 '(x axis never reaches 1)', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(RESULT_DIR, 'chicane_transient_growth.png'), dpi=130)
    plt.close(fig)
    emit('  wrote chicane_transient_growth.png')

    # ---- EXIT transient: decay downstream of each face -----------------------------------
    # §11h set the exit scale from Stupakov & Emma Eq. 10, W ~ 1/(phi_m + 2x) with x the downstream
    # distance in units of R, so the amplitude halves at x = phi_m/2 i.e. d = R phi_m/2 = L_exit.
    # That scale has never been checked against this code's own output; this does that.
    #
    # Measured ON AXIS (the mid-x row), NOT as the global peak over the map. The wake mesh is built
    # from sigma_x, which grows 106 -> 587 um through the drift after B1, and the global peak then
    # lands ON THE MESH EDGE at 6 of 9 sampled positions -- so a "peak" curve tracks where the mesh
    # boundary fell rather than how the wake decayed. The mid-x row sits at a fixed place in the
    # beam frame and does not have that problem.
    emit('')
    emit('  EXIT transient: on-axis |dE| downstream of each face, against Eq. 10')
    emit('  (Eq. 10 shape is phi/(phi + 2 d/R), which is 0.5 at d = L_exit by construction)')
    fig, ax0 = plt.subplots(figsize=(7.5, 5))
    axes = [ax0]
    for name, e, nxt in EXITS:
        sel = [m for m in maps if e - 1e-9 <= m['s'] <= nxt + 1e-9]
        if len(sel) < 3:
            emit(f'    {name}: downstream drift is only '
                 f'{(nxt - e) / L_EXIT:.2f} L_exit, too short to show a decay -- skipped')
            continue
        d = np.array([m['s'] - e for m in sel])
        on = np.array([np.abs(m['dE'][m['dE'].shape[0] // 2, :]).max() for m in sel])
        ref = on[0] if on[0] > 0 else 1.0
        axes[0].plot(d / L_EXIT, on / ref, 'o-', ms=3.5, lw=1.2, label=name)
        emit(f"    {name}: on-axis at the face {ref:.4f} MeV/m, "
             f"at 1 L_exit {np.interp(1.0, d / L_EXIT, on / ref):.3f} of it "
             f"(Eq. 10 predicts 0.500)")
    # Cut the x axis at 6 L_exit. Beyond that the curve stops being an exit decay: B1's drift is
    # 20 L_exit long and DISPERSIVE, so by ~6 L_exit the wake has decayed into the noise and then
    # RISES again (0.0070 -> 0.0587 MeV/m) as the frame shears -- tau sweeps -0.44 to -9.1 and
    # sigma_xi compresses 60 -> 28 um through the drift. That rise is the chirp developing before
    # B2, not the B1 exit transient, and plotting it on this axis would misattribute it.
    axes[0].set_xlim(0, 6)
    xx = np.linspace(0, 8, 200)
    axes[0].plot(xx, PHI / (PHI + 2.0 * xx * L_EXIT / R_BEND), 'k--', lw=1.4,
                 label='Eq. 10  phi/(phi+2x)')
    axes[0].axvline(1.0, color='C7', ls=':', lw=1)
    axes[0].axhline(0.5, color='C7', ls=':', lw=1)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('distance past the exit face / L_exit')
    axes[0].set_ylabel('on-axis |dE| / value at the face')
    axes[0].set_title('exit decay vs Stupakov & Emma Eq. 10')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which='both')
    fig.suptitle('exit decay downstream of each dipole face, all four bends', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(RESULT_DIR, 'chicane_exit_decay.png'), dpi=130)
    plt.close(fig)
    emit('  wrote chicane_exit_decay.png')

    # ---- the exit wake SHAPE downstream, for ALL FOUR bends -------------------------------
    # Previously only B3 got a shape panel, which is the one bend where the decay is cleanest --
    # so it flattered the result. All four are shown now, absolute on top and normalised to each
    # curve's own peak below, so a wake that is merely shrinking is distinguishable from one whose
    # form is changing. Only the first ~2.5 L_exit is drawn: past that B1 and B3 are dominated by
    # the frame shear described above rather than by the exit transient.
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    emit('')
    emit('  exit wake shape: curves drawn per bend (within 2.5 L_exit of the face)')
    for c, (name, e, nxt) in enumerate(EXITS):
        sel = [m for m in maps
               if e - 1e-9 <= m['s'] <= min(nxt, e + 2.5 * L_EXIT) + 1e-9]
        emit(f'    {name}: {len(sel)} kicks within 2.5 L_exit '
             f'(drift is {(nxt - e) / L_EXIT:.2f} L_exit long)')
        cols = cm.plasma(np.linspace(0, 0.88, max(len(sel), 1)))
        for m, col in zip(sel, cols):
            zc = m['zz'][m['zz'].shape[0] // 2, :] * 1e3
            mid = m['dE'][m['dE'].shape[0] // 2, :]
            axes[0, c].plot(zc, mid, color=col, lw=1.2,
                            label=f"{(m['s'] - e) / L_EXIT:.2f}")
            pk = np.abs(mid).max()
            if pk > 0:
                axes[1, c].plot(zc, mid / pk, color=col, lw=1.2)
        axes[0, c].set_title(f'{name} exit at s = {e:.3f} m\n'
                             f'{len(sel)} kicks, drift {(nxt - e) / L_EXIT:.1f} L_exit',
                             fontsize=9)
        axes[0, c].legend(fontsize=6, ncol=2, title='d/L_exit', title_fontsize=6)
        for r in (0, 1):
            axes[r, c].set_xlabel('z  [mm]', fontsize=8)
            axes[r, c].tick_params(labelsize=7)
            axes[r, c].grid(alpha=0.3)
        axes[0, c].set_ylabel('dE/ds  [MeV/m]  (mid-x row)', fontsize=8)
        axes[1, c].set_ylabel('normalised to own peak', fontsize=8)
    fig.suptitle('EXIT transient for all four bends: TOP absolute, BOTTOM shape only '
                 '(dark = at the face, bright = downstream)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(os.path.join(RESULT_DIR, 'chicane_exit_transients_all.png'), dpi=130)
    plt.close(fig)
    emit('  wrote chicane_exit_transients_all.png')

    with open(os.path.join(RESULT_DIR, 'chicane_transients_log.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
