"""
Does the 2D wake in the chicane's FIRST bend depend on how much straight-line history the beam has before
it? The benchmark lattice has a 0.1 m drift (D0) before B1, and the 2D integral cannot reach sources before
the lattice starts (s' < 0), while the 1D entrance transient assumes an infinite straight line. This runs
the 2dR recipe with D0 = <L> m up to 0.35 m past B1's exit and compares the B1 wakes with 2dR and 1d_full.

    PYTHONPATH=<repo root> conda run -n pydfcsr python pyDFCSR_2D/test/chicane_prehistory.py [run|analyze] [L] [n_formation_length] [prehistory] [reach_straight] [reach_slip_max]
"""
import contextlib
import io
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import chicane_2d_vs_1d as C  # noqa: E402

C.OUT = os.path.join(HERE, 'benchmark_results', 'chicane_2d_vs_1d', 'closed')   # the 2dR / 1d_full runs live there
os.makedirs(os.path.join(C.OUT, 'inputs'), exist_ok=True)

L0 = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
NFL = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0      # n_formation_length (reach); 1 is the default
PRE = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0      # CSR_integration.prehistory [m]: synthetic straight history instead of a real drift
RS = float(sys.argv[5]) if len(sys.argv) > 5 else None      # CSR_integration.reach_straight (None = the default)
SLIP = float(sys.argv[6]) if len(sys.argv) > 6 else None    # CSR_integration.reach_slip_max (None = the default)
NAME = (f'2dR_pre{PRE:g}' if PRE else f'2dR_D0_{L0:g}') + (f'_nfl{NFL:g}' if NFL != 1.0 else '') + (f'_rs{RS:g}' if RS is not None else '') + (f'_slip{SLIP:g}' if SLIP is not None else '')


def run():
    f = C.config('2dR')
    cfg = yaml.safe_load(open(f))
    lat = yaml.safe_load(open(cfg['input_lattice']['lattice_input_file']))
    if PRE:
        L0_eff = 0.1                     # the benchmark lattice as is; the history is synthetic
    else:
        L0_eff = L0
    lat['D0']['L'] = L0_eff
    lat['D0']['nsep'] = max(1, int(round(L0_eff / 0.5)))
    lf = os.path.join(C.OUT, 'inputs', f'chicane_lattice_D0_{L0:g}.yaml')
    yaml.safe_dump(lat, open(lf, 'w'), sort_keys=False)
    cfg['input_lattice']['lattice_input_file'] = lf
    cfg['CSR_computation']['write_name'] = NAME
    cfg['CSR_integration']['n_formation_length'] = NFL
    if PRE:
        cfg['CSR_integration']['prehistory'] = PRE
    if RS is not None:
        cfg['CSR_integration']['reach_straight'] = RS
    if SLIP is not None:
        cfg['CSR_integration']['reach_slip_max'] = SLIP
    f2 = os.path.join(C.OUT, 'inputs', f'{NAME}.yaml')
    yaml.safe_dump(cfg, open(f2, 'w'))
    from pyDFCSR_2D import CSR2D
    t0 = time.time()
    sim = CSR2D(input_file=f2)
    sim.run(stop_time=L0_eff + 0.5002 + 0.35)
    print(f'{NAME}: done in {(time.time() - t0) / 60:.1f} min', flush=True)


def analyze():
    import chicane_wake_plots as W
    import plot_bc20_wakes as P
    R0, R1 = P.load2d('2dR'), P.load2d(NAME)
    D1 = W.load1d_full('1d_full')
    sh = 0.0 if PRE else L0 - 0.1
    print(f'B1 (0.1-0.6002 m in the benchmark; shifted by {sh:.2f} m; run {NAME}): charge-weighted mean W and max |W| [MeV/m]')
    print(f'{"s in B1 [m]":>12s} {"<W> 2dR":>9s} {"<W> D0":>9s} {"<W> 1D":>9s} | {"max|W| 2dR":>11s} {"max|W| D0":>10s} {"max|W| 1D":>10s}')
    for r1 in R1:
        s = r1['s'] - sh
        if s < 0.1 - 0.05 or s > 0.6002 + 0.36:
            continue
        r0 = R0[P.nearest(R0, s)]
        d = W.nearest1d(D1, s, tol=0.1)
        if abs(r0['s'] - s) > 0.1:
            continue
        print(f'{s:12.3f} {W.mean_w(r0):+9.4f} {W.mean_w(r1):+9.4f} {W.mean_w1d(d) if d else np.nan:+9.4f} | '
              f'{np.abs(P.proj(r0, r0["W"])).max():11.4f} {np.abs(P.proj(r1, r1["W"])).max():10.4f} {np.abs(d["W"]).max() if d else np.nan:10.4f}'
              f'   (2dR at {r0["s"]:.3f}, 1D at {d["s"] if d else np.nan:.3f})')
    # figure: projected W at B1 mid, exit, +0.3 m
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, s, lab in zip(axs, (0.35, 0.6002, 0.9), ('B1 mid', 'B1 exit', 'D1 +0.3 m')):
        r0, r1 = R0[P.nearest(R0, s)], R1[P.nearest(R1, s + sh)]
        d = W.nearest1d(D1, s, tol=0.15)
        zm = P.zmid(r0)
        ax.axhline(0, color=P.GRID)
        if d is not None:
            ax.plot((d['z'] - zm) * 1e6, d['W'], color=P.COL['1d'], ls='--', label='1D (infinite straight line before B1)')
        ax.plot((r0['z'] - zm) * 1e6, P.proj(r0, r0['W']), color=P.COL['q'], label='2D 2dR, D0 = 0.1 m')
        ax.plot((r1['z'] - P.zmid(r1)) * 1e6, P.proj(r1, r1['W']), color=P.COL['old'], label=(f'2D, prehistory {PRE:g} m' if PRE else f'2D, D0 = {L0:g} m') + (f', n_fl = {NFL:g}' if NFL != 1 else ''))
        ax.set_title(f'{lab}, s = {r0["s"]:.2f} m', loc='left', fontsize=9)
        ax.set_xlabel('z − z_mid [µm]')
    axs[0].set_ylabel('W [MeV/m]')
    axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle('First bend of the benchmark chicane: the 2D wake with 0.1 m and with 2 m of straight-line history before B1', x=0.01, ha='left', fontsize=10)
    fig.tight_layout()
    out = os.path.join(C.OUT, f'prehistory_B1_{NAME[4:]}.png')
    fig.savefig(out, dpi=130)
    print('wrote', out)


if __name__ == '__main__':
    (run if sys.argv[1:2] == ['run'] else analyze)()
