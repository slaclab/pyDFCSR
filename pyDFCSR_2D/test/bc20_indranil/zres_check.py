"""
Deposition z-resolution check for the loss beam's first bend: the wake mesh recomputed on the pickled
0.5 m states of loss_z600 / loss_z1200 (dump_state.py <run> 0.5; 600 / 1200 deposition z bins instead
of 300), against the loss run's stored kick nearest to that s and the 1D wake there.

    conda run -n pydfcsr python ../test/bc20_indranil/zres_check.py

Prints the charge-weighted projected W at the tail horn (min W), the head horn and the rms over z for
each, and writes zres_check_loss.png (projected W of the three resolutions and 1D).
"""
import glob
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'facet_bc20'))
import dump_kick_state as D  # noqa: E402
import plot_bc20_wakes as P  # noqa: E402

RES = os.path.join(HERE, '..', 'benchmark_results', 'bc20_indranil')
RUNS = os.path.join(RES, 'runs')
D.RUNS = P.RUNS = RUNS


def proj_weights(sim):
    xi = np.asarray(sim.CSR_xrange_transformed)
    h = np.gradient(xi)
    n, L = xi.size, xi[-1] - xi[0]
    wx = np.clip(1.0 / ((n - 1) * h) - 0.3 / L, 0, None) * h
    return wx / wx.sum()


def main():
    curves = {}
    s_ref = None
    for run in ('loss_z600', 'loss_z1200'):
        p = D.state_path(run, 0.5)
        if not os.path.exists(p):
            print(run, 'state missing'); continue
        sim = D.load_state(p)
        pos = sim.beam.position; s_ref = pos
        nz = sim.CSR_params.zbins
        W = np.array([sim.get_CSR_wake(pos + z, x)[0] for z, x in zip(sim.CSR_zmesh, sim.CSR_xmesh)]).reshape(-1, nz)
        tr = sim.DF_tracker
        print(f'{run}: s = {pos:.4f}, deposition {tr.xbins}x{tr.zbins}, sz_used {tr.sz_used:.2f} bins, res_z {tr.res_z*1e6:.3f} um; mesh max|W| {np.abs(W).max():.2f}')
        curves[run] = (np.asarray(sim.CSR_zrange), (proj_weights(sim)[:, None] * W).sum(0))
    R = P.load2d('loss')
    r = R[P.nearest(R, s_ref)]
    curves['loss (300)'] = (r['z'], P.proj(r, r['W']))
    print(f"loss stored kick at s = {r['s']:.4f}: max|W| {np.abs(r['W']).max():.2f}")
    D1 = P.load1d('loss_1d')
    s1 = min(D1, key=lambda k: abs(k - s_ref))
    curves['1D'] = D1[s1]
    print(f'1D kick at {s1:.4f}')
    print(f'{"run":12s} {"min W":>8s} {"z(min)":>8s} {"max W":>8s} {"rms W":>8s}  [MeV/m, um]')
    for k, (z, w) in curves.items():
        print(f'{k:12s} {w.min():+8.2f} {z[w.argmin()]*1e6:+8.2f} {w.max():+8.2f} {np.sqrt(np.mean(w**2)):8.3f}')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, (z, w) in curves.items():
        ax.plot(z * 1e6, w, lw=1.2, ls='--' if k == '1D' else '-', label=k)
    ax.axhline(0, color='#e1e0d9', lw=0.8)
    ax.set_xlabel('z [µm] (head at +z)'); ax.set_ylabel('projected W [MeV/m]')
    ax.set_title(f'loss beam, B1LE, s = {s_ref:.3f} m: deposition z bins 300 / 600 / 1200 vs 1D', loc='left', fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(RES, 'zres_check_loss.png'), dpi=140)
    print('wrote zres_check_loss.png')


if __name__ == '__main__':
    main()
