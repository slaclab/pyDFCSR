import os, sys, glob, numpy as np, h5py
T = '/Users/jytang/Library/CloudStorage/GoogleDrive-jytang@stanford.edu/My Drive/Stanford/research/FEL/pyDFCSR_claude/pyDFCSR_near_adapt/pyDFCSR_2D/test'
sys.path.insert(0, os.path.join(T, 'facet_bc20')); sys.path.insert(0, T); sys.path.insert(0, os.path.join(T, '..', '..'))
import plot_bc20_wakes as P
import chicane_wake_plots as W
R = os.path.join(T, 'benchmark_results', 'bc20_indranil', 'runs'); P.RUNS = R
def load2d_file(f):
    keep = P.newest
    P.newest = lambda pat: f
    try:
        return P.load2d('x')
    finally:
        P.newest = keep
for beam, f2 in (('loss', 'loss_v2-2026-11-24T15_35_19-08_00-wakes.h5'), ('gain', 'gain_v2-2026-11-24T16_48_37-08_00-wakes.h5')):
    V2 = load2d_file(os.path.join(R, f2)); V1 = P.load2d(beam); D1 = W.load1d_full(f'{beam}_1d')
    print(f'\n{beam}: v1 (lattice starts at b1le, 80/40 z nodes) vs v2 (prehistory 45 m, straight reach, mesh_z_quantile, loss 160 z nodes) vs 1D; charge-weighted <W> and max|W(z)| [MeV/m]')
    print(f'{"s [m]":>7s} | {"<W> v1":>9s} {"<W> v2":>9s} {"<W> 1D":>9s} | {"max v1":>8s} {"max v2":>8s} {"max 1D":>8s}')
    for s in (0.15, 0.30, 0.53, 0.80, 1.06, 1.3, 1.6, 2.2, 3.5, 6.0, 9.0, 10.6, 11.3, 12.2):
        r1, r2 = V1[P.nearest(V1, s)], V2[P.nearest(V2, s)]
        d = W.nearest1d(D1, r2['s'], tol=0.05)
        if abs(r1['s'] - r2['s']) > 0.03: continue
        print(f'{r2["s"]:7.3f} | {W.mean_w(r1):+9.4f} {W.mean_w(r2):+9.4f} {W.mean_w1d(d) if d else np.nan:+9.4f} | '
              f'{np.abs(P.proj(r1, r1["W"])).max():8.3f} {np.abs(P.proj(r2, r2["W"])).max():8.3f} {np.abs(d["W"]).max() if d else np.nan:8.3f}')
    # integrated energy change from the mean wake, kick by kick (trapezoid in s), over b1le and up to 10.4 m
    for lo, hi, lab in ((0.0, 1.063, 'b1le'), (1.063, 10.414, 'D1..q1el'), (10.414, 12.239, 'b2le')):
        def integ(R2, f):
            s = np.array([r['s'] for r in R2]); m = np.array([f(r) for r in R2]); k = (s >= lo - 1e-6) & (s <= hi + 1e-6)
            return np.trapz(m[k], s[k]) if k.sum() > 1 else np.nan
        e1, e2 = integ(V1, W.mean_w), integ(V2, W.mean_w)
        s1d = sorted(D1); m1d = [W.mean_w1d(D1[k]) for k in s1d]; s1d = np.array([D1[k]['s'] for k in s1d]); k = (s1d >= lo - 1e-6) & (s1d <= hi + 1e-6)
        e1d = np.trapz(np.array(m1d)[k], s1d[k])
        print(f'   integral of <W> over {lab:9s}: v1 {e1:+.4f}  v2 {e2:+.4f}  1D {e1d:+.4f} MeV')
