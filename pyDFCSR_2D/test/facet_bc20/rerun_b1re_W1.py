"""
BC20 with the transverse force reduced to its W1 term from the B1RE entrance on.

    cd pyDFCSR_2D/example
    mpirun -np 6 python ../test/facet_bc20/rerun_b1re_W1.py [base_run]        (pydfcsr-mpi)
    conda run -n pydfcsr python ../test/facet_bc20/rerun_b1re_W1.py --report [base_run]

Replays <base_run> (default dfcsr_2d_peak_1M_sz6.5cap_m40_k115) with its own input, so the beam
and the schedule are identical, up to s = 48.02 m (B1RE entrance). From there every kick is
computed twice at the same beam state: with CSR_integration.transverse_terms 'all' (W1 + W2 + W3,
stored only) and with 'W1' (applied). The run is saved under <base_run>_b1reW1 in the usual
run_dfcsr_bc20 format (wakes.h5 holds the applied, W1-only force in B1RE), plus
b1re_W1_<base_run>.npz with both forces at every B1RE kick.

--report: pedestal roughness and size of F for both forms at each B1RE kick, the end beam against
the base run (compare_1M), and b1re_W1_maps.png.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, '..', 'benchmark_results', 'facet_bc20')
RUNS = os.path.join(RES, 'runs')
S_SWITCH = 48.02          # B1RE entrance
BASE = 'dfcsr_2d_peak_1M_sz6.5cap_m40_k115'


def main(base=BASE):
    from mpi4py import MPI
    import run_dfcsr_bc20 as RD
    from pyDFCSR_2D.CSR import CSR2D
    rank = MPI.COMM_WORLD.Get_rank()
    name = f'{base}_b1reW1'
    RD.RUNS[name] = dict(RD.RUNS[base])
    rec = []
    orig = CSR2D.calculate_2D_CSR_parallel

    def patched(self):
        if self.beam.position < S_SWITCH - 1e-9:
            return orig(self)
        ip = self.integration_params
        ip.transverse_terms = 'all'
        orig(self)
        F_all, W_all = self.x_kick.copy(), self.dE_dct.copy()
        ip.transverse_terms = 'W1'
        orig(self)
        if rank == 0:
            rec.append(dict(s=self.beam.position, z=np.asarray(self.CSR_zrange).copy(),
                            xi=np.asarray(self.CSR_xrange_transformed).copy(), W=W_all, F_all=F_all,
                            F_W1=self.x_kick.copy(), W_W1=self.dE_dct.copy()))
            print(f'  [b1re W1] kick at s = {self.beam.position:.4f} m: max|F| all {np.abs(F_all).max():.3e}, '
                  f'W1 {np.abs(self.x_kick).max():.3e} MeV/m', flush=True)
    CSR2D.calculate_2D_CSR_parallel = patched
    RD.main(name)
    if rank == 0 and rec:
        np.savez(os.path.join(HERE, f'b1re_W1_{base}.npz'), s=np.array([r['s'] for r in rec]),
                 **{f'{k}_{i}': r[k] for i, r in enumerate(rec) for k in ('z', 'xi', 'W', 'F_all', 'F_W1', 'W_W1')})
        print(f'saved b1re_W1_{base}.npz ({len(rec)} kicks)', flush=True)


def report(base=BASE):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import median_filter
    import compare_1M
    d = np.load(os.path.join(HERE, f'b1re_W1_{base}.npz'))
    S = d['s']
    n = S.size
    cmap = LinearSegmentedColormap.from_list('div', ['#0d366b', '#2a78d6', '#9ec5f4', '#f0efec',
                                                     '#f4a3a2', '#e34948', '#7a1f1e'])
    print(f'B1RE, {n} kicks; pedestal = z < spike - 15 um; roughness = rms(F - 3-node median along z) / rms F '
          'over the pedestal cells')
    print(f'{"s [m]":>7s} {"max|F| spike all":>16s} {"W1":>9s} {"ped max|F|/spike: all":>21s} {"W1":>6s} '
          f'{"ped rough all":>13s} {"W1":>6s}')
    rows = []
    for i in range(n):
        z, Fa, Fw, W = d[f'z_{i}'], d[f'F_all_{i}'], d[f'F_W1_{i}'], d[f'W_{i}']
        zs = z[np.argmax(np.abs(W).max(0))]
        ped = z < zs - 15e-6
        sp = np.abs(z - zs) < 5e-6

        def st(F):
            m = np.abs(F[:, sp]).max()
            dev = (F - median_filter(F, size=(1, 3), mode='nearest'))[:, ped]
            return m, np.abs(F[:, ped]).max() / m, np.sqrt(np.mean(dev ** 2)) / np.sqrt(np.mean(F[:, ped] ** 2))
        a, w = st(Fa), st(Fw)
        rows.append((a, w))
        print(f'{S[i]:7.3f} {a[0]:16.4f} {w[0]:9.4f} {a[1]:21.3f} {w[1]:6.3f} {a[2]:13.3f} {w[2]:6.3f}')
    A = np.array([[r[0][k] for k in range(3)] for r in rows])
    Wm = np.array([[r[1][k] for k in range(3)] for r in rows])
    print(f'median: spike max|F| all {np.median(A[:,0]):.4f} vs W1 {np.median(Wm[:,0]):.4f} MeV/m; '
          f'pedestal/spike {np.median(A[:,1]):.3f} vs {np.median(Wm[:,1]):.3f}; '
          f'pedestal roughness {np.median(A[:,2]):.3f} vs {np.median(Wm[:,2]):.3f}')
    print()
    compare_1M.main([f'{base}_b1reW1', base, 'dfcsr_1d_1M'])

    picks = [0, n // 3, 2 * n // 3, n - 1]
    fig, axs = plt.subplots(3, len(picks), figsize=(3.4 * len(picks), 8.2))
    for j, i in enumerate(picks):
        z, xi = d[f'z_{i}'] * 1e6, (d[f'xi_{i}'] - d[f'xi_{i}'].mean()) * 1e6
        Fa, Fw = d[f'F_all_{i}'], d[f'F_W1_{i}']
        a = max(np.abs(Fa).max(), np.abs(Fw).max())
        for r, (M, lab) in enumerate(((Fa, 'F = W1 + W2 + W3'), (Fw, 'F = W1 only'), (Fa - Fw, 'W2 + W3'))):
            ax = axs[r, j]
            ax.pcolormesh(z, xi, M, cmap=cmap, vmin=-a, vmax=a, shading='nearest', rasterized=True)
            ax.text(0.03, 0.04, f'{M.min():+.3g} … {M.max():+.3g}', transform=ax.transAxes, fontsize=6.5,
                    bbox=dict(boxstyle='round,pad=0.1', fc='#fcfcfb', ec='none', alpha=0.85))
            ax.tick_params(labelsize=6.5)
            if j == 0:
                ax.set_ylabel(lab + '\nξ [µm]', fontsize=8)
            if r == 0:
                ax.set_title(f's = {S[i]:.2f} m', loc='left', fontsize=9)
            if r == 2:
                ax.set_xlabel('z [µm]  (head at +z)', fontsize=8)
    fig.suptitle(f'{base}: transverse force in B1RE, same beam state, one colour scale per column (MeV/m)',
                 x=0.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(RES, 'b1re_W1_maps.png'), dpi=140)
    print('wrote b1re_W1_maps.png')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--report':
        report(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
