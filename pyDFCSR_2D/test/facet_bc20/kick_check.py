"""
Quadrature check of one later BC20 2D kick: replays a run (MPI) to the first kick at or after
s_target, then recomputes mesh rows of the transverse force F and the wake W with refined
quadratures, split by region.

    cd pyDFCSR_2D/example
    mpirun -np 6 python ../test/facet_bc20/kick_check.py <run> <s_target> [rows]   (pydfcsr-mpi)
    conda run -n pydfcsr python ../test/facet_bc20/kick_check.py --report <run> <s_target>

Rows default to the two central rows of the x mesh. Variants (near_field_check.VARIANTS style):
  base         the run's settings
  far4/16/40   the far s' region on 4x, 16x, 40x as many uniform nodes (in a bend it is uniform
               at 2 far_zbins = 400): an observer behind a ~3 um spike sees it over only a few
               mm of s', so 400 nodes over ~1 m put 2-3 nodes across it
  sp4/16/64    4x, 16x, 64x finer near-region s' cells (near_cell and near_grade both divided):
               an observer in the pedestal sees the spike AHEAD of it, inside the near region, at
               z_ret ~ z + 2u, so the 3 um spike is crossed over ~1.5 um of s'
  xp4          4x the x' nodes; sp16xp4 both
SWEEP variants are also evaluated along the first row at 0.5 um observer spacing through the
pedestal (spike - 120 um .. spike - 5 um).
(First run, far4/16/40: the far region carries none of the pedestal F, refining it changes nothing.)
Writes kick_check_<run>_s<s>.npz and benchmark_results/facet_bc20/kick_check_<run>_s<s>.png.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, '..', 'benchmark_results', 'facet_bc20')
RUNS = os.path.join(RES, 'runs')

from near_field_check import wake_parts, Stop  # noqa: E402

VARIANTS = {'base': {}, 'adapt_x': {'near_adapt': True, 'near_adapt_x': True},
            'adapt_x_e05': {'near_adapt': True, 'near_adapt_x': True, 'near_adapt_eps': 0.5},
            'adapt_x_dz01': {'near_adapt': True, 'near_adapt_x': True, 'near_adapt_scan_dz': 0.4},
            'adapt_x_e05_dz01': {'near_adapt': True, 'near_adapt_x': True, 'near_adapt_eps': 0.5, 'near_adapt_scan_dz': 0.4},
            'sp64_xp4': dict(near_cell=64, near_grade=64, xbins=4)}
SWEEP = ('base', 'adapt_x', 'adapt_x_e05', 'adapt_x_dz01', 'adapt_x_e05_dz01', 'sp64_xp4')   # swept at 0.5 um through the pedestal


def tag(s_target):
    return f's{float(s_target):.2f}'


def analyse(sim, run, s_target, rows, comm, full_mesh):
    rank, size = comm.Get_rank(), comm.Get_size()
    ip = sim.integration_params
    pos = sim.beam.position
    nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins
    S, X = pos + sim.CSR_zmesh, sim.CSR_xmesh
    # the run's own wake at this kick, for the full mesh (computed in parallel by the replay)
    full_mesh()
    W, F = sim.dE_dct.copy(), sim.x_kick.copy()
    keep = {k: getattr(ip, k) for k in ('near_cell', 'near_grade', 'xbins', 'far_zbins', 'near_adapt', 'near_adapt_x',
                                         'near_adapt_eps', 'near_adapt_scan_dz')}
    far_nodes = sim._far_region_nodes
    pts = [(a, r, iz) for a, r in enumerate(rows) for iz in range(nz)]
    zs = sim.CSR_zrange[np.argmax(np.abs(W[rows]).max(0))]
    zf = np.arange(zs - 120e-6, zs - 5e-6, 0.5e-6)
    xf = sim.CSR_xrange_transformed[rows[0]] + np.polyval(sim.beam.slope, zf) + sim._mesh_ridge(zf)
    res = {}
    for name, fac in VARIANTS.items():
        for k, v in keep.items():
            setattr(ip, k, v)
        sim._far_region_nodes = far_nodes
        for k, f in fac.items():
            if k == '_far':
                n = int(2 * keep['far_zbins'] * f)
                sim._far_region_nodes = lambda s, x, t, a, b, transient=None, n=n: (
                    far_nodes(s, x, t, a, b, transient) if (transient if transient is not None
                                                            else sim._exit_transient())
                    else np.linspace(a, b, n))
            elif k in ('near_adapt', 'near_adapt_x'):
                setattr(ip, k, bool(f))
            elif k in ('near_cell', 'near_grade'):
                setattr(ip, k, keep[k] / f)
            elif isinstance(keep[k], float):
                setattr(ip, k, keep[k] * f)        # near_adapt_eps, near_adapt_scan_dz: factors
            else:
                setattr(ip, k, int(keep[k] * f))
        t0 = time.time()
        out = np.zeros((len(rows), nz, 4, 2))
        for j, (a, r, iz) in enumerate(pts):
            if j % size != rank:
                continue
            i = r * nz + iz
            p = wake_parts(sim, S[i], X[i])
            out[a, iz] = [p['far'], p['near'], p['prev'], p['patch']]
        tot = np.zeros_like(out)
        comm.Allreduce(out, tot)
        res[name] = tot
        if name in SWEEP:
            sw = np.zeros((zf.size, 4, 2))
            for j in range(zf.size):
                if j % size == rank:
                    p = wake_parts(sim, pos + zf[j], xf[j])
                    sw[j] = [p['far'], p['near'], p['prev'], p['patch']]
                    if name != 'sp64_xp4' and j == zf.size // 2:
                        nodes = sim._near_region_nodes(pos + zf[j], *sim._layout_bounds(pos + zf[j], xf[j])[1],
                                                       x=xf[j], t=pos)
                        print(f'  [rank {rank}] {name}: near region at z = {zf[j]*1e6:+.1f} um: {nodes.size} s nodes '
                              f'(adapt: {getattr(sim, "_last_adapt", None)}, x: {getattr(sim, "_last_adapt_x", None)}), '
                              f'cell at u = (z_spike - z)/2: '
                              f'{np.diff(nodes)[np.searchsorted(nodes, pos + zf[j] + 0.5*(zs - zf[j])) - 1]*1e6:.2f} um; '
                              f'sigma_xi {sim.beam._sigma_x_transform*1e6:.2f} um', flush=True)
            swt = np.zeros_like(sw)
            comm.Allreduce(sw, swt)
            res['sweep_' + name] = swt
        if rank == 0:
            print(f'  {name:7s} {(time.time() - t0) * size / len(pts) * 1e3:8.1f} ms/point', flush=True)
    for k, v in keep.items():
        setattr(ip, k, v)
    sim._far_region_nodes = far_nodes
    if rank == 0:
        tr = sim.DF_tracker
        np.savez(os.path.join(HERE, f'kick_check_{run}_{tag(s_target)}.npz'), s=pos, z=sim.CSR_zrange,
                 xi=sim.CSR_xrange_transformed, W=W, F=F, rows=np.array(rows), window=np.array(tr.window),
                 R=sim.R_rec, sweep_z=zf, **{(n if n.startswith('sweep_') else f'{n}_parts'): v
                                             for n, v in res.items()})
        print(f'kick at s = {pos:.4f} m saved', flush=True)
        report(run, s_target)


def report(run, s_target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import median_filter
    d = np.load(os.path.join(HERE, f'kick_check_{run}_{tag(s_target)}.npz'))
    z, rows = d['z'], d['rows']
    names = [k[:-6] for k in d.files if k.endswith('_parts')]
    names = [n for n in names if n in VARIANTS]
    zs = z[np.argmax(np.abs(d['W'][rows]).max(0))]
    ped = z < zs - 15e-6
    print(f'\n{run}, kick at s = {float(d["s"]):.4f} m (R = {float(d["R"]):.2f} m); rows {list(rows)}; spike at '
          f'z = {zs*1e6:+.1f} um; pedestal = z < spike - 15 um')
    print(f'{"variant":8s} {"F ped rough":>11s} {"F ped max/F spike max":>22s} {"F far share (ped rms)":>22s} '
          f'{"F at spike col.":>16s} {"W ped rough":>11s}')
    for n in names:
        P = d[f'{n}_parts']                       # rows, nz, region, (W, F)
        F = P[..., 1].sum(2)
        Wt = P[..., 0].sum(2)
        Ff = P[:, :, 0, 1]
        sp = np.argmin(np.abs(z - zs))
        rough = lambda M: np.sqrt(np.mean((M - median_filter(M, size=(1, 3), mode='nearest'))[:, ped] ** 2)) / \
            np.sqrt(np.mean(M[:, ped] ** 2))
        print(f'{n:8s} {rough(F):11.3f} {np.abs(F[:, ped]).max() / np.abs(F[:, sp]).max():22.3f} '
              f'{np.sqrt(np.mean(Ff[:, ped]**2)) / np.sqrt(np.mean(F[:, ped]**2)):22.3f} '
              f'{F[:, sp].mean():+16.4e} {rough(Wt):11.3f}')
    col = {'base': '#0b0b0b', 'adapt': '#2a78d6', 'adapt_x': '#1baf7a', 'adapt_x_e05': '#eda100',
           'adapt_x_dz01': '#9ec5f4', 'adapt_x_e05_dz01': '#e34948', 'sp64_xp4': '#7a1f1e'}
    fig, axs = plt.subplots(2, len(rows), figsize=(5.5 * len(rows), 6.5), squeeze=False)
    for a, r in enumerate(rows):
        for j, (q, lab) in enumerate(((1, 'transverse force F'), (0, 'longitudinal W'))):
            ax = axs[j, a]
            for n in names:
                ax.plot(z * 1e6, d[f'{n}_parts'][a, :, :, q].sum(1), color=col.get(n, '#898781'),
                        lw=2.0 if n in ('base', 'sp64_xp4') else 1.1, marker='o' if n == 'base' else None, ms=2.5,
                        label=n)
            ax.axvline(zs * 1e6, color='#e1e0d9', lw=0.8)
            ax.axhline(0, color='#e1e0d9', lw=0.8)
            ax.set_title(f'{lab}, mesh row {r}', loc='left', fontsize=9)
            ax.set_xlabel('z [µm]  (head at +z)')
            ax.set_ylabel('[MeV/m]')
    axs[0, 0].legend(fontsize=7, frameon=False)
    if 'sweep_z' in d.files:
        ax = axs[1, -1]
        ax.cla()
        for n in SWEEP:
            if f'sweep_{n}' in d.files:
                ax.plot(d['sweep_z'] * 1e6, d[f'sweep_{n}'][:, :, 1].sum(1), color=col.get(n, '#e34948'),
                        lw=1.3, label=f'{n}, 0.5 µm observers')
        ax.plot(z * 1e6, d['base_parts'][0, :, :, 1].sum(1), 'o', color='#0b0b0b', ms=3, label='mesh nodes (base)')
        ax.set_xlim(d['sweep_z'][0] * 1e6, d['sweep_z'][-1] * 1e6)
        ax.set_title(f'transverse F through the pedestal, row {rows[0]}, resolved in observer z', loc='left',
                     fontsize=9)
        ax.set_xlabel('z [µm]')
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f'{run}: kick at s = {float(d["s"]):.2f} m, mesh rows recomputed with refined far / near / x′ '
                 'quadratures', x=0.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(RES, f'kick_check_{run}_{tag(s_target)}.png'), dpi=130)
    print(f'wrote kick_check_{run}_{tag(s_target)}.png')


def main(run, s_target, rows=None):
    from mpi4py import MPI
    from pyDFCSR_2D import CSR2D
    comm = MPI.COMM_WORLD
    s_target = float(s_target)
    f = os.path.join(RUNS, 'inputs', f'{run}.yaml')
    sim = CSR2D(input_file=f, parallel=comm.Get_size() > 1)
    sim.CSR_params.write_wakes = False
    nx = sim.CSR_params.xbins
    rows = [int(r) for r in rows.split(',')] if rows else [nx // 2 - 1, nx // 2]
    orig = sim.calculate_2D_CSR_parallel if sim.parallel else sim.calculate_2D_CSR

    def kick():
        if sim.beam.position < s_target - 1e-9:
            return orig()
        analyse(sim, run, s_target, rows, comm, orig)
        raise Stop
    if sim.parallel:
        sim.calculate_2D_CSR_parallel = kick
    else:
        sim.calculate_2D_CSR = kick
    try:
        sim.run()
    except Stop:
        pass


if __name__ == '__main__':
    if sys.argv[1] == '--report':
        report(sys.argv[2], sys.argv[3])
    else:
        main(*sys.argv[1:])
