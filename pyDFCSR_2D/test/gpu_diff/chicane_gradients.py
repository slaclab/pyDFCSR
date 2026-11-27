"""
Gradients of the 4-dipole benchmark chicane's CSR effects with the differentiable run
(torch_backend.diff.CSR2DDiff): d(energy loss, emittance growth)/d(bend angle theta, initial chirp).

theta drives all four bends and their edges (B1, B4: +theta, B2, B3: -theta, rectangular edges
E = +-theta as in chicane_lattice.yaml), so the CSR geometry follows it as well. The chirp h is
the linear z-pz correlation of the initial beam: pz -> pz + dh z, so d/dh = sum_i dL/dpz_i z_i
comes from the coordinate gradient, no extra run.

A coarse setup so that it runs on the Mac CPU (N particles, 100 x 150 deposition, an 8 x 24 mesh,
60 x' nodes); it is a demonstration and a check of the gradients, not a converged chicane result.

    python chicane_gradients.py time            # numba CSR2D forward only (cost, reference)
    python chicane_gradients.py grad [N]        # forward + backward, then the theta FD check
Writes benchmark_results/gpu_diff/chicane_gradients.txt (appends) and .npz.
"""
import contextlib
import io
import os
import sys
import tempfile
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
import chicane_2d_vs_1d as C  # noqa: E402

OUT = os.path.join(HERE, '..', 'benchmark_results', 'gpu_diff')
THETA0 = 0.0483
BENDS = {'B1': (+1, 'E2'), 'B2': (-1, 'E1'), 'B3': (-1, 'E2'), 'B4': (+1, 'E1')}


def config(tmp, n):
    beam = yaml.safe_load(open(os.path.join(C.INP, 'chicane_init_beam.yaml')))
    beam['n_particle'] = int(n)
    bf = os.path.join(tmp, 'beam.yaml')
    yaml.safe_dump(beam, open(bf, 'w'))
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in C.BASE.items()}
    cfg['input_beam'] = {'style': 'distgen', 'distgen_input_file': bf}
    cfg['particle_deposition'] = dict(cfg['particle_deposition'], xbins=100, zbins=150, vx_model='affine')
    cfg['CSR_integration'] = {'n_formation_length': 1, 'xbins': 60, 'zbins': 100, 'far_zbins': 100,
                              'drift_cutoff': 0.0}
    cfg['CSR_computation'] = dict(cfg['CSR_computation'], xbins=8, zbins=24, workdir=tmp, write_wakes=False,
                                  torch_chunk=1 << 19)
    f = os.path.join(tmp, 'cfg.yaml')
    yaml.safe_dump(cfg, open(f, 'w'))
    return f


def emittance(x, px, pz):
    """Projected normalised-free emittance of (x, px) with the linear dispersion in pz removed."""
    import torch
    lib = torch if torch.is_tensor(x) else np
    d = pz - pz.mean()

    def detrend(a):
        a = a - a.mean()
        return a - (a * d).mean() / (d * d).mean() * d
    xx, pp = detrend(x), detrend(px)
    return lib.sqrt((xx * xx).mean() * (pp * pp).mean() - (xx * pp).mean() ** 2)


def log(msg):
    os.makedirs(OUT, exist_ok=True)
    print(msg, flush=True)
    with open(os.path.join(OUT, 'chicane_gradients.txt'), 'a') as fh:
        fh.write(msg + '\n')


def main(mode, n=20000):
    import torch
    from pyDFCSR_2D import CSR2D
    from pyDFCSR_2D.torch_backend.diff import CSR2DDiff
    with tempfile.TemporaryDirectory() as tmp:
        f = config(tmp, n)
        if mode == 'time':
            t0 = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                sim = CSR2D(input_file=f)
                sim.run()
            p = sim.beam.particle
            log(f'[time] numba CSR2D, N = {n}: {time.time() - t0:.0f} s, {len(sim.lattice.schedule.s_nodes)} nodes, '
                f'<dE> = {p.pz.mean() * sim.beam.init_energy / 1e3:.3f} keV, eps = {emittance(p.x, p.px, p.pz) * 1e9:.4f} nm')
            return

        def run(theta, c, structure=None):
            params = {}
            for b, (sgn, edge) in BENDS.items():
                params[(b, 'angle')] = sgn * theta
                params[(b, edge)] = sgn * theta
            with contextlib.redirect_stdout(io.StringIO()):
                d = CSR2DDiff(input_file=f, coords=c, params=params, structure=structure, checkpoint=True)
                d.run()
            return d
        with contextlib.redirect_stdout(io.StringIO()):
            ref = CSR2D(input_file=f)
        p0 = ref.beam.particle
        c0 = torch.as_tensor(np.stack([p0.x, p0.px, p0.y, p0.py, p0.z, p0.pz]))
        E0 = ref.beam.init_energy
        eps0 = float(emittance(c0[0], c0[1], c0[5]))
        c = c0.clone().requires_grad_(True)
        th = torch.tensor(THETA0, dtype=torch.float64, requires_grad=True)
        t0 = time.time()
        d = run(th, c)
        t1 = time.time()
        tp = d.tp
        dE = (tp.pz.mean() - c[5].mean()) * E0 / 1e3                      # keV
        deps = (emittance(tp.x, tp.px, tp.pz) - eps0) * 1e9               # nm
        g = {}
        for name, L in (('dE', dE), ('deps', deps)):
            gt, gc = torch.autograd.grad(L, (th, c), retain_graph=(name == 'dE'))
            g[name] = (float(gt), float((gc[5] * c0[4]).sum()))
        t2 = time.time()
        log(f'[grad] N = {n}: forward {t1 - t0:.0f} s, both backward passes {t2 - t1:.0f} s, {len(d.structure)} kicks')
        log(f'  <dE> = {float(dE):.4f} keV   d<dE>/dtheta = {g["dE"][0]:.5e} keV/rad   d<dE>/dh = {g["dE"][1]:.5e} keV m')
        log(f'  deps = {float(deps):.5f} nm   deps/dtheta = {g["deps"][0]:.5e} nm/rad   deps/dh = {g["deps"][1]:.5e} nm m')
        # the theta derivatives against central differences, structure replayed
        h = 1e-4 * THETA0
        T = lambda v: torch.tensor(v, dtype=torch.float64)
        vals = []
        for sgn in (+1, -1):
            dd = run(T(THETA0 + sgn * h), c0, d.record)
            vals.append(((float(dd.tp.pz.mean()) - float(c0[5].mean())) * E0 / 1e3,
                         (float(emittance(dd.tp.x, dd.tp.px, dd.tp.pz)) - eps0) * 1e9))
        fd = [(vals[0][k] - vals[1][k]) / (2 * h) for k in range(2)]
        log(f'  FD (h = 1e-4 theta, structure replayed): d<dE>/dtheta {fd[0]:.5e} (rel {abs(fd[0] - g["dE"][0]) / abs(fd[0]):.1e}), '
            f'deps/dtheta {fd[1]:.5e} (rel {abs(fd[1] - g["deps"][0]) / abs(fd[1]):.1e})')
        # the chirp derivative against central differences in h (pz += dh z)
        hh = 1e-3 * float(c0[5].std() / c0[4].std())
        vals = []
        for sgn in (+1, -1):
            cc = c0.clone()
            cc[5] = cc[5] + sgn * hh * cc[4]
            dd = run(T(THETA0), cc, d.record)
            vals.append(((float(dd.tp.pz.mean()) - float(cc[5].mean())) * E0 / 1e3,
                         (float(emittance(dd.tp.x, dd.tp.px, dd.tp.pz)) - float(emittance(cc[0], cc[1], cc[5]))) * 1e9))
        log(f'  FD in the chirp: d<dE>/dh {(vals[0][0] - vals[1][0]) / (2 * hh):.5e}, '
            f'(eps growth)/dh {(vals[0][1] - vals[1][1]) / (2 * hh):.5e} '
            f'[AD deps/dh above holds the initial emittance fixed; this one subtracts that of the shifted beam]')
        np.savez(os.path.join(OUT, 'chicane_gradients.npz'), dE=float(dE), deps=float(deps),
                 g_dE=g['dE'], g_deps=g['deps'], fd_theta=fd, grad_coords_pz=c.grad if c.grad is not None else 0)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20000)
