"""
GPU benchmark of the torch backend (branch gpu_diff), for an S3DF GPU node (environment-gpu.yml;
batch file gpu_benchmark.slurm next to this script).

  1. one wake mesh at a kick of the coarse 4-dipole chicane (chicane_gradients.config, N particles),
     taken in B2 where the integrals are largest: numba (serial) vs torch on CPU vs torch on CUDA,
     with the agreement and the layout / evaluation split
  2. the differentiable golden run (test/golden), forward + backward on CPU and on CUDA, with the
     gradient's agreement between the two devices

    python gpu_benchmark.py [N] [chunk] [s]      N particles (20000), torch_chunk (2^21), kick at s (6.0 m)
Prints a summary; appends it to benchmark_results/gpu_diff/gpu_benchmark.txt.
"""
import contextlib
import io
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'golden'))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
OUT = os.path.join(HERE, '..', 'benchmark_results', 'gpu_diff')


def log(msg):
    os.makedirs(OUT, exist_ok=True)
    print(msg, flush=True)
    with open(os.path.join(OUT, 'gpu_benchmark.txt'), 'a') as fh:
        fh.write(msg + '\n')


def sync(dev):
    import torch
    if dev.startswith('cuda'):
        torch.cuda.synchronize()


def wake_mesh(n, chunk, stop_s=6.0):
    import torch
    import chicane_gradients as CG
    from pyDFCSR_2D import CSR2D
    from pyDFCSR_2D.torch_backend.evaluate import TorchWake
    with tempfile.TemporaryDirectory() as tmp:
        f = CG.config(tmp, n)
        with contextlib.redirect_stdout(io.StringIO()):
            sim = CSR2D(input_file=f)
            sim.CSR_params.compute_CSR = 1
            sim.run(stop_time=stop_s)                       # in B2 (5.606 - 6.106 m)
        sim.get_CSR_mesh()
        N = sim.CSR_params.xbins * sim.CSR_params.zbins
        ks = np.arange(N)
        t0 = time.time()
        ref = np.array(sim._numba_wakes(ks))
        t_nb = time.time() - t0
        log(f'[mesh] chicane N = {n} at s = {sim.beam.position:.3f} m, {N} mesh points: numba (serial) {t_nb:.1f} s')
        devs = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for dev in devs:
            tw = TorchWake(sim, device=dev, chunk=chunk)
            s = sim.beam.position + np.asarray(sim.CSR_zmesh)[ks]
            x = np.asarray(sim.CSR_xmesh)[ks]
            tw.compute(s[:4], x[:4])                         # warm-up
            sync(dev)
            t0 = time.time()
            dE, xk = tw.compute(s, x)
            sync(dev)
            t = time.time() - t0
            err = max(np.abs(dE - ref[0]).max() / np.abs(ref[0]).max(), np.abs(xk - ref[1]).max() / np.abs(ref[1]).max())
            log(f'  torch {dev:5s}: {t:.1f} s (layout {tw.stats["t_layout"]:.1f} s, evaluation {tw.stats["t_eval"]:.1f} s, '
                f'{tw.stats["columns"]} columns), max rel diff to numba {err:.1e}, speed-up vs numba {t_nb / t:.1f}x '
                f'(evaluation only {(t_nb - tw.stats["t_layout"]) / max(tw.stats["t_eval"], 1e-9):.1f}x)')


def golden_grad(chunk):
    import torch
    import make_golden_2d as G
    from pyDFCSR_2D.torch_backend.diff import CSR2DDiff
    c0 = torch.as_tensor(G.beam_coords().T.copy())
    res = {}
    devs = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    for dev in devs:
        with tempfile.TemporaryDirectory() as tmp:
            f = G.config(tmp, None, {'torch_chunk': chunk}, None)
            c = c0.clone().to(dev).requires_grad_(True)
            ang = torch.tensor(0.05, dtype=torch.float64, device=dev, requires_grad=True)
            sync(dev)
            t0 = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                d = CSR2DDiff(input_file=f, coords=c, params={('B1', 'angle'): ang}, device=dev, checkpoint=True)
                d.run()
            L = d.tp.pz.mean() * 1e6 + d.tp.px.std() * 1e3
            sync(dev)
            t1 = time.time()
            L.backward()
            sync(dev)
            t2 = time.time()
            res[dev] = (float(ang.grad), c.grad.detach().cpu())
            mem = (f', peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB'
                   if dev.startswith('cuda') else '')
            log(f'[grad] golden on {dev}: forward {t1 - t0:.1f} s, backward {t2 - t1:.1f} s{mem}, dL/dangle {res[dev][0]:.10e}')
    if 'cuda' in res:
        a, b = res['cpu'], res['cuda']
        log(f'  cpu vs cuda: dL/dangle rel {abs(a[0] - b[0]) / abs(a[0]):.1e}, '
            f'coordinate gradient rel {float((a[1] - b[1]).abs().max() / a[1].abs().max()):.1e}')


if __name__ == '__main__':
    import torch
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 1 << 21
    name = torch.cuda.get_device_name() if torch.cuda.is_available() else 'no CUDA device'
    log(f'=== gpu_benchmark {time.strftime("%Y-%m-%d %H:%M")}  torch {torch.__version__}  {name}  '
        f'cpu threads {torch.get_num_threads()}')
    wake_mesh(n, chunk, float(sys.argv[3]) if len(sys.argv) > 3 else 6.0)
    if '--no-grad' not in sys.argv:
        golden_grad(min(chunk, 1 << 19))
