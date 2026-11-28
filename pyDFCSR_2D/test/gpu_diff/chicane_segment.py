"""
A 1e6-particle differentiable chicane run with segment checkpointing (CSR2DDiff(segment=k)): the
gradient of the CSR energy loss and emittance growth with respect to the bend angle theta (all four
bends and their edges) and BeamTransform knobs of the initial beam (chirp, bunch-length scale, the
current-profile knots), on the coarse chicane of chicane_gradients.config.

    python chicane_segment.py [N] [k] --device cuda --compile --bands
Appends to benchmark_results/gpu_diff/chicane_segment.txt.
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
import chicane_gradients as CG  # noqa: E402


def main():
    import torch
    import yaml
    from pyDFCSR_2D import CSR2D
    from pyDFCSR_2D.torch_backend.beam_param import BeamTransform
    from pyDFCSR_2D.torch_backend.diff import CSR2DDiff
    args = [a for a in sys.argv[1:] if not a.startswith('--') and a != CG.DEVICE]
    n = int(args[0]) if args else 1000000
    k = int(args[1]) if len(args) > 1 else 22
    dev = CG.DEVICE
    with tempfile.TemporaryDirectory() as tmp:
        f = CG.config(tmp, n)
        cfg = yaml.safe_load(open(f))
        cfg['CSR_computation'].update(torch_bands='--bands' in sys.argv)
        yaml.safe_dump(cfg, open(f, 'w'))
        with contextlib.redirect_stdout(io.StringIO()):
            ref = CSR2D(input_file=f)
        p0 = ref.beam.particle
        base = torch.as_tensor(np.stack([p0.x, p0.px, p0.y, p0.py, p0.z, p0.pz]))
        E0 = ref.beam.init_energy
        bt = BeamTransform(base, device=dev)
        th = torch.tensor(CG.THETA0, dtype=torch.float64, device=dev, requires_grad=True)
        params = {}
        for b, (sgn, edge) in CG.BENDS.items():
            params[(b, 'angle')] = sgn * th
            params[(b, edge)] = sgn * th
        c = bt.beam()
        pz0 = float(c[5].detach().mean())
        if dev.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            d = CSR2DDiff(input_file=f, coords=c, params=params, device=dev, checkpoint=True, segment=k)
            d.run()
        t1 = time.time()
        print(f'forward done in {t1 - t0:.0f} s', flush=True)
        loss = lambda tp: (tp.pz.mean() - pz0) * E0 / 1e3                          # <dE> [keV]
        L = d.segment_gradient(loss, coords=c)
        t2 = time.time()
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev.startswith('cuda') else float('nan')
        q = bt.params['profile_knots'].grad
        lines = [f'=== chicane_segment {time.strftime("%Y-%m-%d %H:%M")}  N = {n}, k = {k}, {dev}'
                 f'{" compiled" if CG.COMPILE else ""}{" GPU bands" if "--bands" in sys.argv else ""}, '
                 f'{len(d.structure)} kicks, {len(d._bounds)} stored beams',
                 f'  forward (no graph) {t1 - t0:.0f} s, segmented gradient {t2 - t1:.0f} s, peak GPU memory {mem:.2f} GB',
                 f'  <dE> = {L:.4f} keV;  d<dE>/dtheta = {float(th.grad):.5e} keV/rad, d/dchirp = {float(bt.params["chirp"].grad):.5e} keV m, '
                 f'd/dz_scale = {float(bt.params["z_scale"].grad):.5e} keV',
                 '  d<dE>/d(profile knot) at w = ' + ', '.join(f'{float(w):+.0f}: {float(g):+.3e}' for w, g in zip(bt.knot_w, q))]
        os.makedirs(CG.OUT, exist_ok=True)
        with open(os.path.join(CG.OUT, 'chicane_segment.txt'), 'a') as fh:
            for line in lines:
                print(line, flush=True)
                fh.write(line + '\n')


if __name__ == '__main__':
    main()
