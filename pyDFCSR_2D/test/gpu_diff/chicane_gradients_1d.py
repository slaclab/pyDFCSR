"""
The chicane gradient demo (chicane_gradients.py) with the 1D model (CSR1DDiff): d<dE>/d(theta, chirp)
on the same coarse 20k-particle chicane, theta driving all four bends and their edges, the chirp
derivative from the coordinate gradient. Linear kick interpolation (CSR_1D.kick_interp), cic
deposit. Appends to benchmark_results/gpu_diff/chicane_gradients.txt.

    python chicane_gradients_1d.py [N] [--device cuda]
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
sys.path.insert(0, HERE)
import chicane_gradients as CG  # noqa: E402


def main():
    import torch
    from pyDFCSR_2D.csr1d import CSR1D
    from pyDFCSR_2D.torch_backend.csr1d_diff import CSR1DDiff
    args = [a for a in sys.argv[1:] if not a.startswith('--') and a != CG.DEVICE]
    n = int(args[0]) if args else 20000
    with tempfile.TemporaryDirectory() as tmp:
        f = CG.config(tmp, n)
        cfg = yaml.safe_load(open(f))
        cfg['CSR_1D'] = dict(kick_interp='linear', **({'bin_mode': 'peak', 'bins_per_fwhm': 20} if '--peak' in sys.argv else {}))
        yaml.safe_dump(cfg, open(f, 'w'))
        with contextlib.redirect_stdout(io.StringIO()):
            ref = CSR1D(input_file=f)
        p0 = ref.beam.particle
        c0 = torch.as_tensor(np.stack([p0.x, p0.px, p0.y, p0.py, p0.z, p0.pz])).to(CG.DEVICE)
        E0 = ref.beam.init_energy
        c = c0.clone().requires_grad_(True)
        th = torch.tensor(CG.THETA0, dtype=torch.float64, device=CG.DEVICE, requires_grad=True)
        params = {}
        for b, (sgn, edge) in CG.BENDS.items():
            params[(b, 'angle')] = sgn * th
            params[(b, edge)] = sgn * th
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            d = CSR1DDiff(input_file=f, coords=c, params=params, device=CG.DEVICE, checkpoint=True)
            d.run()
        t1 = time.time()
        dE = (d.tp.pz.mean() - c[5].mean()) * E0 / 1e3
        gt, gc = torch.autograd.grad(dE, (th, c))
        t2 = time.time()
        CG.log(f'[grad 1D] device {CG.DEVICE}, N = {n}, bins {"peak 20/FWHM" if "--peak" in sys.argv else "extent 300"}: forward {t1 - t0:.0f} s, backward {t2 - t1:.0f} s, {len(d.structure)} kicks')
        CG.log(f'  1D <dE> = {float(dE):.4f} keV   d<dE>/dtheta = {float(gt):.5e} keV/rad   '
               f'd<dE>/dh = {float((gc[5] * c0[4]).sum()):.5e} keV m')


if __name__ == '__main__':
    main()
