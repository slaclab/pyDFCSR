"""
How many x' nodes does the by-parts (ibp) longitudinal form need? Whole wake mesh on the pickled
loss state at B2RE (dump_state.py loss 37.74 35.5), shipped and ibp forms at 200 / 100 / 50 / 25
uniform x' nodes, against the 8x x' reference stored by far_adapt_mesh.py
(far_adapt_mesh_loss_s37.74.npz): max |dW| and rms dW relative to the horn's W, max |dF| relative
to the horn's F, wall time of the mesh.

    conda run -n pydfcsr python ../test/bc20_indranil/ibp_nodes.py [run s_target]
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'facet_bc20'))
import dump_kick_state as D  # noqa: E402

D.RUNS = os.path.join(HERE, '..', 'benchmark_results', 'bc20_indranil', 'runs')


def main(run='loss', s_target=37.74):
    sim = D.load_state(D.state_path(run, s_target))
    ip = sim.integration_params
    pos = sim.beam.position
    nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins
    ref = np.load(os.path.join(D.RUNS, f'far_adapt_mesh_{run}_s{float(s_target):.2f}.npz'))
    Wr, Fr = ref["W_x' x8 (reference)"], ref["F_x' x8 (reference)"]
    horn = np.abs(Wr).max(0).argmax()
    wh, fh = np.abs(Wr[:, horn]).max(), np.abs(Fr[:, horn]).max()
    print(f'{run} at s = {pos:.4f}: reference horn W {wh:.3f}, F {fh:.4f} MeV/m; ibp_j0 {ip.ibp_j0}')
    print(f'{"form":10s} {"x\' nodes":>8s} {"time":>6s} {"max|dW|/W_horn":>15s} {"rms dW/W_horn":>14s} {"max|dF|/F_horn":>15s}')
    for ibp in (False, True):
        for xb in (200, 100, 50, 25):
            ip.ibp, ip.xbins = ibp, xb
            t0 = time.time()
            r = np.array([sim.get_CSR_wake(pos + z, x) for z, x in zip(sim.CSR_zmesh, sim.CSR_xmesh)])
            dt = time.time() - t0
            W, F = r[:, 0].reshape(nx, nz), r[:, 1].reshape(nx, nz)
            print(f'{"ibp" if ibp else "shipped":10s} {xb:8d} {dt:5.0f}s {np.abs(W - Wr).max()/wh*100:14.2f}% {np.sqrt(np.mean((W - Wr)**2))/wh*100:13.3f}% {np.abs(F - Fr).max()/fh*100:14.1f}%', flush=True)
    ip.ibp, ip.xbins = False, 200


if __name__ == '__main__':
    main(*sys.argv[1:])
