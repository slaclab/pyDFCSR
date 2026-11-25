import os, sys, numpy as np
T = '/Users/jytang/Library/CloudStorage/GoogleDrive-jytang@stanford.edu/My Drive/Stanford/research/FEL/pyDFCSR_claude/pyDFCSR_near_adapt/pyDFCSR_2D/test'
sys.path.insert(0, os.path.join(T, '..', '..')); sys.path.insert(0, os.path.join(T, 'facet_bc20'))
import dump_kick_state as D
D.RUNS = os.path.join(T, 'benchmark_results', 'bc20_indranil', 'runs')
def study(sim, label, pts):
    ip = sim.integration_params; pos = sim.beam.position
    print(f'{label}: s={pos:.3f}, inbend {sim.inbend}, Rphi {abs(sim.R_rec*sim.phi_rec):.3f} m')
    for rs in (0, 2, 5, 10, 20, 40, 80, 160):
        ip.reach_straight = float(rs); ip.prehistory = 500.0
        vals = []
        for k in pts:
            dbg = sim.get_CSR_wake(pos + sim.CSR_zmesh[k], sim.CSR_xmesh[k])
            vals.append(dbg[0])
        print(f'   reach_straight {rs:4d} (straight {rs*abs(sim.R_rec*sim.phi_rec):6.1f} m): W at the points ' + ' '.join(f'{v:+.4f}' for v in vals), flush=True)
sim = D.load_state(D.state_path('gain_v2', 0.75)); nz = sim.CSR_params.zbins; nx = sim.CSR_params.xbins
pts = [(nx // 2) * nz + nz // 2, (nx // 2) * nz + int(0.75 * nz), (nx - 1) * nz + int(0.9 * nz)]
study(sim, 'gain, inside B1LE', pts)
import sys; sys.exit()
D.RUNS = os.path.join(T, 'benchmark_results', 'chicane_2d_vs_1d', 'closed', 'runs')
sim = D.load_state(D.state_path('2dR_pre2_rs2', 0.6)); nz = sim.CSR_params.zbins; nx = sim.CSR_params.xbins
pts = [(nx // 2) * nz + nz // 2, (nx // 2) * nz + int(0.75 * nz), (nx - 1) * nz + int(0.9 * nz)]
study(sim, 'chicane, B1 exit (in bend)', pts)
