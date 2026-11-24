import os, sys, time, numpy as np
HERE='/Users/jytang/Library/CloudStorage/GoogleDrive-jytang@stanford.edu/My Drive/Stanford/research/FEL/pyDFCSR_claude/pyDFCSR_near_adapt/pyDFCSR_2D/test/bc20_indranil'
sys.path.insert(0, os.path.join(HERE, '..', '..', '..')); sys.path.insert(0, os.path.join(HERE, '..', 'facet_bc20'))
import dump_kick_state as D
D.RUNS = os.path.join(HERE, '..', 'benchmark_results', 'bc20_indranil', 'runs')
sim = D.load_state(D.state_path('loss', 37.74)); ip = sim.integration_params; pos = sim.beam.position
nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins; ip.xbins = 100
zm, xm = sim.CSR_zmesh, sim.CSR_xmesh
orig = sim._layout_bounds
DMIN = [0.0]
def patched(s, x):
    (s1, s3), (s2b, s4) = orig(s, x)
    d = s - s3
    if d < DMIN[0]:
        s3 = max(0.0, s - DMIN[0]); s1 = max(0.0, s3 - ip.n_formation_length * sim.formation_length)
    return (s1, s3), (s2b, s4)
sim._layout_bounds = patched
rows = {'tail z=-65': np.flatnonzero(np.abs(zm[:nz] + 64.9e-6) < 0.3e-6)[0], 'horn z=-9': np.flatnonzero(np.abs(zm[:nz] + 8.9e-6) < 0.3e-6)[0]}
far_cell = ip.n_formation_length * sim.formation_length / (2 * 200)
print(f'far cell at far_zbins 200: {far_cell*1e3:.2f} mm; near_grade {ip.near_grade}; sigma_x {sim.beam._sigma_x*1e6:.0f} um, mean x {sim.beam._mean_x*1e6:+.0f} um')
for lab, iz in rows.items():
    idx = np.arange(nx) * nz + iz
    print(f'\n{lab} um row, x rows [um]: ' + ' '.join(f'{xm[k]*1e6:+7.0f}' for k in idx))
    ref = None
    for dmin_lab, dmin in (('as is', 0.0), ('d>=far_cell/grade/2', far_cell / ip.near_grade / 2), ('d>=far_cell/grade', far_cell / ip.near_grade)):
        DMIN[0] = dmin
        for fz in (200, 400):
            ip.far_zbins = fz
            t1 = time.time(); W = np.array([sim.get_CSR_wake(pos + zm[k], xm[k])[0] for k in idx]); dt = time.time() - t1
            nn = sim._last_region_nodes
            if ref is None: ref = W
            print(f'  {dmin_lab:22s} far_zbins {fz}: ' + ' '.join(f'{v:+7.3f}' for v in W) + f'   | nodes far/near {nn[0]}/{nn[-1]}, {dt:4.1f} s')
