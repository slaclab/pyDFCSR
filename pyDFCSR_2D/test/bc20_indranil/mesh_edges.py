import os, sys, numpy as np
HERE='/Users/jytang/Library/CloudStorage/GoogleDrive-jytang@stanford.edu/My Drive/Stanford/research/FEL/pyDFCSR_claude/pyDFCSR_near_adapt/pyDFCSR_2D/test/bc20_indranil'
sys.path.insert(0, os.path.join(HERE, '..', '..', '..')); sys.path.insert(0, os.path.join(HERE, '..', 'facet_bc20'))
import dump_kick_state as D
from pyDFCSR_2D.peakfind import charge_nodes
D.RUNS = os.path.join(HERE, '..', 'benchmark_results', 'bc20_indranil', 'runs')
def spacing_report(label, nodes, q):
    h = np.diff(nodes); frac = np.histogram(q, bins=nodes)[0] / q.size
    print(f'   {label}: span [{nodes[0]*1e6:+.1f}, {nodes[-1]*1e6:+.1f}] um, {nodes.size} nodes; spacing edge/core: first 3 {h[:3]*1e6}, last 3 {h[-3:]*1e6}, min {h.min()*1e6:.2f}, median {np.median(h)*1e6:.2f} um')
    print(f'      charge per cell [%]: first 3 {np.round(frac[:3]*100, 3)}, last 3 {np.round(frac[-3:]*100, 3)}, max {frac.max()*100:.2f}; cells with < 0.05% charge: {(frac < 5e-4).sum()}')
for run, st in (('loss', 37.74), ('gain', 11.30), ('loss', 0.15)):
    sim = D.load_state(D.state_path(run, st)); cp = sim.CSR_params; pos = sim.beam.position
    xt = sim.beam.x_transform - sim._mesh_ridge(sim.beam.z); z = sim.beam.z
    res_z, res_x = sim._deposition_resolution()
    print(f'{run} s={pos:.2f}: frame {sim.DF_tracker.frame}, mesh_x_quantile {cp.mesh_x_quantile}, sigma_x_t {xt.std()*1e6:.0f} um, sigma_z {z.std()*1e6:.1f} um, res_x {res_x*1e6:.1f} res_z {res_z*1e6:.2f} um')
    lo, hi = np.quantile(xt, [cp.mesh_x_quantile, 1 - cp.mesh_x_quantile])
    xq_clip = np.clip(xt, lo, hi); xq_drop = xt[(xt >= lo) & (xt <= hi)]
    spacing_report('x, clipped (current)', charge_nodes(xq_clip, cp.xbins, cp.mesh_uniform_frac, smooth=cp.mesh_smooth * res_x, pad=cp.mesh_pad, max_ratio=cp.mesh_max_ratio), xt)
    spacing_report('x, dropped outside   ', charge_nodes(xq_drop, cp.xbins, cp.mesh_uniform_frac, smooth=cp.mesh_smooth * res_x, pad=cp.mesh_pad, max_ratio=cp.mesh_max_ratio), xt)
    spacing_report('z, all particles     ', charge_nodes(z, cp.zbins, cp.mesh_uniform_frac, smooth=cp.mesh_smooth * res_z, pad=cp.mesh_pad, max_ratio=cp.mesh_max_ratio), z)
    zl, zh = np.quantile(z, [0.001, 0.999]); zq = z[(z >= zl) & (z <= zh)]
    spacing_report('z, 0.1% quantile drop', charge_nodes(zq, cp.zbins, cp.mesh_uniform_frac, smooth=cp.mesh_smooth * res_z, pad=cp.mesh_pad, max_ratio=cp.mesh_max_ratio), z)
    print(f'   z extremes: min {z.min()*1e6:+.1f}, 0.1% {zl*1e6:+.1f}, 99.9% {zh*1e6:+.1f}, max {z.max()*1e6:+.1f} um')
