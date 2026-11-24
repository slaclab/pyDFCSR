import os, sys, time, numpy as np
HERE='/Users/jytang/Library/CloudStorage/GoogleDrive-jytang@stanford.edu/My Drive/Stanford/research/FEL/pyDFCSR_claude/pyDFCSR_near_adapt/pyDFCSR_2D/test/bc20_indranil'
sys.path.insert(0, os.path.join(HERE, '..', '..', '..')); sys.path.insert(0, os.path.join(HERE, '..', 'facet_bc20'))
import dump_kick_state as D
D.RUNS = os.path.join(HERE, '..', 'benchmark_results', 'bc20_indranil', 'runs')
for run, st in (('loss', 37.74), ('loss', 48.73), ('gain', 11.30)):
    sim = D.load_state(D.state_path(run, st)); ip = sim.integration_params; pos = sim.beam.position
    nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins; ip.xbins = 100
    if not hasattr(ip, 'seam_ratio'): ip.seam_ratio = 2.0
    xi = np.asarray(sim.CSR_xrange_transformed); h = np.gradient(xi); n, L = xi.size, xi[-1]-xi[0]
    wx = np.clip(1.0/((n-1)*h) - 0.3/L, 0, None)*h; wx /= wx.sum()
    print(f'{run} s={pos:.3f}: L_f {sim.formation_length:.3f} m, far cell {sim.formation_length/400*1e3:.2f} mm, seam floor {sim.formation_length/400/ip.near_grade/2*1e3:.0f} mm')
    ref = {}
    for seam, fz in ((0.0, 200), (0.0, 400), (2.0, 200), (2.0, 400)):
        ip.seam_ratio, ip.far_zbins = seam, fz
        t1 = time.time(); r = np.array([sim.get_CSR_wake(pos + z, x) for z, x in zip(sim.CSR_zmesh, sim.CSR_xmesh)]); dt = time.time()-t1
        W, F = r[:, 0].reshape(nx, nz), r[:, 1].reshape(nx, nz); pw = (wx[:, None]*W).sum(0)
        ref[(seam, fz)] = (W, F, pw)
        W0, F0, pw0 = ref[(0.0, 200)]
        print(f'   seam {seam} far_zbins {fz}: {dt:.0f} s, <W> {pw.mean():+.4f} ({(pw.mean()-pw0.mean())/abs(pw0.mean())*100:+.2f}% vs as-is), max|dW| {np.abs(W-W0).max():.3f} ({np.abs(W-W0).max()/np.abs(W0).max()*100:.1f}% of max|W|), max|dF| {np.abs(F-F0).max()/np.abs(F0).max()*100:.1f}% of max|F|; near nodes {sim._last_region_nodes[-1]}', flush=True)
    a, b = ref[(2.0, 200)], ref[(2.0, 400)]
    print(f'   seam 2: 200 vs 400 far cols: max|dW| {np.abs(a[0]-b[0]).max()/np.abs(a[0]).max()*100:.2f}% of max|W|, max|dF| {np.abs(a[1]-b[1]).max()/np.abs(a[1]).max()*100:.2f}% of max|F|, <W> {(a[2].mean()-b[2].mean())/abs(a[2].mean())*100:+.3f}%')
    ip.seam_ratio, ip.far_zbins = 2.0, 200
