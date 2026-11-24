import os, sys, time, numpy as np
HERE='/Users/jytang/Library/CloudStorage/GoogleDrive-jytang@stanford.edu/My Drive/Stanford/research/FEL/pyDFCSR_claude/pyDFCSR_near_adapt/pyDFCSR_2D/test/bc20_indranil'
sys.path.insert(0, os.path.join(HERE, '..', '..', '..')); sys.path.insert(0, os.path.join(HERE, '..', 'facet_bc20'))
import dump_kick_state as D
D.RUNS = os.path.join(HERE, '..', 'benchmark_results', 'bc20_indranil', 'runs')
for run, st in (('loss', 37.74), ('gain', 11.30)):
    sim = D.load_state(D.state_path(run, st)); ip = sim.integration_params; pos = sim.beam.position
    nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins; ip.xbins = 100
    for k, v in (('seam_ratio', 2.0), ('prehistory', 0.0), ('reach_straight', 2.0), ('reach_slip_max', 2.0)):
        if not hasattr(ip, k): setattr(ip, k, v)
    a = pos - sim.bend_start_rec; R = abs(sim.R_rec); slip = a**3/(6*R*R)
    print(f'{run} s={pos:.3f}: a={a:.3f} m into the bend, R={R:.1f}, slip {slip*1e6:.1f} um = {slip/sim.beam._sigma_z:.2f} sigma_z, L_f {sim.formation_length:.3f}; geo reach {sim._geo_reach(pos)}')
    xi = np.asarray(sim.CSR_xrange_transformed); h = np.gradient(xi); n, L = xi.size, xi[-1]-xi[0]
    wx = np.clip(1.0/((n-1)*h) - 0.3/L, 0, None)*h; wx /= wx.sum()
    ref = None
    for rs, slip in ((0.0, 2.0), (40.0, 2.0), (40.0, 5.0)):
        ip.reach_straight = rs; ip.reach_slip_max = slip
        t1 = time.time(); r = np.array([sim.get_CSR_wake(pos + z, x) for z, x in zip(sim.CSR_zmesh, sim.CSR_xmesh)]); dt = time.time()-t1
        W = r[:, 0].reshape(nx, nz); pw = (wx[:, None]*W).sum(0)
        (s1, s3), _ = sim._layout_bounds(pos, sim.CSR_xmesh[nz//2])
        if ref is None: ref = (W, pw)
        print(f'   reach_straight {rs} slip_max {slip}: reach {pos - s1:.3f} m, far nodes {sim._last_region_nodes[0]}, {dt:.0f} s, <W> {pw.mean():+.4f} ({(pw.mean()-ref[1].mean())/abs(ref[1].mean())*100:+.3f}%), max|dW| {np.abs(W-ref[0]).max()/np.abs(ref[0]).max()*100:.2f}% of max|W|', flush=True)
