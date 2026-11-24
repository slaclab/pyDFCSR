import os, sys, time, numpy as np
HERE='/Users/jytang/Library/CloudStorage/GoogleDrive-jytang@stanford.edu/My Drive/Stanford/research/FEL/pyDFCSR_claude/pyDFCSR_near_adapt/pyDFCSR_2D/test/bc20_indranil'
sys.path.insert(0, os.path.join(HERE, '..', '..', '..')); sys.path.insert(0, os.path.join(HERE, '..', 'facet_bc20'))
import dump_kick_state as D
D.RUNS = os.path.join(HERE, '..', 'benchmark_results', 'bc20_indranil', 'runs')
for run, st in (('loss', 37.74), ('loss', 48.73)):
    sim = D.load_state(D.state_path(run, st)); ip = sim.integration_params; pos = sim.beam.position; tr = sim.DF_tracker
    nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins; ip.xbins = 100
    t0 = float(tr.time_log[0]); Lf = sim.formation_length
    xi = np.asarray(sim.CSR_xrange_transformed); h = np.gradient(xi); n, L = xi.size, xi[-1]-xi[0]
    wx = np.clip(1.0/((n-1)*h) - 0.3/L, 0, None)*h; wx /= wx.sum()
    print(f'{run} s={pos:.3f}: sigma_z {sim.beam.sigma_z*1e6:.1f} um, formation length {Lf:.3f} m, history back to {t0:.3f} m ({(pos-t0)/Lf:.2f} L_f)')
    ref = None
    for nfl, fz in ((1, 200), (1, 400), (2, 400), (3, 600), (1, 100)):
        ip.n_formation_length = nfl; ip.far_zbins = fz
        # where does the far region start for a central observer?
        k = (nx//2)*nz + nz//2; (s1, s3), _ = sim._layout_bounds(pos + sim.CSR_zmesh[k], sim.CSR_xmesh[k])
        t1 = time.time(); r = np.array([sim.get_CSR_wake(pos + z, x) for z, x in zip(sim.CSR_zmesh, sim.CSR_xmesh)]); dt = time.time()-t1
        W = r[:, 0].reshape(nx, nz); pw = (wx[:, None]*W).sum(0)
        if ref is None: ref = (W, pw)
        print(f'   n_fl {nfl} far_zbins {fz:3d}: far region starts {pos - s1:.3f} m behind (clipped at the history? {s1 <= t0 + 1e-9}); {dt:.0f} s; <W> {pw.mean():+.4f} MeV/m ({(pw.mean()-ref[1].mean())/abs(ref[1].mean())*100:+.2f}% vs n_fl 1); max|W - W(1)| {np.abs(W-ref[0]).max():.4f} at z={sim.CSR_zmesh[np.abs(r[:,0]-ref[0].ravel()).argmax()]*1e6:+.1f} um ({np.abs(W-ref[0]).max()/np.abs(ref[0]).max()*100:.2f}% of max|W|)', flush=True)
    ip.n_formation_length = 1; ip.far_zbins = 200
