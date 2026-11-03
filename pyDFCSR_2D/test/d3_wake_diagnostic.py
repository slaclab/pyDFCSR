"""
Diagnose the 2D far-drift wake failure in the chicane's D3 (progress.md top TODO).

The chicane is tracked with CSR OFF but debug=True, so the density history is recorded and the beam is
the CSR-off reference (no CSR feedback). The run is stopped at a D3 position; get_CSR_wake can then be
evaluated on that frozen history at any (x, z) and with any Integration_params, without retracking.
With drift_cutoff = 0 (the cutoff is applied in run(), not in get_CSR_wake, so it does not matter here).

Three hypotheses (integration_region.md section 8):
  H1  far-region layout: for |tau| >~ 40 the near region saturates at ~200 sigma_x and the live chirp
      branch runs into the far region, whose layout assumes a narrow strip far upstream
  H2  near-parallel branches: the two Eq 4.24 bands overlap partly; the subtraction leaves a sliver
      whose width jumps from column to column
  H3  cancellation x deposition noise: the pointwise cancellation is largest at high tilt and amplifies
      noise in d rho/dx

Usage (pydfcsr env):
  python d3_wake_diagnostic.py check   <s_stop>   -> 2D wake map at s_stop on the frozen CSR-off history,
                                                    against the stored CSR-on run's map (sanity check)
  python d3_wake_diagnostic.py anatomy <s_stop>   -> per-column bands / contributions at a few points
  python d3_wake_diagnostic.py sweep   <s_stop>   -> wake on a z-cut for each parameter variant
Each command rebuilds the frozen state (~1.5 min of CSR-off tracking with the history recorded);
the CSR2D object holds numba closures and so is not pickled.
"""
import contextlib
import copy
import io
import os
import pickle
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'benchmark_results', 'd3_diagnostic')
BASE_CFG = os.path.join(HERE, 'benchmark_results', 'chicane_2d_vs_1d', 'inputs', '2d.yaml')
os.makedirs(OUT, exist_ok=True)


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def build(s_stop, overrides=None):
    """CSR2D on the chicane config, CSR off, history recorded, stopped at s_stop (a scheduled node)."""
    from pyDFCSR_2D import CSR2D
    cfg = yaml.safe_load(open(BASE_CFG))
    cfg['CSR_computation'].update(compute_CSR=0, apply_CSR=0, write_wakes=False, write_beam=None,
                                  workdir=OUT, write_name='d3diag')
    cfg['CSR_integration']['drift_cutoff'] = 0.0
    for sec, d in (overrides or {}).items():
        cfg.setdefault(sec, {}).update(d)
    # per-process file: two runs in parallel used to share 'cfg_freeze.yaml', and one read the other's
    # n_formation_length (caught because its printed reach did not match its label)
    f = os.path.join(OUT, f'cfg_freeze_{os.getpid()}.yaml'); yaml.safe_dump(cfg, open(f, 'w'))
    sim = quiet(CSR2D, input_file=f)
    sim.dump_beam = lambda label: None
    sim.write_statistics = lambda: None
    quiet(sim.run, stop_time=s_stop, debug=True)
    return sim


def wake_mesh(sim):
    """The run's own 12 x 40 wake mesh at this s (get_CSR_mesh), as (xmesh, zmesh) flattened."""
    sim.get_CSR_mesh()
    return sim.CSR_xmesh.copy(), sim.CSR_zmesh.copy()


def zcut(sim, row, nz=None):
    """Points of one x-row of the run's wake mesh (row in 0..11; 5/6 straddle the axis)."""
    xm, zm = wake_mesh(sim)
    nx, nzm = sim.CSR_params.xbins, sim.CSR_params.zbins
    X, Z = xm.reshape(nx, nzm), zm.reshape(nx, nzm)
    return X[row], Z[row]


def evaluate(sim, xs, zs, params=None):
    """dE_dct [MeV/m] at the points, with Integration_params overridden by `params`."""
    ip0 = sim.integration_params
    if params:
        sim.integration_params = copy.copy(ip0)
        for k, v in params.items():
            setattr(sim.integration_params, k, v)
    try:
        out = np.array([sim.get_CSR_wake(sim.beam.position + z, x)[0] for x, z in zip(xs, zs)])
    finally:
        sim.integration_params = ip0
    return out


if __name__ == '__main__':
    cmd, s_stop = sys.argv[1], float(sys.argv[2])
    if cmd == 'freeze':
        freeze(s_stop)
    elif cmd == 'check':
        import h5py, glob
        t0 = time.time(); sim = build(s_stop); t1 = time.time()
        print(f'built at s = {sim.beam.position:.6f} in {(t1 - t0)/60:.1f} min; tau(frame) {sim._frame_tilt():+.2f}, '
              f'sigma_x {sim.beam._sigma_x*1e3:.3f} mm, sigma_z {sim.beam._sigma_z*1e6:.2f} um, '
              f'sigma_xi {sim.beam._sigma_x_transform*1e6:.2f} um, L_f {sim.formation_length:.4f} m, '
              f'snapshots {len(sim.DF_tracker.time_log)}, afterbend {sim.afterbend}, inbend {sim.inbend}', flush=True)
        xm, zm = wake_mesh(sim)
        W = evaluate(sim, xm, zm).reshape(sim.CSR_params.xbins, sim.CSR_params.zbins)
        print(f'map in {(time.time() - t1)/60:.1f} min: W min/max {W.min():.3f}/{W.max():.3f} MeV/m')
        ref = h5py.File(sorted(glob.glob(os.path.join(HERE, 'benchmark_results', 'chicane_1d2d', '2d_chic_on_wakes.h5')))[0])
        k = min(ref.keys(), key=lambda k: abs(ref[k].attrs['position'] - sim.beam.position))
        Wr = ref[k]['longitudinal/dE_dct'][()]
        print(f'stored CSR-on map at s = {ref[k].attrs["position"]:.4f}: W min/max {Wr.min():.3f}/{Wr.max():.3f}; '
              f'rel L2 (frozen CSR-off vs stored CSR-on) {np.linalg.norm(W - Wr)/np.linalg.norm(Wr):.3f}')
        np.savez(os.path.join(OUT, f'check_{s_stop:.4f}.npz'), W=W, Wr=Wr, xm=xm, zm=zm)
        u = np.linspace(-4, 4, 12); wx = np.exp(-u**2/2); wx /= wx.sum()
        for name, M in [('frozen', W), ('stored', Wr)]:
            print(f'  {name}: x-avg wake at z = -2,-1,0,1,2 sigma: ' + ' '.join(f'{v:+.3f}' for v in (wx[:, None]*M).sum(0)[[10, 15, 20, 25, 30]])
                  + f'   | x-row spread at z=0: {M[:, 20].min():+.3f}..{M[:, 20].max():+.3f}')
    elif cmd == 'anatomy':
        sim = build(s_stop); t = sim.beam.position; ip = sim.integration_params
        xm, zm = wake_mesh(sim); nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins
        X, Z = xm.reshape(nx, nz), zm.reshape(nx, nz)
        sz = sim.beam._sigma_z
        L = [f'D3 anatomy at s = {t:.4f}: tau(frame) {sim._frame_tilt():+.2f}, sigma_x {sim.beam._sigma_x*1e3:.3f} mm, '
             f'sigma_xi {sim.beam._sigma_x_transform*1e6:.1f} um, sigma_z {sz*1e6:.2f} um, L_f {sim.formation_length:.4f} m']
        res = {}
        for row in (2, 5, 6, 9):
            for col in (15, 20, 25):
                x, z = X[row, col], Z[row, col]; s = t + z
                (s1, s3), (_, s4) = sim._layout_bounds(s, x)
                sp_far = sim._far_region_nodes(s, x, t, s1, s3)
                sp_near = sim._near_region_nodes(s, s3, s4)
                radii = sim._near_patch_radii(s, s3, s4)
                parts = {}
                for name, sp, tp in [('far', sp_far, None), ('near', sp_near, None if radii is None else (*radii, False))]:
                    raw = sim._retarded_xi_bands(s, x, t, sp); bands = sim._disjoint_bands(raw)
                    diag = sim._branch_diag
                    # per band: contribution, its own cancellation, live columns, width stats
                    nxb = ip.xbins; frac = np.linspace(0, 1, nxb); per = []
                    for j, (lo, hi) in enumerate(bands):
                        width = hi - lo; alive = width > 0
                        if not alive.any(): per.append(dict(dE=0.0, live=0)); continue
                        xp_mesh = lo[None, :] + frac[:, None] * width[None, :]
                        sp_mesh = np.broadcast_to(sp[None, :], xp_mesh.shape).copy()
                        gz, _ = sim.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh, sp=sp_mesh, ignore_vx=False, taper=tp)
                        col_int = np.where(alive, np.trapz(gz, axis=0) * width / (nxb - 1), 0.0)
                        dE = -sim.CSR_scaling * np.trapz(col_int, x=sp)
                        w_sp = np.gradient(sp)
                        node = -sim.CSR_scaling * (gz * (width / (nxb - 1))[None, :] * w_sp[None, :])
                        per.append(dict(dE=dE, live=int(alive.sum()), C=np.abs(node).sum() / max(abs(dE), 1e-30),
                                        w_med=np.median(width[alive]) / sim.beam._sigma_x_transform,
                                        w_min=width[alive].min() / sim.beam._sigma_x_transform,
                                        jump=np.max(np.abs(np.diff(width[alive]))) / sim.beam._sigma_x_transform if alive.sum() > 1 else 0,
                                        u_lo=(s - sp[alive].max()) / sim.formation_length, u_hi=(s - sp[alive].min()) / sim.formation_length))
                    live_raw = [int(np.sum(d['live'])) for d in diag['branches']]
                    parts[name] = dict(n=len(sp), per=per, live_raw=live_raw, span=(s - sp.min(), s - sp.max()))
                patch = sim._integrate_near_patch(s, x, t, *radii, False)[0] if radii else 0.0
                tot = sum(b['dE'] for p in parts.values() for b in p['per']) + patch
                res[(row, col)] = dict(parts=parts, patch=patch, tot=tot, d=s - s3)
                L.append(f'\n(x row {row}, z = {(z - sim.beam._mean_z)/sz:+.1f} sigma_z): total {tot:+.4f} MeV/m; near length d = {(s - s3)*1e3:.1f} mm '
                         f'= {(s - s3)/sim.beam._sigma_x:.0f} sigma_x; patch {patch:+.4f}')
                for name, p in parts.items():
                    L.append(f'   {name:4s} {p["n"]:4d} s\' nodes over u = {p["span"][1]*1e3:.1f}..{p["span"][0]*1e3:.1f} mm; live columns per raw branch {p["live_raw"]}')
                    for j, b in enumerate(p['per']):
                        if b['live']:
                            L.append(f'        band {j}: dE {b["dE"]:+.4f}  live {b["live"]:4d}  cancellation C {b["C"]:8.1f}  width median {b["w_med"]:.1f} / min {b["w_min"]:.2f} sigma_xi, '
                                     f'max col-to-col jump {b["jump"]:.2f} sigma_xi;  live over u/L_f = {b["u_lo"]:.3f}..{b["u_hi"]:.3f}')
                        else:
                            L.append(f'        band {j}: dead')
        open(os.path.join(OUT, f'anatomy_{s_stop:.4f}.txt'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
    elif cmd == 'sweep':
        group = sys.argv[3] if len(sys.argv) > 3 else 'integration'
        u = np.linspace(-4, 4, 12); wx = np.exp(-u ** 2 / 2); wx /= wx.sum()

        def summarise(W):
            avg = (wx[:, None] * W).sum(0)
            xs = np.sqrt((wx[:, None] * (W - avg) ** 2).sum(0))          # x-spread at each z, charge weighted
            return avg, xs

        variants = {
            'integration': [('baseline', {}),
                            ('near_grade 0.02', dict(near_grade=0.02)), ('near_grade 0.01', dict(near_grade=0.01)),
                            ('near_grade 0.005', dict(near_grade=0.005)),
                            ('near_cell 0.25', dict(near_cell=0.25)), ('near_cell 0.1', dict(near_cell=0.1)),
                            ('xbins 400', dict(xbins=400)), ('xbins 800', dict(xbins=800)),
                            ('margin 3', dict(xi_band_margin=3.0)), ('margin 4', dict(xi_band_margin=4.0)),
                            ('branch_sin_min 0.2', dict(branch_sin_min=0.2)), ('branch_sin_min 0.02', dict(branch_sin_min=0.02)),
                            ('branch_sin_min 0.01', dict(branch_sin_min=0.01)),
                            ('near_patch 10', dict(near_patch=10.0)), ('near_patch 2.5', dict(near_patch=2.5)),
                            ('far_cell 3', dict(far_cell=3.0)), ('far_window 2', dict(far_window=2.0)),
                            ('ALL fine: grade 0.005, cell 0.1, xbins 800, margin 3',
                             dict(near_grade=0.005, near_cell=0.1, xbins=800, xi_band_margin=3.0))],
            'deposition': [('baseline', {}),
                           ('dep 600x600', {'particle_deposition': dict(xbins=600, zbins=600)}),
                           ('dep smoothing 1.5', {'particle_deposition': dict(smoothing_sigma=1.5)}),
                           ('dep smoothing 6', {'particle_deposition': dict(smoothing_sigma=6.0)}),
                           ('dep xlim/zlim 7', {'particle_deposition': dict(xlim=7, zlim=7)})],
        }[group]
        L = [f'D3 wake sweep ({group}) at s = {s_stop}; full 12 x 40 map per variant; rel L2 vs baseline over the map; '
             f'x-avg = charge-weighted average over the 12 x-rows; x-spread = charge-weighted rms over x at fixed z']
        maps = {}
        sim = build(s_stop) if group == 'integration' else None
        xm = zm = None
        for name, p in variants:
            t0 = time.time()
            if group == 'deposition':
                sim = build(s_stop, p) if p else build(s_stop)
                pp = {}
            else:
                pp = p
            if xm is None:
                xm, zm = wake_mesh(sim)
            W = evaluate(sim, xm, zm, pp).reshape(sim.CSR_params.xbins, sim.CSR_params.zbins)
            maps[name] = W
            avg, xs = summarise(W)
            b = maps['baseline']; ab, xb = summarise(b)
            L.append(f"{name:52s} rel L2 vs base {np.linalg.norm(W - b) / np.linalg.norm(b):6.3f} | x-avg peak {np.abs(avg).max():.4f}, "
                     f"rel L2(x-avg) {np.linalg.norm(avg - ab) / np.linalg.norm(ab):6.3f} | x-spread rms {np.sqrt(np.mean(xs**2)):.4f} "
                     f"| row-spread at z=0 {W[:, 20].min():+.3f}..{W[:, 20].max():+.3f} | {time.time() - t0:5.1f} s")
            print(L[-1], flush=True)
        # 1D reference on the same frozen beam: line-charge wake at the same position
        from pyDFCSR_2D.csr1d import line_density, wake_after_bend, CSR1DGeometry
        g = CSR1DGeometry(sim.lattice.lattice_config, sim.lattice.distance).at(sim.beam.position, sim.lattice.current_element)
        S, ds, lam, dl = line_density(sim.beam.z, 300, abs(sim.beam.charge) / 1.602176634e-19, 2)
        W1 = wake_after_bend(S, ds, lam, dl, g['R'], g['phim'], g['x_R']) / 1e6
        zc = zm.reshape(sim.CSR_params.xbins, sim.CSR_params.zbins)[0]
        W1z = np.interp(zc, S, W1)
        L.append(f'\n1D line-charge wake on the same beam (x_R = {g["x_R"]:.3f}): peak {np.abs(W1z).max():.4f} MeV/m; '
                 f'x-avg 2D vs 1D at z = -2,-1,0,1,2 sigma:')
        for name in ('baseline', next(n for n, _ in variants if n.startswith('ALL')) if group == 'integration' else 'baseline'):
            avg, _ = summarise(maps[name])
            L.append(f'   {name:52s} 2D x-avg: ' + ' '.join(f'{v:+.4f}' for v in avg[[10, 15, 20, 25, 30]]))
        L.append(f'   {"1D":52s}           ' + ' '.join(f'{v:+.4f}' for v in W1z[[10, 15, 20, 25, 30]]))
        np.savez(os.path.join(OUT, f'sweep_{group}_{s_stop:.4f}.npz'), names=np.array(list(maps)),
                 maps=np.array(list(maps.values())), xm=xm, zm=zm, W1z=W1z)
        open(os.path.join(OUT, f'sweep_{group}_{s_stop:.4f}.txt'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L[-4:]))
    elif cmd == 'reach':
        # Does the far-drift wake fail because the integration (and the retained history) no longer
        # reaches the bend? Freeze the SAME CSR-off state with a larger n_formation_length, so both the
        # s' reach (s1 = s3 - n_fl*L_f) and the history retention extend back over B3.
        from pyDFCSR_2D.csr1d import line_density, wake_after_bend, CSR1DGeometry
        u = np.linspace(-4, 4, 12); wx = np.exp(-u ** 2 / 2); wx /= wx.sum()
        nfls = [float(v) for v in (sys.argv[3].split(',') if len(sys.argv) > 3 else ['1', '2', '4', '6'])]
        tag = '' if nfls == [1.0, 6.0] else '_nfl' + '_'.join(f'{n:g}' for n in nfls)   # never overwrite the 1-vs-6 result
        L = [f'drift wake vs n_formation_length at s = {s_stop} (frozen CSR-off history)']
        res = {}
        for n in nfls:
            t0 = time.time()
            sim = build(s_stop, {'CSR_integration': dict(n_formation_length=n)})
            xm, zm = wake_mesh(sim)
            W = evaluate(sim, xm, zm).reshape(sim.CSR_params.xbins, sim.CSR_params.zbins)
            avg = (wx[:, None] * W).sum(0); xs = np.sqrt((wx[:, None] * (W - avg) ** 2).sum(0))
            tr = sim.DF_tracker; s = sim.beam.position
            (s1, s3), _ = sim._layout_bounds(s, 0.0)
            geo = CSR1DGeometry(sim.lattice.lattice_config, sim.lattice.distance)
            last = next(e for e in geo.elements if e['name'] == geo.at(s, sim.lattice.current_element)['bend'])
            res[n] = W
            L.append(f'n_fl {n:4.1f}: reach s\' >= {s1:.3f} m (last bend {last["name"]} {last["start"]:.3f}..{last["end"]:.3f}; '
                     f'reach covers its exit: {s1 < last["end"]}), history {tr.time_log[0]:.3f}..{tr.time_log[-1]:.3f} m '
                     f'({len(tr.time_log)} snapshots) | x-avg at z = -2,-1,0,+1,+2 sigma: ' + ' '.join(f'{v:+.4f}' for v in avg[[10, 15, 20, 25, 30]])
                     + f' | x-spread rms {np.sqrt(np.mean(xs ** 2)):.4f} | row range z=0 {W[:, 20].min():+.3f}..{W[:, 20].max():+.3f} | {time.time() - t0:.0f} s')
            print(L[-1], flush=True)
        g = CSR1DGeometry(sim.lattice.lattice_config, sim.lattice.distance).at(sim.beam.position, sim.lattice.current_element)
        S, ds, lam, dl = line_density(sim.beam.z, 300, abs(sim.beam.charge) / 1.602176634e-19, 2)
        W1z = np.interp(zm.reshape(sim.CSR_params.xbins, sim.CSR_params.zbins)[0], S, wake_after_bend(S, ds, lam, dl, g['R'], g['phim'], g['x_R']) / 1e6)
        L.append(f'1D line charge (x_R = {g["x_R"]:.3f})          x-avg at z = -2,-1,0,+1,+2 sigma: ' + ' '.join(f'{v:+.4f}' for v in W1z[[10, 15, 20, 25, 30]]))
        np.savez(os.path.join(OUT, f'reach_{s_stop:.4f}{tag}.npz'), nfl=np.array(nfls), maps=np.array([res[n] for n in nfls]), xm=xm, zm=zm, W1z=W1z)
        open(os.path.join(OUT, f'reach_{s_stop:.4f}{tag}.txt'), 'w').write('\n'.join(L) + '\n'); print(L[-1])
    elif cmd == 'novx':
        # Is the growing x-structure in the drift carried by the transverse-velocity terms? Same frozen
        # state (reach covering the bend), map with ignore_vx False vs True. ignore_vx zeroes vx, vx_ret
        # and d vx/dx in the integrand (CSR.py get_CSR_integrand); get_CSR_wake always passes False, so it
        # is forced here by wrapping the integrand.
        from pyDFCSR_2D.csr1d import line_density, wake_after_bend, CSR1DGeometry
        u = np.linspace(-4, 4, 12); wx = np.exp(-u ** 2 / 2); wx /= wx.sum()
        nfl = float(sys.argv[3])
        sim = build(s_stop, {'CSR_integration': dict(n_formation_length=nfl)})
        xm, zm = wake_mesh(sim)
        (s1, _), _ = sim._layout_bounds(sim.beam.position, 0.0)
        L = [f'drift wake at s = {sim.beam.position:.4f}, n_fl {nfl:g} (reach s\' >= {s1:.3f} m), transverse-velocity terms on vs off']
        orig = sim.get_CSR_integrand; maps = {}
        for name, flag in (('vx terms ON', False), ('vx terms OFF (ignore_vx)', True)):
            sim.get_CSR_integrand = (lambda *a, _f=flag, **k: orig(*a, **{**k, 'ignore_vx': _f}))
            t0 = time.time()
            W = evaluate(sim, xm, zm).reshape(sim.CSR_params.xbins, sim.CSR_params.zbins); maps[name] = W
            avg = (wx[:, None] * W).sum(0); xs = np.sqrt((wx[:, None] * (W - avg) ** 2).sum(0))
            L.append(f'{name:26s} x-avg at z = -2,-1,0,+1,+2 sigma: ' + ' '.join(f'{v:+.4f}' for v in avg[[10, 15, 20, 25, 30]])
                     + f' | x-spread rms {np.sqrt(np.mean(xs ** 2)):.4f} | row range z=0 {W[:, 20].min():+.3f}..{W[:, 20].max():+.3f} | {time.time() - t0:.0f} s')
            print(L[-1], flush=True)
        sim.get_CSR_integrand = orig
        g = CSR1DGeometry(sim.lattice.lattice_config, sim.lattice.distance).at(sim.beam.position, sim.lattice.current_element)
        S, ds, lam, dl = line_density(sim.beam.z, 300, abs(sim.beam.charge) / 1.602176634e-19, 2)
        W1z = np.interp(zm.reshape(sim.CSR_params.xbins, sim.CSR_params.zbins)[0], S, wake_after_bend(S, ds, lam, dl, g['R'], g['phim'], g['x_R']) / 1e6)
        # how fast is the beam converging here: d ln sigma_x / ds from the frozen beam's x-x' correlation
        b = sim.beam; conv = np.mean((b.x - b.x.mean()) * (b.px - b.px.mean())) / np.var(b.x)
        L.append(f'{"1D line charge":26s} x-avg at z = -2,-1,0,+1,+2 sigma: ' + ' '.join(f'{v:+.4f}' for v in W1z[[10, 15, 20, 25, 30]])
                 + f'   | beam convergence <x x\'>/<x^2> = {conv:+.3f} /m, sigma_x {b._sigma_x*1e3:.3f} mm, tau {sim._frame_tilt():+.1f}')
        np.savez(os.path.join(OUT, f'novx_{s_stop:.4f}.npz'), names=np.array(list(maps)), maps=np.array(list(maps.values())), xm=xm, zm=zm, W1z=W1z)
        open(os.path.join(OUT, f'novx_{s_stop:.4f}.txt'), 'w').write('\n'.join(L) + '\n'); print(L[-1])
