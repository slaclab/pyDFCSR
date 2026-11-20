"""
The one-snapshot wake glitch at s = 19.11 m in drift D5 (progress_facet_bc20.md 9m): is it a band
switch? On the pickled state (dump_kick_state.py <run> 19.11 --dense) the whole 12 x 40 mesh is
recomputed with the run's settings (must reproduce the dense run's stored wake) and with variants
of the band location / integration layout; the charge-weighted projected W is compared with the
smooth trend of the neighbouring kicks (~ +0.2 MeV/m at 19.0-19.3 m in the dense run).

    conda run -n pydfcsr python pyDFCSR_2D/test/facet_bc20/glitch_1911.py

Variants: base; no tangent_bands; band_jacobian_min 0 / 0.5; xi_band_margin 3; branch_sin_min 0.2;
near_adapt + near_adapt_x; far region 4x. Also printed: the frame tilt tau, sin 2a, and per branch the
fraction of live far/near columns for a central observer. Writes glitch_1911.png.
"""
import glob
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
from dump_kick_state import load_state, state_path  # noqa: E402
import plot_bc20_wakes as P  # noqa: E402

RES = os.path.join(HERE, '..', 'benchmark_results', 'facet_bc20')
RUN = 'dfcsr_2d_peak_1M_sz6.5cap_m40_k115'
VARIANTS = {
    'base': {},
    'no_tangent_bands': dict(tangent_bands=False),
    'jac0': dict(band_jacobian_min=0.0),
    'jac05': dict(band_jacobian_min=0.5),
    'margin3': dict(xi_band_margin=6.0),
    'sinmin02': dict(branch_sin_min=0.2),
    'adapt_x': dict(near_adapt=True, near_adapt_x=True),
    'far4': dict(far_zbins=800),
}


def mesh_wake(sim):
    pos = sim.beam.position
    r = np.array([sim.get_CSR_wake(pos + z, x) for z, x in zip(sim.CSR_zmesh, sim.CSR_xmesh)])
    nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins
    return r[:, 0].reshape(nx, nz), r[:, 1].reshape(nx, nz)


def proj_weights(sim):
    xi = np.asarray(sim.CSR_xrange_transformed)
    h = np.gradient(xi)
    n, L = xi.size, xi[-1] - xi[0]
    wx = np.clip(1.0 / ((n - 1) * h) - 0.3 / L, 0, None) * h
    return wx / wx.sum()


def main():
    sim = load_state(state_path(RUN, 19.11))
    pos = sim.beam.position
    print(f'state at s = {pos:.4f} m; frame tilt tau = {sim._frame_tilt():+.4f}, sin 2a = {2*sim._frame_tilt()/(1+sim._frame_tilt()**2):+.4f}; '
          f'sigma_z {sim.beam.sigma_z*1e6:.1f} um, sigma_xi {sim.beam._sigma_x_transform*1e6:.1f} um, in bend: {sim.inbend}, exit transient: {sim._exit_transient()}')
    # the dense run's stored wake at this kick
    f = h5py.File(sorted(glob.glob(os.path.join(RES, 'runs', f'{RUN}_kdense-2*-wakes.h5')), key=os.path.getmtime)[-1], 'r')
    ks = sorted(f, key=lambda k: f[k].attrs['position'])
    pk = [k for k in ks if abs(f[k].attrs['position'] - pos) < 1e-6]
    W_stored = f[pk[0]]['longitudinal/dE_dct'][()] if pk else None
    nb = [k for k in ks if 18.9 < f[k].attrs['position'] < 19.4]
    wx = proj_weights(sim)
    zr = np.asarray(sim.CSR_zrange)
    print('dense run, charge-weighted <W> at the kicks 18.9-19.4 m:')
    for k in nb:
        W = f[k]['longitudinal/dE_dct'][()]
        print(f"   s = {f[k].attrs['position']:.3f}: <W> {np.mean((wx[:, None] * W).sum(0)):+.4f} MeV/m, max {W.max():+.3f}, min {W.min():+.3f}"
              + ('   <- this state' if abs(f[k].attrs['position'] - pos) < 1e-6 else ''))
    ip = sim.integration_params
    keep = {k: getattr(ip, k) for k in ('tangent_bands', 'band_jacobian_min', 'xi_band_margin', 'branch_sin_min',
                                         'near_adapt', 'near_adapt_x', 'far_zbins')}
    out = {}
    for name, fac in VARIANTS.items():
        for k, v in keep.items():
            setattr(ip, k, v)
        for k, v in fac.items():
            setattr(ip, k, v)
        W, F = mesh_wake(sim)
        out[name] = (W, F)
        pw = (wx[:, None] * W).sum(0)
        extra = f'   max|W - stored| {np.abs(W - W_stored).max():.2e}' if (name == 'base' and W_stored is not None) else ''
        print(f'{name:18s}: <W> {pw.mean():+.4f} MeV/m, max {W.max():+.3f}, min {W.min():+.3f}; <F> {np.mean((wx[:, None] * F).sum(0)):+.4e}{extra}', flush=True)
    for k, v in keep.items():
        setattr(ip, k, v)
    # branch liveness for a central observer, far and near regions
    i = sim.CSR_zmesh.size // 2
    s, x = pos + sim.CSR_zmesh[i], sim.CSR_xmesh[i]
    (s1, s3), (_, s4) = sim._layout_bounds(s, x)
    for lab, sp in (('far', sim._far_region_nodes(s, x, pos, s1, s3)), ('near', sim._near_region_nodes(s, s3, s4, x=x, t=pos))):
        sim._retarded_xi_bands(s, x, pos, sp)
        d = sim._branch_diag
        print(f'central observer, {lab} region ({sp.size} columns): live fraction per branch '
              + ', '.join(f'{b["live"].mean():.2f}' for b in d['branches'])
              + f'; both live: {np.mean(d["branches"][0]["live"] & d["branches"][1]["live"]):.2f}')
    np.savez(os.path.join(HERE, 'glitch_1911.npz'), z=zr, xi=sim.CSR_xrange_transformed, wx=wx,
             **{f'W_{n}': v[0] for n, v in out.items()}, **{f'F_{n}': v[1] for n, v in out.items()})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for n, (W, F) in out.items():
        ax.plot(zr * 1e6, (wx[:, None] * W).sum(0), lw=2.0 if n == 'base' else 1.0, label=n)
    ax.axhline(0, color='#e1e0d9', lw=0.8)
    ax.set_xlabel('z [µm]  (head at +z)')
    ax.set_ylabel('projected W [MeV/m]')
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title(f'{RUN}: the s = {pos:.2f} m snapshot (drift D5) under band-location / layout variants', loc='left', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'glitch_1911.png'), dpi=140)
    print('wrote glitch_1911.png')


if __name__ == '__main__' and len(sys.argv) == 1:
    main()


def step_state(sim):
    """
    Advance the pickled state by one schedule step (the run loop's body for one node): track the
    beam through the current element by sched.dl, refresh the formation length, deposit a snapshot
    and rebuild the interpolant. Used to reach the 19.123 m node from the 19.110 m pickle; the
    dense run also applied a 0.23 MeV/m x 12.5 mm kick at 19.110 (3 keV), which is not reproduced.
    """
    sched = sim.lattice.schedule
    s_nodes = np.asarray(sched.s_nodes)
    i = int(np.argmin(np.abs(s_nodes - sim.beam.position)))
    assert abs(s_nodes[i] - sim.beam.position) < 1e-9, 'beam is not on a schedule node'
    dist = np.asarray(sim.lattice.distance)
    keys = [k for k in sim.lattice.lattice_config if k != 'step_size']
    ele_count = int(np.searchsorted(dist, sim.beam.position + 1e-9))
    ele = keys[ele_count]
    DL = float(sched.dl[i + 1])
    assert sim.beam.position + DL <= dist[ele_count] + 1e-9, 'the step crosses an element boundary'
    sim.lattice.update(ele)
    element = sim.get_bmadx_element(ele=ele, DL=DL, entrance=False,
                                    exit=sim._ends_element(sim.beam.position + DL, ele_count))
    sim.beam.track(element, DL)
    typ = sim.lattice.lattice_config[ele]['type']
    R = sim.lattice.lattice_config[ele]['L'] / sim.lattice.lattice_config[ele]['angle'] if typ == 'dipole' else float('inf')
    sim._refresh_formation_length(R)
    sim.DF_tracker.get_DF(x=sim.beam.x, z=sim.beam.z, px=sim.beam.px, t=sim.beam.position)
    sim.DF_tracker.append_DF()
    sim._thin_drift_history()
    sim.DF_tracker.append_interpolant(formation_length=sim.formation_length,
                                      n_formation_length=sim.integration_params.n_formation_length,
                                      min_start_time=sim._retention_floor())
    sim.DF_tracker.build_interpolant()
    sim.get_CSR_mesh()
    return sim.beam.position


def main_step():
    sim = load_state(state_path(RUN, 19.11))
    pos = step_state(sim)
    print(f'stepped to s = {pos:.4f} m; tau = {sim._frame_tilt():+.4f}, sin 2a = {2*sim._frame_tilt()/(1+sim._frame_tilt()**2):+.4f}')
    f = h5py.File(sorted(glob.glob(os.path.join(RES, 'runs', f'{RUN}_kdense-2*-wakes.h5')), key=os.path.getmtime)[-1], 'r')
    ks = {round(float(f[k].attrs['position']), 4): k for k in f}
    W_stored = f[ks[round(pos, 4)]]['longitudinal/dE_dct'][()] if round(pos, 4) in ks else None
    wx = proj_weights(sim)
    if W_stored is not None:
        print(f'dense run at this node: <W> {np.mean((wx[:, None] * W_stored).sum(0)):+.4f}, max {W_stored.max():+.3f}, min {W_stored.min():+.3f}')
    ip = sim.integration_params
    keep = {k: getattr(ip, k) for k in ('tangent_bands', 'band_jacobian_min', 'xi_band_margin', 'branch_sin_min',
                                         'near_adapt', 'near_adapt_x', 'far_zbins')}
    out = {}
    for name in ('base', 'no_tangent_bands', 'margin3', 'sinmin02', 'adapt_x'):
        for k, v in keep.items():
            setattr(ip, k, v)
        for k, v in VARIANTS[name].items():
            setattr(ip, k, v)
        W, F = mesh_wake(sim)
        out[name] = (W, F)
        extra = f'   max|W - stored| {np.abs(W - W_stored).max():.2e}' if (name == 'base' and W_stored is not None) else ''
        print(f'{name:18s}: <W> {np.mean((wx[:, None] * W).sum(0)):+.4f} MeV/m, max {W.max():+.3f}, min {W.min():+.3f}{extra}', flush=True)
    for k, v in keep.items():
        setattr(ip, k, v)
    i = sim.CSR_zmesh.size // 2
    s, x = sim.beam.position + sim.CSR_zmesh[i], sim.CSR_xmesh[i]
    (s1, s3), (_, s4) = sim._layout_bounds(s, x)
    for lab, sp in (('far', sim._far_region_nodes(s, x, pos, s1, s3)), ('near', sim._near_region_nodes(s, s3, s4, x=x, t=pos))):
        sim._retarded_xi_bands(s, x, pos, sp)
        d = sim._branch_diag
        print(f'central observer, {lab} region ({sp.size} columns): live fraction per branch '
              + ', '.join(f'{b["live"].mean():.2f}' for b in d['branches'])
              + f'; both live: {np.mean(d["branches"][0]["live"] & d["branches"][1]["live"]):.2f}')
    np.savez(os.path.join(HERE, 'glitch_19123.npz'), z=np.asarray(sim.CSR_zrange), xi=sim.CSR_xrange_transformed, wx=wx,
             W_stored=W_stored, **{f'W_{n}': v[0] for n, v in out.items()})


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == '--step':
    main_step()
