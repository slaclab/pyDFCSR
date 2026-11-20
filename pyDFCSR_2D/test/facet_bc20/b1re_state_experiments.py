"""
Quadrature experiments on the pickled B1RE kick state (dump_kick_state.py), in seconds each.

    conda run -n pydfcsr python pyDFCSR_2D/test/facet_bc20/b1re_state_experiments.py [variant ...]

The observers are the kick_check sweep (row 5 of the wake mesh, 0.5 um through the pedestal), so
the stored brute-force sweep (kick_check_<run>_s48.39.npz, sweep_sp64_xp4) is the reference and the
stored 'base' sweep checks that the pickled state reproduces the replay. Variants set
Integration_params attributes on the loaded state (VARIANTS). Prints noise (rms of F minus its
5 um smoothing) and rms(F - reference), both / the spike's max|F|, and ms per observer; writes
b1re_state_experiments.npz / .png.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
from dump_kick_state import load_state, state_path  # noqa: E402

RES = os.path.join(HERE, '..', 'benchmark_results', 'facet_bc20')
RUN = 'dfcsr_2d_peak_1M_sz6.5cap_m40_k115'
ROW = 5
S_TARGET = float(os.environ.get('B1RE_S', '48.38'))      # which pickled kick state (dump_kick_state)
VARIANTS = {
    'base': {},
    'adapt': dict(near_adapt=True),
    'adapt_x': dict(near_adapt=True, near_adapt_x=True),
    'adapt_x_e05': dict(near_adapt=True, near_adapt_x=True, near_adapt_eps=0.05),
    'adapt_x_e02': dict(near_adapt=True, near_adapt_x=True, near_adapt_eps=0.02),
    'adapt_x_dz01': dict(near_adapt=True, near_adapt_x=True, near_adapt_scan_dz=0.1),
    'adapt_x_r12': dict(near_adapt=True, near_adapt_x=True, near_adapt_max_ratio=1.2),
    'adapt_x_xmax8': dict(near_adapt=True, near_adapt_x=True, near_adapt_x_max=8),
    'adapt_xp4': dict(near_adapt=True, xbins=800),
    # combinations, after the first round of results
    'adapt_e02_xp4': dict(near_adapt=True, near_adapt_eps=0.02, xbins=800),
    'adapt_x_e02_xp2': dict(near_adapt=True, near_adapt_x=True, near_adapt_eps=0.02, xbins=400),
    'adapt_x_e01': dict(near_adapt=True, near_adapt_x=True, near_adapt_eps=0.01),
    'adapt_x_e02_r4': dict(near_adapt=True, near_adapt_x=True, near_adapt_eps=0.02, near_adapt_x_rounds=4),
    'adapt_xp8': dict(near_adapt=True, xbins=1600),
    'adapt_e02_xp8': dict(near_adapt=True, near_adapt_eps=0.02, xbins=1600),
    # the x' convergence check (/tmp/xconv, 2026-11-19) showed the 64x/800-x' brute force is NOT
    # converged in x' (an aliasing bump of 0.1 spike F at z - z_spike = -90 um that 1600 and 3200 x'
    # nodes remove); adapt_e02_xp16 reproduces the 64x/3200-x' values and is the reference now
    'adapt_x_e10_r2': dict(near_adapt=True, near_adapt_x=True, near_adapt_eps=0.1, near_adapt_x_rounds=2),   # the old defaults
    'adapt_e02_xp16': dict(near_adapt=True, near_adapt_eps=0.02, xbins=3200),
    'sp64_xp16': dict(near_cell=0.25 / 64, near_grade=0.05 / 64, xbins=3200),
}
REF_VARIANT = os.environ.get('B1RE_REF', 'adapt_e02_xp16')    # stored variant used as the reference
KEYS = ('near_adapt', 'near_adapt_x', 'near_adapt_eps', 'near_adapt_scan_dz', 'near_adapt_max_ratio',
        'near_adapt_x_max', 'near_adapt_x_rounds', 'near_cell', 'near_grade', 'xbins')


def main(variants):
    sim = load_state(state_path(RUN, S_TARGET))
    pos = sim.beam.position
    zr = np.asarray(sim.CSR_zrange)
    tag = '' if abs(S_TARGET - 48.38) < 1e-6 else f'_s{S_TARGET:.2f}'
    ref_file = os.path.join(HERE, f'kick_check_{RUN}_s48.39.npz') if not tag else None
    if ref_file and os.path.exists(ref_file):
        ref = np.load(ref_file)
        zs = zr[np.argmax(np.abs(ref['W'][ref['rows']]).max(0))]
        zf = np.arange(zs - 120e-6, zs - 5e-6, 0.5e-6)
        assert np.allclose(zf, ref['sweep_z']), 'observer sweep differs from the stored one'
        Fref = ref['sweep_sp64_xp4'][:, :, 1].sum(1)
        Fbase_stored = ref['sweep_base'][:, :, 1].sum(1)
        spk = float(np.abs(ref['sp64_xp4_parts'][:, :, :, 1].sum(2)[:, np.abs(zr - zs) < 5e-6]).max())
    else:
        # no stored check at this kick: locate the spike from W along the mesh row, reference computed below
        xr = sim.CSR_xrange_transformed[ROW] + np.polyval(sim.beam.slope, zr) + sim._mesh_ridge(zr)
        Wrow = np.array([sim.get_CSR_wake(pos + z, x)[0] for z, x in zip(zr, xr)])
        zs = zr[np.argmax(np.abs(Wrow))]
        zf = np.arange(zs - 120e-6, zs - 5e-6, 0.5e-6)
        Fref = Fbase_stored = np.full(zf.size, np.nan)
        spk = np.nan
    xf = sim.CSR_xrange_transformed[ROW] + np.polyval(sim.beam.slope, zf) + sim._mesh_ridge(zf)
    print(f'state at s = {pos:.4f} m, spike at z = {zs*1e6:+.1f} um, spike max|F| {spk:.4f} MeV/m, {zf.size} observers')
    ip = sim.integration_params
    keep = {k: getattr(ip, k) for k in KEYS}
    npz = os.path.join(HERE, f'b1re_state_experiments{tag}.npz')
    out = dict(np.load(npz)) if os.path.exists(npz) else {}
    out.update({'zf': zf, 'F_ref': Fref, 'F_base_stored': Fbase_stored})
    from scipy.ndimage import gaussian_filter1d
    for v in variants:
        for k, val in keep.items():
            setattr(ip, k, val)
        for k, val in VARIANTS[v].items():
            setattr(ip, k, val)
        t0 = time.time()
        F = np.array([sim.get_CSR_wake(pos + z, x)[1] for z, x in zip(zf, xf)])
        ms = (time.time() - t0) / zf.size * 1e3
        out[f'F_{v}'] = F
        out[f'ms_{v}'] = ms
        if v == 'base' and np.isfinite(Fbase_stored).all():
            print(f'base: max|F - stored base sweep| = {np.abs(F - Fbase_stored).max():.2e}', flush=True)
        print(f'  {v:16s} done, {ms:6.1f} ms/obs', flush=True)
    for k, val in keep.items():
        setattr(ip, k, val)
    np.savez(npz, **out)
    Fr = out.get(f'F_{REF_VARIANT}', Fref)
    if not np.isfinite(spk):      # spike scale from the reference's force at the spike
        ip = sim.integration_params
        for k, val in keep.items():
            setattr(ip, k, val)
        for k, val in VARIANTS[REF_VARIANT].items():
            setattr(ip, k, val)
        spk = abs(sim.get_CSR_wake(pos + zs, sim.CSR_xrange_transformed[ROW] + np.polyval(sim.beam.slope, zs)
                                   + sim._mesh_ridge(np.array([zs]))[0])[1])
        for k, val in keep.items():
            setattr(ip, k, val)
    print(f'\nreference: {REF_VARIANT if f"F_{REF_VARIANT}" in out else "kick_check sp64_xp4"}; '
          f'noise = rms(F - 5 um smoothing), diff = rms(F - reference), both / spike max|F| = {spk:.4f}')
    print(f'{"variant":16s} {"noise":>7s} {"diff":>7s} {"ms/obs":>7s}')
    for v in sorted(k[2:] for k in out if k.startswith('F_') and k not in ('F_ref', 'F_base_stored')):
        F = out[f'F_{v}']
        sm = gaussian_filter1d(F, 10.0, mode='nearest')
        print(f'{v:16s} {np.sqrt(np.mean((F - sm)**2))/spk:7.4f} {np.sqrt(np.mean((F - Fr)**2))/spk:7.4f} '
              f'{float(out.get(f"ms_{v}", np.nan)):7.1f}')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if np.isfinite(Fref).all():
        ax.plot((zf - zs) * 1e6, Fref, color='#c3c2b7', lw=1.0, label='64× s′, 4× x′ (not converged in x′)')
    if f'F_{REF_VARIANT}' in out:
        ax.plot((zf - zs) * 1e6, out[f'F_{REF_VARIANT}'], color='#7a1f1e', lw=2.2, label=f'reference: {REF_VARIANT}')
    import itertools
    cols = itertools.cycle(['#0b0b0b', '#2a78d6', '#1baf7a', '#eda100', '#e34948', '#9ec5f4', '#4a3aa7', '#898781', '#eb6834'])
    for v, c in zip(variants, cols):
        if f'F_{v}' in out:
            ax.plot((zf - zs) * 1e6, out[f'F_{v}'], color=c, lw=1.0, label=v)
    ax.axhline(0, color='#e1e0d9', lw=0.8)
    ax.set_xlabel('z − z_spike [µm]')
    ax.set_ylabel('F [MeV/m]')
    ax.legend(fontsize=7, frameon=False, ncol=3)
    ax.set_title(f'{RUN}, kick s = {pos:.2f} m: transverse force through the pedestal (mesh row {ROW}), 0.5 µm observers',
                 loc='left', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, f'b1re_state_experiments{tag}.png'), dpi=140)
    print(f'wrote b1re_state_experiments{tag}.png')


if __name__ == '__main__':
    main(sys.argv[1:] or list(VARIANTS))
