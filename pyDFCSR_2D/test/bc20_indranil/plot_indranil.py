"""
Wake maps, projected wakes, roughness and the end-beam table for Indranil's gain / loss beams
(run_indranil.py), reusing the BC20 plot machinery with this lattice's landmarks.

    conda run -n pydfcsr python pyDFCSR_2D/test/bc20_indranil/plot_indranil.py <beam> [cmp_run] [run]

<beam> gain | loss: the 2D run; cmp_run (default the same run) is the second 2D run drawn in the
projections (e.g. gain_k2); the 1D run <beam>_1d is drawn at its nearest kick. Writes to
benchmark_results/bc20_indranil/: wake_maps_long_<run>.png, wake_maps_trans_<run>.png,
wake_maps_spike_<run>.png, wake_zproj_<run>.png, wakes_along_s_<run>.png, and prints the roughness
of the projected wake by section and the energy / emittance table against <beam>_1d and <beam>_off.
"""
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'facet_bc20'))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
import plot_bc20_wakes as P  # noqa: E402

RES = os.path.join(HERE, '..', 'benchmark_results', 'bc20_indranil')
P.RES, P.RUNS = RES, os.path.join(RES, 'runs')
# lattice_bc20w_c_0.060000_000000: b1le 0-1.06, b2le 10.41-12.24, b3le 23.62-24.15, b3re 24.94-25.47,
# b2re 36.85-38.67, b1re 48.02-49.08, end 51.08
P.PICK = [(0.53, 'b1le mid'), (1.06, 'b1le exit'), (3.5, 'drift D1'), (11.3, 'b2le mid'), (12.24, 'b2le exit'),
          (14.6, 'quads q3'), (18.0, 'drift D9'), (24.15, 'b3le exit'), (25.47, 'b3re exit'), (31.0, 'drift D24'),
          (36.65, 'q2er exit'), (37.3, 'b2re 1/4'), (37.76, 'b2re 1/2'), (38.2, 'b2re 3/4'), (38.67, 'b2re exit'),
          (40.0, 'drift D30'), (42.8, 'q1er'), (46.0, 'drift D31'), (48.37, 'b1re 1/3'), (48.73, 'b1re 2/3'),
          (49.08, 'b1re exit'), (50.5, 'final drift')]
SECTIONS = ((0, 10.414, 'b1le, D1, q1el'), (10.414, 12.239, 'b2le'), (12.239, 23.616, 'sextupole section'),
            (23.616, 25.467, 'b3 pair'), (25.467, 36.845, 'sextupole section 2'), (36.845, 38.670, 'b2re'),
            (38.670, 48.021, 'drift, q1er'), (48.021, 49.084, 'b1re'), (49.084, 51.1, 'final drift'))
ENDS = ('q2er', 'b2re', 'D31', 'b1re', 'D32')


def rough(r):
    z, w = r['z'], P.proj(r, r['W'])
    g = np.arange(z[0], z[-1], 2e-6)
    if g.size < 20:
        return np.nan
    wi = np.interp(g, z, w)
    sm = gaussian_filter1d(wi, 10.0, mode='nearest')
    core = slice(int(0.1 * g.size), int(0.9 * g.size))
    return np.sqrt(np.mean((wi - sm)[core] ** 2)) / np.sqrt(np.mean(sm[core] ** 2))


def table(beam, run):
    sys.path.insert(0, os.path.join(HERE, '..', 'facet_bc20'))
    import compare_csr_runs as C
    from slice_emittance import slices
    d = np.load(os.path.join(P.RUNS, f'{run}.npz'))
    off = np.load(os.path.join(P.RUNS, f'{beam}_off.npz'))
    d1 = np.load(os.path.join(P.RUNS, f'{beam}_1d.npz')) if os.path.exists(os.path.join(P.RUNS, f'{beam}_1d.npz')) else None
    p0c = float(off['p0c'])
    names = list(off['names'])
    k = [names.index(e) for e in ENDS]
    print(f'\n{beam}: energy change per section [MeV] (vs CSR off) and the end beam')
    print(f'{"run":10s} {"->q2er":>8s} {"b2re":>8s} {"drift":>8s} {"b1re":>8s} {"D32":>8s} {"total":>8s} {"rms dE":>7s} {"enx 1st/3rd":>12s} {"spike slice":>12s} {"kicks":>5s} {"min":>5s}')
    rows = [(run, d)]
    if run != beam and os.path.exists(os.path.join(P.RUNS, f'{beam}.npz')):
        rows.append((beam, np.load(os.path.join(P.RUNS, f'{beam}.npz'))))      # the earlier (v1) 2D run
    if d1 is not None:
        rows.append((f'{beam}_1d', d1))
    for lab, x in rows:
        dE = (x['exit_mean'][:, 5] - off['exit_mean'][:, 5]) * p0c / 1e6
        cum = dE[k]
        sec = np.diff(np.concatenate([[0.0], cum]))
        c, co = x['coords_end'], off['coords_end']
        sl = slices(c, p0c)
        sp = (sl[:, 2] - sl[:, 1]) < 1e-6
        print(f'{lab:10s} ' + ' '.join(f'{v:+8.3f}' for v in sec) + f' {cum[-1]:+8.3f} {np.std((c[5]-co[5])*p0c)/1e6:7.2f} '
              f'{C.emit_n(c, p0c, 1)*1e6:6.2f}/{C.emit_n(c, p0c, 3)*1e6:5.2f} '
              f'{(sl[sp, 3].mean()*1e6 if sp.any() else np.nan):6.1f}/{(sl[sp, 4].mean()*1e6 if sp.any() else np.nan):5.1f} '
              f'{int(x["n_kicks"]):5d} {float(x["runtime"])/60:5.0f}')
    co = off['coords_end']
    sl = slices(co, p0c)
    sp = (sl[:, 2] - sl[:, 1]) < 1e-6
    print(f'{"CSR off":10s} {"":53s} {C.emit_n(co, p0c, 1)*1e6:6.2f}/{C.emit_n(co, p0c, 3)*1e6:5.2f} '
          f'{(sl[sp, 3].mean()*1e6 if sp.any() else np.nan):6.1f}/{(sl[sp, 4].mean()*1e6 if sp.any() else np.nan):5.1f}')
    # final current
    for lab, c in ((run, d['coords_end']), ('CSR off', co)):
        z = c[4]
        h, _ = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.5e-6, 0.5e-6))
        I = gaussian_filter1d(h.astype(float), 1.0) * 1.6e-9 / z.size / (0.5e-6 / 299792458.0)
        half = np.flatnonzero(I > I.max() / 2)
        print(f'   {lab:10s}: final sigma_z {z.std()*1e6:.1f} um, I_pk {I.max()/1e3:.1f} kA, FWHM {(half[-1]-half[0]+1)*0.5:.1f} um')


def main(beam, cmp_run=None, run=None):
    # run: the 2D run drawn (default <beam>); cmp_run: the second 2D run in the projections
    # (default the drawn run). E.g. main('loss', 'loss', 'loss_v2') draws loss_v2 against loss.
    run = run or beam
    cmp_run = cmp_run or run
    P.LABEL.update({run: f'2D {run}', cmp_run: f'2D {cmp_run}', f'{beam}_1d': f'1D {beam}_1d'})
    P.RUN, P.CMP, P.RUN1 = run, cmp_run, f'{beam}_1d'
    Rq, Ro = P.load2d(run), P.load2d(cmp_run)
    D1 = P.load1d(f'{beam}_1d') if os.path.exists(os.path.join(P.RUNS, f'{beam}_1d.npz')) else {}
    tag = f'_{run}'
    smax = Rq[-1]['s']
    picks = [p for p in P.PICK if p[0] <= smax + 1e-6]
    P.maps(Rq, 'W', f'Longitudinal wake dE/ds, {beam} beam, {len(Rq)} kicks', f'wake_maps_long{tag}.png', picks=picks)
    P.maps(Rq, 'F', f'Transverse force, {beam} beam', f'wake_maps_trans{tag}.png', picks=picks)
    P.maps(Rq, 'W', f'Longitudinal wake zoomed on the current peak (±12 µm), {beam}', f'wake_maps_spike{tag}.png', zoom=12.0,
           picks=[p for p in picks if p[0] >= 36.8])
    if D1:
        P.zproj(Rq, Ro, D1, f'wake_zproj{tag}.png')
        P.along_s(Rq, Ro, D1, f'wakes_along_s{tag}.png')
    print(f'{run}: {len(Rq)} kicks to s = {smax:.2f} m; roughness of the projected W (rms deviation from a 20 um smoothing / rms W), median by section:')
    for lo, hi, lab in SECTIONS:
        rr = [rough(r) for r in Rq if lo < r['s'] <= hi]
        if rr:
            print(f'   {lab:22s} {np.nanmedian(rr):.3f}  ({len(rr)} kicks)')
    if os.path.exists(os.path.join(P.RUNS, f'{beam}_off.npz')) and os.path.exists(os.path.join(P.RUNS, f'{run}.npz')):
        table(beam, run)


if __name__ == '__main__':
    main(*sys.argv[1:])
