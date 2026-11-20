"""Convert Indranil's openPMD beams (flat layout, basePath "/", unreadable by the installed pmd_beamphysics) to bmad-x text: P0C = mean p, z from position/z (or -c (t - <t>) if z is flat).  python convert_beams.py <dir>"""
import h5py, numpy as np, os, glob, sys
d = sys.argv[1]
C0 = 299792458.0
for f in sorted(glob.glob(os.path.join(d, 'beam_*.h5'))):
    h = h5py.File(f, 'r')
    x, y, z = (h['position/' + k][()] for k in 'xyz')
    px, py, pzm = (h['momentum/' + k][()] for k in 'xyz')          # eV/c
    t, w = h['time'][()], h['weight'][()]
    p = np.sqrt(px**2 + py**2 + pzm**2)
    p0 = p.mean()
    zb = z if np.ptp(z) > 0 else -C0 * (t - t.mean())
    coords = np.column_stack([x, px / p0, y, py / p0, zb, p / p0 - 1.0])
    tag = 'gain' if 'gain' in f else 'loss'
    out = os.path.join(d, f'beam_{tag}_bmadx.txt')
    np.savetxt(out, coords)
    Q = float(np.sum(w))
    print(f'{os.path.basename(f)} -> {os.path.basename(out)}: N {x.size}, Q {Q*1e9:.4f} nC, P0C {p0/1e9:.5f} GeV/c (mean p), sigma_p/p {p.std()/p0:.3e}')
    print(f'   ptp position/z {np.ptp(z)*1e6:.1f} um, ptp c*t {np.ptp(t)*C0*1e6:.1f} um -> z from {"position/z" if np.ptp(z) > 0 else "time"}')
    print(f'   sigma_z {zb.std()*1e6:.1f} um, span {zb.min()*1e6:.0f}..{zb.max()*1e6:.0f}; sigma_x {x.std()*1e6:.1f} um, sigma_y {y.std()*1e6:.1f} um; dx/dz {np.polyfit(zb, x, 1)[0]:+.4f}')
    hh, e = np.histogram(zb, bins=np.arange(zb.min(), zb.max() + 0.5e-6, 0.5e-6))
    I = hh * Q / x.size / (0.5e-6 / C0)
    c = np.polyfit((zb - zb.mean()) / zb.std(), coords[:, 5], 3)
    print(f'   I_pk {I.max()/1e3:.2f} kA; delta(w) = {c[3]:+.2e} {c[2]:+.2e} w {c[1]:+.2e} w^2 {c[0]:+.2e} w^3   (chirp h = {c[2]/zb.std():+.1f} /m)')
