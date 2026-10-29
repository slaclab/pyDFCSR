"""2D (pyDFCSR) side of the 1D-vs-2D comparisons.  Usage: tilt_scan_run.py <tag> on|off [--handoff S] [--outdir NAME]
  --handoff S : position where the beam is written for the 1D code (default 0.1 m = dipole entrance)
  --outdir    : results subfolder (default tilt_scan_1d2d)
  --regions B : comma list of region edges in s (e.g. chicane element boundaries); K and K_z are also
                accumulated per region (kick position in [edge_i, edge_i+1)), plus separately for any kick
                listed in --single (e.g. one suspect wake map), to attribute the slice spread
  --dump-at S : also save (x, z, pz) of all particles at the node nearest S
  tag -> example/input/ts_<tag>_config.yaml (written by tilt_scan_setup.py)
Records, into benchmark_results/tilt_scan_1d2d/:
  2d_<tag>_<on|off>.npz : per step  -- s, slice stats (40 final-z slices over +-3 sigma_z: n, mean E,
                          sigma_E, linear-detrended sigma_E, sigma_x), projected sigma_E/x/z;
                          at ~0.1 m spacing -- (z,delta) and (x,x') charge-density histograms;
                          per particle -- total CSR energy kick K, and K_z = kick from the wake taken at the
                          beam-centre x along the particle's own z path (K - K_z = wake x-dependence);
                          flag for particles that ever left the wake mesh (they get zero kick).
  2d_<tag>_on_wakes.h5  : the 2D wake mesh at every kick (pyDFCSR's own write_wakes format)
  /tmp/csr1d_wrap/ts_<tag>.csr (off run only): the beam at the bend entrance (s = 0.1 m), in
                          csr_calc's file format, so the 1D code starts from the same particles."""
import os, sys, shutil, numpy as np
from scipy.interpolate import RegularGridInterpolator
from pyDFCSR_2D import CSR2D

HERE = os.path.dirname(os.path.abspath(__file__))
def _opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
HANDOFF = float(_opt('--handoff', 0.1))
EDGES = [float(v) for v in _opt('--regions', '').split(',') if v]
SINGLE = [float(v) for v in _opt('--single', '').split(',') if v]
DUMP_AT = float(_opt('--dump-at', 'nan'))
OUT = os.path.join(HERE, 'benchmark_results', _opt('--outdir', 'tilt_scan_1d2d')); os.makedirs(OUT, exist_ok=True)
WRAP = '/tmp/csr1d_wrap'
os.chdir(os.path.join(HERE, '..', 'example'))
tag, mode = sys.argv[1], sys.argv[2]
NSLICE, ZR, NH, HR, MC2 = 40, 3.0, 200, 5.0, 510998.95

sim = CSR2D(input_file=f'input/ts_{tag}_config.yaml')
sim.CSR_params.write_name = f'ts_{tag}_{mode}'
sim.prefix = f'{sim.CSR_params.write_name}-{sim.timestamp}'   # prefix is fixed in __init__
if mode == 'off':
    sim.CSR_params.compute_CSR = 0; sim.CSR_params.apply_CSR = 0; sim.CSR_params.write_wakes = False
b = sim.beam; N = len(b.particle.x); E0 = b.init_energy; Q = b.charge
L_LAT = sim.lattice.lattice_length if hasattr(sim.lattice, 'lattice_length') else sim.lattice.schedule.lattice_length
hist_at = np.round(np.linspace(0, L_LAT, 17), 6)    # positions at which phase-space histograms are kept (nearest node)
hist_nodes = {float(sim.lattice.schedule.s_nodes[np.argmin(abs(sim.lattice.schedule.s_nodes - h))]) for h in hist_at}
rec = {k: [] for k in ['s', 'zc', 'n', 'mE', 'sE', 'sEd', 'sx', 'projE', 'projx', 'projz']}
H = {k: [] for k in ['s', 'H_zd', 'ez', 'ed', 'H_xxp', 'ex', 'exp']}

def hist2(a, c):
    ea = np.linspace(a.mean() - HR * a.std(), a.mean() + HR * a.std(), NH + 1)
    ec = np.linspace(c.mean() - HR * c.std(), c.mean() + HR * c.std(), NH + 1)
    h, _, _ = np.histogram2d(a, c, bins=[ea, ec])
    return h * Q / len(a) / np.diff(ea)[0] / np.diff(ec)[0], ea, ec

def record():
    p = b.particle; z = p.z - p.z.mean(); sz = z.std(); e = p.pz * E0
    edges = np.linspace(-ZR * sz, ZR * sz, NSLICE + 1); idx = np.digitize(z, edges) - 1
    cols = {k: np.zeros(NSLICE) for k in ['n', 'mE', 'sE', 'sEd', 'sx']}
    for i in range(NSLICE):
        m = idx == i; cols['n'][i] = m.sum()
        if m.sum() < 3: continue
        zi, ei = z[m], e[m]
        cols['mE'][i] = ei.mean(); cols['sE'][i] = ei.std()
        cols['sEd'][i] = np.std(ei - np.polyval(np.polyfit(zi, ei, 1), zi)); cols['sx'][i] = p.x[m].std()
    rec['s'].append(b.position); rec['zc'].append(0.5 * (edges[1:] + edges[:-1]))
    for k, v in cols.items(): rec[k].append(v)
    rec['projE'].append(e.std()); rec['projx'].append(p.x.std()); rec['projz'].append(sz)
    if min(abs(h - b.position) for h in hist_nodes) < 1e-6:
        H['s'].append(b.position)
        for (h, e1, e2), names in [(hist2(p.z, p.pz), ('H_zd', 'ez', 'ed')), (hist2(p.x, p.px), ('H_xxp', 'ex', 'exp'))]:
            H[names[0]].append(h); H[names[1]].append(e1); H[names[2]].append(e2)
    if DUMP_AT == DUMP_AT and abs(b.position - DUMP_AT) < 1e-6 and not dump:
        dump.update(dx=p.x.copy(), dz=p.z.copy(), dpz=p.pz.copy(), ds=b.position)
    if mode == 'off' and abs(b.position - HANDOFF) < 1e-9:  # bend entrance: hand the beam to the 1D code
        g = (1 + p.pz) * p.p0c / MC2                         # p ~ E at gamma ~ 1e4
        arr = np.column_stack([p.x, p.px, p.y, p.py, -p.z, g])   # csr_calc negates z (Elegant convention)
        os.makedirs(WRAP, exist_ok=True); np.savetxt(os.path.join(WRAP, f'ts_{tag}.csr'), arr, fmt='%.12e')
        print('wrote 1D input beam at s =', b.position, flush=True)

K, Kz, out_mesh = np.zeros(N), np.zeros(N), np.zeros(N, bool)
NR = max(len(EDGES) - 1, 0) + len(SINGLE)
Kr, Kzr = np.zeros((NR, N), np.float32), np.zeros((NR, N), np.float32)
dump = {}
def region_of(pos):
    for j, sp in enumerate(SINGLE):
        if abs(pos - sp) < 5e-4: return len(EDGES) - 1 + j
    for j in range(len(EDGES) - 1):
        if EDGES[j] <= pos < EDGES[j + 1] or (j == len(EDGES) - 2 and pos >= EDGES[-1] - 1e-9): return j
    return None
_apply = b.apply_wakes
def apply_wakes(dE_dct, x_kick, xrange, zrange, step_size, transverse_on):
    global K, Kz
    pz0 = b.particle.pz.copy(); xt, z = b.x_transform, b.particle.z
    f = RegularGridInterpolator((xrange, zrange), step_size * dE_dct * 1e6 / E0, fill_value=0.0, bounds_error=False)
    kz = f(np.array([np.full(N, xt.mean()), z]).T); Kz += kz
    out_mesh[(xt < xrange[0]) | (xt > xrange[-1]) | (z < zrange[0]) | (z > zrange[-1])] = True
    _apply(dE_dct, x_kick, xrange, zrange, step_size, transverse_on)
    k = b.particle.pz - pz0; K += k
    r = region_of(b.position) if NR else None
    if r is not None: Kr[r] += k; Kzr[r] += kz
b.apply_wakes = apply_wakes
_upd = sim.update_statistics
def update_statistics(step):
    _upd(step); record()
sim.update_statistics = update_statistics
p = b.particle; x0, xp0, z0, d0 = p.x.copy(), p.px.copy(), p.z.copy(), p.pz.copy()
record()                     # s = 0 (step-0 statistics were taken in the constructor)
sim.run()
p = b.particle
SUFFIX = _opt('--suffix', '')
np.savez(os.path.join(OUT, f'2d_{tag}_{mode}{SUFFIX}.npz'), E0=E0, charge=Q,
         **{k: np.array(v) for k, v in rec.items()}, **{f'hist_{k}': np.array(v) for k, v in H.items()},
         K=K, Kz=Kz, out_mesh=out_mesh, Kr=Kr, Kzr=Kzr, edges=np.array(EDGES), single=np.array(SINGLE), **{f'dump_{k}': v for k, v in dump.items()}, x0=x0, xp0=xp0, z0=z0, d0=d0, xf=p.x, zf=p.z, df=p.pz)
if mode == 'on':
    shutil.copy(f'output/{sim.prefix}-wakes.h5', os.path.join(OUT, f'2d_{tag}_on{SUFFIX}_wakes.h5'))
print(f'saved; {len(rec["s"])} steps; fraction ever outside wake mesh: {out_mesh.mean():.4f}', flush=True)
