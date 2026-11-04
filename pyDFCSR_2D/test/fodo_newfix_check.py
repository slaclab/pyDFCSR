"""
Regression check for the 2026-11-03 options on the FODO case (test_fodo_auto.py, progress_comoving_interp 11t).
It is the one lattice where the whole wake is the W2 / div(v) term: there the wake peak / (-alpha/beta) was
0.1328..0.1356 (2.1% spread) through the sign change at the waist. The same config is run twice:
  base   defaults as recorded (tangent_bands False, vx_model none)
  new    tangent_bands True, vx_model affine
and every kick's wake map and the W2 ratio are compared.   Usage: python fodo_newfix_check.py
"""
import os, sys, time
import numpy as np
import yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_fodo_auto as T
from pyDFCSR_2D import CSR2D

OUT = os.path.join(T.RESULT_DIR, 'newfix_check'); os.makedirs(OUT, exist_ok=True)
NX, NZ = T.MESH_XBINS, T.MESH_ZBINS


def run(tag, integ, dep):
    cache = os.path.join(OUT, f'{tag}.npz')
    if os.path.exists(cache):
        d = np.load(cache); return d['s'], d['dE'], d['xk'], d['sa'], d['a'], d['b'], float(d['wall'])
    os.chdir(T.EXAMPLE_DIR)
    p = T.write_config(f'newfix_{tag}')
    cfg = yaml.safe_load(open(p)); cfg['CSR_integration'].update(integ); cfg['particle_deposition'].update(dep)
    yaml.safe_dump(cfg, open(p, 'w'))
    csr = CSR2D(input_file=p); S, E, X = [], [], []; orig = csr.calculate_2D_CSR

    def cap():
        orig(); S.append(float(csr.beam.position))
        E.append(csr.dE_dct.reshape(NX, NZ).copy()); X.append(csr.x_kick.reshape(NX, NZ).copy())
    csr.calculate_2D_CSR = cap
    t0 = time.time(); csr.run(debug=False); wall = time.time() - t0
    st = csr.statistics
    np.savez(cache, s=S, dE=E, xk=X, sa=np.asarray(csr.lattice.schedule.s_nodes[:len(st['sigma_z'])], float), a=st['twiss']['alpha_x'], b=st['twiss']['beta_x'], wall=wall)
    return run(tag, integ, dep)


def ratios(s, dE, sa, a, b):
    out = []
    for sm, m in zip(s, dE):
        mid = m[NX // 2]; pk = mid[np.argmax(np.abs(mid))]
        q = -np.interp(sm, sa, a) / np.interp(sm, sa, b)
        out.append(pk / q if abs(q) > 1e-3 else np.nan)
    return np.array(out)


if __name__ == '__main__':
    A = run('base', dict(tangent_bands=False), dict(vx_model='none'))
    B = run('new', dict(tangent_bands=True), dict(vx_model='affine'))
    assert np.allclose(A[0], B[0]), 'kick positions differ'
    L = [__doc__.strip(), '']
    dE_a, dE_b = np.array(A[1]), np.array(B[1]); xk_a, xk_b = np.array(A[2]), np.array(B[2])
    L.append(f'{len(A[0])} kicks; runtime base {A[6]:.0f} s, new {B[6]:.0f} s')
    L.append(f'longitudinal maps: rel L2 change {np.linalg.norm(dE_b - dE_a) / np.linalg.norm(dE_a):.4f}, '
             f'max |change| {np.abs(dE_b - dE_a).max():.3e} of peak {np.abs(dE_a).max():.3e} MeV/m')
    L.append(f'transverse maps:   rel L2 change {np.linalg.norm(xk_b - xk_a) / max(np.linalg.norm(xk_a), 1e-300):.4f}, '
             f'max |change| {np.abs(xk_b - xk_a).max():.3e}')
    for tag, R in (('base', A), ('new', B)):
        r = ratios(R[0], np.array(R[1]), R[3], R[4], R[5]); r = r[np.isfinite(r)]
        L.append(f'W2 ratio wake peak / (-alpha/beta), {tag:4s}: {r.min():.4f} .. {r.max():.4f}, spread {np.ptp(r) / np.mean(r) * 100:.1f} % '
                 f'over {r.size} kicks')
    L.append('per kick: s, rel change of the map')
    for sm, a, b in zip(A[0], dE_a, dE_b):
        L.append(f'  {sm:8.4f}  {np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-300):.4f}')
    open(os.path.join(OUT, 'summary.txt'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
