"""
Package the BC20 lattice and beam for someone running the same case in another code.

    conda run -n pydfcsr python pyDFCSR_2D/test/facet_bc20/export_for_sharing.py

Writes test/facet_bc20/share/facet_bc20_package/ and facet_bc20_package.zip:
  README.txt                    coordinates, units, lattice conventions
  beam_BEGBC20.txt              the 20 000 particles at BEGBC20 (the pyDFCSR input beam)
  lattice_dfcsr.yaml            the pyDFCSR lattice file (verbatim)
  lattice_table.txt             the same lattice as a flat table, one element per row
  optics_nocsr.txt              beam moments and Twiss at BEGBC20 and every element exit,
                                every collective effect off (Tao, 5 mm sextupole steps)
  beam_ENDBC20_nocsr.txt        the particles at ENDBC20 with every collective effect off, same
                                particle order as beam_BEGBC20.txt
The reference data are Tao's (bc20_export.npz, coords_fine). pyDFCSR with CSR off reproduces
them to 1e-4 sigma_x and 1e-5 sigma_z (progress_facet_bc20.md section 5).
"""
import json
import os
import shutil
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(ROOT, '..'))
from pyDFCSR_2D.twiss import twiss_dispersion_calc  # noqa: E402

OUT = os.path.join(HERE, 'share', 'facet_bc20_package')
LAT = os.path.join(ROOT, 'example', 'input', 'facet_bc20_lattice.yaml')
BEAM = os.path.join(ROOT, 'example', 'input', 'facet_bc20_beam.txt')
CHARGE = 1.6000000000003584e-09
MC2 = 510998.95069

BEAM_HEADER = """FACET-II BC20 beam, {where}. {n} macroparticles, total charge {q:.10g} C (equal weights).
Bmad phase-space coordinates relative to the reference momentum P0C = {p0c:.6f} eV (electrons):
  x [m], px = P_x/P0, y [m], py = P_y/P0, z [m], pz = (P - P0)/P0
px, py are canonical (not x' = dx/ds). z = -beta*c*(t - t_ref) at fixed s: z > 0 is the bunch
HEAD (ahead of the reference particle). The beam's mean energy is 1.14% below P0C (<pz> = {mpz:.6e}).
Columns: x px y py z pz"""


def lattice_table(L):
    cols = ['index', 'name', 'type', 's_start', 's_end', 'L', 'angle', 'G', 'DG', 'E1', 'E2',
            'FINT', 'HGAP', 'FINTX', 'HGAPX', 'K1', 'K2', 'X_OFFSET', 'TILT', 'NUM_STEPS']
    rows, s = [], 0.0
    for i, (k, v) in enumerate((k, v) for k, v in L.items() if k != 'step_size'):
        r = [str(i + 1), k, v['type'], f'{s:.12f}', f'{s + v["L"]:.12f}']
        for c in cols[5:]:
            r.append(repr(float(v[c])) if c in v else ('0' if c != 'NUM_STEPS' else '-'))
        rows.append(r)
        s += v['L']
    w = [max(len(c), *(len(r[j]) for r in rows)) for j, c in enumerate(cols)]
    fmt = lambda r: '  '.join(x.rjust(w[j]) for j, x in enumerate(r))
    head = ['# FACET-II BC20 (BEGBC20 -> ENDBC20) in pyDFCSR form, one element per row; see README.txt',
            '# lengths and offsets [m], angles [rad], G and DG [1/m], K1 [1/m^2], K2 [1/m^3]',
            '# ' + fmt(cols)]
    return '\n'.join(head + ['  ' + fmt(r) for r in rows]) + '\n'


def moments(c, p0c):
    x, px, y, py, z, pz = c[:6]
    tx = twiss_dispersion_calc(np.cov([x, px, pz]))
    ty = twiss_dispersion_calc(np.cov([y, py, pz]))
    g = p0c / MC2
    return [x.mean(), y.mean(), z.mean(), pz.mean(), x.std(), y.std(), z.std(), pz.std(),
            tx['beta'], tx['alpha'], tx['eta'], tx['etap'], tx['emit'], tx['emit'] * g,
            ty['beta'], ty['alpha'], ty['eta'], ty['etap'], ty['emit'], ty['emit'] * g]


def main():
    X = np.load(os.path.join(HERE, 'bc20_export.npz'))
    els = json.loads(str(X['elements']))
    s_tao = np.array([e['s'] for e in els]) - els[0]['s']
    p0c = float(X['p0c'])
    tao = X['coords_fine']
    L = yaml.safe_load(open(LAT))
    names = [k for k in L if k != 'step_size']
    ends = np.cumsum([L[k]['L'] for k in names])
    jt = [0] + [int(np.flatnonzero(np.abs(s_tao - e) < 1e-9)[-1]) for e in ends]
    beam0 = np.loadtxt(BEAM)
    # equal to round-off (pz: the beam file's P0C differs from the lattice's in the 17th digit)
    assert np.allclose(beam0, tao[0][:6].T, rtol=1e-12, atol=1e-15), 'input beam differs from Tao BEGBC20'
    assert tao[-1][6].all(), 'particles lost in the reference'

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    shutil.copy(LAT, os.path.join(OUT, 'lattice_dfcsr.yaml'))
    open(os.path.join(OUT, 'lattice_table.txt'), 'w').write(lattice_table(L))
    np.savetxt(os.path.join(OUT, 'beam_BEGBC20.txt'), beam0, fmt='%.17e',
               header=BEAM_HEADER.format(where='at BEGBC20 (start of the lattice)', n=len(beam0),
                                         q=CHARGE, p0c=p0c, mpz=beam0[:, 5].mean()))
    end = tao[-1][:6].T
    np.savetxt(os.path.join(OUT, 'beam_ENDBC20_nocsr.txt'), end, fmt='%.17e',
               header=BEAM_HEADER.format(where='at ENDBC20, every collective effect OFF (Tao reference), '
                                               'same particle order as beam_BEGBC20.txt',
                                         n=len(end), q=CHARGE, p0c=p0c, mpz=end[:, 5].mean()))

    cols = ['s', 'mean_x', 'mean_y', 'mean_z', 'mean_pz', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_pz',
            'beta_x', 'alpha_x', 'eta_x', 'etap_x', 'emit_x', 'emit_nx',
            'beta_y', 'alpha_y', 'eta_y', 'etap_y', 'emit_y', 'emit_ny']
    rows = []
    for nm, s, j in zip(['BEGBC20'] + names, np.r_[0.0, ends], jt):
        rows.append((nm, [s] + moments(tao[j], p0c)))
    w = max(len(n) for n, _ in rows)
    lines = ['# FACET-II BC20 with every collective effect OFF (no CSR, ISR, space charge, wakes):',
             '# beam moments at BEGBC20 and at the exit of every element of lattice_table.txt.',
             '# Reference: Bmad/Tao (Bmad 20261031), sextupoles at 5 mm steps, bends FRINGE_TYPE Full.',
             '# From the 20 000 particles: means and rms in Bmad coordinates (see README.txt);',
             '# Twiss and dispersion from the beam covariance of (x, px, pz) and (y, py, pz), with',
             '# the linear correlation with pz removed (eta = <x pz>/<pz^2>, etap = <px pz>/<pz^2>):',
             '# beta [m], alpha [1], eta [m], etap [1], emit = geometric [m], emit_n = emit * P0C/mc^2 [m].',
             '# px is canonical (P_x/P0), so these are not exactly the x\' = dx/ds values for a 1% energy spread.',
             '# s [m] from BEGBC20. Columns: name ' + ' '.join(cols)]
    for nm, v in rows:
        lines.append(nm.ljust(w) + ' ' + ' '.join(f'{x: .10e}' for x in v))
    open(os.path.join(OUT, 'optics_nocsr.txt'), 'w').write('\n'.join(lines) + '\n')

    shutil.copy(os.path.join(HERE, 'share', 'README_template.txt'), os.path.join(OUT, 'README.txt'))
    zp = shutil.make_archive(os.path.join(HERE, 'share', 'facet_bc20_package'), 'zip',
                             root_dir=os.path.join(HERE, 'share'), base_dir='facet_bc20_package')
    print(f'{zp}: {sorted(os.listdir(OUT))}')
    print('ENDBC20, CSR off: sigma_x %.4g um, sigma_z %.5g um, emit_nx %.4g um' %
          (rows[-1][1][5] * 1e6, rows[-1][1][7] * 1e6, rows[-1][1][14] * 1e6))


if __name__ == '__main__':
    main()
