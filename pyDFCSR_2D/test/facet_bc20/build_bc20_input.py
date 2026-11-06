"""
Step 2 of 3: build the pyDFCSR input for the FACET-II W chicane (BC20) from the Tao export.

Reads   test/facet_bc20/bc20_export.npz            (export_bc20.py, Tao side)
        example/bmad_Wchicane_Sergei/.../beam_BEGBC20.h5
Writes  example/input/facet_bc20_lattice.yaml       BEGBC20 -> ENDBC20, 48.08 m
        example/input/facet_bc20_beam.txt           Bmad coordinates at BEGBC20
        example/input/facet_bc20_config.yaml        CSR run (2D, comoving deposition, auto steps)
        example/input/facet_bc20_nocsr_config.yaml  tracking only, for the Bmad comparison

Conversion rules (see progress.md, FACET-II BC20, for why):
  * The reference momentum is Bmad's P0C in BC20 (10116.23 MeV), NOT the beam's mean energy
    (10001.07 MeV). The beam is given as `style: from_file` with `energy: P0C`, so pyDFCSR's
    coordinates are Bmad's coordinates and every K1, K2, G, DG is copied verbatim. With
    `style: ParticleGroup` pyDFCSR would (a) take the mean energy as the reference, so every
    strength would have to be rescaled, and (b) read pg.z, which is identically 0 in a Tao beam
    file (Tao writes the beam at fixed s, with the longitudinal position in t).
  * Superimposed elements (bends and S1/S2 sextupoles split by markers and kickers) are merged
    back into their lords. Drifts, markers, monitors, collimators, zero patches and correctors
    between magnets are merged into one drift; the script asserts every corrector is at 0.
  * Bends: Bmad FRINGE_TYPE = Full has no bmad-x equivalent. hard_edge_only (with FINT, HGAP) is
    the closest: it reproduces Tao's Full fringe to 1e-12 m in x and 4e-11 m in z, 2e-9 in y.
  * The vertical wiggler WIGE1-3 (REF_TILT = pi/2) becomes `type: vbend`: tracked as a tilted
    SBend, but straight for the reference orbit and the CSR integral, which are (x, z) only.
  * Sextupoles get NUM_STEPS for a 5 mm integration step. Sergei's lattice has num_steps = 1,
    which is 34 um off the converged answer in x at ENDBC20.
"""
import json
import os

import h5py
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
SRC = os.path.join(ROOT, 'example', 'bmad_Wchicane_Sergei', 'S2E_reference', 'E320_17418_2026-10-19')
INPUT = os.path.join(ROOT, 'example', 'input')
C_LIGHT = 299792458.0
M_ELECTRON = 510998.95069        # Bmad's value (bmadx.M_ELECTRON)

STRAIGHT = ('Drift', 'Marker', 'Monitor', 'HKicker', 'VKicker', 'RCollimator', 'Patch')


def num(v):
    """A float YAML (1.1, which PyYAML implements) reads back as a float: '2e-05' is a STRING
    there, so every exponent form needs a decimal point."""
    s = repr(float(v))
    if 'e' in s and '.' not in s.split('e')[0]:
        s = s.replace('e', '.0e')
    return s


def check_straight(e):
    a = e['attrs']
    if e['key'] in ('HKicker', 'VKicker'):
        assert a['KICK'] == 0.0, f"{e['name']} has KICK = {a['KICK']}: add it as an element"
    if e['key'] == 'Patch':
        moved = [k for k in ('X_OFFSET', 'Y_OFFSET', 'Z_OFFSET', 'X_PITCH', 'Y_PITCH', 'TILT',
                             'E_TOT_OFFSET', 'T_OFFSET') if a.get(k, 0.0) != 0.0]
        assert not moved, f"{e['name']} is a nonzero patch: {moved}"


def magnet(name, key, a, n_steps_sext):
    if key == 'SBend':
        vertical = abs(a.get('REF_TILT', 0.0)) > 0
        d = dict(type='vbend' if vertical else 'dipole', L=a['L'], angle=a['G'] * a['L'], G=a['G'],
                 DG=a['DG'], E1=a['E1'], E2=a['E2'], FINT=a['FINT'], HGAP=a['HGAP'],
                 FINTX=a['FINTX'], HGAPX=a['HGAPX'], FRINGE_TYPE='hard_edge_only')
        if vertical:
            d['TILT'] = a['REF_TILT']
        assert a.get('TILT', 0.0) == 0.0, f'{name}: misaligned bend'
    elif key == 'Quadrupole':
        d = dict(type='quad', L=a['L'], K1=a['K1'], NUM_STEPS=int(a['NUM_STEPS']))
        for k in ('X_OFFSET', 'Y_OFFSET', 'TILT'):
            if a.get(k, 0.0) != 0.0:
                d[k] = a[k]
    elif key == 'Sextupole':
        d = dict(type='sextupole', L=a['L'], K2=a['K2'], NUM_STEPS=int(n_steps_sext))
        for k in ('X_OFFSET', 'Y_OFFSET', 'TILT'):
            if a.get(k, 0.0) != 0.0:
                d[k] = a[k]
    else:
        raise ValueError(f'{name}: {key} not handled')
    d['nsep'] = 1
    return d


def build_lattice(elements, lords, sext_steps, step_size):
    """[(name, dict)] in beam order, BEGBC20 -> ENDBC20."""
    out, run, i = [], [], 0
    def flush():
        if run:
            L = sum(e['attrs'].get('L', 0.0) for e in run)
            names = [e['name'] for e in run if e['key'] == 'Drift']
            if L > 0:
                out.append((f"{names[0]}_{names[-1]}" if len(names) > 1 else names[0],
                            dict(type='drift', L=L, nsep=1)))
            run.clear()
    while i < len(elements):
        e = elements[i]
        if e['key'] in STRAIGHT:
            check_straight(e)
            run.append(e)
            i += 1
            continue
        flush()
        if e['lord']:                          # superimposed: merge the slaves into the lord
            # the slaves are split by zero-length elements (the trim corrector XCB2LE at the
            # centre of B2LE, a marker at the centre of S1EL): those must be inert to be dropped
            j = i
            while j < len(elements) and (elements[j]['lord'] == e['lord'] or
                                         (not elements[j]['lord'] and elements[j]['key'] in STRAIGHT
                                          and elements[j]['attrs'].get('L', 0.0) == 0.0
                                          and any(x['lord'] == e['lord'] for x in elements[j:]))):
                if not elements[j]['lord']:
                    check_straight(elements[j])
                j += 1
            slaves = [x for x in elements[i:j] if x['lord'] == e['lord']]
            name, a = e['lord'], lords[e['lord']]
            assert abs(sum(s['attrs']['L'] for s in slaves) - a['L']) < 1e-12, name
            n_sext = sum(sext_steps.get(s['name'], 0) for s in slaves)
            out.append((name, magnet(name, e['key'], a, n_sext)))
            i = j
        else:
            out.append((e['name'], magnet(e['name'], e['key'], e['attrs'], sext_steps.get(e['name'], 0))))
            i += 1
    flush()
    # a merged drift is named after its first and last Bmad drift; the same Bmad drift (DQ4E,
    # DS3E, ...) recurs, and the YAML is a mapping, so repeated names get a counter
    seen, uniq = {}, []
    for n, d in out:
        seen[n] = seen.get(n, 0) + 1
        uniq.append((n if seen[n] == 1 else f'{n}__{seen[n]}', d))
    assert all(d['type'] == 'drift' for n, d in uniq if '__' in n), 'a magnet name repeats'
    return uniq


def write_lattice(path, lattice, step_size, p0c, header):
    lines = [f'# {h}' for h in header] + ['', f'step_size: {num(step_size)}', '']
    for name, d in lattice:
        lines.append(f'{name}:')
        for k, v in d.items():
            lines.append(f'  {k}: {v}' if isinstance(v, (int, str)) else f'  {k}: {num(v)}')
        lines.append('')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def bmad_coords_from_h5(path, p0c):
    """
    Bmad (x, px, y, py, z, pz) from a Tao openPMD beam file: z = -beta c (t - t_ref).

    p0c is the LATTICE's P0C. The file stores its own copy, which differs from it in the 17th
    digit (1e-16 relative), so the momenta are normalised to the lattice value.
    """
    with h5py.File(path, 'r') as f:
        g = f['data/00001/particles/electron']
        p0c_file = float(g['totalMomentumOffset'].attrs['value'][0])
        assert abs(p0c_file / p0c - 1) < 1e-14, (p0c_file, p0c)
        P = p0c_file + g['totalMomentum'][()]
        beta = P / np.sqrt(P ** 2 + M_ELECTRON ** 2)
        coords = np.column_stack([g['position/x'][()], g['momentum/x'][()] / p0c,
                                  g['position/y'][()], g['momentum/y'][()] / p0c,
                                  -beta * C_LIGHT * g['time'][()], P / p0c - 1.0])
        charge = float(g.attrs['totalCharge'][0])
        status = g['particleStatus'].attrs['value'][0]
    assert status == 1
    return coords, charge


def main():
    X = np.load(os.path.join(HERE, 'bc20_export.npz'))
    elements = json.loads(str(X['elements']))
    lords = json.loads(str(X['lords']))
    sext_steps = json.loads(str(X['sext_steps_fine']))
    p0c = float(X['p0c'])

    coords, charge = bmad_coords_from_h5(os.path.join(SRC, 'beam_BEGBC20.h5'), p0c)
    tao0 = X['coords_fine'][0][:6].T
    err = np.abs(coords - tao0).max(0)
    assert err.max() < 1e-15, f'beam conversion differs from Tao: {err}'
    np.savetxt(os.path.join(INPUT, 'facet_bc20_beam.txt'), coords)   # %.18e: exact round trip

    step_size = 0.05
    lattice = build_lattice(elements[1:], lords, sext_steps, step_size)   # [0] is BEGBC20
    L_tot = sum(d['L'] for _, d in lattice)
    L_bmad = elements[-1]['s'] - elements[0]['s']
    assert abs(L_tot - L_bmad) < 1e-9, (L_tot, L_bmad)
    header = [
        'FACET-II W chicane BC20, BEGBC20 -> ENDBC20, from Sergei Kladov\'s tuned DAQ simulation',
        'E320_17418 (2026-10-19, FACET2-S2E commit 3bc1b17), lattice.bmad as written by Tao.',
        'Generated by pyDFCSR_2D/test/facet_bc20/build_bc20_input.py -- do not edit by hand.',
        f'Strengths are Bmad\'s, normalised to P0C = {p0c!r} eV: the beam file uses the same',
        'reference (input_beam.energy). They are NOT normalised to the beam energy (10001 MeV).',
        'Bends: FRINGE_TYPE Full in Bmad -> hard_edge_only here (the closest bmad-x has).',
        'vbend = the vertical wiggler, tracked tilted, straight for the 2D CSR geometry.',
        f'Total length {L_tot!r} m.',
    ]
    write_lattice(os.path.join(INPUT, 'facet_bc20_lattice.yaml'), lattice, step_size, p0c, header)

    # read back: every number must be a float, and in the same order
    back = yaml.safe_load(open(os.path.join(INPUT, 'facet_bc20_lattice.yaml')))
    for name, d in lattice:
        for k, v in d.items():
            assert type(back[name][k]) is type(v) and back[name][k] == v, (name, k, back[name][k], v)

    beam = dict(style='from_file', beamfile='input/facet_bc20_beam.txt', charge=charge, energy=p0c)
    def config(nocsr):
        c = dict(
            input_beam=beam,
            input_lattice=dict(lattice_input_file='input/facet_bc20_lattice.yaml'),
            step_control=dict(mode='manual' if nocsr else 'auto', kick_interval='midpoint'),
            particle_deposition=dict(method='bspline_comoving', xbins=300, zbins=300, xlim=5, zlim=5,
                                     smoothing_sigma=3.0, poly_degree=1, velocity_threhold=1000),
            CSR_integration=dict(n_formation_length=1, xbins=200, zbins=200),
            CSR_computation=dict(compute_CSR=0 if nocsr else 1, apply_CSR=0 if nocsr else 1,
                                 transverse_on=1, xbins=12, zbins=40, xlim=4, zlim=4,
                                 write_beam=None, write_wakes=not nocsr,
                                 write_name='facet_bc20_nocsr' if nocsr else 'facet_bc20',
                                 workdir='./output'))
        if nocsr:
            c['step_control']['default_steps'] = None
        return c
    for nocsr, fn in ((False, 'facet_bc20_config.yaml'), (True, 'facet_bc20_nocsr_config.yaml')):
        c = config(nocsr)
        if nocsr:
            del c['step_control']['default_steps']
        txt = yaml.safe_dump(c, sort_keys=False)
        txt = txt.replace(f'energy: {p0c!r}', f'energy: {num(p0c)}').replace(
            f'charge: {charge!r}', f'charge: {num(charge)}')
        with open(os.path.join(INPUT, fn), 'w') as f:
            f.write('# FACET-II BC20, generated by pyDFCSR_2D/test/facet_bc20/build_bc20_input.py\n' + txt)
        r = yaml.safe_load(txt)['input_beam']
        assert isinstance(r['energy'], float) and isinstance(r['charge'], float)

    kinds = {}
    for _, d in lattice:
        kinds[d['type']] = kinds.get(d['type'], 0) + 1
    print(f'{len(lattice)} elements {kinds}, L = {L_tot:.6f} m (Bmad {L_bmad:.6f}),'
          f' {coords.shape[0]} particles, Q = {charge:.3e} C, P0C = {p0c:.6e} eV')


if __name__ == '__main__':
    main()
