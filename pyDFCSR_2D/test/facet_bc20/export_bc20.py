"""
Step 1 of 3 (Tao side): export the FACET-II W chicane (BC20) from Sergei Kladov's tuned DAQ lattice.

Reads   example/bmad_Wchicane_Sergei/S2E_reference/E320_17418_2026-10-19/{lattice.bmad, beam_BEGBC20.h5}
Writes  test/facet_bc20/bc20_export.npz:
          elements  -- every tracking element BEGBC20..ENDBC20 (slaves) and the lords of the
                       superimposed ones, as JSON
          coords    -- Tao's beam at every element exit, collective effects all OFF, for
                       sextupole num_steps = 1 (Sergei's lattice as is) and = SEXT_FINE
          p0c       -- Bmad reference momentum in BC20

Runs in a conda env with Bmad + pytao (pytao 1.4 / Bmad 20261031 here; the `bmad` env on this
machine has a broken numpy), e.g.
    conda create -n bmad-wch -c conda-forge "bmad=*=nompi*" pytao openpmd-beamphysics
    conda run -n bmad-wch python pyDFCSR_2D/test/facet_bc20/export_bc20.py

Tao truncates long paths and this tree has spaces in it, so the lattice and beam are copied to a
temporary directory first.
"""
import json
import os
import shutil
import tempfile

import numpy as np
from pytao import Tao

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', 'example', 'bmad_Wchicane_Sergei', 'S2E_reference',
                   'E320_17418_2026-10-19')
OUT = os.path.join(HERE, 'bc20_export.npz')
COORDS = ('x', 'px', 'y', 'py', 'z', 'pz', 'state')
# sextupole integration converged to < 1e-9 m (step 0.005 m): see progress.md
SEXT_DS = 0.005


def tao_bc20(tmp, sext_ds=None):
    tao = Tao(f"-lat {tmp}/lattice.bmad -noplot -noinit")
    for c in ["set bmad_com csr_and_space_charge_on = F",
              "set bmad_com radiation_fluctuations_on = F",
              "set bmad_com radiation_damping_on = F",
              "set bmad_com sr_wakes_on = F", "set bmad_com lr_wakes_on = F",
              "set bmad_com aperture_limit_on = F"]:
        tao.cmd(c)
    if sext_ds:
        tao.cmd(f"set ele sextupole::BEGBC20:ENDBC20 ds_step = {sext_ds}")
    for c in [f"set beam_init position_file = {tmp}/beam_BEGBC20.h5",
              "set beam track_start = BEGBC20", "set beam track_end = ENDBC20",
              "set beam saved_at = BEGBC20:ENDBC20", "set global track_type = beam"]:
        tao.cmd(c)
    return tao


def main():
    tmp = tempfile.mkdtemp(prefix='bc20_')
    for f in ('lattice.bmad', 'beam_BEGBC20.h5'):
        shutil.copy(os.path.join(SRC, f), tmp)

    tao = tao_bc20(tmp)
    i0 = tao.ele_head('BEGBC20')['ix_ele']
    i1 = tao.ele_head('ENDBC20')['ix_ele']
    ix = list(range(i0, i1 + 1))

    elements, lords = [], {}
    for i in ix:
        h = tao.ele_head(str(i))
        attrs = {k: v for k, v in tao.ele_gen_attribs(str(i)).items() if not k.startswith('units#')}
        lord = [r['name'] for r in tao.ele_lord_slave(str(i)) if r['type'] == 'Lord']
        elements.append(dict(ix=i, name=h['name'], key=h['key'], s=h['s'], s_start=h['s_start'],
                             lord=lord[0] if lord else None, attrs=attrs))
        for name in lord:
            if name not in lords:
                lords[name] = {k: v for k, v in tao.ele_gen_attribs(name).items()
                               if not k.startswith('units#')}

    def coords(t):
        return np.stack([np.array([t.bunch1(str(i), k) for k in COORDS]) for i in ix])

    c_as_is = coords(tao)
    tao_fine = tao_bc20(tmp, SEXT_DS)
    c_fine = coords(tao_fine)
    steps_fine = {e['name']: tao_fine.ele_gen_attribs(e['name'])['NUM_STEPS']
                  for e in elements if e['key'] == 'Sextupole'}

    np.savez(OUT, ix=np.array(ix), p0c=tao.ele_gen_attribs('BEGBC20')['P0C'],
             elements=json.dumps(elements, default=str), lords=json.dumps(lords, default=str),
             coords_as_is=c_as_is, coords_fine=c_fine, sext_ds=SEXT_DS,
             sext_steps_fine=json.dumps(steps_fine))
    alive = int(c_as_is[-1, 6].sum())
    print(f'wrote {OUT}: {len(ix)} elements, {c_as_is.shape[2]} particles, {alive} alive at ENDBC20')
    shutil.rmtree(tmp)


if __name__ == '__main__':
    main()
