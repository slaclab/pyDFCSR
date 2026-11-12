"""
Tao runs of BC20 (BEGBC20 -> ENDBC20) from Sergei Kladov's BEGBC20 beam, sextupoles at 5 mm steps.

    conda run -n bmad-wch python pyDFCSR_2D/test/facet_bc20/tao_bc20.py <name>

  tao_csr   1D CSR only (csr_method 1_dim in BC20), everything else off: the like-for-like
            reference for the DFCSR 1D/2D runs, which compute CSR only
  tao_all   Sergei's BC20 settings (S2E_guide.pdf section 5.4): ISR fluctuations, 1D CSR, slice
            space charge, wakes; n_bin 128, ds_track_step 0.05 m

Common to both, as in Sergei's Activate_CSR.tao: beam_chamber_height 0.024 m, n_shield_images 0
(free space), particle_bin_span 2, lsc_sigma_cutoff 0.1, apertures off.

Writes test/benchmark_results/facet_bc20/runs/<name>.npz: the beam (x, px, y, py, z, pz) at every
element exit, element names and s.
"""
import os
import shutil
import sys
import tempfile
import time

import numpy as np
from pytao import Tao

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', 'example', 'bmad_Wchicane_Sergei', 'S2E_reference',
                   'E320_17418_2026-10-19')
OUT = os.path.join(HERE, '..', 'benchmark_results', 'facet_bc20', 'runs')
SEXT_DS = 0.005          # as export_bc20.py and the pyDFCSR lattice
COORDS = ('x', 'px', 'y', 'py', 'z', 'pz', 'state')

RUNS = {
    'tao_csr': dict(isr=False, csr=True, sc=False, wakes=False),
    'tao_all': dict(isr=True, csr=True, sc=True, wakes=True),
}
# 1D CSR only, binning scan: n_bin and particle_bin_span (defaults above: 128 and 2)
for _nb in (64, 256, 512, 1024, 2048):
    RUNS[f'tao_csr_nb{_nb}'] = dict(isr=False, csr=True, sc=False, wakes=False, n_bin=_nb)
for _sp in (1, 4, 8):
    RUNS[f'tao_csr_span{_sp}'] = dict(isr=False, csr=True, sc=False, wakes=False, span=_sp)
    RUNS[f'tao_csr_nb1024_span{_sp}'] = dict(isr=False, csr=True, sc=False, wakes=False, n_bin=1024, span=_sp)


def main(name):
    r = RUNS[name]
    tf = lambda b: 'T' if b else 'F'
    tmp = tempfile.mkdtemp(prefix='bc20_')
    for f in ('lattice.bmad', 'beam_BEGBC20.h5'):
        shutil.copy(os.path.join(SRC, f), tmp)
    tao = Tao(f"-lat {tmp}/lattice.bmad -noplot -noinit")
    cmds = [
        f"set bmad_com radiation_fluctuations_on = {tf(r['isr'])}",
        "set bmad_com radiation_damping_on = F",
        f"set bmad_com csr_and_space_charge_on = {tf(r['csr'] or r['sc'])}",
        f"set bmad_com sr_wakes_on = {tf(r['wakes'])}", f"set bmad_com lr_wakes_on = {tf(r['wakes'])}",
        "set bmad_com aperture_limit_on = F",
        "set ele * csr_method = off",
        f"set ele BEGBC20:ENDBC20 csr_method = {'1_dim' if r['csr'] else 'off'}",
        f"set ele * space_charge_method = {'slice' if r['sc'] else 'off'}",
        "set ele * csr_ds_step = 0",
        f"set space_charge_com n_bin = {r.get('n_bin', 128)}", "set space_charge_com ds_track_step = 0.05",
        "set space_charge_com beam_chamber_height = 0.024", "set space_charge_com n_shield_images = 0",
        f"set space_charge_com particle_bin_span = {r.get('span', 2)}", "set space_charge_com lsc_sigma_cutoff = 0.1",
        f"set ele sextupole::BEGBC20:ENDBC20 ds_step = {SEXT_DS}",
        "set global random_seed = 1234",
        f"set beam_init position_file = {tmp}/beam_BEGBC20.h5",
        "set beam track_start = BEGBC20", "set beam track_end = ENDBC20",
        "set beam saved_at = BEGBC20:ENDBC20",
    ]
    for c in cmds:
        tao.cmd(c)
    t0 = time.time()
    tao.cmd("set global track_type = beam")
    runtime = time.time() - t0
    i0 = tao.ele_head('BEGBC20')['ix_ele']; i1 = tao.ele_head('ENDBC20')['ix_ele']
    ix = list(range(i0, i1 + 1))
    heads = [tao.ele_head(str(i)) for i in ix]
    coords = np.stack([np.array([tao.bunch1(str(i), k) for k in COORDS]) for i in ix])
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f'{name}.npz'), coords=coords, p0c=tao.ele_gen_attribs('BEGBC20')['P0C'],
             names=np.array([h['name'] for h in heads]), s=np.array([h['s'] for h in heads]) - heads[0]['s'],
             runtime=runtime, settings=repr(r))
    alive = int(coords[-1, 6].sum())
    print(f'{name}: tracked in {runtime:.1f} s, {alive} alive, '
          f"<pz> end {coords[-1, 5].mean():+.6e}, sigma_z end {coords[-1, 4].std() * 1e6:.3f} um")
    shutil.rmtree(tmp)


if __name__ == '__main__':
    main(sys.argv[1])
