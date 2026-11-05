import sys, os, time, contextlib, io, yaml, numpy as np
sys.path.insert(0, os.getcwd())
from d3_wake_diagnostic import build, wake_mesh, evaluate, OUT, quiet
from pyDFCSR_2D import CSR2D
which = sys.argv[1]; s_stop = float(sys.argv[2])
if which == 'dipole':
    ex = os.path.join(os.getcwd(), '..', 'example'); os.chdir(ex)
    cfg = yaml.safe_load(open('input/sesp_cm_config.yaml'))
    cfg['CSR_computation'].update(compute_CSR=0, apply_CSR=0, write_wakes=False, write_beam=None)
    cfg['CSR_integration'].update(drift_cutoff=0.0, drift_reach=True, drift_snap_thin=0.01)
    f = f'input/_fc_{os.getpid()}.yaml'; yaml.safe_dump(cfg, open(f, 'w'))
    sim = quiet(CSR2D, input_file=f); sim.dump_beam = lambda l: None; sim.write_statistics = lambda: None
    quiet(sim.run, stop_time=s_stop, debug=True)
else:
    sim = build(s_stop, {'CSR_integration': dict(n_formation_length=1.0, drift_reach=True, drift_snap_thin=0.01)})
xm, zm = wake_mesh(sim); nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins
res = {}
for name, p in (('far_cell 10', {}), ('cap 300', dict(far_fine_max=300)), ('cap 500', dict(far_fine_max=500)), ('cap 800', dict(far_fine_max=800)),
                ('ref cell 3, window 2', dict(far_cell=3.0, far_window=2.0))):
    t0 = time.time(); W = evaluate(sim, xm, zm, p).reshape(nx, nz); res[name] = (W, time.time() - t0, sim._last_region_nodes[0])
ref = res['ref cell 3, window 2'][0]
print(f'{which} s = {sim.beam.position:.4f} ({sim.lattice.current_element}, exit transient {sim._exit_transient()}), peak |W| {np.abs(ref).max():.4g} MeV/m: '
      + '; '.join(f'{n}: {nn} nodes, {dt/W.size*1e3:.0f} ms/pt, rel L2 {np.linalg.norm(W-ref)/np.linalg.norm(ref):.2e}' for n, (W, dt, nn) in res.items() if not n.startswith('ref')), flush=True)
