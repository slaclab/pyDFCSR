"""
Replacing the 3-region s' layout with 2 regions: does it hold at a bend EXIT?

Section 11s claimed s2 = s3 - 200 sigma_z was vestigial, "only partitioning the node
budget". That was wrong, and this test is what showed it. Because every region got the
same node COUNT while their lengths differed ~45x, s2 set the CELL SIZE -- measured
45.7 / 1.0 / 0.1 sigma_z across the three regions. It was a crude two-level grading,
and a naive uniform merge was 3-10x WORSE at matched cost.

The author then raised the case that decides the design: at a bend EXIT transient the
source is the part of the trajectory still inside the dipole, and that sits entirely in
the far region. Measured 0.2 m past a 1 rad bend, the dipole overlaps region 1 by 77%
and regions 2 and 3 by 0%. So grading the far region by distance from s' = s puts its
COARSEST cells on the only part that contributes:

  10% -> 99% of the far-region integral lies within ~0.3 L_f above the causal edge,
  at u/L_f ~ 2.7-3.1, and plain log grading spans that with ONE 30 mm cell.

Hence the shipped design, which this test measures:
  - the far region starts at the CAUSAL EDGE (lowest s' with nonzero band width),
    found by bisection, since below it every column is identically zero
  - uniform at far_cell*sigma_z over far_window*L_f above that edge
  - graded by near_grade above the window

Two confounds are controlled, both of which have bitten this area before:

  1. RESOLUTION vs PARTITION. A merge that halves the node count measures coarsening,
     not repartitioning. Node counts are therefore reported alongside every error, and
     the new layout is held to <= the old layout's node count.

  2. A PARTITION-INDEPENDENT REFERENCE. Refining the 3-region split and comparing the
     3-region split to it flatters the 3-region split. The reference here is a single
     uniform 40000-node quadrature of the whole far domain. Verified: three refined
     quadratures of different structure (3-region 4000+4000, 1-region uniform 40000,
     1-region log g=0.0008) agree to 4e-6 rel L2, so one limit exists.

A third trap, recorded because it cost a wrong conclusion here: RELATIVE error at the
+z tail is meaningless, because far and near nearly cancel there (far/total reaches
6.25) and the wake passes through zero. The worst pointwise relative error sits where
|dE| is 0.4% of peak. Errors are reported as rel L2 over the cut.
"""
import sys
import os
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyDFCSR_2D.CSR import CSR2D

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results',
                          'merge_far_regions')
EXAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'example'))

# 0.50 is mid-bend (steady state); 1.20 and 1.60 are 0.1 m and 0.5 m past the exit,
# which is the case the whole design turns on.
OBS_S = [0.50, 1.20, 1.60]
DEP_BINS = 200
N_PARTICLE = 200000
HIST_NFL = 4.0
INT_BINS = dict(xbins=200, zbins=200)


def write_inputs(shear):
    beam = {
        'n_particle': N_PARTICLE, 'species': 'electron',
        'px_dist': {'sigma_px': {'units': 'keV/c', 'value': 25.0}, 'type': 'gaussian'},
        'py_dist': {'sigma_py': {'units': 'keV/c', 'value': 25.0}, 'type': 'gaussian'},
        'pz_dist': {'avg_pz': {'units': 'GeV/c', 'value': 5},
                    'sigma_pz': {'units': 'MeV/c', 'value': 0.5}, 'type': 'gaussian'},
        'x_dist': {'sigma_x': {'units': 'um', 'value': 50}, 'type': 'gaussian'},
        'y_dist': {'sigma_y': {'units': 'um', 'value': 5}, 'type': 'gaussian'},
        'random': {'type': 'hammersley'},
        'start': {'tstart': {'units': 'sec', 'value': 0}, 'type': 'time'},
        'total_charge': {'units': 'nC', 'value': 1},
        'z_dist': {'avg_z': {'units': 'mm', 'value': 0},
                   'sigma_z': {'units': 'um', 'value': 50}, 'type': 'gaussian'},
    }
    if shear:
        beam['transforms'] = {'s1': {'shear_coefficient':
                                     {'units': 'dimensionless', 'value': float(shear)},
                                     'type': 'shear z:x'}}
    bp = f'input/mfr_beam_s{shear:g}.yaml'
    with open(os.path.join(EXAMPLE_DIR, bp), 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)
    config = {
        'input_beam': {'style': 'distgen', 'distgen_input_file': bp},
        'input_lattice': {'lattice_input_file': 'input/dipole_lattice.yaml'},
        'particle_deposition': {'method': 'bspline_comoving',
                                'xbins': DEP_BINS, 'zbins': DEP_BINS,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': 1, 'velocity_threhold': 1000},
        'CSR_integration': dict(n_formation_length=HIST_NFL, **INT_BINS),
        'CSR_computation': {'compute_CSR': 0, 'apply_CSR': 0, 'transverse_on': 1,
                            'xbins': 5, 'zbins': 40, 'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': 'mfr', 'workdir': './output'},
    }
    cp = f'input/mfr_config_s{shear:g}.yaml'
    with open(os.path.join(EXAMPLE_DIR, cp), 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return cp


DIPOLE = (0.1, 1.1)          # dipole_lattice.yaml: drift 0.1, dipole 1.0, drift 0.5


def far_only(csr, s, x, sp):
    """Far-region contribution on an explicit node set, no patch, no taper.

    The near region is identical for every variant, so excluding it stops a shared
    term from diluting the comparison.
    """
    return csr._integrate_xi_region(s, x, csr.beam.position, sp, False, None)[0]


def legacy_three(csr, s, x):
    """The OLD 3-region s' layout, rebuilt from the current 2-region bounds.

    _layout_bounds now returns ((s1, s3), (s3, s4)); the old version inserted
    s2 = s3 - 200 sigma_z and reached back n_fl*L_f from THERE. Reconstructed here so
    the comparison survives the change rather than being deleted with it.
    """
    (_, s3), (_, s4) = csr._layout_bounds(s, x)
    ip = csr.integration_params
    s2 = s3 - 200.0 * csr.beam._sigma_z
    s1 = max(0.0, s2 - ip.n_formation_length * csr.formation_length)
    return ((s1, s2), (s2, s3), (s3, s4))


def log_nodes(a, b, s, g):
    """Nodes graded geometrically in u = s - s'. The variant that FAILS at an exit."""
    u_lo, u_hi = max(s - b, 1e-12), max(s - a, 1e-11)
    if u_hi <= u_lo:
        return np.array([a, b])
    n = int(np.ceil(np.log(u_hi / u_lo) / np.log1p(g)))
    return np.sort(s - np.minimum(u_lo * (1.0 + g) ** np.arange(n + 1), u_hi))


def dipole_overlap(csr, s, x):
    """How much of each OLD region lies inside the dipole. The author's question."""
    out = []
    for a, b in legacy_three(csr, s, x):
        ov = max(0.0, min(b, DIPOLE[1]) - max(a, DIPOLE[0]))
        out.append(ov / (b - a) if b > a else 0.0)
    return out


def mass_band(csr, s, x, a, b, n=4000):
    """Where the far-region integral actually lives: u/L_f at the 10/50/90 percentile."""
    sp = np.linspace(a, b, n)
    col = np.trapz(csr._integrate_xi_region(s, x, csr.beam.position, sp,
                                            False, None)[4], axis=0)
    c = np.concatenate([[0.0], np.cumsum(np.abs(0.5 * (col[1:] + col[:-1]))
                                        * np.diff(sp))])
    if c[-1] <= 0:
        return None
    c = c / c[-1]
    return [(s - sp[int(np.argmin(np.abs(c - f)))]) / csr.formation_length
            for f in (0.1, 0.5, 0.9)]


def variants(csr, s, x):
    """Every far-region node set under comparison, as {label: nodes}."""
    b3 = legacy_three(csr, s, x)
    (s1, s3), _ = csr._layout_bounds(s, x)
    nz = csr.integration_params.far_zbins
    return {
        '3 region (old)': np.unique(np.concatenate(
            [np.linspace(a, b, nz) for a, b in b3[:-1]])),
        'merge uniform': np.linspace(s1, s3, 2 * nz),
        'merge log': log_nodes(s1, s3, s, csr.integration_params.near_grade),
        'SHIPPED': csr._far_region_nodes(s, x, csr.beam.position, s1, s3),
    }


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    lines = []

    def emit(t=''):
        print(t, flush=True)
        lines.append(t)

    emit('=' * 100)
    emit("Dropping s2: the 2-region far layout, tested at a bend EXIT transient")
    emit('=' * 100)
    emit('  reference = ONE uniform 40000-node quadrature of the whole far domain,')
    emit('  so it shares no structure with any variant. Errors are rel L2 over the z')
    emit('  cut; pointwise relative error is meaningless at the +z tail where far and')
    emit('  near cancel. Node counts are reported because a merge that costs more is')
    emit('  not a simplification.')
    emit('')

    cfg = write_inputs(0.0)
    os.chdir(EXAMPLE_DIR)
    labels = ['3 region (old)', 'merge uniform', 'merge log', 'SHIPPED']
    curves = {}

    for s_stop in OBS_S:
        csr = CSR2D(input_file=cfg)
        # debug=True records the density history; compute_CSR = 0 alone records none,
        # which _assert_history now catches instead of returning near-zeros (11d).
        csr.run(stop_time=s_stop, debug=True)
        csr.get_CSR_mesh()
        s0 = csr.beam.position
        nz = csr.CSR_params.zbins
        ix = csr.CSR_params.xbins // 2
        Lf = csr.formation_length
        where = 'in bend' if DIPOLE[0] < s0 <= DIPOLE[1] else 'past exit'
        emit(f'  s = {s0:.3f} m ({where}, {s0 - DIPOLE[1]:+.3f} m from the exit),'
             f'  L_f = {Lf * 1e3:.1f} mm,  sigma_z = {csr.beam._sigma_z * 1e6:.1f} um')

        ov = dipole_overlap(csr, s0, 0.0)
        note = ('   <- past the exit the source sits in region 1 ALONE'
                if s0 > DIPOLE[1] else '   (in bend: all regions are inside the dipole)')
        emit(f'    dipole overlap of the OLD regions 1/2/3: '
             f'{ov[0] * 100:.0f}% / {ov[1] * 100:.0f}% / {ov[2] * 100:.0f}%' + note)
        (s1, s3), _ = csr._layout_bounds(s0, 0.0)
        mb = mass_band(csr, s0, 0.0, s1, s3)
        if mb:
            # mb is (u at 10%, 50%, 90% of the cumulative integral); u DECREASES as the
            # percentile rises, so the 90% entry is the small-u end. Print low..high.
            emit(f'    far-region mass spans u/L_f = {min(mb):.2f} .. {max(mb):.2f} '
                 f'(median {mb[1]:.2f}), where u = s - s\'. Distance grading is '
                 f'coarsest at LARGE u.')

        got = {k: [] for k in labels}
        nodes = {k: [] for k in labels}
        ref = []
        for j in range(nz):
            k = ix * nz + j
            s = s0 + csr.CSR_zmesh[k]
            x = csr.CSR_xmesh[k]
            (a, b), _ = csr._layout_bounds(s, x)
            for lab, sp in variants(csr, s, x).items():
                got[lab].append(far_only(csr, s, x, sp))
                nodes[lab].append(len(sp))
            ref.append(far_only(csr, s, x, np.linspace(a, b, 40000)))
        ref = np.asarray(ref)
        n2 = np.linalg.norm(ref)
        emit(f"    {'variant':>16} {'nodes':>7} {'rel L2':>11}")
        for lab in labels:
            e = np.linalg.norm(np.asarray(got[lab]) - ref) / n2
            emit(f'    {lab:>16} {int(np.mean(nodes[lab])):>7} {e:>11.3e}')
        curves[s0] = (np.asarray([csr.CSR_zmesh[ix * nz + j] for j in range(nz)]),
                      {lab: np.asarray(got[lab]) for lab in labels}, ref)
        emit('')
        del csr

    fig, axes = plt.subplots(1, len(curves), figsize=(5.0 * len(curves), 4.0),
                             squeeze=False)
    for ax, (s0, (zz, gg, ref)) in zip(axes[0], sorted(curves.items())):
        ax.plot(zz * 1e3, ref, '-', color='k', lw=2.2, alpha=0.3, label='reference')
        for lab, st in zip(labels, ['-', '--', ':', '-']):
            ax.plot(zz * 1e3, gg[lab], st, lw=1.3, label=lab)
        ax.set_title(f's = {s0:.3f} m', fontsize=9)
        ax.set_xlabel('z [mm]')
        ax.set_ylabel('far-region dE/ds [MeV/m]')
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, 'merge_far_regions.png'), dpi=130)
    plt.close(fig)
    emit('  wrote merge_far_regions.png')

    with open(os.path.join(RESULT_DIR, 'merge_far_regions_log.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
