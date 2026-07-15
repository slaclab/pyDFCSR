"""Compare CSR integrand at step 17, z=0, x=4930um for the tilted beam case."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'example'))

from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.deposit_smooth import DF_tracker_smooth
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dep_config = {'method': 'bspline_fft', 'xbins': 100, 'zbins': 100,
              'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
              'poly_degree': 3, 'velocity_threhold': 1000}

# Run legacy to step 17
csr_leg = CSR2D(input_file='input/single_dipole_tilted_config.yaml')
csr_leg.CSR_params.apply_CSR = 0
csr_leg.run(stop_time=0.86)

# Run new to step 17
csr_new = CSR2D(input_file='input/single_dipole_tilted_config.yaml')
csr_new.CSR_params.apply_CSR = 0
csr_new.DF_tracker = DF_tracker_smooth(dep_config)
csr_new.use_smooth_deposit = True
csr_new.DF_tracker.get_DF(x=csr_new.beam.x, z=csr_new.beam.z, px=csr_new.beam.px, t=csr_new.beam.position)
csr_new.DF_tracker.append_DF()
csr_new.DF_tracker.append_interpolant(formation_length=float('inf'),
                                       n_formation_length=csr_new.integration_params.n_formation_length)
csr_new.run(stop_time=0.86)

s = csr_leg.beam.position
x_query = 4930e-6
print(f'Beam position: {s:.4f}')
print(f'Query: s={s:.4f}, x={x_query*1e6:.1f}um')
print(f'Beam slope: {np.polyfit(csr_leg.beam.z, csr_leg.beam.x, 1)[0]:.4f}')

# Get integrand in debug mode
r_leg = csr_leg.get_CSR_wake(s, x_query, debug=True)
r_new = csr_new.get_CSR_wake(s, x_query, debug=True)

print(f'\nDebug output length: {len(r_leg)}')

# chirp_band mode returns 15 elements:
# xp1, xp2, xp3, xp4, sp1, sp2, sp3, iz1, ix1, iz2, ix2, iz3, ix3, iz4, ix4
out_dir = '../test/benchmark_results/single_dipole_tilted'

if len(r_leg) == 15:
    xp1_leg, xp2_leg, xp3_leg, xp4_leg = r_leg[0], r_leg[1], r_leg[2], r_leg[3]
    sp1_leg, sp2_leg, sp3_leg = r_leg[4], r_leg[5], r_leg[6]
    iz1_leg, ix1_leg = r_leg[7], r_leg[8]
    iz2_leg, ix2_leg = r_leg[9], r_leg[10]
    iz3_leg, ix3_leg = r_leg[11], r_leg[12]
    iz4_leg, ix4_leg = r_leg[13], r_leg[14]

    xp1_new, xp2_new, xp3_new, xp4_new = r_new[0], r_new[1], r_new[2], r_new[3]
    sp1_new, sp2_new, sp3_new = r_new[4], r_new[5], r_new[6]
    iz1_new, ix1_new = r_new[7], r_new[8]
    iz2_new, ix2_new = r_new[9], r_new[10]
    iz3_new, ix3_new = r_new[11], r_new[12]
    iz4_new, ix4_new = r_new[13], r_new[14]

    print(f'Chirp band mode (4 regions):')
    print(f'  Region 1 (far): xp4=[{xp4_leg[0]*1e6:.0f},{xp4_leg[-1]*1e6:.0f}]um, sp1=[{(sp1_leg[0]-s)*1e6:.0f},{(sp1_leg[-1]-s)*1e6:.0f}]um, shape={iz1_leg.shape}')
    print(f'  Region 2 (mid): xp3=[{xp3_leg[0]*1e6:.0f},{xp3_leg[-1]*1e6:.0f}]um, sp2=[{(sp2_leg[0]-s)*1e6:.0f},{(sp2_leg[-1]-s)*1e6:.0f}]um, shape={iz2_leg.shape}')
    print(f'  Region 3 (chirp band 1): xp1=[{xp1_leg[0]*1e6:.0f},{xp1_leg[-1]*1e6:.0f}]um, sp3=[{(sp3_leg[0]-s)*1e6:.0f},{(sp3_leg[-1]-s)*1e6:.0f}]um, shape={iz3_leg.shape}')
    print(f'  Region 4 (chirp band 2): xp2=[{xp2_leg[0]*1e6:.0f},{xp2_leg[-1]*1e6:.0f}]um, sp3=[{(sp3_leg[0]-s)*1e6:.0f},{(sp3_leg[-1]-s)*1e6:.0f}]um, shape={iz4_leg.shape}')

    # Plot all 4 regions
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle(f'CSR integrand_z (chirp_band mode), z=0, x={x_query*1e6:.0f}um', fontsize=14)

    regions = [
        (iz1_leg, iz1_new, xp4_leg, sp1_leg, 'Region 1 (far history)'),
        (iz2_leg, iz2_new, xp3_leg, sp2_leg, 'Region 2 (mid)'),
        (iz3_leg, iz3_new, xp1_leg, sp3_leg, 'Region 3 (chirp band 1)'),
        (iz4_leg, iz4_new, xp2_leg, sp3_leg, 'Region 4 (chirp band 2)'),
    ]

    for row, (iz_l, iz_n, xp, sp, label) in enumerate(regions):
        sp_um = (sp - s) * 1e6
        xp_um = xp * 1e6
        vmax = max(abs(iz_l).max(), abs(iz_n).max(), 1e-30)

        im = axes[row, 0].imshow(iz_l.T, origin='lower', extent=[xp_um[0], xp_um[-1], sp_um[0], sp_um[-1]],
                                  aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[row, 0].set_title(f'{label} legacy'); plt.colorbar(im, ax=axes[row, 0])

        im = axes[row, 1].imshow(iz_n.T, origin='lower', extent=[xp_um[0], xp_um[-1], sp_um[0], sp_um[-1]],
                                  aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[row, 1].set_title(f'{label} new'); plt.colorbar(im, ax=axes[row, 1])

        diff = iz_n - iz_l
        im = axes[row, 2].imshow(diff.T, origin='lower', extent=[xp_um[0], xp_um[-1], sp_um[0], sp_um[-1]],
                                  aspect='auto', cmap='RdBu_r')
        axes[row, 2].set_title(f'{label} diff'); plt.colorbar(im, ax=axes[row, 2])

    for ax in axes[:, 0]:
        ax.set_ylabel("s' - s (um)")
    for ax in axes[3, :]:
        ax.set_xlabel("x' (um)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'integrand_z_step17_z0_4regions.png'), dpi=150)
    plt.close()

    # Print integrated values
    print(f'\nIntegrated dE/dct per region:')
    dE1_leg = np.trapz(y=np.trapz(y=iz1_leg, x=xp4_leg, axis=0), x=sp1_leg)
    dE1_new = np.trapz(y=np.trapz(y=iz1_new, x=xp4_new, axis=0), x=sp1_new)
    dE2_leg = np.trapz(y=np.trapz(y=iz2_leg, x=xp3_leg, axis=0), x=sp2_leg)
    dE2_new = np.trapz(y=np.trapz(y=iz2_new, x=xp3_new, axis=0), x=sp2_new)
    dE3_leg = np.trapz(y=np.trapz(y=iz3_leg, x=xp1_leg, axis=0), x=sp3_leg)
    dE3_new = np.trapz(y=np.trapz(y=iz3_new, x=xp1_new, axis=0), x=sp3_new)
    dE4_leg = np.trapz(y=np.trapz(y=iz4_leg, x=xp2_leg, axis=0), x=sp3_leg)
    dE4_new = np.trapz(y=np.trapz(y=iz4_new, x=xp2_new, axis=0), x=sp3_new)
    total_leg = dE1_leg + dE2_leg + dE3_leg + dE4_leg
    total_new = dE1_new + dE2_new + dE3_new + dE4_new

    print(f'  Region 1: leg={dE1_leg:.6e}, new={dE1_new:.6e}, rel_diff={abs(dE1_new-dE1_leg)/max(abs(dE1_leg),1e-30):.4f}')
    print(f'  Region 2: leg={dE2_leg:.6e}, new={dE2_new:.6e}, rel_diff={abs(dE2_new-dE2_leg)/max(abs(dE2_leg),1e-30):.4f}')
    print(f'  Region 3: leg={dE3_leg:.6e}, new={dE3_new:.6e}, rel_diff={abs(dE3_new-dE3_leg)/max(abs(dE3_leg),1e-30):.4f}')
    print(f'  Region 4: leg={dE4_leg:.6e}, new={dE4_new:.6e}, rel_diff={abs(dE4_new-dE4_leg)/max(abs(dE4_leg),1e-30):.4f}')
    print(f'  Total:    leg={total_leg:.6e}, new={total_new:.6e}, rel_diff={abs(total_new-total_leg)/max(abs(total_leg),1e-30):.4f}')

    # Cancellation
    sum_abs = np.abs(iz1_leg).sum() + np.abs(iz2_leg).sum() + np.abs(iz3_leg).sum() + np.abs(iz4_leg).sum()
    net = abs(iz1_leg.sum() + iz2_leg.sum() + iz3_leg.sum() + iz4_leg.sum())
    print(f'\n  Cancellation (legacy): |net|/sum|f| = {net/sum_abs:.6f}')

else:
    print('Non-chirp-band mode — not expected for slope=-8')

print(f'\nPlot: {out_dir}/integrand_z_step17_z0_4regions.png')

# Integrated values per region
print(f'\nIntegrated dE/dct per region:')
dE1_leg = np.trapz(y=np.trapz(y=iz1_leg, x=xp_w_leg, axis=0), x=sp1_leg)
dE1_new = np.trapz(y=np.trapz(y=iz1_new, x=xp_w_new, axis=0), x=sp1_new)
dE2_leg = np.trapz(y=np.trapz(y=iz2_leg, x=xp_n_leg, axis=0), x=sp2_leg)
dE2_new = np.trapz(y=np.trapz(y=iz2_new, x=xp_n_new, axis=0), x=sp2_new)
dE3_leg = np.trapz(y=np.trapz(y=iz3_leg, x=xp_n_leg, axis=0), x=sp3_leg)
dE3_new = np.trapz(y=np.trapz(y=iz3_new, x=xp_n_new, axis=0), x=sp3_new)
total_leg = dE1_leg + dE2_leg + dE3_leg
total_new = dE1_new + dE2_new + dE3_new

print(f'  Region 1: leg={dE1_leg:.6e}, new={dE1_new:.6e}, rel_diff={abs(dE1_new-dE1_leg)/max(abs(dE1_leg),1e-30):.4f}')
print(f'  Region 2: leg={dE2_leg:.6e}, new={dE2_new:.6e}, rel_diff={abs(dE2_new-dE2_leg)/max(abs(dE2_leg),1e-30):.4f}')
print(f'  Region 3: leg={dE3_leg:.6e}, new={dE3_new:.6e}, rel_diff={abs(dE3_new-dE3_leg)/max(abs(dE3_leg),1e-30):.4f}')
print(f'  Total:    leg={total_leg:.6e}, new={total_new:.6e}, rel_diff={abs(total_new-total_leg)/max(abs(total_leg),1e-30):.4f}')

sum_abs_leg = np.abs(iz1_leg).sum() + np.abs(iz2_leg).sum() + np.abs(iz3_leg).sum()
net_leg = abs(iz1_leg.sum() + iz2_leg.sum() + iz3_leg.sum())
print(f'\n  Cancellation (legacy): |net|/sum|f| = {net_leg/sum_abs_leg:.6f}')
print(f'\nPlot: {out_dir}/integrand_z_step17_z0.png')
