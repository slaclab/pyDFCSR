"""
Diagnostic: compare CSR integrand components at a specific (x,z) point
between legacy and new methods at a step where wakes diverge.

Uses chirp=500, step 6 (slope ~ -10, dE/dct differs by 0.1166 vs exact).
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'example'))

from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.deposit_smooth import DF_tracker_smooth

dep_config = {
    'method': 'bspline_fft',
    'xbins': 200, 'zbins': 200,
    'xlim': 5, 'zlim': 5,
    'smoothing_sigma': 3.0,
    'poly_degree': 3,
    'velocity_threhold': 1000,
}

TARGET_STEP = 7  # step where dE/dct starts diverging significantly


def run_to_step(method):
    """Run until target step, then return the CSR object ready for integrand inspection."""
    csr = CSR2D(input_file='input/dipole_chirp500_config.yaml')

    if method == 'bspline_fft':
        csr.DF_tracker = DF_tracker_smooth(dep_config)
        csr.use_smooth_deposit = True
        csr.DF_tracker.get_DF(x=csr.beam.x, z=csr.beam.z, px=csr.beam.px, t=csr.beam.position)
        csr.DF_tracker.append_DF()
        csr.DF_tracker.append_interpolant(formation_length=float('inf'),
                                           n_formation_length=csr.integration_params.n_formation_length)

    csr.run(stop_time=(TARGET_STEP + 0.5) * 0.1)
    return csr


def compare_integrand_at_point(csr_leg, csr_new, ix, iz):
    """Compare the integrand at a specific CSR mesh point between legacy and new."""
    # Get the observation point
    xbins = csr_leg.CSR_params.xbins
    zbins = csr_leg.CSR_params.zbins
    k = ix * zbins + iz

    s_leg = csr_leg.beam.position + csr_leg.CSR_zmesh[k]
    x_leg = csr_leg.CSR_xmesh[k]
    s_new = csr_new.beam.position + csr_new.CSR_zmesh[k]
    x_new = csr_new.CSR_xmesh[k]

    print(f"\nObservation point (ix={ix}, iz={iz}):")
    print(f"  Legacy: s={s_leg:.6f}, x={x_leg*1e6:.2f} um")
    print(f"  New:    s={s_new:.6f}, x={x_new*1e6:.2f} um")

    # Call get_CSR_wake in debug mode to get the integrands
    result_leg = csr_leg.get_CSR_wake(s_leg, x_leg, debug=True)
    result_new = csr_new.get_CSR_wake(s_new, x_new, debug=True)

    # Handle chirp_band (15 values) vs no-chirp (11 values) mode
    if len(result_leg) == 15:
        # chirp_band mode: xp1, xp2, xp3, xp4, sp1, sp2, sp3, iz1, ix1, iz2, ix2, iz3, ix3, iz4, ix4
        xp1_leg, xp2_leg, xp3_leg, xp4_leg, sp1_leg, sp2_leg, sp3_leg = result_leg[:7]
        iz1_leg, ix1_leg, iz2_leg, ix2_leg, iz3_leg, ix3_leg, iz4_leg, ix4_leg = result_leg[7:]
        xp1_new, xp2_new, xp3_new, xp4_new, sp1_new, sp2_new, sp3_new = result_new[:7]
        iz1_new, ix1_new, iz2_new, ix2_new, iz3_new, ix3_new, iz4_new, ix4_new = result_new[7:]

        print(f"\n  Mode: chirp_band (|tan_theta| > 1)")
        print(f"  Integration regions:")
        print(f"  Legacy: sp1=[{sp1_leg[0]:.4f},{sp1_leg[-1]:.4f}], sp2=[{sp2_leg[0]:.4f},{sp2_leg[-1]:.4f}], sp3=[{sp3_leg[0]:.4f},{sp3_leg[-1]:.4f}]")
        print(f"  New:    sp1=[{sp1_new[0]:.4f},{sp1_new[-1]:.4f}], sp2=[{sp2_new[0]:.4f},{sp2_new[-1]:.4f}], sp3=[{sp3_new[0]:.4f},{sp3_new[-1]:.4f}]")
        print(f"  Legacy: xp4=[{xp4_leg[0]*1e6:.1f},{xp4_leg[-1]*1e6:.1f}], xp3=[{xp3_leg[0]*1e6:.1f},{xp3_leg[-1]*1e6:.1f}], xp1=[{xp1_leg[0]*1e6:.1f},{xp1_leg[-1]*1e6:.1f}]um")
        print(f"  New:    xp4=[{xp4_new[0]*1e6:.1f},{xp4_new[-1]*1e6:.1f}], xp3=[{xp3_new[0]*1e6:.1f},{xp3_new[-1]*1e6:.1f}], xp1=[{xp1_new[0]*1e6:.1f},{xp1_new[-1]*1e6:.1f}]um")

        CSR_scaling = csr_leg.CSR_scaling

        dE1_leg = -CSR_scaling * np.trapz(y=np.trapz(y=iz1_leg, x=xp4_leg, axis=0), x=sp1_leg)
        dE1_new = -CSR_scaling * np.trapz(y=np.trapz(y=iz1_new, x=xp4_new, axis=0), x=sp1_new)
        dE2_leg = -CSR_scaling * np.trapz(y=np.trapz(y=iz2_leg, x=xp3_leg, axis=0), x=sp2_leg)
        dE2_new = -CSR_scaling * np.trapz(y=np.trapz(y=iz2_new, x=xp3_new, axis=0), x=sp2_new)
        dE3_leg = -CSR_scaling * np.trapz(y=np.trapz(y=iz3_leg, x=xp1_leg, axis=0), x=sp3_leg)
        dE3_new = -CSR_scaling * np.trapz(y=np.trapz(y=iz3_new, x=xp1_new, axis=0), x=sp3_new)
        dE4_leg = -CSR_scaling * np.trapz(y=np.trapz(y=iz4_leg, x=xp2_leg, axis=0), x=sp3_leg)
        dE4_new = -CSR_scaling * np.trapz(y=np.trapz(y=iz4_new, x=xp2_new, axis=0), x=sp3_new)

        print(f"\n  Integrated dE/dct (MeV/m) per region:")
        print(f"  {'Region':<10} | {'Legacy':<15} | {'New':<15} | {'Ratio':<10}")
        print(f"  {'-'*55}")
        for name, leg_val, new_val in [("R1(sp1)", dE1_leg, dE1_new), ("R2(sp2)", dE2_leg, dE2_new),
                                        ("R3(sp3a)", dE3_leg, dE3_new), ("R4(sp3b)", dE4_leg, dE4_new)]:
            ratio = new_val / leg_val if abs(leg_val) > 1e-30 else float('inf')
            print(f"  {name:<10} | {leg_val:<15.6e} | {new_val:<15.6e} | {ratio:<10.4f}")
        total_leg = dE1_leg + dE2_leg + dE3_leg + dE4_leg
        total_new = dE1_new + dE2_new + dE3_new + dE4_new
        print(f"  {'Total':<10} | {total_leg:<15.6e} | {total_new:<15.6e} | {total_new/total_leg if abs(total_leg)>1e-30 else float('inf'):<10.4f}")

        # Integrand stats
        for name, iz_l, iz_n in [("Region 1 (sp1,xp4)", iz1_leg, iz1_new),
                                   ("Region 2 (sp2,xp3)", iz2_leg, iz2_new)]:
            print(f"\n  Integrand_z {name}:")
            print(f"    Legacy: max={iz_l.max():.4e}, min={iz_l.min():.4e}, L2={np.linalg.norm(iz_l):.4e}")
            print(f"    New:    max={iz_n.max():.4e}, min={iz_n.min():.4e}, L2={np.linalg.norm(iz_n):.4e}")
            if iz_l.shape == iz_n.shape:
                diff = iz_n - iz_l
                rel = np.linalg.norm(diff) / max(np.linalg.norm(iz_l), 1e-30)
                print(f"    Rel L2 diff: {rel:.4f}")

    else:
        # no-chirp mode: xp_w, xp_n, sp1, sp2, sp3, iz1, ix1, iz2, ix2, iz3, ix3
        xp_w_leg, xp_n_leg, sp1_leg, sp2_leg, sp3_leg = result_leg[:5]
        iz1_leg, ix1_leg, iz2_leg, ix2_leg, iz3_leg, ix3_leg = result_leg[5:]
        xp_w_new, xp_n_new, sp1_new, sp2_new, sp3_new = result_new[:5]
        iz1_new, ix1_new, iz2_new, ix2_new, iz3_new, ix3_new = result_new[5:]

        print(f"\n  Mode: no chirp_band (|tan_theta| <= 1)")
        print(f"  Integration regions:")
        print(f"  Legacy: sp1=[{sp1_leg[0]:.4f},{sp1_leg[-1]:.4f}], sp2=[{sp2_leg[0]:.4f},{sp2_leg[-1]:.4f}], sp3=[{sp3_leg[0]:.4f},{sp3_leg[-1]:.4f}]")
        print(f"  New:    sp1=[{sp1_new[0]:.4f},{sp1_new[-1]:.4f}], sp2=[{sp2_new[0]:.4f},{sp2_new[-1]:.4f}], sp3=[{sp3_new[0]:.4f},{sp3_new[-1]:.4f}]")

        CSR_scaling = csr_leg.CSR_scaling
        dE1_leg = -CSR_scaling * np.trapz(y=np.trapz(y=iz1_leg, x=xp_w_leg, axis=0), x=sp1_leg)
        dE1_new = -CSR_scaling * np.trapz(y=np.trapz(y=iz1_new, x=xp_w_new, axis=0), x=sp1_new)
        dE2_leg = -CSR_scaling * np.trapz(y=np.trapz(y=iz2_leg, x=xp_n_leg, axis=0), x=sp2_leg)
        dE2_new = -CSR_scaling * np.trapz(y=np.trapz(y=iz2_new, x=xp_n_new, axis=0), x=sp2_new)
        dE3_leg = -CSR_scaling * np.trapz(y=np.trapz(y=iz3_leg, x=xp_n_leg, axis=0), x=sp3_leg)
        dE3_new = -CSR_scaling * np.trapz(y=np.trapz(y=iz3_new, x=xp_n_new, axis=0), x=sp3_new)

        print(f"\n  Integrated dE/dct (MeV/m) per region:")
        print(f"  {'Region':<10} | {'Legacy':<15} | {'New':<15} | {'Ratio':<10}")
        print(f"  {'-'*55}")
        for name, leg_val, new_val in [("R1(sp1)", dE1_leg, dE1_new), ("R2(sp2)", dE2_leg, dE2_new),
                                        ("R3(sp3)", dE3_leg, dE3_new)]:
            ratio = new_val / leg_val if abs(leg_val) > 1e-30 else float('inf')
            print(f"  {name:<10} | {leg_val:<15.6e} | {new_val:<15.6e} | {ratio:<10.4f}")
        total_leg = dE1_leg + dE2_leg + dE3_leg
        total_new = dE1_new + dE2_new + dE3_new
        print(f"  {'Total':<10} | {total_leg:<15.6e} | {total_new:<15.6e} | {total_new/total_leg if abs(total_leg)>1e-30 else float('inf'):<10.4f}")

        for name, iz_l, iz_n in [("Region 1 (sp1)", iz1_leg, iz1_new),
                                   ("Region 2 (sp2)", iz2_leg, iz2_new)]:
            print(f"\n  Integrand_z {name}:")
            print(f"    Legacy: max={iz_l.max():.4e}, min={iz_l.min():.4e}, L2={np.linalg.norm(iz_l):.4e}")
            print(f"    New:    max={iz_n.max():.4e}, min={iz_n.min():.4e}, L2={np.linalg.norm(iz_n):.4e}")
            if iz_l.shape == iz_n.shape:
                diff = iz_n - iz_l
                rel = np.linalg.norm(diff) / max(np.linalg.norm(iz_l), 1e-30)
                print(f"    Rel L2 diff: {rel:.4f}")

    return result_leg, result_new


def main():
    print("="*70)
    print(f"Integrand Diagnostic: chirp=500, step={TARGET_STEP}")
    print("="*70)

    print("\nRunning legacy...")
    csr_leg = run_to_step('legacy')
    print("\nRunning new...")
    csr_new = run_to_step('bspline_fft')

    # Check beam state is same
    print(f"\nBeam state at step {TARGET_STEP}:")
    print(f"  Legacy: position={csr_leg.beam.position:.4f}, slope={csr_leg.beam._slope}")
    print(f"  New:    position={csr_new.beam.position:.4f}, slope={csr_new.beam._slope}")
    print(f"  Legacy: sigma_x={csr_leg.beam._sigma_x*1e6:.2f}um, sigma_z={csr_leg.beam._sigma_z*1e6:.2f}um")
    print(f"  New:    sigma_x={csr_new.beam._sigma_x*1e6:.2f}um, sigma_z={csr_new.beam._sigma_z*1e6:.2f}um")

    # Compare at center point and a few others
    xbins = csr_leg.CSR_params.xbins
    zbins = csr_leg.CSR_params.zbins
    print(f"\nCSR mesh: {xbins} x {zbins}")

    # Center point
    compare_integrand_at_point(csr_leg, csr_new, xbins // 2, zbins // 2)

    # Off-center point
    compare_integrand_at_point(csr_leg, csr_new, xbins // 4, zbins // 2)


if __name__ == '__main__':
    main()
