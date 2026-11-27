"""
The longitudinal main term integrated by parts along each column (CSR_integration.ibp) and the
shipped-versus-by-parts consistency check (ibp_check); docs/DFCSR_2D_methods.md §5.7.

Mixed into CSR.CSR2D; the methods use its state (beam, lattice, DF_tracker, integration_params).
"""
import numpy as np


class IBPMixin:
    @staticmethod
    def _ibp_chi(J, J0):
        """Partition weight of the shipped form: 1 for |J| <= J0, 0 for |J| >= 2 J0, C2 between."""
        if not J0:
            return np.zeros_like(J)
        u = np.clip((np.abs(J) - J0) / J0, 0.0, 1.0)
        return 1.0 - u * u * u * (u * (6.0 * u - 15.0) + 10.0)

    def _ibp_Q(self, ex, t, delta, taper, J0, A=None):
        """
        Q = (1 - chi(J)) G0 / J at the nodes shifted by delta in x', G0 = h' (v.n') T(R) / R.
        Pure geometry and frame (A from the field-free kernel unless given), no density.
        Returns (Q, J, chi, G0, R_x).
        """
        rx = ex['rx'] - delta * ex['nx']
        ry = ex['ry'] - delta * ex['ny']
        R = np.sqrt(rx * rx + ry * ry)
        Rx = -(rx * ex['nx'] + ry * ex['ny']) / R
        xp = ex['xp'] + delta
        if A is None:
            A = self._comoving_fields_ibp(xp, ex['sp'] - (t - R), t - R, want_fields=False)[5]
        J = 1.0 - A * Rx
        chi = self._ibp_chi(J, J0)
        G0 = (1.0 + xp * ex['kappa']) * ex['vn'] * self._radial_taper(R, taper) / R
        safe = np.where(chi < 1.0, J, 1.0)
        Q = np.where(chi < 1.0, (1.0 - chi) * G0 / safe, 0.0)
        return Q, J, chi, G0, Rx

    def _ibp_integrand(self, gz, ex, t, taper, shape):
        """
        The longitudinal integrand with its main term integrated by parts along each column.

        Node integrand  gz - (1 - chi) G0 rho_x - Q' rho - Q B R_x, and per column the boundary
        term Q rho |last node - Q rho |first node. Their column integral equals that of gz up to
        quadrature error (progress_cancellation.md §2). Q' is a central difference with step
        1e-3 of the smallest sigma_xi in the history, on geometry and frame only. Columns where
        |J| <= 2 J0 on a node with charge keep gz (see below).
        """
        J0 = self.integration_params.ibp_j0
        h = 1e-3 * float(np.min(self.DF_tracker.sigma_xi_arr))
        Q0, J, chi, G0, Rx = self._ibp_Q(ex, t, 0.0, taper, J0, A=ex['A'])
        Qp = self._ibp_Q(ex, t, +h, taper, J0)[0]
        Qm = self._ibp_Q(ex, t, -h, taper, J0)[0]
        dQ = (Qp - Qm) / (2.0 * h)
        rho = ex['rho']
        main = G0 * ex['rho_x']
        g = (gz.ravel() - (1.0 - chi) * main - dQ * rho - Q0 * ex['B'] * Rx).reshape(shape)
        Qr = (Q0 * rho).reshape(shape)
        bnd = Qr[-1] - Qr[0]
        # Column-wise choice of form. The identity holds column by column, so a column can keep
        # the shipped form without any boundary term. Where J gets small on a node that carries
        # charge, the chi partition inside the column becomes a steep ramp in x' (J changes at
        # A (1 - R_x^2)/R near the observer, A ~ the beam tilt) whose chi and 1 - chi pieces
        # cancel each other, so the by-parts column is far less converged than the shipped one.
        # Measured on the chicane, frozen at B3 (tau = -30) and D3 (tau = -49): with the
        # partition alone the by-parts wake was off by up to 30x at 3200 nodes. So a column uses
        # the by-parts form only if |J| > 2 J0 at every node with charge; chi then vanishes there
        # except on empty nodes, where it only guards the 1/J.
        #
        # J must also keep ONE SIGN over the whole column. Along a column R_x is monotonic and A
        # varies slowly, so J crosses zero at most once, but at high tilt it can do so between
        # two nodes: chicane D3 (tau = -49), a column 40 um from the observer has J from -20 to
        # +20, its 200 nodes miss the crossing (smallest sampled |J| 3.0), and the 1/J pole
        # between them made that one column 100x too large. A sign test sees it at any node count.
        live = (rho > 1e-6 * max(float(rho.max()), 1e-300)).reshape(shape)
        Jr = J.reshape(shape)
        Jc = np.where(live, np.abs(Jr), np.inf).min(axis=0)
        one_sign = np.all(Jr > 0.0, axis=0) | np.all(Jr < 0.0, axis=0)
        col = (Jc > 2.0 * J0) & one_sign
        g = np.where(col[None, :], g, gz)
        bnd = np.where(col, bnd, 0.0)
        # |main term| per node, and the part of it in columns kept in the shipped form, for
        # the consistency check (the share where the cancellation is still being summed)
        am = np.abs(main).reshape(shape)
        zone = np.broadcast_to(~col[None, :], shape)
        return g, bnd, am, zone

    def _ibp_consistency_check(self):
        """
        Both forms at a few points of the mesh's centre row, at ibp_check_xbins x' nodes.
        See Integration_params.ibp_check. Appends (s, relative L2 difference, points used,
        points skipped, largest zone share) to ibp_check_log.

        Also recorded per kick: the largest share of the main term's sum|node| in columns that
        keep the shipped form (|J| <= 2 J0 on a node with charge). There the two forms coincide,
        so a share near 1 means the check sees little. Points above ibp_check_zone_max are
        skipped (default 1: none).
        """
        ip = self.integration_params
        n = ip.ibp_check
        if n <= 0:
            return
        nz = self.CSR_params.zbins
        ix = self.CSR_params.xbins // 2
        js = np.unique(np.round(np.linspace(0, nz - 1, n + 2)[1:-1]).astype(int))
        saved = (ip.ibp, ip.xbins)
        ip.ibp, ip.xbins = True, ip.ibp_check_xbins
        a, b, fz = [], [], []
        skipped = 0
        try:
            for j in js:
                k = ix * nz + j
                dE, _ = self.get_CSR_wake(self.beam.position + self.CSR_zmesh[k], self.CSR_xmesh[k])
                f = self._ibp_zone_acc / max(self._ibp_gross_acc, 1e-300)
                fz.append(f)
                if f > ip.ibp_check_zone_max:
                    skipped += 1
                    continue
                a.append(self._last_wake_shipped)
                b.append(dE)
        finally:
            ip.ibp, ip.xbins = saved
        a, b = np.array(a), np.array(b)
        d = (float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)) if len(a)
             else float('nan'))
        # The warning is judged against the mesh's largest wake, not against the checked points
        # themselves: the centre-row points can sit where W is ~0 (the gain beam's first bend,
        # W 0.002 MeV/m against a mesh max of 0.02), and there a 4% relative difference was
        # 1e-4 MeV/m, which the beam never sees. d (relative to the points) stays in the log.
        wmax = float(np.abs(self.dE_dct).max()) if getattr(self, 'dE_dct', None) is not None else 0.0
        d_mesh = float(np.abs(a - b).max() / max(wmax, 1e-300)) if len(a) else float('nan')
        if not hasattr(self, 'ibp_check_log'):
            self.ibp_check_log = []
        self.ibp_check_log.append((float(self.beam.position), d, len(a), skipped, max(fz), d_mesh))
        if len(a) and d_mesh > ip.ibp_check_tol:
            self._note('ibp_check', f'  [CSR2D] warning: at s = {self.beam.position:.4f} the shipped and the '
                  f'integrated-by-parts longitudinal wakes differ by {d_mesh:.1e} of the mesh\'s '
                  f'largest |W| ({d:.1e} relative to the {len(a)} checked points). They are the '
                  f'same integral: rho and d rho/dx are inconsistent somewhere (grid edge, '
                  f'blending). Further kicks are logged in ibp_check_log, not printed.')
