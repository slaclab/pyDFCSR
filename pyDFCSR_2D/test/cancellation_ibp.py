"""
Prototype: the longitudinal W1 term integrated by parts along each s' column.

Diagnostic only, nothing here is called by the shipped code. See progress_cancellation.md §2-§3.

The shipped longitudinal integrand is (get_CSR_integrand)

    g_z = h' K.grad'rho / R + (div v term) / R,     K = v - (v.v')v'

and K.grad'rho = (K.n') rho_x + (K.tau') rho_z / h'. Its main piece is

    G0 rho_x,      G0 = h' (v.n') T(R) / R          (T = near-patch taper, 1 if none)

G0 is pure geometry, so it has no noise, while rho_x has + and - lobes on the two sides of the
beam. The rest, -h'(v.v')vx' rho_x/R + (K.tau') rho_z/R + div v term, is smaller by |vx'| ~ 1e-4
or is not a derivative lobe, and is left as it is.

Along one column (fixed s') the retarded point moves with x' as
    z_ret = s' - t + R(x'),   t_ret = t - R(x'),   so  dz_ret/dx' = R_x',  dt_ret/dx' = -R_x'
with R_x' = dR/dx' = -n'.(r - r')/R. With the beam frame's ridge c(z, t) (centre line, tilt
and curvature included) and xi = x' - c:

    d rho_ret / dx' = J rho_x + D,     J = 1 - dc/dx'|column,
    D = the change of rho along the column at FIXED xi (the slice-to-slice change, no lobes)

so  rho_x = (d rho_ret/dx' - D) / J, and per column

    int G0 rho_x dx' = [G0 rho_ret / J]_lo^hi - int d(G0/J)/dx' rho_ret dx' - int (G0/J) D dx'.

The boundary term is ~0, because the bands are +-10 sigma_xi wide. d(G0/J)/dx' is geometry and
the frame only, so it is noise-free. D is a finite difference of the interpolant at fixed xi.
J and D are central finite differences with a step of 1e-3 sigma_xi along the column path. The
interpolant is a C2 cubic B-spline, so that is O(eps^2) accurate.

J -> 0 where a column runs along a tilted beam's ridge (the Eq 4.24 tangent case, possible
once the beam tilt |tau| exceeds 1). There the form is singular. A smooth partition of unity
in J handles it:

    G0 rho_x = chi(J) G0 rho_x  +  (1 - chi(J)) G0 rho_x,
    chi = 1 for |J| <= J0, 0 for |J| >= 2 J0, C2 smootherstep in between.

The chi part stays in the shipped form, and the rest is integrated by parts with
Q = (1 - chi) G0 / J, which is bounded. chi depends on x' only through J(x'), which is smooth,
so the finite-difference dQ/dx' covers it.
"""
import numpy as np

from pyDFCSR_2D.interp1D import interpolate1D

import cancellation_lib as L


def _lat(csr, sv):
    lat = csr.lattice
    sv = np.atleast_1d(np.asarray(sv, float))
    f = lambda d: interpolate1D(xval=sv, data=d, min_x=lat.min_x, delta_x=lat.delta_x)
    return dict(X=f(lat.coords[:, 0]), Y=f(lat.coords[:, 1]),
                nx=f(lat.n_vec[:, 0]), ny=f(lat.n_vec[:, 1]),
                tx=f(lat.tau_vec[:, 0]), ty=f(lat.tau_vec[:, 1]))


def _kappa(csr, sp):
    lat = csr.lattice
    k = np.zeros(sp.shape)
    for c in range(lat.Nelement):
        if c == 0:
            k[sp < lat.distance[c]] = lat.rho[c]
        else:
            k[(sp < lat.distance[c]) & (sp >= lat.distance[c - 1])] = lat.rho[c]
    return k


def _ridge(csr, z, t):
    poly, xi_bar, _, _, _ = csr._comoving_frame_at(t)
    return csr._eval_poly_rows(poly, z) + xi_bar + csr._curv_shift_at(t, z)


class Column:
    """Geometry and frame quantities at nodes (xp, sp), all noise-free except where noted."""

    def __init__(self, csr, s, x):
        self.csr, self.s, self.x, self.t = csr, s, x, csr.beam.position
        o = _lat(csr, s)
        self.o = {k: v[0] for k, v in o.items()}
        vx = csr._comoving_fields(np.array([x]), np.array([s - self.t]),
                                  np.array([self.t]))[3][0]
        self.vx_obs = float(vx)

    def geom(self, xp, sp):
        o, p = self.o, _lat(self.csr, sp)
        rx = o['X'] - p['X'] + self.x * o['nx'] - xp * p['nx']
        ry = o['Y'] - p['Y'] + self.x * o['ny'] - xp * p['ny']
        R = np.sqrt(rx * rx + ry * ry)
        Rx = -(rx * p['nx'] + ry * p['ny']) / R
        vX = o['tx'] + self.vx_obs * o['nx']
        vY = o['ty'] + self.vx_obs * o['ny']
        v_n = vX * p['nx'] + vY * p['ny']
        h = 1.0 + xp * _kappa(self.csr, sp)
        return R, Rx, h, v_n

    def G0(self, xp, sp, taper):
        R, Rx, h, v_n = self.geom(xp, sp)
        T = 1.0
        if taper is not None:
            R1, R2, keep_inner = taper
            u = np.clip((R - R1) / (R2 - R1), 0.0, 1.0)
            T = u * u * u * (u * (6.0 * u - 15.0) + 10.0)
            if keep_inner:
                T = 1.0 - T
        return h * v_n * T / R, R, Rx

    def J_D(self, xp, sp, eps_x, want_D=True):
        """J = 1 - dc/dx' along the column; D = d rho/dx' along the column at fixed xi."""
        R, Rx, _, _ = self.geom(xp, sp)
        z0, t0 = sp - self.t + R, self.t - R
        c0 = _ridge(self.csr, z0, t0)
        cs, rs = [], []
        for sgn in (+1.0, -1.0):
            z1, t1 = z0 + sgn * eps_x * Rx, t0 - sgn * eps_x * Rx
            c1 = _ridge(self.csr, z1, t1)
            cs.append(c1)
            if want_D:
                rs.append(self.csr._comoving_fields(xp + (c1 - c0), z1, t1)[0])
        J = 1.0 - (cs[0] - cs[1]) / (2.0 * eps_x)
        D = (rs[0] - rs[1]) / (2.0 * eps_x) if want_D else None
        return J, D

    def Q(self, xp, sp, taper, eps_x, J0):
        g0, _, _ = self.G0(xp, sp, taper)
        J, _ = self.J_D(xp, sp, eps_x, want_D=False)
        return (1.0 - chi(J, J0)) * g0 / J


def chi(J, J0):
    """1 for |J| <= J0, 0 for |J| >= 2 J0, C2 smootherstep between; J0 = 0 gives 0."""
    if not J0:
        return np.zeros_like(J)
    u = np.clip((np.abs(J) - J0) / J0, 0.0, 1.0)
    return 1.0 - u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def ibp_pieces(csr, s, x, regions=('far', 'near', 'prev'), J0=0.25):
    """
    Per node contributions to dE (MeV/m), shipped form AND integrated-by-parts form, on the
    shipped nodes. Regions not in `regions` and the polar patch keep the shipped form in both.
    Returns dict with the two node arrays per piece, the boundary terms, and min|J|.
    """
    t = csr.beam.position
    col = Column(csr, s, x)
    sxi = float(csr.DF_tracker.sigma_xi_arr.min())
    eps = 1e-3 * sxi
    regs, radii = L.region_layout(csr, s, x)
    out = []
    jmin = np.inf
    for reg in regs:
        for b in L.band_meshes(csr, s, x, reg['sp'], reg['scale_bands']):
            xp, sp, w = b['xp'], b['sp'], b['w']
            gz, _ = csr.get_CSR_integrand(s=s, t=t, x=x, xp=xp, sp=sp, taper=reg['taper'])
            ship = -csr.CSR_scaling * w * gz
            if reg['name'] not in regions:
                out.append(dict(region=reg['name'], ship=ship, ibp=ship, bnd=0.0))
                continue
            f = [a.ravel() for a in (xp, sp)]
            rho, rho_x, _, _, _ = csr._comoving_fields(f[0], f[1] - (t - col.geom(*f)[0]),
                                                       t - col.geom(*f)[0])
            g0, _, _ = col.G0(f[0], f[1], reg['taper'])
            J, D = col.J_D(f[0], f[1], eps)
            dQ = (col.Q(f[0] + eps, f[1], reg['taper'], eps, J0)
                  - col.Q(f[0] - eps, f[1], reg['taper'], eps, J0)) / (2.0 * eps)
            ch = chi(J, J0)
            shp = xp.shape
            # the part left in shipped form: the small terms, and the chi share of the main one
            rest = gz - ((1.0 - ch) * g0 * rho_x).reshape(shp)
            main_ibp = (-(dQ * rho) - ((1.0 - ch) * g0 / J) * D).reshape(shp)
            # boundary [Q rho] at both ends of every live column, times the s' weight
            ws = L.trapz_weights(sp[0])
            Qr = ((1.0 - ch) * g0 / J * rho).reshape(shp)
            bnd_col = -csr.CSR_scaling * np.where(b['alive'], (Qr[-1] - Qr[0]) * ws, 0.0)
            bnd = float(np.sum(bnd_col))
            out.append(dict(region=reg['name'], ship=ship,
                            ibp=-csr.CSR_scaling * w * (rest + main_ibp), bnd=bnd,
                            main_ship=-csr.CSR_scaling * w * (g0 * rho_x).reshape(shp),
                            main_ibp=-csr.CSR_scaling * w * main_ibp, xp=xp, sp=sp,
                            J=J.reshape(shp), rho=rho.reshape(shp), bnd_col=bnd_col))
            live = b['alive'][None, :] & (rho.reshape(shp) > 1e-6 * rho.max())
            if live.any():
                jmin = min(jmin, float(np.abs(J.reshape(shp))[live].min()))
    if radii is not None:
        p = L.patch_mesh(csr, s, x, radii)
        gz, _ = csr.get_CSR_integrand(s=s, t=t, x=x, xp=p['xp'], sp=p['sp'], taper=p['taper'])
        ship = -csr.CSR_scaling * p['w'] * gz
        out.append(dict(region='patch', ship=ship, ibp=ship, bnd=0.0))
    return out, jmin


def totals(ps):
    ship = sum(float(p['ship'].sum()) for p in ps)
    ibp = sum(float(p['ibp'].sum()) + p['bnd'] for p in ps)
    C_ship = L.cancel(np.concatenate([p['ship'].ravel() for p in ps]))[0]
    C_ibp = L.cancel(np.concatenate([p['ibp'].ravel() for p in ps]
                                    + [np.atleast_1d(p['bnd']) for p in ps]))[0]
    bnd = sum(p['bnd'] for p in ps)
    return dict(ship=ship, ibp=ibp, C_ship=C_ship, C_ibp=C_ibp, bnd=bnd)


def wake_ibp(csr, s, x, J0=0.25):
    """dE from the integrated-by-parts form (the transverse kick is unchanged)."""
    ps, _ = ibp_pieces(csr, s, x, J0=J0)
    return totals(ps)['ibp']
