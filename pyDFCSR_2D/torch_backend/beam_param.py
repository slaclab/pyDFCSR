"""
Parameterised initial beams for the differentiable run (torch_backend.diff.CSR2DDiff(coords=...)).

Gradients of a CSR result with respect to single particles are not useful: 6N unknowns, each
particle's effect on the collective field is O(1/N), and the per-particle derivative is only
piecewise smooth. What is useful is the gradient with respect to the parameters p of the beam's
DISTRIBUTION. Each class here keeps its randomness fixed (a given beam, or draws made once) and
builds the particles as a differentiable function x_i = g(xi_i; p), so the loss is a smooth,
deterministic function of p and dL/dp = sum_i dL/dx_i dx_i/dp averages over all particles.

  BeamTransform   (a) a fixed beam (from a file, distgen, an upstream code) with physical knobs:
                  extra chirp and curvature, bunch length and position, uncorrelated energy spread,
                  and a monotonic map of z that reshapes the current profile (moves, grows or
                  shrinks horns) while keeping every correlation of the beam
  BeamFromParams  (b) a beam built from physical parameters: Twiss and emittance in x and y, a
                  current profile (Gaussian, or a spline log-density sampled by inverse CDF), chirp,
                  curvature and energy spread
  NeuralBeam      (c) a neural-network generator on fixed normal draws (a 6D -> 6D MLP, initialised
                  to the identity, i.e. a Gaussian with the given rms sizes), for flexible
                  distributions such as phase-space reconstruction

Coordinates are bmadx's (x, px, y, py, z, pz), shape (6, N), float64. Parameters are plain leaf
tensors with requires_grad (BeamTransform, BeamFromParams: self.params) or the module's weights
(NeuralBeam). The structure of the run (schedule, windows, mesh, layout) is decided from the beam
CSR2DDiff sees; when an optimisation moves the beam far, rebuild it (run without `structure=`).
"""
import math

import numpy as np
import torch

F64 = torch.float64


def _leaf(v, device):
    return torch.tensor(float(v), dtype=F64, device=device, requires_grad=True)


def _interp(xq, xp, fp):
    """Linear interpolation of fp(xp) at xq (xp increasing; the indices detached, the weights not)."""
    i = (torch.searchsorted(xp.detach(), xq.detach()) - 1).clamp(0, xp.numel() - 2)
    t = (xq - xp[i]) / (xp[i + 1] - xp[i])
    return fp[i] + t * (fp[i + 1] - fp[i])


def _cumulative(w, f):
    """Cumulative trapezoid of f on the grid w, starting at 0."""
    inc = 0.5 * (f[1:] + f[:-1]) * (w[1:] - w[:-1])
    return torch.cat([torch.zeros(1, dtype=f.dtype, device=f.device), torch.cumsum(inc, 0)])


# ----------------------------------------------------------------------------------------- (a)
class BeamTransform:
    """
    (a) A fixed beam with differentiable knobs. All are at their identity values by default, so
    beam() returns the given beam (to round-off: the profile map is tabulated). Applied in this
    order, about the beam's own centroid (z0 = <z>, the linear pz(z) fit of the fixed beam):

      z_shift, z_scale      z -> z0 + z_shift + z_scale (z - z0)            bunch position, length
      profile_knots         z -> z0 + sigma_z phi((z - z0) / sigma_z), phi monotonic on [-W, W]:
                            phi(w) = -W + 2W C(w)/C(W), C(w) = int_-W^w exp(q), q piecewise linear in
                            the knots (q = 0: identity; phi(+-W) = +-W, identity outside). The
                            current becomes lambda(phi(w)) phi'(w) = lambda(w) -> a knot q_k > 0
                            stretches the beam there (lower current), q_k < 0 compresses it.
      espread_scale         pz -> fit(z) + espread_scale (pz - fit(z))     uncorrelated energy spread
      chirp, chirp2         pz -> pz + chirp dz + chirp2 dz^2, dz the new z - z0 - z_shift (1/m, 1/m^2)

    `params` holds the knobs as leaf tensors; pass p=... to override (e.g. finite differences).
    """

    def __init__(self, coords, n_knots=9, width=4.0, device=None):
        c = torch.as_tensor(coords, dtype=F64)
        self.device = torch.device(device) if device is not None else c.device
        self.base = c.to(self.device).detach()
        z, pz = self.base[4], self.base[5]
        self.z0 = z.mean()
        self.sigma_z = z.std(correction=0)
        zc = z - self.z0
        slope = (zc * (pz - pz.mean())).sum() / (zc * zc).sum()
        self.fit = (pz.mean(), slope)                   # linear pz(z) of the fixed beam
        self.width = float(width)
        self.knot_w = torch.linspace(-width, width, n_knots, dtype=F64, device=self.device)
        self.params = dict(z_shift=_leaf(0.0, self.device), z_scale=_leaf(1.0, self.device),
                           profile_knots=torch.zeros(n_knots, dtype=F64, device=self.device, requires_grad=True),
                           espread_scale=_leaf(1.0, self.device), chirp=_leaf(0.0, self.device),
                           chirp2=_leaf(0.0, self.device))

    def profile_map(self, w, q, n_grid=4001):
        """phi(w) for the knot values q (see the class doc)."""
        W = self.width
        g = torch.linspace(-W, W, n_grid, dtype=F64, device=self.device)
        C = _cumulative(g, torch.exp(_interp(g, self.knot_w, q)))
        phi_g = -W + 2.0 * W * C / C[-1]
        inside = w.abs() <= W
        return torch.where(inside, _interp(w.clamp(-W, W), g, phi_g), w)

    def beam(self, p=None):
        p = dict(self.params, **(p or {}))
        x, px, y, py, z, pz = self.base.unbind(0)
        dz = z - self.z0
        pz_fit = self.fit[0] + self.fit[1] * dz
        pz = pz_fit + p['espread_scale'] * (pz - pz_fit)
        if bool((p['profile_knots'] != 0).any()) or p['profile_knots'].requires_grad:
            dz = self.sigma_z * self.profile_map(dz / self.sigma_z, p['profile_knots'])
        dz = p['z_scale'] * dz
        z = self.z0 + p['z_shift'] + dz
        pz = pz + p['chirp'] * dz + p['chirp2'] * dz * dz
        return torch.stack([x, px, y, py, z, pz])


# ----------------------------------------------------------------------------------------- (b)
class BeamFromParams:
    """
    (b) A beam from physical parameters, on N fixed draws (seeded): 6 standard normals per particle
    and one uniform for the longitudinal position.

      x, px:   x = sqrt(eps_x beta_x) n1,  px = sqrt(eps_x / beta_x) (n2 - alpha_x n1)   (geometric eps)
      y, py:   the same with (eps_y, beta_y, alpha_y)
      z:       profile 'gaussian': z = sigma_z n5
               profile 'spline':   the current lambda(z) proportional to exp(q(z)), q piecewise linear
                                   in `profile_knots` on [-W, W] sigma_z_ref, sampled by inverse
                                   CDF: z = F^-1(u). Knots all 0 = flat top over the window.
      pz:      pz = chirp z + chirp2 z^2 + sigma_pz n6 (+ pz_mean)

    `params` holds them as leaf tensors (initial values from the keyword arguments).
    """

    def __init__(self, n, eps_x=1e-9, beta_x=10.0, alpha_x=0.0, eps_y=1e-9, beta_y=10.0, alpha_y=0.0,
                 sigma_z=50e-6, sigma_pz=1e-3, chirp=0.0, chirp2=0.0, pz_mean=0.0,
                 profile='gaussian', n_knots=11, width=4.0, seed=0, device='cpu'):
        self.device = torch.device(device)
        g = torch.Generator().manual_seed(int(seed))
        self.normal = torch.randn(6, int(n), generator=g, dtype=F64).to(self.device)
        self.uniform = torch.rand(int(n), generator=g, dtype=F64).to(self.device)
        if profile not in ('gaussian', 'spline'):
            raise ValueError(f"profile must be 'gaussian' or 'spline', got {profile!r}")
        self.profile = profile
        self.width = float(width)
        self.sigma_z_ref = float(sigma_z)              # the spline window's scale (fixed)
        L = lambda v: _leaf(v, self.device)
        self.params = dict(eps_x=L(eps_x), beta_x=L(beta_x), alpha_x=L(alpha_x),
                           eps_y=L(eps_y), beta_y=L(beta_y), alpha_y=L(alpha_y),
                           sigma_pz=L(sigma_pz), chirp=L(chirp), chirp2=L(chirp2), pz_mean=L(pz_mean))
        if profile == 'gaussian':
            self.params['sigma_z'] = L(sigma_z)
        else:
            w = torch.linspace(-width, width, n_knots, dtype=F64, device=self.device)
            self.knot_w = w
            # start from the Gaussian of rms sigma_z: q = -w^2/2 at the knots
            self.params['profile_knots'] = (-0.5 * w * w).detach().clone().requires_grad_(True)

    def z_of_u(self, q, n_grid=4001):
        """Inverse CDF of the spline current profile at the fixed uniforms."""
        W = self.width
        g = torch.linspace(-W, W, n_grid, dtype=F64, device=self.device)
        F = _cumulative(g, torch.exp(_interp(g, self.knot_w, q)))
        F = F / F[-1]
        return self.sigma_z_ref * _interp(self.uniform, F, g)

    def beam(self, p=None):
        p = dict(self.params, **(p or {}))
        n1, n2, n3, n4, n5, n6 = self.normal.unbind(0)

        def plane(a, b, eps, beta, alpha):
            return torch.sqrt(eps * beta) * a, torch.sqrt(eps / beta) * (b - alpha * a)
        x, px = plane(n1, n2, p['eps_x'], p['beta_x'], p['alpha_x'])
        y, py = plane(n3, n4, p['eps_y'], p['beta_y'], p['alpha_y'])
        z = p['sigma_z'] * n5 if self.profile == 'gaussian' else self.z_of_u(p['profile_knots'])
        pz = p['pz_mean'] + p['chirp'] * z + p['chirp2'] * z * z + p['sigma_pz'] * n6
        return torch.stack([x, px, y, py, z, pz])


# ----------------------------------------------------------------------------------------- (c)
class NeuralBeam(torch.nn.Module):
    """
    (c) A neural-network generator: particles = mean + scale * (xi + MLP(xi)), xi fixed 6D standard
    normal draws, `scale` the rms sizes (x, px, y, py, z, pz) in physical units. The MLP's last layer
    starts at zero, so the initial beam is exactly the uncorrelated Gaussian of those sizes; training
    the weights (self.parameters()) shapes any distribution, correlations included.
    """

    def __init__(self, n, scale, mean=None, hidden=64, layers=3, seed=0, device='cpu'):
        super().__init__()
        dev = torch.device(device)
        g = torch.Generator().manual_seed(int(seed))
        self.register_buffer('xi', torch.randn(int(n), 6, generator=g, dtype=F64).to(dev))
        self.register_buffer('scale', torch.as_tensor(scale, dtype=F64).to(dev))
        self.register_buffer('mean', torch.zeros(6, dtype=F64, device=dev) if mean is None
                             else torch.as_tensor(mean, dtype=F64).to(dev))
        mods, d = [], 6
        for _ in range(layers - 1):
            mods += [torch.nn.Linear(d, hidden, dtype=F64), torch.nn.Tanh()]
            d = hidden
        last = torch.nn.Linear(d, 6, dtype=F64)
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
        self.net = torch.nn.Sequential(*mods, last).to(dev)

    def beam(self):
        u = self.xi + self.net(self.xi)
        return (self.mean + self.scale * u).T.contiguous()

    def forward(self):
        return self.beam()
