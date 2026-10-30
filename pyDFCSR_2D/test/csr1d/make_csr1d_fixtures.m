function make_csr1d_fixtures(outdir)
% Reference wakes from Paul Emma's ORIGINAL csr_4case_calc.m, for pyDFCSR_2D/test/test_csr1d.py.
% Each case is a prescribed line density lambda(S) on a uniform grid (no particles, no tracking)
% and a geometry (in-bend phi, or post-bend x_R). The routine is called with nsmooth = 1 (no
% smoothing) so both codes see the SAME lambda; the comparison is then of the kernels alone.
% Written as one small .mat per case: S, lam, R, phi, phim, x_R, gam, W.
% Run once, in MATLAB, with the sandbox off:
%   matlab -batch "addpath('<this dir>'); make_csr1d_fixtures('<this dir>/fixtures')"
  ORIG = '/Users/jytang/Library/CloudStorage/GoogleDrive-jytang@stanford.edu/My Drive/Stanford/research/FEL/CSR/Paul Emma CSR code/CSR code';
  % make sure the ORIGINAL routine is used, not the drift-cutoff shim in this directory
  here = fileparts(mfilename('fullpath')); rmpath(here); addpath(ORIG);
  assert(strcmp(fileparts(which('csr_4case_calc')), ORIG), 'csr_4case_calc must resolve to the original');
  if ~exist(outdir, 'dir'), mkdir(outdir); end
  N = 6.2415e9; sz = 50e-6; gam = 5e9/510998.95; OPT = optimset('TolFun', 1e-12, 'TolX', 1e-14);
  dists = {'gauss', 'flat', 'asym', 'dhorn'};
  % geometries: R [m], phi or x_R; phim for post-bend. phi_t = (24 sz/R)^(1/3) is the transient angle.
  R = 10.0; phit = (24*sz/R)^(1/3); phim = 0.1;
  geoms = struct('name', {'bend_phi_0p3t', 'bend_phi_1t', 'bend_phi_5t', 'drift_xR_1e-3', 'drift_xR_1e-2', 'drift_xR_1e-1', 'drift_xR_1'}, ...
                 'phi', {0.3*phit, phit, 5*phit, 0, 0, 0, 0}, 'x_R', {0, 0, 0, 1e-3, 1e-2, 1e-1, 1});
  for nb = [300 1000]
    for d = 1:numel(dists)
      S = linspace(-8*sz, 8*sz, nb); ds = S(2) - S(1);
      switch dists{d}
        case 'gauss', f = exp(-S.^2/(2*sz^2));
        case 'flat',  f = 0.5*(tanh((S + 1.7*sz)/(0.15*sz)) - tanh((S - 1.7*sz)/(0.15*sz)));  % flat-top, smooth edges
        case 'asym',  f = exp(-S.^2/(2*sz^2)) .* (1 + erf(3*S/sz));                            % skew-normal
        case 'dhorn', f = exp(-S.^2/(2*(1.5*sz)^2)) + 1.2*exp(-(S - 1.6*sz).^2/(2*(0.25*sz)^2)) + 0.8*exp(-(S + 1.6*sz).^2/(2*(0.25*sz)^2));
      end
      lam = N * f / (sum(f) * ds);
      for g = 1:numel(geoms)
        G = geoms(g);
        if G.x_R == 0
          W = csr_4case_calc(S, lam, ds, R, G.phi, phim, 0, gam, 1, OPT);
        else
          W = csr_4case_calc(S, lam, ds, R, 0, phim, G.x_R, gam, 1, OPT);
        end
        phi = G.phi; x_R = G.x_R;  %#ok<NASGU>
        save(fullfile(outdir, sprintf('emma_%s_%s_n%d.mat', dists{d}, G.name, nb)), 'S', 'lam', 'R', 'phi', 'phim', 'x_R', 'gam', 'W', '-v7');
      end
    end
  end
  fprintf('wrote %d fixtures to %s\n', numel(dir(fullfile(outdir, 'emma_*.mat'))), outdir);
end
