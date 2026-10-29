function run_ts(tag, outdir, fac_in, nsb, cutoff)
% cutoff: pyDFCSR-style drift_cutoff in units of R*phi (default 0 = off); output suffix _cut<cutoff> if > 0
% nsb: number of CSR slices in the 1 m bend (default 100, i.e. 1 cm); output gets suffix _nsb<N> if not 100
% fac_in = 0: compute the 1D wake but do not apply it (CSR-off reference with identical 1D tracking)
% Batch driver for Paul Emma's csr_calc.m on one tilt-scan case. Saves, per CSR slice (every 1 cm):
% the 1D wake dE/dct(z) [eV/m] on its bin grid Si, the smoothed line density fi, and s;
% plus the initial and final particles, to outdir/1d_<tag>.mat
  ORIG = '/Users/jytang/Library/CloudStorage/GoogleDrive-jytang@stanford.edu/My Drive/Stanford/research/FEL/CSR/Paul Emma CSR code/CSR code';
  W = '/tmp/csr1d_wrap'; if ~exist(W, 'dir'), mkdir(W); end; HERE = fileparts(mfilename('fullpath')); addpath(ORIG); addpath(W); addpath(HERE); cd(W);
  if nargin < 3, fac_in = 1; end
  if nargin < 4, nsb = 100; end
  if nargin < 5, cutoff = 0; end
  global DRIFT_CUTOFF; DRIFT_CUTOFF = cutoff;
  % verbatim copy of the original wake routine under a new name, so the shim csr_4case_calc.m in HERE
  % can wrap it (both names on the path; HERE is added last-first so the shim shadows the original)
  osrc = fileread(fullfile(ORIG, 'csr_4case_calc.m'));
  osrc = regexprep(osrc, 'function\s*\[dE_dct,lams,dlam_ds\]\s*=\s*csr_4case_calc\(', 'function [dE_dct,lams,dlam_ds] = csr_4case_calc_orig(', 'once');
  assert(contains(osrc, 'csr_4case_calc_orig('), 'could not rename the original csr_4case_calc');
  fid = fopen(fullfile(W, 'csr_4case_calc_orig.m'), 'w'); fprintf(fid, '%s', osrc); fclose(fid); rehash;
  assert(strcmp(fileparts(which('csr_4case_calc')), HERE), 'the cutoff shim must shadow the original');
  tmpl = fullfile(HERE, ['ts_' tag '_template_csr.m']); if ~exist(tmpl, 'file'), tmpl = fullfile(HERE, 'ts_template_csr.m'); end
  src = fileread(tmpl); src = strrep(src, 'TAG', ['ts_' tag]); src = strrep(src, 'NSB', sprintf('%d', nsb));
  src = [src sprintf('\nfac = %d;\n', fac_in)];
  fn_tmp = sprintf('ts_%s_n%d', tag, nsb); fid = fopen(fullfile(W, [fn_tmp '_csr.m']), 'w'); fprintf(fid, '%s', src); fclose(fid);
  set(0, 'DefaultFigureVisible', 'off');
  fn = fn_tmp; fn_yn = 'n';
  Xs_init = [];
  try
    csr_calc;
  catch e
    fprintf('post-processing error caught (slice-emittance block): %s\n', e.message);
  end
  close all;
  fprintf('1D %s done: %d particles, s_end = %.3f m\n', tag, size(Xs, 2), s(end));
  suf = ''; if fac_in == 0, suf = '_off'; end
  if nsb ~= 100, suf = [suf sprintf('_nsb%d', nsb)]; end
  if cutoff > 0, suf = [suf sprintf('_cut%g', cutoff)]; end
  save(fullfile(outdir, ['1d_' tag suf '.mat']), 's', 'Si', 'fi', 'dE_dct', 'dE_E', 'Xs', 'sigZ', 'sigE', 'sigX', ...
       'Ebar_CSR', 'Erms_CSR', 'R56', 'R51', 'R52', 'gam0', 'E0', 'N', '-v7.3');
end
