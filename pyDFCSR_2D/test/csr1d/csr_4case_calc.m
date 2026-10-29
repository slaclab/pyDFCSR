function [dE_dct,lams,dlam_ds] = csr_4case_calc(S,lam,ds,R,phi,phim,x_R,gam,nsmooth,OPT)
% Shim around Paul Emma's csr_4case_calc.m that applies pyDFCSR's drift_cutoff rule:
% the wake is zeroed once the bunch is more than DRIFT_CUTOFF * R*phim past the last bend exit
% (x_R = x/R > DRIFT_CUTOFF*phim, same |R*phi| scale as CSR.py). DRIFT_CUTOFF is a global set by
% run_ts; empty or 0 = off, which reproduces the original exactly.
% The original is called through csr_4case_calc_orig.m, a verbatim copy that run_ts writes to
% /tmp/csr1d_wrap with only the function name changed; the user's file is never modified.
  global DRIFT_CUTOFF
  [dE_dct,lams,dlam_ds] = csr_4case_calc_orig(S,lam,ds,R,phi,phim,x_R,gam,nsmooth,OPT);
  if ~isempty(DRIFT_CUTOFF) && DRIFT_CUTOFF > 0 && x_R > DRIFT_CUTOFF * phim
    dE_dct = zeros(size(dE_dct));
  end
end
