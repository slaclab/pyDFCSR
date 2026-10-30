% CSR_CALC input for the pyDFCSR 1D-mode benchmark (benchmark_csr1d_vs_emma.py). The beam is
% pyDFCSR's own, written at the first bend entrance; the lattice and the 1D settings are written by
% the benchmark into TAG_latt.m (a 'latt' matrix plus E0, N, nbins, nsmooth), so each case is
% defined once, in Python, for both codes. ISR off.
ugf = 'f'; rematch = 0; fncsr = 'TAG'; isamp = 1;
emitxN0 = 1; betax0 = 1; alphax0 = 0;          % not used for ugf='f', rematch=0 (must still pass checks)
npart = 1000000; sigZ0 = 50e-6; sigEu0 = 1e-4; sigE0 = 1e-4; corsign = -1; cor2 = 0; order = 2;
run(fullfile('/tmp/csr1d_wrap', 'TAG_latt.m'));   % sets latt, E0, N, charge, nbins, nsmooth
