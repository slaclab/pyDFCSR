% CSR_CALC input for the 1D side of the tilt scan. Beam comes from pyDFCSR at the bend entrance
% (s = 0.1 m), written by tilt_scan_run.py; the 0.1 m drift before it carries no CSR in 1D.
ugf = 'f'; rematch = 0; fncsr = 'TAG'; isamp = 1;
emitxN0 = 1; betax0 = 1; alphax0 = 0;          % not used for ugf='f', rematch=0 (must still pass checks)
npart = 1000000; E0 = 5e9; sigZ0 = 50e-6; sigEu0 = 1e-4; sigE0 = 1e-4; corsign = -1; cor2 = 0;
N = 6.2415e9; charge = 1;                      % 1 nC
nbins = 300; nsmooth = 2; order = 2;
% bend 1 m / 1 rad then 0.5 m drift, ISR off, 0.01 m slices (100 + 50)
latt = [1.0  1.0  0  0  0  NSB  0;
        0.5  0    0  0  0   50  0];
