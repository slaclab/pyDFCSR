% CSR_CALC input for the 1D side of the chicane comparison. Lattice copied element by element from
% pyDFCSR's example/input/chicane_lattice.yaml (same L, angle, E1, E2). The beam comes from pyDFCSR at
% the B1 entrance (s = 0.1 m), written by tilt_scan_run.py, so the 0.1 m D0 drift is not repeated here.
% Slices: NSB per 0.5002 m bend; drifts at 10x the bend slice length. ISR off.
ugf = 'f'; rematch = 0; fncsr = 'TAG'; isamp = 1;
emitxN0 = 1; betax0 = 1; alphax0 = 0;          % not used for ugf='f', rematch=0 (must still pass checks)
npart = 1000000; E0 = 5e9; sigZ0 = 200e-6; sigEu0 = 1e-4; sigE0 = 1e-4; corsign = -1; cor2 = 0;
N = 6.2415e9; charge = 1;                      % 1 nC
nbins = 300; nsmooth = 2; order = 2;
Lb = 0.5002; ang = 0.0483; nb = NSB; dl = Lb/nb*10;
nd = @(L) max(1, round(L/dl));
%       L        angle   k   E1     E2     N-slice   ISR
latt = [Lb       ang     0   0      ang    nb        0;    % B1
        5.0058   0       0   0      0      nd(5.0058) 0;   % D1
        Lb      -ang     0  -ang    0      nb        0;    % B2
        1.0      0       0   0      0      nd(1.0)   0;    % D2
        Lb      -ang     0   0     -ang    nb        0;    % B3
        5.0058   0       0   0      0      nd(5.0058) 0;   % D3
        Lb       ang     0   ang    0      nb        0;    % B4
        0.2      0       0   0      0      nd(0.2)   0];   % Df
