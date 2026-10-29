% Batch-mode shim for Paul Emma's csr_check_input.m: skips the interactive input() prompts when
% the caller has already set fn / fn_yn, then runs the original checks unchanged.
if ~exist('fn', 'var'),    fn = input('CSR_CALC input file descriptor (*_csr.m assumed): ', 's'); end
if ~exist('fn_yn', 'var'), fn_yn = 'n'; end
ORIG = '/Users/jytang/Library/CloudStorage/GoogleDrive-jytang@stanford.edu/My Drive/Stanford/research/FEL/CSR/Paul Emma CSR code/CSR code/csr_check_input.m';
src = fileread(ORIG);
src = regexprep(src, "fn = input\([^\n]*\n", "\n", 'once');
src = regexprep(src, "fn_yn = input\([^\n]*\n", "\n", 'once');
eval(src);
