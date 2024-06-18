from pyDFCSR_2D import CSR2D

DtestCSR = CSR2D(input_file='input/dipole_config_chirp.yaml')
DtestCSR.CSR_params.compute_CSR = 0
DtestCSR.run()

