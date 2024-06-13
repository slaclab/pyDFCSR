from pyDFCSR_2D import CSR2D

DtestCSR = CSR2D(input_file='input/fodo_config.yaml')
#DtestCSR.CSR_params.compute_CSR = 0
DtestCSR.run()
