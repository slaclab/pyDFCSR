from pyDFCSR_2D import CSR2D

DtestCSR = CSR2D(input_file='input/FACETII_BC14_config.yaml')
DtestCSR.CSR_params.compute_CSR = 0
DtestCSR.run()

