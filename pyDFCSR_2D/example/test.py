from CSR import *
import matplotlib.pyplot as plt
import matplotlib.colors as color

DtestCSR = CSR2D(input_file='input/dipole_facet_config.yaml')
#DtestCSR.CSR_params.compute_CSR = 0
DtestCSR.run()