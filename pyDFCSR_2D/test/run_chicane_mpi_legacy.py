"""Run chicane with legacy method via MPI."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'example'))
from pyDFCSR_2D.CSR import CSR2D
csr = CSR2D(input_file='input/chicane_config_highres300_noCSR.yaml', parallel=True)
csr.run()
