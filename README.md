# pyDFCSR

Fast 2D/3D CSR simulation with Python


## Installation
### Requirements

- distgen
- h5py
- matplotlib
- mpi4py
- numba
- numpy
- pyyaml
- scipy

For the examples:

- jupyterlab
- ipympl
### Using conda

To install pyDFCSR using conda, perform the following:

```bash
git clone https://github.com/jy-tang/pyDFCSR
cd pyDFCSR
conda env create -n pydfcsr -f environment.yml
conda activate pydfcsr
```

The new environment ``pydfcsr`` contains JupyterLab. You can then try the examples in
the directory [pyDFCSR_2D/example/](pyDFCSR_2D/example/).

```bash

cd pyDFCSR_2D/example
jupyter lab
```
