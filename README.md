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
- Bmad-X is required from https://github.com/bmad-sim/Bmad-X

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

To install Bmad-X

```bash
git clone https://github.com/bmad-sim/Bmad-X.git
cd Bmad-X
pip install --no-dependencies -e .
```

The new environment ``pydfcsr`` contains JupyterLab. You can then try the examples in
the directory [pyDFCSR_2D/example/](pyDFCSR_2D/example/).

```bash

cd pyDFCSR_2D/example
jupyter lab
```

### Developing pyDFCSR

To develop pyDFCSR, perform the following after setting up the environment as
in the previous section:

```python
cd /path/to/pyDFCSR  # replace this with the top-level directory of your clone
python -m pip install -e .
```

The ``pip install`` line will make a "development" install of this library so
that you don't have to reinstall the library after each modification to its
code.
