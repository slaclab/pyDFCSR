from yaml_parser import *
import numpy as np
from distgen import Generator
import matplotlib.pyplot as plt
from tracking import *
from beams import *
from scipy.interpolate import RegularGridInterpolator

with open('input/config.yaml') as f:
    config = ordered_load(f)

input_beam = config['input_beam']
Xs = generate_beams(input_beam)

#--------------------------------------------------------------

input_lattice = config['input_lattice']
filename = input_lattice['lattice_input_file']
with open(filename) as f:
    lattice_config = ordered_load(f)

s, coords, tau_vec, n_vec, rho, distance = get_referece_traj(lattice_config)
# ----test get_reference_trajectory--------
#plt.plot(coords[:,0],coords[:,1])
#plt.quiver(coords[::20,0], coords[::20,1], tau_vec[::20,0], tau_vec[::20,1])
#plt.quiver(coords[::20,0], coords[::20,1], n_vec[::20,0], n_vec[::20,1])
#plt.show()





