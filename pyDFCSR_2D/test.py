from yaml_parser import *
import numpy as np
from distgen import Generator
import matplotlib.pyplot as plt
from tracking import *
from scipy.interpolate import RegularGridInterpolator

with open('input/config.yaml') as f:
    config = ordered_load(f)

input_beam = config['input_beam']
if input_beam['style'] == 'from_file':
    filename = input_beam['beamfile']
    Xs = np.loadtxt(filename)
#-------------------------------------------------------------
# To Do: add distgen input type


#elif input_beam['style'] == 'distgen':
#    filename = input_beam['distgen_input_file']
#    gen = Generator(filename)
#    gen.run()
#    pg = gen.particles
#    #pg.plot('z', 'pz')
#    #plt.show()
#    px = pg.px
#    print(px)

#--------------------------------------------------------------

input_lattice = config['input_lattice']
filename = input_lattice['lattice_input_file']
with open(filename) as f:
    lattice_config = ordered_load(f)
    # Todo: test get_reference_trajectory
    s, coords, tau_vec, n_vec, rho, distance = get_referece_traj(lattice_config)





