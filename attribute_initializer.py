import numpy as np

from utility_functions import full_path
from beam import Beam
from lattice import Lattice
from df_tracker import DF_Tracker

"""
Module Name: parameter_initialization.py

Initializes some dictionaries for CSR3D and creates a Beam objects
"""

def init_attributes(input_dict = {}):
    """
    Given the input dictionary, creates and returns the integration parameter and 
    csr parameter dictionaries along with the Beam, Lattice, and DF_Tracker objects. 
    """
    integration_params = init_integration_params(**input_dict.get("CSR_integration", {}))
    histo_mesh_params = init_histo_params(**input_dict.get("particle_deposition", {}))
    csr_params = init_CSR_params(**input_dict.get("CSR_computation", {}))

    beam = Beam(input_dict["input_beam"])
    lattice = Lattice(input_dict["input_lattice"], beam.init_energy)
    df_tracker = DF_Tracker(lattice.total_steps, histo_mesh_params, csr_params)

    statistics = init_stats(lattice.total_steps)
    wake_history = init_wake_history(lattice.total_steps, csr_params)

    return statistics, integration_params, csr_params, lattice, beam, df_tracker, wake_history

def init_stats(total_steps):
    """
    Initialize the statitics dictionary, which stores the twiss of the beam at each step
    (also a dictionary) and other various statistics. All statistics are a np array which have an
    element for each 'step' in the lattice.
    Parameters:
        total_steps: the number of steps in the lattice
    """

    # Preallocate size based upon Nstep of np arrays for speed
    statistics = {}
    statistics['twiss'] = {'alpha_x': np.zeros(total_steps),
                                'beta_x': np.zeros(total_steps),
                                'gamma_x': np.zeros(total_steps),
                                'emit_x': np.zeros(total_steps),
                                'eta_x': np.zeros(total_steps),
                                'etap_x': np.zeros(total_steps),
                                'norm_emit_x': np.zeros(total_steps),
                                'alpha_y': np.zeros(total_steps),
                                'beta_y': np.zeros(total_steps),
                                'gamma_y': np.zeros(total_steps),
                                'emit_y': np.zeros(total_steps),
                                'eta_y': np.zeros(total_steps),
                                'etap_y': np.zeros(total_steps),
                                'norm_emit_y': np.zeros(total_steps)}

    statistics['slope'] = np.zeros(total_steps)
    statistics['sigma_x'] = np.zeros(total_steps)
    statistics['sigma_z'] = np.zeros(total_steps)
    statistics['sigma_energy'] = np.zeros(total_steps)
    statistics['mean_x']  = np.zeros(total_steps)
    statistics['mean_z'] = np.zeros(total_steps)
    statistics['mean_energy'] = np.zeros(total_steps)

    return statistics

def init_histo_params(pbins=100, obins=100, plim=5, olim=5, filter_order=1, filter_window=5, velocity_threshold=1000):
    """
    Initialize the histogram parameters, this dictionary is fed into the step snapshot initalizer
    """
    keys = ["pbins", "obins", "plim", "olim", "filter_order", "filter_window", "velocity_threshold"]
    values = [pbins, obins, plim, olim, filter_order, filter_window, velocity_threshold]

    histo_mesh_params = {k: v for k, v in zip(keys, values)}

    return histo_mesh_params

def init_integration_params(n_formation_length = 4, zbins = 200, xbins = 200):
    """ Initializes the integration_params dictionary"""

    keys = ["n_formation_length", "zbins", "xbins"]
    values = [n_formation_length, zbins, xbins]

    integration_params = {k: v for k, v in zip(keys, values)}
    
    return integration_params

def init_CSR_params(workdir = '.', apply_CSR = 1, compute_CSR = 1, transverse_on = 1, 
                    pbins = 20, obins = 30, plim = 5, olim = 5,
                    write_beam = None, write_wakes = True, write_name = ''):
    """ Initializes the CSR_params dictionary"""

    keys = ["workdir", "apply_CSR", "compute_CSR", "transverse_on", "pbins", "obins", "plim", "olim",
            "write_beam", "write_wakes", "write_name"]
    values = [full_path(workdir), apply_CSR, compute_CSR, transverse_on, pbins, obins, plim, olim,
            write_beam, write_wakes, write_name]

    CSR_params = {k: v for k, v in zip(keys, values)}

    if CSR_params["write_beam"] == "None":
        CSR_params["write_beam"] = []

    return CSR_params

def init_wake_history(total_step_num, csr_mesh_params):
    """ Initializes the dictionary of two arrays where the wakes are stored """
    # Extract some parameters
    pbins = csr_mesh_params["pbins"]
    obins = csr_mesh_params["obins"]

    # Create the numpy arrays
    dE_history = np.zeros((total_step_num-1, obins, pbins), dtype=np.float64)
    x_kick_history = np.zeros((total_step_num-1, obins, pbins), dtype=np.float64)

    # Create dictionary
    wake_history = {"dE_history": dE_history,
                    "x_kick_history": x_kick_history}
    
    return wake_history
        
