import numpy as np
import h5py
import matplotlib.pyplot as plt


def read_old_2D_stats(filepath):
    """
    Reads the statistics from the given HDF5 file and returns the data in a structured format.
    
    Parameters:
    h5_filename (str): The path to the HDF5 file.

    Returns:
    dict: A dictionary containing the statistics and lattice data.
    """
    data = {}

    # Open the HDF5 file in read mode
    with h5py.File(filepath, 'r') as hf:
        # Read the lattice data
        data['step_positions'] = np.array(hf['step_positions'])
        data['coords'] = np.array(hf['coords'])
        data['n_vec'] = np.array(hf['n_vec'])
        data['tau_vec'] = np.array(hf['tau_vec'])
        
        # Initialize a dictionary for the statistics
        data['statistics'] = {
            'mean_energy': np.array(hf['mean_energy']),
            'mean_x': np.array(hf['mean_x']),
            'mean_z': np.array(hf['mean_z']),
            'sigma_energy': np.array(hf['sigma_energy']),
            'sigma_x': np.array(hf['sigma_x']),
            'sigma_z': np.array(hf['sigma_z']),
            'slope': np.array(hf['slope']),
            'twiss': {}
        }

        # Load the 'twiss' sub-group
        twiss_group = hf['twiss']
        for key in twiss_group.keys():
            data['statistics']['twiss'][key] = np.array(twiss_group[key])

    return data['statistics']

def read_new_2D_stats(filepath):
    data = {}

    # Open the HDF5 file in read mode
    with h5py.File(filepath, 'r') as f:
        # Navigate to the 'statistics' group
        stats_grp = f['statistics']

        # Helper function to recursively load datasets
        def recursively_load(group, data_dict):
            for key, item in group.items():
                if isinstance(item, h5py.Dataset):
                    data_dict[key] = item[()]  # Load dataset as a NumPy array
                elif isinstance(item, h5py.Group):
                    data_dict[key] = {}
                    recursively_load(item, data_dict[key])

        # Load all datasets and groups under 'statistics'
        recursively_load(stats_grp, data)

    return data

def read_CSRtrack_stats(directory, step_num):
    N = step_num
    CSRtrack_stats = {'t': np.zeros((N,)), 'sigma_x': np.zeros((N,)), 'sigma_z': np.zeros((N,)), 
                    'emit_x': np.zeros((N,)), 'norm_emit_x': np.zeros((N,)),
                    'beta_x': np.zeros((N,)), 'alpha_x': np.zeros((N,))}
    
    for step_index in range(N):
        filename = f'/dipole_{step_index+1:04d}.fmt3'
        data = np.loadtxt(directory + filename)
        z = data[1:, 4]
        delta = data[1:, 5]

        x = data[1:, 0]
        xp = data[1:, 1]

        t = data[0,0]
        gam0 = data[0,1]

        sigma_x = np.std(x)
        sigma_z = np.std(z)
        
        cov3 = np.cov([x, xp, delta])
        twiss = twiss_dispersion_calc(cov3)
        twiss['norm_emit'] = twiss['emit'] * gam0
        
        CSRtrack_stats['t'][step_index] = t
        CSRtrack_stats['sigma_x'][step_index] = sigma_x
        CSRtrack_stats['sigma_z'][step_index] = sigma_z
        CSRtrack_stats['emit_x'][step_index] = twiss['emit']
        CSRtrack_stats['norm_emit_x'][step_index] = twiss['norm_emit']
        CSRtrack_stats['beta_x'][step_index] = twiss['beta']
        CSRtrack_stats['alpha_x'][step_index] = twiss['alpha']

    return CSRtrack_stats

def twiss_dispersion_calc(sigma3):
    """
    Twiss and Dispersion calculation from a 3x3 sigma (covariance) matrix from particles
    x, p, delta

    From https://github.com/ChristopherMayes/openPMD-beamphysics/blob/master/pmd_beamphysics/statistics.py

    Formulas from:
        https://uspas.fnal.gov/materials/19Knoxville/g-2/creation-and-analysis-of-beam-distributions.html

    Returns a dict with:
        alpha
        beta
        gamma
        emit
        eta
        etap

    """

    # Collect terms

    delta2 = sigma3[2, 2]
    xd = sigma3[0, 2]
    pd = sigma3[1, 2]

    eb = sigma3[0, 0] - xd ** 2 / delta2
    eg = sigma3[1, 1] - pd ** 2 / delta2
    ea = -sigma3[0, 1] + xd * pd / delta2

    emit = np.sqrt(eb * eg - ea ** 2)

    # Form the output dict
    d = {}

    d['alpha'] = ea / emit
    d['beta'] = eb / emit
    d['gamma'] = eg / emit
    d['emit'] = emit
    d['eta'] = xd / delta2
    d['etap'] = pd / delta2

    return d

def plot_statistics(stats, labels, figure_title="", axis_title=""):

    colors = ["blue", "red", "green"]
    for i, stat in enumerate(stats):
        # Plot the first array
        plt.plot(stat, label=labels[i], color = colors[i], marker='o')

    # Add labels and title
    plt.xlabel('Step index')
    plt.ylabel(axis_title)
    plt.title(figure_title)

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()