import numpy as np
from numba import jit
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from histogram_function import histogram_cic_2d

"""
Module Name: post_processor.py

Helps with data processing of already computed CSR wake data
"""

def load_3D_wakes_new(filepath):
    """
    Given a filepath to an h5 file containg the wake data, unpacks it
    Parameters:
        filepath: string, the filepath of the data file
    Returns:
        the csr mesh grids, and the dE, x_kick, & y_kick history
    """
    
    with h5py.File(filepath, 'r') as f:
            
        groupname = "wake_history"
        g = f[groupname]
        
        csr_mesh_coords = g["csr_mesh_coords"][:]
        dE_history = g["dE_history"][:]
        x_kick_history = g["x_kick_history"][:]
        y_kick_history = g["y_kick_history"][:]

    return csr_mesh_coords, dE_history, x_kick_history, y_kick_history

def load_2D_wakes_new(filepath):
    """
    Given a filepath to an h5 file containg the wake data, unpacks it
    Parameters:
        filepath: string, the filepath of the data file
    Returns:
        the csr mesh grids, and the dE, x_kick, & y_kick history
    """
    
    with h5py.File(filepath, 'r') as f:
            
        groupname = "wake_history"
        g = f[groupname]
        
        csr_mesh_coords = g["csr_mesh_coords"][:]
        dE_history = g["dE_history"][:]
        x_kick_history = g["x_kick_history"][:]
        wake_indices = g["wake steps"][:]

    return csr_mesh_coords, dE_history, x_kick_history, wake_indices

def load_CSR_track_wakes(directory, total_step_num):
    """
    Loads and returns the wakes produced by CSRtrack
    Parameters:
        directory: the path to the directory where the CSR track files are stored
        step_num: number of CSR .fmt files to parse
    """
    # Initalize the wake history
    nx = 10
    nz = 30
    xlim = 3
    zlim = 3

    mesh_coords_history = np.zeros((total_step_num, nx, nz, 2), dtype=np.float64)
    dE_history = np.zeros((total_step_num, nx, nz), dtype=np.float64)
    x_kick_history = np.zeros((total_step_num, nx, nz), dtype=np.float64)

    # Parse through all CSR output files
    for step_index in range(total_step_num):
        filename = f'/dipole_{step_index+1:04d}.fmt3'
        data = np.loadtxt(directory + filename)

        z = data[1:, 4]
        x = data[1:, 0]
        fz = data[1:, 7]
        fx = data[1:, 8]

        meanx = np.mean(x)
        meanz = np.mean(z)
        x -= meanx
        z -= meanz
        sig_x = np.std(x)
        sig_z = np.std(z)

        xrange = np.linspace(meanx-xlim*sig_x, meanx+xlim*sig_x, nx)
        zrange = np.linspace(meanz-zlim*sig_z, meanz+zlim*sig_z, nz)
        xmesh, zmesh = np.meshgrid(xrange, zrange, indexing = 'ij')
        mesh_coords = np.stack([xmesh, zmesh], axis=-1)

        fz_count = histogram_cic_2d(q1 = x, q2 = z, w = fz/1e6,
                            nbins_1 = nx, bins_start_1 = meanx-xlim*sig_x, bins_end_1 = meanx + xlim*sig_x,
                            nbins_2 = nz, bins_start_2 = meanz-zlim*sig_z, bins_end_2 = meanz + zlim*sig_z)
        fx_count = histogram_cic_2d(q1 = x, q2 = z, w = fx/1e6,
                            nbins_1 = nx, bins_start_1 = meanx-xlim*sig_x, bins_end_1 = meanx + xlim*sig_x,
                            nbins_2 = nz, bins_start_2 = meanz-zlim*sig_z, bins_end_2 = meanz + zlim*sig_z)

        density_count = histogram_cic_2d(q1 = x, q2 = z, w = np.ones(z.shape),
                            nbins_1 = nx, bins_start_1 = meanx-xlim*sig_x, bins_end_1 = meanx + xlim*sig_x,
                            nbins_2 = nz, bins_start_2 = meanz-zlim*sig_z, bins_end_2 = meanz+ zlim*sig_z)
        
        # The minimum particle number of particles in a bin for that bin to have non zero beta_x value
        #threshold = np.max(fz_count) / 1000
        #fz_count[density_count <= threshold] = 0
        #fx_count[density_count <= threshold] = 0
        
        fz_bin = fz_count/density_count
        fx_bin = fx_count/density_count        

        mesh_coords_history[step_index, :, :, :] = mesh_coords
        dE_history[step_index, :, :] = fz_bin
        x_kick_history[step_index, :, :] = fx_bin
    
    return mesh_coords_history, dE_history, x_kick_history

def load_CSR_track_wakes2(directory, total_step_num):
    # Initalize the wake history
    nx = 20
    nz = 30
    xlim = 3
    zlim = 3

    mesh_coords_history = np.zeros((total_step_num, nx, nz, 2), dtype=np.float64)
    dE_history = np.zeros((total_step_num, nx, nz), dtype=np.float64)
    x_kick_history = np.zeros((total_step_num, nx, nz), dtype=np.float64)

    # Parse through all CSR output files
    for step_index in range(total_step_num):
        filename = f'/dipole_{step_index+1:04d}.fmt3'
        data = np.loadtxt(directory + filename)

        z = data[1:, 4]
        y = data[1:, 2]
        delta = data[1:, 5]
        Q = data[1:, 6]
        x = data[1:, 0]
        xp = data[1:, 1]
        t = data[0,0]
        gam0 = data[0,1]
        fz = data[1:, 7]
        fx = data[1:, 8]
        fy = data[1:, 9]

        x -= np.mean(x)
        z -= np.mean(z)
        n = z.shape[0]
        sig_x = np.std(x)
        sig_z = np.std(z)

        xrange = np.linspace(-3*sig_x, 3*sig_x, 20)
        zrange = np.linspace(-3*sig_z, 3*sig_z, 30)
        xmesh, zmesh = np.meshgrid(xrange, zrange, indexing = 'ij')
        mesh_coords = np.stack([xmesh, zmesh], axis=-1)

        fz_count = histogram_cic_2d(q1 = x, q2 = z, w = fz/1e6,
                            nbins_1 = 20, bins_start_1 = -3*sig_x, bins_end_1 = 3*sig_x,
                            nbins_2 = 30, bins_start_2 = -3*sig_z, bins_end_2 = 3*sig_z)
        fx_count = histogram_cic_2d(q1 = x, q2 = z, w = fx/1e6,
                            nbins_1 = 20, bins_start_1 = -3*sig_x, bins_end_1 = 3*sig_x,
                            nbins_2 = 30, bins_start_2 = -3*sig_z, bins_end_2 = 3*sig_z)
        density_count = histogram_cic_2d(q1 = x, q2 = z, w = np.ones(z.shape),
                            nbins_1 = 20, bins_start_1 = -3*sig_x, bins_end_1 = 3*sig_x,
                            nbins_2 = 30, bins_start_2 = -3*sig_z, bins_end_2 = 3*sig_z)
        fz_bin = fz_count/density_count
        fx_bin = fx_count/density_count

        threshold = np.max(fz_count) / 1000
        fz_count[density_count <= threshold] = 0
        fx_count[density_count <= threshold] = 0

        mesh_coords_history[step_index, :, :, :] = mesh_coords
        dE_history[step_index, :, :] = fz_bin
        x_kick_history[step_index, :, :] = fx_bin
    
    return mesh_coords_history, dE_history, x_kick_history


def load_2D_wakes_old(filepath):
    """
    Given a filepath to an h5 file containg the wake data, unpacks it
    Parameters:
        filepath: string, the filepath of the data file
    Returns:
        the csr mesh grids, and the dE and x_kick history
    """

    with h5py.File(filepath, 'r') as hf:

        # In some cases the wake may not be computed at every step, so we need to find the true number of steps

        max_index = 0
        # Iterate over each group in the file
        for groupname in hf.keys():
            step_group = hf[groupname]
            index = step_group.attrs["step"]

            if index > max_index:
                max_index = index

        csr_mesh_coords = np.empty(max_index, dtype=object)
        dE_history = np.empty(max_index, dtype=object)
        x_kick_history = np.empty(max_index, dtype=object)


        # Iterate over each group in the file
        for groupname in hf.keys():
            step_group = hf[groupname]

            index = step_group.attrs["step"]

            longitudinal_group = step_group['longitudinal']
            x_grids = longitudinal_group['x_grids'][:]
            z_grids = longitudinal_group['z_grids'][:]
            dE_dct = longitudinal_group['dE_dct'][:]

            transverse_group = step_group['transverse']
            xkicks = transverse_group['xkicks'][:]

            csr_mesh_coords[index-1] = np.dstack((x_grids, z_grids))
            dE_history[index-1] = dE_dct
            x_kick_history[index-1] = xkicks

        # Delete the unpopulated entries
        mask = np.array([coord is not None for coord in csr_mesh_coords])

        # Keep only the elements where mesh_coord is not None
        csr_mesh_coords = csr_mesh_coords[mask]
        dE_history = dE_history[mask]
        x_kick_history = x_kick_history[mask]

    return csr_mesh_coords, dE_history, x_kick_history

def plot_wakes2(mesh_coords, dEs, x_kicks, y_kicks, step_index):
    X = mesh_coords[step_index][:,:,0]
    Z = mesh_coords[step_index][:,:,1]
    dE = dEs[step_index]
    x_kick = x_kicks[step_index]
    y_kick = y_kicks[step_index]

    # Create a new figure
    fig = plt.figure(figsize=(10,10))

    make_wake_fig(fig, 221, Z, X, dE, title="dE", cmap="viridis")
    make_wake_fig(fig, 222, Z, X, x_kick, title="x_kick", cmap="plasma")
    make_wake_fig(fig, 223, Z, X, y_kick, title="y_kick", cmap="plasma")

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_wakes(mesh_coords, dEs, x_kicks, step_index):
    X = mesh_coords[step_index][:,:,0]
    Z = mesh_coords[step_index][:,:,1]
    dE = dEs[step_index]
    x_kick = x_kicks[step_index]

    # Create a new figure
    fig = plt.figure(figsize=(12,5))

    make_wake_fig(fig, 121, Z, X, dE, title="dE at step "+str(step_index), cmap="viridis")
    make_wake_fig(fig, 122, Z, X, x_kick, title="x_kick at step "+str(step_index), cmap="plasma")

    # Show the plot
    plt.tight_layout()
    plt.show()

def make_wake_fig(fig, subplot_position, Z, X, wake, title="", cmap="plasma"):
    """
    Helper function to plot_wakes()
    """
    ax = fig.add_subplot(subplot_position, projection='3d')
    surface2 = ax.plot_surface(Z, X, wake, cmap=cmap)
    fig.colorbar(surface2, ax=ax, shrink=0.4, aspect=5)
    ax.view_init(elev=0, azim=90)
    ax.set_title(title)
    ax.set_xlabel('Z axis')
    ax.set_ylabel('X axis')
    ax.set_zlabel(title)

def plot_wakes_together(mesh_coords, wakes, step_index, title="", labels=["","",""], wake_steps=None):

    Xs = []
    Zs = []

    for mesh_coord in mesh_coords:
        Xs.append(mesh_coord[step_index][:,:,0])
        Zs.append(mesh_coord[step_index][:,:,1])

    # Create a new figure
    fig = plt.figure()

    # Add a 3D subplot
    ax = fig.add_subplot(111, projection='3d')

    colors = ["red", "blue", "green"]
    for i in range(len(Xs)):
    #for i in range(1):
        X = Xs[i]
        Z = Zs[i]

        # To view unrotated and uncompressed mesh

        #Z_values = np.arange(0,Z.shape[1])
        #X_values = np.arange(0,X.shape[0])
        #Z, X = np.meshgrid(Z_values, X_values)

        # Plot the first surface
        surface1 = ax.plot_surface(Z, X, wakes[i][step_index], color=colors[i], alpha=0.4, label=labels[i])

    # Set axis labels
    ax.set_xlabel('Z axis')
    ax.set_ylabel('X axis')
    ax.set_zlabel('')
    if wake_steps is None:
        ax.set_title(title + " at step "+ str(step_index+1))
    else:
        ax.set_title(title + " at step "+ str(wake_steps[step_index]))

    plt.legend()

    plt.show()

# colormesh rather than surface
def plot_wakes_together2(mesh_coords, wakes, step_index, title="", labels=["", "", ""]):
    Xs = []
    Zs = []

    for mesh_coord in mesh_coords:
        Xs.append(mesh_coord[step_index][:, :, 0])
        Zs.append(mesh_coord[step_index][:, :, 1])

    # Create a new figure with subplots
    num_plots = len(Xs)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5), constrained_layout=True)

    if num_plots == 1:  # Handle single plot case
        axes = [axes]

    for i in range(num_plots):
        X = Xs[i]
        Z = Zs[i]

        # Generate a 2D colormesh plot
        c = axes[i].pcolormesh(Z, X, wakes[i][step_index], shading='auto', cmap='viridis')

        # Add a colorbar for each subplot
        fig.colorbar(c, ax=axes[i], label=labels[i])

        # Set axis labels
        axes[i].set_xlabel('Z axis')
        axes[i].set_ylabel('X axis')
        axes[i].set_title(f"{title} (Plot {i + 1})")

    plt.suptitle(title + f" at step {step_index + 1}")
    plt.show()

def plot_histos(histo_data, step_index):
    X = histo_data["X"][step_index]
    Z = histo_data["Z"][step_index]
    histos = [histo_data["density"][step_index], histo_data["beta_x"][step_index], histo_data["partial_density_x"][step_index],
              histo_data["partial_density_z"][step_index], histo_data["partial_beta_x"][step_index]]
    
    titles = ["density", "beta_x", "partial_density_x",
              "partial_density_z", "partial_beta_x"]


    # Create a new figure
    fig = plt.figure(figsize=(15, 10))

    # Define the number of rows and columns
    rows, cols = 2, 3

    # Create subplots
    for i in range(5):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        surface = ax.plot_surface(Z, X, histos[i], cmap='plasma', edgecolor='none')
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
        ax.set_title(titles[i])
        ax.set_xlabel('Z axis')
        ax.set_ylabel('X axis')
        ax.view_init(elev=90, azim=270)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_histo_together(histo_data_old, histo_data_new, step_index, histo_type, label1="", label2=""):

    X_1 = histo_data_old["h_coords"][step_index][:,:,0]
    Z_1 = histo_data_old["h_coords"][step_index][:,:,1]

    X_2 = histo_data_new["X"][step_index]
    Z_2 = histo_data_new["Z"][step_index]

    # Create a new figure
    fig = plt.figure()

    # Add a 3D subplot
    ax = fig.add_subplot(111, projection='3d')

    # Plot the first surface
    surface1 = ax.plot_surface(Z_1, X_1, histo_data_old[histo_type][step_index], color="red", alpha=0.5, label=label1)
    
    # Plot the second surface
    surface2 = ax.plot_surface(Z_2, X_2, histo_data_new[histo_type][step_index], color='blue', alpha=0.5, label=label2)

    # Set axis labels
    ax.set_xlabel('Z axis')
    ax.set_ylabel('X axis')
    ax.set_zlabel('')

    # Adjust x and y axis limits to ensure they cover the same range
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()

    # Determine the maximum range to apply equal scaling
    max_range = max(x_limits[1] - x_limits[0], y_limits[1] - y_limits[0])

    ax.set_xlim([x_limits[0], x_limits[0] + max_range])
    ax.set_ylim([y_limits[0], y_limits[0] + max_range])
    
    #plt.show()
    plt.legend()


def plot_all_histos_together(histo_data_old, histo_data_new, step_index, label1="", label2=""):
    """
    Plots all available histo types together in subplots.

    Parameters:
    - histo_data_old: dict, contains old histo data.
    - histo_data_new: dict, contains new histo data.
    - step_index: int, index of the step to plot.
    - label1: str, label for the old data.
    - label2: str, label for the new data.
    """

    X_1 = histo_data_old["h_coords"][step_index][:,:,0]
    Z_1 = histo_data_old["h_coords"][step_index][:,:,1]

    X_2 = histo_data_new["X"][step_index]
    Z_2 = histo_data_new["Z"][step_index]

    # Create a new figure
    fig = plt.figure(figsize=(10, 10))

    # Number of subplots
    # Example usage
    histo_types = ["density", "beta_x", "partial_density_x", "partial_density_z", "partial_beta_x"]
    n_histos = len(histo_types)

    # Define the number of rows and columns for subplots
    rows = (n_histos + 1) // 2
    cols = 2

    for i, histo_type in enumerate(histo_types):
        # Add a 3D subplot
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')

        # Plot the first surface
        surface1 = ax.plot_surface(Z_1, X_1, histo_data_old[histo_type][step_index], color="red", alpha=0.5, label=label1)
        
        # Plot the second surface
        surface2 = ax.plot_surface(Z_2, X_2, histo_data_new[histo_type][step_index], color='blue', alpha=0.5, label=label2)

        # Set axis labels
        ax.set_xlabel('Z axis')
        ax.set_ylabel('X axis')
        ax.set_zlabel(histo_type)

        # Set the title of the subplot
        ax.set_title(histo_type)

        # Adjust x and y axis limits to ensure they cover the same range
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()

        # Determine the maximum range to apply equal scaling
        max_range = max(x_limits[1] - x_limits[0], y_limits[1] - y_limits[0])

        ax.set_xlim([x_limits[0], x_limits[0] + max_range])
        ax.set_ylim([y_limits[0], y_limits[0] + max_range])

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the legend
    plt.legend()

    # Show the plot
    plt.show()

def plot_3histos_together(histo_data1, histo_data2, histo_data3, step_index, histo_type, label1="", label2="", label3=""):

    X_1 = histo_data1["X"][step_index]
    Z_1 = histo_data1["Z"][step_index]

    X_2 = histo_data2["X"][step_index]
    Z_2 = histo_data2["Z"][step_index]

    X_3 = histo_data3["X"][step_index]
    Z_3 = histo_data3["Z"][step_index]

    # Create a new figure
    fig = plt.figure()

    # Add a 3D subplot
    ax = fig.add_subplot(111, projection='3d')

    # Plot the first surface
    surface1 = ax.plot_surface(Z_1, X_1, histo_data1[histo_type][step_index], color='red', alpha=0.4, label=label1)

    # Plot the second surface
    surface2 = ax.plot_surface(Z_2, X_2, histo_data2[histo_type][step_index], color='blue', alpha=0.4, label=label2)

    surface3 = ax.plot_surface(Z_3, X_3, histo_data3[histo_type][step_index], color='green', alpha=0.4, label=label3)

    # Set axis labels
    ax.set_xlabel('Z axis')
    ax.set_ylabel('X axis')
    ax.set_zlabel('')

    # Adjust x and y axis limits to ensure they cover the same range
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()

    # Determine the maximum range to apply equal scaling
    max_range = max(x_limits[1] - x_limits[0], y_limits[1] - y_limits[0])

    ax.set_xlim([x_limits[0], x_limits[0] + max_range])
    ax.set_ylim([y_limits[0], y_limits[0] + max_range])
    
    plt.legend()

    plt.show()


def create_beam_gif(Xs, Zs, intensity_vals):
    """
    Creates an gif showing how the beam propagates with time
    """
    fig, ax = plt.subplots()
    Z = Zs[0]
    X = Xs[0]
    
    pcm = ax.pcolormesh(Z, X, intensity_vals[0], shading='auto', cmap='viridis')
    cb = plt.colorbar(pcm, ax=ax, label='Values')
    plt.axis("equal")
    plt.xlabel("z")
    plt.ylabel("x")

    animate = lambda frame: update(frame, cb, ax, Xs, Zs, intensity_vals)

    ani = animation.FuncAnimation(fig, animate, frames=intensity_vals.shape[0], interval=200, blit=True)

    # Save the animation as a GIF file
    ani.save('dipole animation.gif', writer='imagemagick', fps=2)

    """
    mesh_coords_list = [None]*len(step_snapshots)
    density_list = [None]*len(step_snapshots)
    for index, step_snapshot in enumerate(step_snapshots):
        mesh_coords_list[index] = step_snapshot.mesh_coords
        density_list[index] = step_snapshot.density
    """
    
# Function to update the plot for each frame
def update(frame, cb, ax, Xs, Zs, intensity_vals):
    ax.clear()  # Clear the previous plot

    intensity_val = intensity_vals[frame]

    pcm = ax.pcolormesh(Zs[frame], Xs[frame], intensity_val, shading='auto', cmap='viridis')
    
    cb.update_normal(pcm)
    plt.axis("equal")
    plt.title("beam distribution at step "+str(frame))
    plt.xlabel("z")
    plt.ylabel("x")

    return pcm,
