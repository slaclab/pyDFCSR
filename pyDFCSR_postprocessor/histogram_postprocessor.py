import h5py
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Function to load data from the "new" format
def load_new_histo_data(file_path_histos):
    # Load the data from the HDF5 file
    with h5py.File(file_path_histos, 'r') as f:
        groupname = "histogram_data"
        g = f[groupname]

        # Load the datasets into a dictionary
        h_coords = g["histo_mesh_coords"][:]

        X = h_coords[:, :, :, 0]  # Extract X from the stacked h_coords
        Z = h_coords[:, :, :, 1]  # Extract Z from the stacked h_coords
        
        density = g["density"][:]
        beta_x = g["beta_x"][:]
        partial_density_x = g["partial_density_x"][:]
        partial_density_z = g["partial_density_z"][:]
        partial_beta_x = g["partial_beta_x"][:]

        step_num = density.shape[0]

        histo_data = []
        for i in range(step_num):
            histo_data.append({
                "X": X[i],
                "Z": Z[i],
                "density": density[i],
                "beta_x": beta_x[i],
                "partial_density_x": partial_density_x[i],
                "partial_density_z": partial_density_z[i],
                "partial_beta_x": partial_beta_x[i]
            })
    
    return histo_data

# Function to load data from the "old" format (multi-step)
def load_old_histo_data(file_path_histos):
    # Dictionary to store data for all steps
    histo_data = {}

    # Load the data from the HDF5 file
    with h5py.File(file_path_histos, 'r') as f:
        # Iterate through all groups in the file (assuming each group is named 'step_X')
        for groupname in f.keys():
            g = f[groupname]
            
            # Extract the step index from the group's attributes
            step_index = g.attrs['step_index']
            
            # Load the datasets
            h_coords = g['histo_mesh_coords'][:]
            X = h_coords[:, :, 0]  # Extract X from the stacked h_coords
            Z = h_coords[:, :, 1]  # Extract Z from the stacked h_coords
            
            # Store data for this step in the dictionary
            histo_data[step_index] = {
                "X": X,
                "Z": Z,
                "density": g['density'][:],
                "beta_x": g['beta_x'][:],
                "partial_density_x": g['partial_density_x'][:],
                "partial_density_z": g['partial_density_z'][:],
                "partial_beta_x": g['partial_beta_x'][:]
            }

    return histo_data

def plot_histos(histo_data):
    X = histo_data["X"]
    Z = histo_data["Z"]

    histos = [histo_data["density"], histo_data["beta_x"], histo_data["partial_density_x"],
              histo_data["partial_density_z"], histo_data["partial_beta_x"]]
    
    titles = ["density", "beta_x", "partial_density_x",
              "partial_density_z", "partial_beta_x"]

    # To view unrotated and uncompressed mesh
    Z_values = np.arange(0,histos[0].shape[0])
    X_values = np.arange(0,histos[0].shape[1])
    Z, X = np.meshgrid(Z_values, X_values)

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
        ax.set_xlabel('X axis')
        ax.set_ylabel('Z axis')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_one_histo_together(histo_data1, histo_data2, histo_type, step_index, title="", label1="", label2=""):
    step_data1 = histo_data1[step_index]
    step_data2 = histo_data2[step_index]

    # Extract X and Z for both histo_data sets
    X1, Z1 = step_data1["X"], step_data1["Z"]
    X2, Z2 = step_data2["X"], step_data2["Z"]

    # Extract histograms for both datasets
    histo1 = step_data1[histo_type]
    histo2 = step_data2[histo_type]

    # Create a new figure
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Plot first dataset's surface
    surface1 = ax.plot_surface(Z1, X1, histo1, color='red', alpha=0.3, label=label1)

    # Plot second dataset's surface
    surface2 = ax.plot_surface(Z2, X2, histo2, color='blue', alpha=0.3, label=label2)

    # Set axis labels with microns (µm)
    ax.set_xlabel(r'$x$ axis ($\mu m$)', fontsize=10, labelpad=10)
    ax.set_ylabel(r'$z$ axis ($\mu m$)', fontsize=10, labelpad=10)
    ax.set_zlabel(r'Density ($\mu \text{m}^{-3}$)', fontsize=10, labelpad=10)

    # Create custom legend handles
    legend_elements = [
        Line2D([0], [0], color='red', lw=4, label=label1),
        Line2D([0], [0], color='blue', lw=4, label=label2)
    ]
    ax.legend(handles=legend_elements, loc="best")
    ax.set_title(title, fontsize=10)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_one_histo_together2(histo_data1, histo_data2, histo_type, step_index, title="", label1="", label2=""):
    step_data1 = histo_data1[step_index]
    step_data2 = histo_data2[step_index]

    # Extract X and Z for both histo_data sets
    X1, Z1 = step_data1["X"], step_data1["Z"]
    X2, Z2 = step_data2["X"], step_data2["Z"]

    # Extract histograms for both datasets
    histo1 = step_data1[histo_type]
    histo2 = step_data2[histo_type]

    print(f"Z1 shape: {np.shape(Z1)}, X1 shape: {np.shape(X1)}, histo1 shape: {np.shape(histo1)}")
    print(f"Z2 shape: {np.shape(Z2)}, X2 shape: {np.shape(X2)}, histo2 shape: {np.shape(histo2)}")

    # Create a new figure with high DPI for publication quality
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot first dataset's surface
    surface1 = ax.plot_surface(Z1, X1, histo1, alpha=0.5, color="green")

    # Plot second dataset's surface
    surface2 = ax.plot_surface(Z2, X2, histo2, alpha=0.5, color="orange")

    # Set axis labels with microns (µm)
    ax.set_xlabel(r'$x$ axis ($\mu m$)', fontsize=14, labelpad=10)
    ax.set_ylabel(r'$z$ axis ($\mu m$)', fontsize=14, labelpad=10)
    ax.set_zlabel(histo_type, fontsize=14, labelpad=10)

    # Create custom legend handles
    legend_elements = [
        Line2D([0], [0], color='green', lw=4, label=label1),
        Line2D([0], [0], color='orange', lw=4, label=label2)
    ]
    ax.legend(handles=legend_elements, loc="best")

    # Set the title
    ax.set_title(title, fontsize=16, pad=15)

    # Adjust layout to ensure everything fits
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_all_histos_together(histo_data1, histo_data2, step_index, label1="", label2=""):
    step_data1 = histo_data1[step_index]
    step_data2 = histo_data2[step_index]

    # Extract X and Z for both histo_data sets
    X1, Z1 = step_data1["X"], step_data1["Z"]
    X2, Z2 = step_data2["X"], step_data2["Z"]

    # Extract histograms for both datasets
    histos1 = [step_data1["density"], step_data1["beta_x"], step_data1["partial_density_x"],
               step_data1["partial_density_z"], step_data1["partial_beta_x"]]
    histos2 = [step_data2["density"], step_data2["beta_x"], step_data2["partial_density_x"],
               step_data2["partial_density_z"], step_data2["partial_beta_x"]]

    titles = ["density", "beta_x", "partial_density_x",
              "partial_density_z", "partial_beta_x"]

    # Create a new figure
    fig = plt.figure(figsize=(15, 10))

    # Define the number of rows and columns
    rows, cols = 2, 3

    # Create subplots
    for i in range(5):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')

        # Plot first dataset's surface
        surface1 = ax.plot_surface(Z1, X1, histos1[i], color='red', alpha=0.5, label=label1)

        # Plot second dataset's surface
        surface2 = ax.plot_surface(Z2, X2, histos2[i], color='blue', alpha=0.5, label=label2)

        ax.set_title(titles[i])
        ax.set_xlabel('X axis')
        ax.set_ylabel('Z axis')

        plt.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()
