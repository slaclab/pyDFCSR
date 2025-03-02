import numpy as np
import matplotlib.pyplot as plt


def plot_wakes(wake_data):
    X = wake_data["X"]
    Z = wake_data["Z"]
    dE = wake_data["dE"]
    x_kick = wake_data["x_kick"]

    # Create a new figure
    fig = plt.figure(figsize=(12, 6))

    # Add first 3D subplot
    ax1 = fig.add_subplot(121, projection='3d')
    surface1 = ax1.plot_surface(Z, X, dE, cmap='viridis')
    fig.colorbar(surface1, ax=ax1, shrink=0.5, aspect=5)
    ax1.set_title('dE')
    ax1.set_xlabel('Z axis')
    ax1.set_ylabel('X axis')
    ax1.set_zlabel('dE')

    # Add second 3D subplot
    ax2 = fig.add_subplot(122, projection='3d')
    surface2 = ax2.plot_surface(Z, X, x_kick, cmap='plasma')
    fig.colorbar(surface2, ax=ax2, shrink=0.5, aspect=5)
    ax2.set_title('x_kick')
    ax2.set_xlabel('Z axis')
    ax2.set_ylabel('X axis')
    ax2.set_zlabel('x_kick')

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_histos(histo_data):
    X = histo_data["X"]
    Z = histo_data["Z"]
    histos = [histo_data["density"], histo_data["beta_x"], histo_data["partial_density_x"],
              histo_data["partial_density_z"], histo_data["partial_beta_x"]]
    
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
        ax.set_xlabel('X axis')
        ax.set_ylabel('Z axis')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_histos_together(histo_data1, histo_data2):
    # Extract X and Z for both histo_data sets
    X1, Z1 = histo_data1["X"], histo_data1["Z"]
    X2, Z2 = histo_data2["X"], histo_data2["Z"]

    # Extract histograms for both datasets
    histos1 = [histo_data1["density"], histo_data1["beta_x"], histo_data1["partial_density_x"],
               histo_data1["partial_density_z"], histo_data1["partial_beta_x"]]
    histos2 = [histo_data2["density"], histo_data2["beta_x"], histo_data2["partial_density_x"],
               histo_data2["partial_density_z"], histo_data2["partial_beta_x"]]

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
        surface1 = ax.plot_surface(Z1, X1, histos1[i], cmap='plasma', edgecolor='none', alpha=0.7)
        fig.colorbar(surface1, ax=ax, shrink=0.5, aspect=5)

        # Plot second dataset's surface
        surface2 = ax.plot_surface(Z2, X2, histos2[i], cmap='viridis', edgecolor='none', alpha=0.7)
        fig.colorbar(surface2, ax=ax, shrink=0.5, aspect=5)

        ax.set_title(titles[i])
        ax.set_xlabel('X axis')
        ax.set_ylabel('Z axis')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_densities(old_histo_data, new_histo_data):
    X_new = new_histo_data["X"]
    Z_new = new_histo_data["Z"]
    density_new = new_histo_data["density"]

    X_old = old_histo_data["X"]
    Z_old = old_histo_data["Z"]
    density_old = old_histo_data["density"]

    # Create a new figure
    fig = plt.figure()

    # Add a 3D subplot
    ax = fig.add_subplot(111, projection='3d')

    # Plot the first surface
    surface1 = ax.plot_surface(Z_new, X_new, density_new, color='red', alpha=0.4, label='density new')

    # Plot the second surface
    #surface2 = ax.plot_surface(Z_old, X_old, density_old, color='blue', alpha=0.4, label='density old')

    # Set axis labels
    ax.set_xlabel('Z axis')
    ax.set_ylabel('X axis')
    ax.set_zlabel('Density')
    plt.legend()

    plt.show()