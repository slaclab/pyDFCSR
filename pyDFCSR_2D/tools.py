import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import datetime
def full_path(path):
    """
    From C. Mayes
    Helper function to expand enviromental variables and return the absolute path
    """
    return os.path.abspath(os.path.expandvars(path))

"""UTC to ISO 8601 with Local TimeZone information without microsecond"""
def isotime():
    """
    From C. Mayes.
    Get time stamp for filename
    :return:
    """
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).astimezone().replace(microsecond=0).isoformat().replace(':','_')

def plot_surface(x, y, z):
    """
    :param x: 1D x grid (Nx,)
    :param y: 1D y grid (Ny, )
    :param z: 2D z data (Nx, Ny)
    :return:
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X, Y = np.meshgrid(x, y)


    # Plot the surface.
    surf = ax.plot_surface(X, Y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def find_nearest_ind(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx