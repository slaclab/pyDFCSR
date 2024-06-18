import os
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import cm
import datetime
import numpy as np
from scipy.interpolate import griddata
import h5py

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

def plot_surface(x, y, z, title = 'None'):
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

def plot_2D_contour(x, y, z, title = None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    surf = ax.imshow(z, extent=(np.min(x)*1e6, np.max(x)*1e6, np.min(y)*1e6, np.max(y)*1e6), origin='lower',  cmap='seismic')

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel('y ($\mu m$)')
    plt.ylabel('x ($\mu m$)')
    if title:
        plt.title(title)
    plt.show()
    plt.close()

def find_nearest_ind(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def dict2hdf5(hf, dic, group=None):
    for key, item in dic.items():
        if not isinstance(item, dict):
            if group:
                group.create_dataset(key, data=item)
            else:
                hf.create_dataset(key, data=item)
        else:
            dict2hdf5(hf, item, hf.create_group(key))


