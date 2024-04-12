import numpy as np
from numba import jit, vectorize, float64
from numba.experimental import jitclass
from numba import int32, float32, double
from numba import njit, prange
spec = [
    ('min_x', double),
    ('min_y', double), # a simple scalar field
    ('min_z', double),
    ('max_x', double),
    ('max_y', double),
    ('max_z', double),
    ('delta_x', double),
    ('delta_y', double),
    ('delta_z', double),
    ('data', double[:, :, :]),          # an array field
]

@jit(nopython = True,  cache = True)
def interpolate3D(xval, yval, zval, data, min_x, min_y, min_z,  delta_x, delta_y, delta_z):
    result = np.zeros(len(xval))
    x_size, y_size, z_size = data.shape[0], data.shape[1], data.shape[2]
    xval = (xval - min_x) / delta_x
    yval = (yval - min_y) / delta_y
    zval = (zval - min_z) / delta_z
    for i in range(len(xval)):
        x = xval[i]
        y = yval[i]
        z = zval[i]

        x0 = int(x)
        if x0 == x_size - 1:
            x1 = x0
        else:
            x1 = x0 + 1

        y0 = int(y)
        if y0 == y_size - 1:
            y1 = y0
        else:
            y1 = y0 + 1

        z0 = int(z)
        if z0 == z_size - 1:
            z1 = z0
        else:
            z1 = z0 + 1

        xd = x - x0
        yd = y - y0
        zd = z - z0

        if x0 >= 0 and y0 >= 0 and z0 >= 0 and x1 < x_size and y1 < y_size and z1 < z_size:
            c00 = data[x0, y0, z0] * (1 - xd) + data[x1, y0, z0] * xd
            c01 = data[x0, y0, z1] * (1 - xd) + data[x1, y0, z1] * xd
            c10 = data[x0, y1, z0] * (1 - xd) + data[x1, y1, z0] * xd
            c11 = data[x0, y1, z1] * (1 - xd) + data[x1, y1, z1] * xd

            c0 = c00 * (1 - yd) + c10 * yd
            c1 = c01 * (1 - yd) + c11 * yd

            result[i] = c0 * (1 - zd) + c1 * zd

        else:
            result[i] = 0

    return result

@jit(nopython = True,  cache = True)
def interpolate_3d_vectorized(data, x, y, z, min_x, min_y, min_z,  delta_x, delta_y, delta_z):
    """
    Perform linear interpolation for multiple points (x, y, z) within a 3D space defined by 'data'.
    Extrapolated values outside the dataset boundaries are set to zero.

    Parameters:
        data (numpy.ndarray): The 3D numpy array containing data values.
        x (numpy.ndarray): The x-coordinates of the interpolation points.
        y (numpy.ndarray): The y-coordinates of the interpolation points.
        z (numpy.ndarray): The z-coordinates of the interpolation points.

    Returns:
        numpy.ndarray: The interpolated values or zero if the points are outside the data boundaries.
    """
    #nx, ny, nz = data.shape
    #interpolated_values = np.zeros(x.shape)

    nx, ny, nz = data.shape
    output = np.zeros_like(x)

    # Clamp coordinates to valid index ranges
    x = np.clip(x, 0, nx - 1)
    y = np.clip(y, 0, ny - 1)
    z = np.clip(z, 0, nz - 1)

    # Convert floating indices to integer indices
    ix = np.floor(x).astype(np.int32)
    iy = np.floor(y).astype(np.int32)
    iz = np.floor(z).astype(np.int32)

    # Calculate linear indices for the corners of the interpolation cube
    i000 = ix + iy * nx + iz * nx * ny
    i001 = ix + iy * nx + (iz + 1) * nx * ny
    i010 = ix + (iy + 1) * nx + iz * nx * ny
    i011 = ix + (iy + 1) * nx + (iz + 1) * nx * ny
    i100 = (ix + 1) + iy * nx + iz * nx * ny
    i101 = (ix + 1) + iy * nx + (iz + 1) * nx * ny
    i110 = (ix + 1) + (iy + 1) * nx + iz * nx * ny
    i111 = (ix + 1) + (iy + 1) * nx + (iz + 1) * nx * ny

    # Flatten the data to use linear indexing
    data_flat = data.ravel()

    # Retrieve values using linear indices
    v000 = data_flat[i000]
    v001 = data_flat[i001]
    v010 = data_flat[i010]
    v011 = data_flat[i011]
    v100 = data_flat[i100]
    v101 = data_flat[i101]
    v110 = data_flat[i110]
    v111 = data_flat[i111]

    # Fractional parts for interpolation
    xd = x - np.floor(x)
    yd = y - np.floor(y)
    zd = z - np.floor(z)

    # Interpolate along z-axis
    c00 = v000 * (1 - zd) + v001 * zd
    c10 = v010 * (1 - zd) + v011 * zd
    c01 = v100 * (1 - zd) + v101 * zd
    c11 = v110 * (1 - zd) + v111 * zd

    # Interpolate along y-axis
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    # Final interpolation along x-axis
    output = c0 * (1 - xd) + c1 * xd

    return output



@jitclass(spec)
class TrilinearInterpolator:
    def __init__(self, data, x, y, z):
        self.data = data
        self.min_x, self.max_x = x[0], x[-1]
        self.min_y, self.max_y = y[0], y[-1]
        self.min_z, self.max_z = z[0], z[-1]
        self.delta_x = (self.max_x - self.min_x) / (x.shape[0] - 1)
        self.delta_y = (self.max_y - self.min_y) / (y.shape[0] - 1)
        self.delta_z = (self.max_z - self.min_z) / (z.shape[0] - 1)

    def interp(self, xval, yval, zval):
        return interpolate3D(xval, yval, zval, self.data,
                           self.min_x, self.min_y, self.min_z,
                           self.delta_x, self.delta_y, self.delta_z)



class TrilinearInterpolator_vec:
    def __init__(self, data, x, y, z):
        self.data = data
        self.min_x, self.max_x = x[0], x[-1]
        self.min_y, self.max_y = y[0], y[-1]
        self.min_z, self.max_z = z[0], z[-1]
        self.delta_x = (self.max_x - self.min_x) / (x.shape[0] - 1)
        self.delta_y = (self.max_y - self.min_y) / (y.shape[0] - 1)
        self.delta_z = (self.max_z - self.min_z) / (z.shape[0] - 1)

    def interp(self, xval, yval, zval):
        return interpolate3D_vec(xval, yval, zval, self.data,
                           self.min_x, self.min_y, self.min_z,
                           self.delta_x, self.delta_y, self.delta_z)