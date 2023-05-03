import numpy as np
from numba import jit
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
    v_c = data
    for i in prange(len(xval)):
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
            c00 = v_c[x0, y0, z0] * (1 - xd) + v_c[x1, y0, z0] * xd
            c01 = v_c[x0, y0, z1] * (1 - xd) + v_c[x1, y0, z1] * xd
            c10 = v_c[x0, y1, z0] * (1 - xd) + v_c[x1, y1, z0] * xd
            c11 = v_c[x0, y1, z1] * (1 - xd) + v_c[x1, y1, z1] * xd

            c0 = c00 * (1 - yd) + c10 * yd
            c1 = c01 * (1 - yd) + c11 * yd

            result[i] = c0 * (1 - zd) + c1 * zd

        else:
            result[i] = 0

    return result


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