import numpy as np
from numba import jit
from numba.experimental import jitclass
from numba import double
from numba import prange
spec = [
    ('min_x', double),
    ('max_x', double),
    ('delta_x', double),
    ('data', double[:]),          # an array field
]

@jit(nopython = True,  cache = True)
def interpolate1D(xval, data, min_x, delta_x):
    result = np.zeros(xval.shape)
    x_size = data.shape[0]
    xval = (xval - min_x) / delta_x
    v_c = data
    for i in prange(len(xval)):
        x = xval[i]
        x0 = int(x)
        if x0 == x_size - 1:
            x1 = x0
        else:
            x1 = x0 + 1


        xd = x - x0


        if x0 >= 0 and x1 < x_size:
            result[i] = v_c[x0] * (1 - xd) + v_c[x1] * xd
        else:
            result[i] = 0

    return result


@jitclass(spec)
class LinearInterpolator:
    def __init__(self, data, x):
        self.data = data
        self.min_x, self.max_x = x[0], x[-1]

        self.delta_x = (self.max_x - self.min_x) / (x.shape[0] - 1)


    def interp(self, xval):
        return interpolate1D(xval, self.data, self.min_x, self.delta_x)