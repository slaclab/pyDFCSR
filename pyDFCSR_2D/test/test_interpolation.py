from pyDFCSR_2D.interp3D import  TrilinearInterpolator
import numpy as np
from scipy.interpolate import RegularGridInterpolator
def f(x, y, z):
    return 2 * x**3 + 3 * y**2 - z

# input points
x = np.linspace(1, 4, 11)
y = np.linspace(4, 7, 22)
z = np.linspace(7, 9, 33)
xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij')
data = f(xg, yg, zg)

m = 1000000
xval = np.linspace(1, 4,m)
yval = np.linspace(4, 7, m)
zval = np.linspace(7, 9, m)

def test1(n = 100):
    interp = RegularGridInterpolator((x, y, z), data)
    for _ in range(n):

        result1 = interp((xval, yval, zval))



def test2(n = 100):
    t2 = TrilinearInterpolator(data, x, y, z)
    for _ in range(n):
        result2 = t2.interp(xval, yval, zval)


import time
n = 600
#t0 = time.time()
#test1(n)
#print('test1', time.time() - t0)

t0 = time.time()
test2(n)
print('test2', time.time() - t0)

t0 = time.time()
test2(n)
print('test2', time.time() - t0)

