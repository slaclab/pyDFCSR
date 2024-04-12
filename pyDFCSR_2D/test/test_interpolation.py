from pyDFCSR_2D.interp3D import  TrilinearInterpolator, interpolate3D,interpolate_3d_vectorized
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import time
def f(x, y, z):
    return 2 * x**3 + 3 * y**2 - z

# input points
x = np.linspace(1, 4, 200)
y = np.linspace(4, 7, 200)
z = np.linspace(7, 9, 100)
xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij')
data = f(xg, yg, zg)

m = 40000
xval = np.linspace(1, 4,m)
yval = np.linspace(4, 7, m)
zval = np.linspace(7, 9, m)

n = 1500

#interp = RegularGridInterpolator((x, y, z), data)

#t0 = time.time()
#for _ in range(n):
#    result1 = interp((xval, yval, zval))
#print('RegularGridInterpolator', time.time() - t0)


t2 = TrilinearInterpolator(data, x, y, z)

print("Starting")
t0 = time.time()
for _ in range(n):
    result2 = t2.interp(xval, yval, zval)
print('TrilinearInterpolator', time.time() - t0)

t0 = time.time()
for _ in range(n):
    result2 = t2.interp(xval, yval, zval)
print('TrilinearInterpolator', time.time() - t0)

t0 = time.time()
for _ in range(n):
    result2 = t2.interp(xval, yval, zval)
print('TrilinearInterpolator', time.time() - t0)

#t0 = time.time()

#n = 1500
#for _ in range(n):
#    interpolate_3d_vectorized(data, xval, yval, zval, min(x), min(y), min(z), np.mean(np.diff(x)), np.mean(np.diff(y)), np.mean(np.diff(z)))
#print('interpolate_3d_vectorized', time.time() - t0)


#t0 = time.time()

#n = 1500
#for _ in range(n):
#    interpolate3D(xval, yval, zval, data, min(x), min(y), min(z), np.mean(np.diff(x)), np.mean(np.diff(y)), np.mean(np.diff(z)))
#print('interpolate3D', time.time() - t0)
#
#t0 = time.time()
#
#n = 1500
#for _ in range(n):
#    interpolate3D(xval, yval, zval, data, min(x), min(y), min(z), np.mean(np.diff(x)), np.mean(np.diff(y)), np.mean(np.diff(z)))
#print('interpolate3D', time.time() - t0)
#

t0 = time.time()

n = 1500
for _ in range(n):
    interpolate3D(xval, yval, zval, data, min(x), min(y), min(z), np.mean(np.diff(x)), np.mean(np.diff(y)), np.mean(np.diff(z)))
print('interpolate3D', time.time() - t0)




