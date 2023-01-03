import numpy as np


def r_gen6(L, angle, k1=0, roll=0, E1=0, E2=0, hgap=0):
    """
    Return a general 6*6 R matrix. Translated from Paul Emma's matlab codes.
    :param L:  magnetic length [meters]. If L = 0, return a rotation matrix defined by roll and all other params are ignored.
    :param angle: bending angle [rads]
    :param k1: (opt, DEF = 0) quad strength [1/meter^2]
    :param roll: (opt, DEF=0) roll angle around longitudinal axis [rads].
                 roll > 0   is clockwise bout positibe Z-axis. i.e. clockwise next element rotation as beam leaves the observer.
                 (NOTE: if L=0, then R = rotation through 'roll' angle)
    :param E1:  (opt,DEF = 0) Pole-face rotation at bend entrance [rads]
    :param E2:  (opt, DEF = 0) Pole-face rotation at bend exit [rads]
    :param hgap: (opt, DEF= 0) Vertical half-gap of bend magnet used for fringe field vertical focusing (uses K1 = 0.5 - see MAD manual)[m]
    :return: 6*6 R matrix (x, x', y, y', z, dp/p)
    e.g.
    drift:                  r_gen6(L = L, angle = 0)
    pure sector dipole:     r_gen6(L = L, angle = angle)
    rectangular bend:       r_gen6(L = L , angle = angle, E1 = angle/2, E2 = angle/2)
    pure quadrupole:        r_gen6(L = L, angle = 0, k1 = k1)
    skew quadurpole:        r_gen6(L = L, angle = 0, k = k1, roll = np.pi/4)
    combo function magnet   r_gen6(L = L, angle = angle, k1 = k1)
    coords rotation matrix  r_gen6(L = 0, angle = 0, roll = roll)
    rotated combo func mag (like in SLC ars) r_gen6(L = L, angle = angle, k1 = k1, roll = roll)
    have been benchmared against matlab version
    """

    if roll:
        c = np.cos(-roll)
        s = np.sin(-roll)
        O = np.array([[c, 0, s, 0, 0, 0],
                      [0, c, 0, s, 0, 0],
                      [-s,0, c, 0, 0, 0],
                      [0,-s, 0, c, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])
    else:
        O = np.eye(6, 6)

    if not L:
        return O

    h = angle/L
    kx2 = (k1 + h**2)
    ky2 = -k1

    if angle:
        psi1 = 0.50*2*h*hgap*(1 + np.sin(E1)**2)/np.cos(E1)
        psi2 = 0.50*2*h*hgap*(1 + np.sin(E2)**2)/np.cos(E2)
        Rpr1 = np.eye(6, 6)
        Rpr2 = np.eye(6, 6)
        Rpr1[1, 0] = np.tan(E1)*h
        Rpr2[1, 0] = np.tan(E2)*h
        Rpr1[3, 2] = -np.tan(E1 - psi1)*h
        Rpr2[3, 2] = -np.tan(E2 - psi2)*h   # seems to be a typo on Paul's codes?

    # Horizontal plane
    kx = np.sqrt(np.abs(kx2))
    phix = kx*L

    if np.abs(phix) < 1.0e-12:
        Rx = np.array([[1, L],
              [0, 1]])
        Dx = np.zeros((2, 2))
        R56 = 0

    else:
        if kx2 > 0:
            co = np.cos(phix)
            si = np.sin(phix)
            Rx = np.array([[co, si/kx],
                           [-kx*si, co]])

        else:
            co = np.cosh(phix)
            si = np.sinh(phix)
            Rx = np.array([[co, si/kx],
                           [kx*si, co]])

        Dx = np.array([[0, h*(1 - co)/kx2],
                       [0, h*si/kx]])
        R56 = -(h**2)*(phix - kx*Rx[0,1])/kx**3


    # vertical plane
    ky = np.sqrt(np.abs(ky2))
    phiy = ky * L

    if np.abs(phiy) < 1.0e-12:
        Ry = np.array([[1, L],
                       [0, 1]])

    else:
        if ky2 > 0:
            co = np.cos(phiy)
            si = np.sin(phiy)
            Ry = np.array([[co, si / ky],
                           [-ky * si, co]])

        else:
            co = np.cosh(phiy)
            si = np.sinh(phiy)
            Ry = np.array([[co, si / ky],
                           [ky * si, co]])

    R = np.zeros((6, 6))
    R[0:2, 0:2] = Rx
    R[2:4, 2:4] = Ry
    R[0:2, 4:6] = Dx
    R[4, 0] = -Dx[1, 1]
    R[4, 1] = -Dx[0,1]
    R[4, 4] = 1
    R[4, 5] = R56
    R[5, 5] = 1

    R = np.matmul(O, np.matmul(R, O.T))

    if angle:
        R = np.matmul(Rpr2, np.matmul(R, Rpr1))

    return R







