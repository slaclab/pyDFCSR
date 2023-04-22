import numpy as np
def twiss_R(R, beta0, alpha0):
    """
    Returns Beta and Alpha and Gamma after a transport through a 2*2 R-matrix from point 0 to point 1.
    The initial Beta0 and Aplha0 are required)
    Translated from P. Emma's matlab codes
    :param R: 2*2 transfer matrix
    :param beta0: initial beta
    :param alpha0: initla alpha
    :return: beta1, alhpa1, gamma1 after transport
    """
    assert R.shape == (2,2),"x must be a 2*2 matrix"

    R11 = R[0,0]
    R12 = R[0,1]
    R21 = R[1,0]
    R22 = R[1,1]

    M = np.array([[R11**2,      -2*R11*R12,       R12**2],
                  [-R11*R21,    1 + 2*R12*R21,   -R12*R22],
                  [R21**2,      -2*R21*R22,       R22**2]])

    gamma0 = (1 + alpha0**2)/beta0

    v0 = np.array([beta0, alpha0, gamma0]).T

    v1 = np.matmul(M, v0)

    return v1[0], v1[1], v1[2]