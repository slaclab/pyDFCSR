import numpy as np

def get_referece_traj(lattice_config, Nsample = 100, Ndim = 2):
    """
    A function to get the reference trajectory of partices with given lattice configuration
    :param lattice_config: dictionary
           Nsample: the number of points in each element to calculate the traj
           Ndim:  the dimension of the trajectory
    :return:
      s : longitudindal coordinates (Nsamle,)
      coords :  coordinate of the trajectory, array (Nsample, Ndim). coord[:,0] = x, coord[:,1] = y
      tau_vec, n_vec: tangential and normal vectors along the trajectory, array (Nsamp, Ndim).
                        tau_vec[:, 0] x component. tau_vec[:, 1] y component
      rho: bending radius of the trajectory, array (Nsample,)
      distance: (Nelement,): distance[i] is the distance from the lattice entrance to the end of ith element
    """
    Nelement = len(lattice_config)
    distance = np.zeros(Nelement)        # distance[i] is the distance of the end of ith element
    s = np.zeros(Nelement*Nsample + 1)
    count = 0
    for key in lattice_config.keys():
        current_element = lattice_config[key]
        L = current_element['L']
        if count == 0:
            distance[count] = L
            s[(count - 1)*Nsample: count*Nsample] = np.linspace(0, L, Nsample + 1)[:-1]
        else:
            distance[count] = L + distance[count - 1]
            # this is to make sure that the edge of each elements is sampled in s
            s[(count - 1) * Nsample: count * Nsample] = np.linspace(distance[count - 1], distance[count], Nsample + 1)[:-1]
        count += 1
    s[-1] = distance[-1]

    N_elements = len(distance)   # number of elements in the lattice
    L_lattice = distance[-1]     # total length of the lattice
    s = np.linspace(0, L_lattice, Nsample)                     # coordinates along s to calculate the reference traj
    interval = np.mean(np.diff(s))
    coords = np.zeros((Nsample, Ndim))
    tau_vec = np.zeros((Nsample, Ndim))
    n_vec = np.zeros((Nsample, Ndim))
    rho = np.zeros(Nsample)

    element_infer = 0      # pointer to which element we are in
    count = 1
    theta_0 = 0  # te angle between the traj tangential and x axis
    for st in s[1:]:
        if st > distance[element_infer]:
            element_infer += 1

        delta_s = s[count] - s[count - 1]
        ele_name = 'element_' + str(count)


        # for dipole
        if lattice_config[ele_name] == 'dipole':
            L = lattice_config[ele_name]['L']
            angle = lattice_config[ele_name]['angle']
            phi = delta_s/L*angle
            r = L/angle

            # seems to be stupid. Todo
            x0 = coords[count - 1, 0] - r*np.sin(theta_0)
            y0 = coords[count - 1, 1] + r*np.cos(theta_0)

            coords[count, 0] = x0 + r*np.sin(phi + theta_0)
            coords[count, 1] = y0 - r*np.cos(phi + theta_0)
            rho[count]  = angle/L

        # for drift, quad, sext
        else:
            phi = 0
            coords[count, 0] = coords[count - 1, 0] + delta_s *np.cos(theta_0)
            coords[count, 1] = coords[count - 1, 1] + delta_s *np.sin(theta_0)
            rho[count] = 0

        # Todo: check other elements (quad, sextupole)

        theta_0 += phi
        count += 1

        tau_vec[count, 0] = np.cos(theta_0)
        tau_vec[count, 1] = np.sin(theta_0)

        # Todo: check the sign of n_vec
        n_vec[count, 0] = np.sin(theta_0)
        n_vec[count, 1] = -1*np.cos(theta_0)

    return s, coords, tau_vec, n_vec, rho, distance





