import numpy as np
from numba import jit
from numba import prange
import math
from line_profiler import profile

"""
Module Name: interpolation.py

This file contains the interpolation functions which allow for interpolation between mesh slices
"""

@jit(nopython = True,  cache = True)
def interpolate1D(xval, data, min_x, delta_x):
    """
    Quick interpolator for 1D values
    Parameters:
        xvals: np array of positions to interpolate values at
        data: the value of the function at each slice
        min_x: the smallest slice position
        delta_x: the change in position between slices
    """
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

@jit(nopython = True,  cache = True)
def translate_points(ret_tvals, svals, xvals, t1_h, C_inv_h, R_inv_h, t2_h, step_ranges, p_indices, translated_points):
    """
    A bit complicated... TODO: write this docstring
    Parameters:
        ret_tvals, svals, xvals: arrays containing the retarded time, s, and x coordinates at which to interpolate
        t1, C_inv, R_inv, t2: matrix transformation arrays for each step
        step_ranges: [(start_of_step_i, start_of_step_i+1), (start_of_step_i+1, start_of_step_i+2), ...]
        p_indices: literally a 1D array counting the indices of each point in the list (so np.arrange(0, len(svals)))
        translated_points: an array full of zeros which is to be populated and returned
    """
    # Stack the points and their indices, this method is faster than np.column_stack
    num_points = len(ret_tvals)
    points = np.zeros((num_points, 4))
    points[:, 0] = ret_tvals
    points[:, 1] = svals
    points[:, 2] = xvals
    points[:, 3] = p_indices

    # Initialize lists for groups
    point_groups = []

    # We only populate groups that will have a non zero amount of points
    # Compute the step indices of the leftmost and rightmost step
    step_size = step_ranges[0][1] - step_ranges[0][0]
    lower_index_bound = int(np.min(ret_tvals)/step_size)
    upper_index_bound = math.ceil(np.max(ret_tvals)/step_size)

    # Compute the maximum step and then remove the uneeded ranges from step_ranges
    max_index = len(step_ranges)
    step_ranges = step_ranges[lower_index_bound:upper_index_bound+1] # May not need the +1

    # Populate all the groups
    for i, (low, high) in enumerate(step_ranges):
        if i == max_index - 1:
            # For the last range, include points equal to the highest value
            mask = (points[:, 0] >= low) & (points[:, 0] <= high)
        else:
            mask = (points[:, 0] >= low) & (points[:, 0] < high)
        point_groups.append(points[mask])

    for i, group in enumerate(point_groups):
        # Compute the true step index
        step_index = i + lower_index_bound

        if step_index != upper_index_bound:
            # Coordinate transfer all s,x points in each group, group[:, 1:3] is all points in the group, s and x coordinates
            # step_index is the index to the left step of the group
            trans_left = (C_inv_h[step_index] @ (R_inv_h[step_index] @ (group[:,1:3].T - t2_h[step_index][:, np.newaxis]))) + t1_h[step_index][:, np.newaxis]
            trans_right = (C_inv_h[step_index+1] @ (R_inv_h[step_index+1] @ (group[:,1:3].T - t2_h[step_index+1][:, np.newaxis]))) + t1_h[step_index+1][:, np.newaxis]
        
        # In the case that the step_index is equal to the upper index bound, the matrices at step_index+1 have not yet been populated
        # Here we are right on the border, so we just need to make sure that we are not indexing non populated matrices
        else:
            trans_left = (C_inv_h[step_index] @ (R_inv_h[step_index] @ (group[:,1:3].T - t2_h[step_index][:, np.newaxis]))) + t1_h[step_index][:, np.newaxis]
            trans_right = trans_left

        # Loop through each point in the loop and populate the respective translated points value
        for group_index, total_index in enumerate(group[:, 3]):

            total_index = int(total_index)
            translated_points[total_index][0] = group[group_index][0]           # The t_ret time of the point
            translated_points[total_index][1] = trans_left[1][group_index]      # Index position of point in the left step, 1st dimension
            translated_points[total_index][2] = trans_left[0][group_index]      # Index position of point in the left step, 2st dimension
            translated_points[total_index][3] = trans_right[1][group_index]     # Index position of point in the right step, 1st dimension
            translated_points[total_index][4] = trans_right[0][group_index]     # Index position of point in the right step, 2st dimension

    return translated_points

@jit(nopython = True,  cache = True)
def interpolate3D(translated_points, data_list, step_size, result):      
    """
    Interpolates values in between steps
    Parameters:
        translated points: output from translated_points()
        data_list: numpy array of each 3D dataset (ex: density, beta_x, etc)
        step_size: float, the s spacing between all steps
        result: initalized array that will be populated and returned
    """

    # Divide all t_ret values by step_size to normalize them
    translated_points[:,0] = translated_points[:,0]/step_size

    # Get dimension of data
    step_num, obins, pbins = data_list[0].shape[0], data_list[0].shape[1], data_list[0].shape[2]

    # Loop over all points we wish to interpolate
    for point_index in range(len(translated_points)):
        # Index the coordinate space values
        t_ret_normalized = translated_points[point_index][0]
        cl0 = translated_points[point_index][1]
        cl1 = translated_points[point_index][2]
        cr0 = translated_points[point_index][3]
        cr1 = translated_points[point_index][4]

        # Compute the indices
        t_ret_coord_l = int(t_ret_normalized)
        if t_ret_coord_l == step_num - 1:
            t_ret_coord_r = t_ret_coord_l
        else:
            t_ret_coord_r = t_ret_coord_l + 1

        cl0_index_small = int(cl0)
        if cl0_index_small == obins - 1:
            cl0_index_large = cl0_index_small
        else:
            cl0_index_large = cl0_index_small + 1

        cl1_index_small = int(cl1)
        if cl1_index_small == pbins - 1:
            cl1_index_large = cl1_index_small
        else:
            cl1_index_large = cl1_index_small + 1

        cr0_index_small = int(cr0)
        if cr0_index_small == obins - 1:
            cr0_index_large = cr0_index_small
        else:
            cr0_index_large = cr0_index_small + 1

        cr1_index_small = int(cr1)
        if cr1_index_small == pbins - 1:
            cr1_index_large = cr1_index_small
        else:
            cr1_index_large = cr1_index_small + 1

        in_bounds_left = ((cl0_index_small >= 0) and (cl1_index_small >= 0) and (cl0_index_large <= obins) and (cl1_index_large <= pbins))
        in_bounds_right = ((cr0_index_small >= 0) and (cr1_index_small >= 0) and (cr0_index_large <= obins) and (cr1_index_large <= pbins))

        if (in_bounds_left and in_bounds_right):
            # Interpolate all data
            for data_index, data in enumerate(data_list):
                # Left step
                wl0 = data[t_ret_coord_l][cl0_index_small][cl1_index_small] * (1-(cl0-cl0_index_small)) + data[t_ret_coord_l][cl0_index_large][cl1_index_small] * (cl0-cl0_index_small)
                wl1 = data[t_ret_coord_l][cl0_index_small][cl1_index_large] * (1-(cl0-cl0_index_small)) + data[t_ret_coord_l][cl0_index_large][cl1_index_large] * (cl0-cl0_index_small)
                wl = (wl0 * (1 - (cl1-cl1_index_small))) + (wl1 * (cl1-cl1_index_small))

                # Right step
                wr0 = data[t_ret_coord_r][cr0_index_small][cr1_index_small] * (1-(cr0-cr0_index_small)) + data[t_ret_coord_r][cr0_index_large][cr1_index_small] * (cr0-cr0_index_small)
                wr1 = data[t_ret_coord_r][cr0_index_small][cr1_index_large] * (1-(cr0-cr0_index_small)) + data[t_ret_coord_r][cr0_index_large][cr1_index_large] * (cr0-cr0_index_small)
                wr = (wr0 * (1 - (cr1-cr1_index_small))) + (wr1 * (cr1-cr1_index_small))

                # Right vs left step
                cl = (t_ret_normalized - t_ret_coord_l)

                # Both steps together to compute the ith value of result
                wi = (wl * (1-cl)) + (wr * cl)

                result[data_index][point_index] = wi

        elif in_bounds_left and not in_bounds_right:
            # Interpolate all data
            for data_index, data in enumerate(data_list):
                # Left step
                wl0 = data[t_ret_coord_l][cl0_index_small][cl1_index_small] * (1-(cl0-cl0_index_small)) + data[t_ret_coord_l][cl0_index_large][cl1_index_small] * (cl0-cl0_index_small)
                wl1 = data[t_ret_coord_l][cl0_index_small][cl1_index_large] * (1-(cl0-cl0_index_small)) + data[t_ret_coord_l][cl0_index_large][cl1_index_large] * (cl0-cl0_index_small)
                wl = (wl0 * (1 - (cl1-cl1_index_small))) + (wl1 * (cl1-cl1_index_small))

                result[:,point_index] = wl

        elif in_bounds_right and not in_bounds_left:
            # Interpolate all data
            for data_index, data in enumerate(data_list):
                # Right step
                wr0 = data[t_ret_coord_r][cr0_index_small][cr1_index_small] * (1-(cr0-cr0_index_small)) + data[t_ret_coord_r][cr0_index_large][cr1_index_small] * (cr0-cr0_index_small)
                wr1 = data[t_ret_coord_r][cr0_index_small][cr1_index_large] * (1-(cr0-cr0_index_small)) + data[t_ret_coord_r][cr0_index_large][cr1_index_large] * (cr0-cr0_index_small)
                wr = (wr0 * (1 - (cr1-cr1_index_small))) + (wr1 * (cr1-cr1_index_small))

                result[:,point_index] = wr

        else:
            result[:,point_index] = 0.0

    return result

