
import numpy as np

from bmadx import  Drift, SBend, Quadrupole, Sextupole

"""
Module Name: create_reference_traj.py

Used by Lattice Class __init__ to create a reference trajectory. A bit hefty so it deserves its own file
"""
   
def get_step_characteristics(lattice_config, element_distances, CSR_step_seperation):
        """
        Calcuates the locations on the reference trajectory (the steps) where the CSR wake will be computed
        Parameters:
            lattice_config: dictionary
            element_distances: the distance from the beginning of the lattice to the end of each element
            CSR_step_seperation: compute CSR at every nspep[i] slice in the ith element
        Retuns:
            The following variables in a dictionary
            step_size: distance between each step
            total_steps: number of steps in the lattice
            step_position: the s val of each step in the lattice
            compute_CSR: len(total_steps), true if we should compute CSR at compute_CSR[step_index], false otherwise
            steps_since_CSR_applied: ith element is how many steps it has been since csr wake has been applied
        """
        # Total length of the lattice
        lattice_length = element_distances[-1]

        # Total number of elements in the lattice
        element_num = len(element_distances)

        # The distance between each step
        step_size = lattice_config['step_size']

        #Todo: Deal with the endpoint
        # Array with nth element containing the nth step position
        step_position = np.round(np.arange(0, lattice_length + step_size/2, step_size), 10)

        # Total number of steps in the lattice
        total_steps = len(step_position)

        # The indices of the steps which to compute the CSR wake at
        CSR_steps_index = np.array([])

        # Initialize with Nelement zeros, records how many steps are inside each lattice element
        steps_per_element = np.zeros((element_num,), dtype = int)

        # The lattice element's step position from the prevoius loop iteration
        prev_ind = 0

        # Populate _CSR_steps_index and steps_per_element
        # Loop through each element in lattice - d will be the distance from lattice entrance to end of the element
        for count, d in enumerate(element_distances):
            # Find where the lattice element ends in the _positions array
            ind = np.searchsorted(step_position, d, side = 'right')    #a[ind-1]<= d<a[ind], a is positions_record

            # The number of steps in this lattice element
            nsep_t = CSR_step_seperation[count]

            # Create a smaller array for this specific lattice element
            new_index = np.arange(prev_ind, ind, nsep_t)

            # Append new_index to the overall array
            CSR_steps_index = np.append(CSR_steps_index, new_index)

            # Populate steps_per_element
            if count == 0:
                steps_per_element[count] = ind - prev_ind - 1 # s = 0
            else:
                steps_per_element[count] = ind - prev_ind

            prev_ind = ind

        # Don't want to mess with the above code so I'll just convert the steps index
        compute_CSR = [False]*total_steps
        for step in CSR_steps_index:
            compute_CSR[int(step)] = True

        # Populate steps_since_CSR_applied
        steps_since_CSR_applied = [0]

        step_tracker = 1 # Tracks how many steps since CSR was computed
        for step_index in range(1,len(compute_CSR)):
            steps_since_CSR_applied.append(step_tracker)

            if compute_CSR[step_index]:
                step_tracker = 1
            else:
                step_tracker += 1
        
        # Compute the step ranges
        step_ranges = np.array([(step_position[i], step_position[i + 1]) for i in range(len(step_position) - 1)], dtype=np.float64)

        return step_size, total_steps, step_position, step_ranges, compute_CSR, steps_since_CSR_applied

def get_element_characteristics(lattice_config):
    """
    Loops through each element given in lattice_config and creates all needed
    arrays relating to element characteristics
    Parameters:
        lattice_config: dictionary of lattice configuration
    Returns:
        element_distances: distance[i] is the distance between the entrance and the end of ith lattice element
        element_rho_vals: the effective radius of each lattice element
        CSR_step_seperation: compute CSR at every nsep[i] slice in the ith element 
    """

    # The number of lattice elements
    element_num = len(lattice_config) - 1

    # distance[i] is the distance between the entrance and the end of ith lattice element
    element_distances = np.zeros(element_num)

    # The effective radius of each lattice element
    element_rho_vals = np.zeros(element_num)

    # Compute CSR at every nsep[i] slice in the ith element
    CSR_step_seperation = np.zeros(element_num)

    # List of lattice elements
    elements = list(lattice_config.keys())[1:]

    # Loop through each lattice element (ie drift, quad, drift, etc) and pull charcteristics fom lattice_config
    for count, key in enumerate(elements):
        # The element which we are processing
        current_element = lattice_config[key]

        # length of the current element
        length = current_element['L']

        CSR_step_seperation[count] = current_element['nsep']

        # Need to account for reference trajectory curving through dipole
        if current_element['type'] == 'dipole':
            angle = current_element['angle']
            element_rho_vals[count] = angle/length

        # Populate the distance array
        if count == 0:
            element_distances[count] = length

        else:
            element_distances[count] = length + element_distances[count - 1]

    return element_num, element_distances, element_rho_vals, CSR_step_seperation

def get_trajectory_characteristics(lattice_config, element_distances, sample_num=2000):
    """
    Computes the trajectory characteristics at sample_num of points along the reference trajactory
    Parameters: see above function docstring
    Returns:
        The following variables in a dictionary:
        sample_s_vals: s value for each sample point
        lab_frame_sample_coords: lab frame coordinates for each sample point
        step_n_vecs, step_tau_vecs: normal and tangential vectors in lab frame for each sample point
    """
    # List of lattice elements
    elements = list(lattice_config.keys())[1:]

    # Total length of the lattice
    lattice_length = element_distances[-1]

    # Initialize coordinates along s to calculate the reference traj
    sample_s_vals = np.linspace(0, lattice_length, sample_num)

    # Initialize [x, y] pair values for each sample (lab frame)
    lab_frame_sample_coords = np.zeros((sample_num, 2))

    # Initalize reference trajectory vector arrays
    sample_tau_vecs = np.zeros((sample_num, 2))
    sample_n_vecs = np.zeros((sample_num, 2))

    # Populate 1st element of reference trajectory vectors
    theta_0 = 0  # the angle between the traj tangential and x axis
    sample_tau_vecs[0, 0] = np.cos(theta_0)
    sample_tau_vecs[0, 1] = np.sin(theta_0)
    # Todo: High Priority! check the sign of n_vec
    sample_n_vecs[0, 0] = np.sin(theta_0)
    sample_n_vecs[0, 1] = -1 * np.cos(theta_0)

    # Pointer to which element we are in
    current_ele_index = 0

    # For each sample reference trajectory point (RTP)
    count = 1
    for st in sample_s_vals[1:]:
    #for count, st in enumerate(sample_s_vals[1:]):
        # Check to see if current RTP is outside current lattice element
        if st > element_distances[current_ele_index]:
            current_ele_index += 1

        # Change in s from previous RTP (should be uniform)
        delta_s = sample_s_vals[count] - sample_s_vals[count - 1]

        # The name of the current lattice element
        ele_name = elements[current_ele_index]

        # Populate the reference trajectory variables depending on the type of the current element
        # for dipole (account for curve)
        if lattice_config[ele_name]['type'] == 'dipole':
            length = lattice_config[ele_name]['L']
            angle = lattice_config[ele_name]['angle']
            phi = delta_s/length*angle
            r = length/angle

            # seems to be stupid. Todo
            x0 = lab_frame_sample_coords[count - 1, 0] - r*np.sin(theta_0)
            y0 = lab_frame_sample_coords[count - 1, 1] + r*np.cos(theta_0)

            lab_frame_sample_coords[count, 0] = x0 + r*np.sin(phi + theta_0)
            lab_frame_sample_coords[count, 1] = y0 - r*np.cos(phi + theta_0)

        # for drift, quad, sext
        else:
            phi = 0
            lab_frame_sample_coords[count, 0] = lab_frame_sample_coords[count - 1, 0] + delta_s *np.cos(theta_0)
            lab_frame_sample_coords[count, 1] = lab_frame_sample_coords[count - 1, 1] + delta_s *np.sin(theta_0)

        # Todo: check other elements (quad, sextupole)

        # Update our current angle
        theta_0 += phi

        # Populate the reference trajectory vectors
        sample_tau_vecs[count, 0] = np.cos(theta_0)
        sample_tau_vecs[count, 1] = np.sin(theta_0)
        # TODO: High Priority! check the sign of n_vec
        sample_n_vecs[count, 0] = np.sin(theta_0)
        sample_n_vecs[count, 1] = -1*np.cos(theta_0)

        count += 1

    return sample_s_vals, lab_frame_sample_coords, sample_n_vecs, sample_tau_vecs

def get_bmdax_elements(lattice_config, e_dists, step_positions, init_energy):
    """
    Creates the bmadx elements that each step propagates through, a bit complicated but very clean :)
    Note that we have to round some algebra bc the floating point numbers were a bit off and messing up the
    entrance/exit mechanisms
    Parameters:
        element_distances: distance[i] is the distance between the entrance and the end of ith lattice element
        step_positions: the s val of each step in the lattice
        init_energy: the energy of the reference trajectory, necessary for bmadx element creation 
    Returns:
        bmadx_elements: array holding bmadx elements, len = step_size-1, corresponds to the 
                        lattice element from the step_index to the next step_index, may contain
                        two elements per index, so each element is in an array.
    """
    # List of lattice elements names, used to index the lattice_config dictionary
    element_names = list(lattice_config.keys())[1:]

    # Initialize bmadx_array
    bmadx_elements = [None]*(len(step_positions)-1)

    # Insert zero into the element_distances array
    e_dists = np.insert(e_dists, 0, 0.0)

    # Helper conditional functions, returns true if s is within the element bounds
    condition1 = lambda element_left, element_right, s: element_left <= s and element_right > s
    condition2 = lambda element_left, element_right, s: element_left < s and element_right >= s

    # Keeps track of the element index of the previous slice, used for the construction of dipole elements
    # where start/end is revelent
    previous_e = -1

    # Loop through each step and create bmadx elements
    for i in range(len(step_positions)-1):

        # Given a position on the nominal trajectory, finds the index of the lattice element that it is within
        current_e = next(e_index for e_index in range(len(e_dists)-1) if condition1(e_dists[e_index], e_dists[e_index+1], step_positions[i]))
	    
        # Find the element that the next slice is in
        next_e = next(e_index for e_index in range(len(e_dists)-1) if condition2(e_dists[e_index], e_dists[e_index+1], step_positions[i+1]))
        
        # Get the distance between the two slices
        slice_length = round(step_positions[i+1] - step_positions[i], 10)

        # If the step cross an element boundary, we have to make two or more (in the case of skipping) bmadx elements
        if current_e != next_e:
            # List of bmadx elements for this step
            step_elements = []

            # Keeps track of where we are in the lattice
            current_distance = step_positions[i]

            # Loop through all elements that the step covers
            for e in range(current_e, next_e):

                # The distance remaining in the first element
                dist1 = round(e_dists[e+1] - current_distance, 10)

                # The distance remaining in the second element (either it goes through the next element or it does not)
                dist2 = round(min(e_dists[e+2] - e_dists[e+1], step_positions[i+1] - e_dists[e+1]), 10)

                #TODO: Bmadx seems to have some problems when the distance is very small
                if dist1 > 1.0e-6 and dist2 > 1.0e-6:
                    # Create bmadx element object for the previous element and append it to the bmadx_array
                    bmadx_element1 = create_bmadx_element(lattice_config, element_names[e], dist1, init_energy, exit=True)

                    # Sometimes while looping we start to jump through entire elements, in this case we need to set entrance and exit as true
                    exit_status = (dist2 == round(e_dists[e+2] - e_dists[e+1], 10))
                    bmadx_element2 = create_bmadx_element(lattice_config, element_names[e+1], dist2, init_energy, entrance=True, exit=exit_status)

                    step_elements.append(bmadx_element1)
                    step_elements.append(bmadx_element2)
                
                elif dist1 > 1.0e-6 and dist2 < 1.0e-6:
                    bmadx_element1 = create_bmadx_element(lattice_config, element_names[e], dist1, init_energy, exit=True)
                    step_elements.append(bmadx_element1)

                elif dist1 < 1.0e-6 and dist2 > 1.0e-6:
                    # Sometimes while looping we start to jump through entire elements, in this case we need to set entrance and exit as true
                    exit_status = (dist2 == round(e_dists[e+2] - e_dists[e+1], 10))
                    bmadx_element2 = create_bmadx_element(lattice_config, element_names[e+1], dist2, init_energy, entrance=True, exit=exit_status)

                    step_elements.append(bmadx_element2)

                else:
                    print("ERROR: STEP TOO SMALL")

                # Update the current distance
                current_distance += round(dist1 + dist2, 10)

            # We need to update the current element becuase we crossed an element boundary
            current_e = next_e

            bmadx_elements[i] = step_elements

        else:
            # This is annoying... but we have to use condition1 to test what the actual next element is
            next_e = next((e_index for e_index in range(len(e_dists)-1) if condition1(e_dists[e_index], e_dists[e_index+1], step_positions[i+1])), None)

            # Check if we are entering a new element perfectly (the current step is positioned at the start of the next element)
            if current_e != previous_e:
                bmadx_element1 = create_bmadx_element(lattice_config, element_names[current_e], slice_length, init_energy, entrance=True)
            
            # Check if we are exiting a new element perfeclty
            elif current_e != next_e:
                bmadx_element1 = create_bmadx_element(lattice_config, element_names[current_e], slice_length, init_energy, exit=True)

            else:
                bmadx_element1 = create_bmadx_element(lattice_config, element_names[current_e], slice_length, init_energy)

            bmadx_elements[i] = [bmadx_element1]
        
        previous_e = current_e
        
    return bmadx_elements

def create_bmadx_element(lattice_config, ele, DL, beam_energy, entrance = False, exit = False):
        """
        Creates a bmadx_element object from the given element parameters
        Parameters:
            lattice_config, dictionary of lattice elements
            ele: the desired element, string, used to index lattice_config
            DL: distance of the element (if many steps in an element this may be smaller than the entire element)
            entrance, exit: booleans, if the current slice of the lattice element contains an entrance or exit
                            relevant only for dipoles
        Returns:
            element: bmadx_element object
        """

        L = lattice_config[ele]['L']
        type = lattice_config[ele]['type']

        # For each element type bmadx has a different class
        if type == 'dipole':
            # The degree through which the dipole curves the nominal trajectory
            angle = lattice_config[ele]['angle']

            # E1 and E2 are the entrance and exit angles respectively
            E1 = lattice_config[ele]['E1']
            E2 = lattice_config[ele]['E2']

            # Radius of curvature
            G = angle/L

            # If the dipole element we want to create includes an entrance and/or exit it is constructed slightly differently
            element = SBend(L = DL, P0C = beam_energy, G = G, E1 = E1 if entrance else 0.0, E2 = E2 if exit else 0.0)

        elif type == 'drift':
            element = Drift(L = DL)

        elif type == 'quad':
            K1 = lattice_config[ele]['strength']
            element = Quadrupole(L=DL, K1=K1)

        elif type == 'sextupole':
            K2 = lattice_config[ele]['strength']
            element = Sextupole(L=DL, K2=K2)

        return element
    
        

