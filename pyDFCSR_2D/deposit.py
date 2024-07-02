from numba import jit
import math
import numpy as np

# deque is a 'double ended quote' data structure that allows for efficient append and pop operations from both ends of the queue
from collections import deque
#from SGolay_filter import *

from scipy.interpolate import RegularGridInterpolator
from scipy.signal import savgol_filter

@jit(nopython = True)
def histogram_cic_1d(q1, w, nbins, bins_start, bins_end):
    """
    Return an 1D histogram of the values in `q1` weighted by `w`,
    consisting of `nbins` evenly-spaced bins between `bins_start`
    and `bins_end`. Contribution to each bins is determined by the
    CIC weighting scheme (i.e. linear weights).
    Source: https://github.com/openPMD/openPMD-viewer/blob/dev/openpmd_viewer/openpmd_timeseries/utilities.py
    """
    # Define various scalars
    bin_spacing = (bins_end - bins_start) / nbins
    inv_spacing = 1. / bin_spacing
    n_ptcl = len(w)

    # Allocate array for histogrammed data
    hist_data = np.zeros(nbins, dtype=np.float64)

    # Go through particle array and bin the data
    for i in range(n_ptcl):
        # Calculate the index of lower bin to which this particle contributes
        q1_cell = (q1[i] - bins_start) * inv_spacing
        i_low_bin = int(math.floor(q1_cell))
        # Calculate corresponding CIC shape and deposit the weight
        S_low = 1. - (q1_cell - i_low_bin)
        if (i_low_bin >= 0) and (i_low_bin < nbins):
            hist_data[i_low_bin] += w[i] * S_low
        if (i_low_bin + 1 >= 0) and (i_low_bin + 1 < nbins):
            hist_data[i_low_bin + 1] += w[i] * (1. - S_low)

    return (hist_data)

@jit(nopython = True)
def histogram_cic_2d(q1, q2, w,
                     nbins_1, bins_start_1, bins_end_1,
                     nbins_2, bins_start_2, bins_end_2):
    """
    Return an 2D histogram of the values in `q1` and `q2` weighted by `w`,
    consisting of `nbins_1` bins in the first dimension and `nbins_2` bins
    in the second dimension.
    Contribution to each bins is determined by the
    CIC weighting scheme (i.e. linear weights).
    Source:https://github.com/openPMD/openPMD-viewer/blob/dev/openpmd_viewer/openpmd_timeseries/utilities.py
    """
    # Define various scalars
    bin_spacing_1 = (bins_end_1 - bins_start_1) / nbins_1
    inv_spacing_1 = 1. / bin_spacing_1
    bin_spacing_2 = (bins_end_2 - bins_start_2) / nbins_2
    inv_spacing_2 = 1. / bin_spacing_2
    n_ptcl = len(w)

    # Allocate array for histogrammed data
    hist_data = np.zeros((nbins_1, nbins_2), dtype=np.float64)

    # Go through particle array and bin the data
    for i in range(n_ptcl):

        # Calculate the index of lower bin to which this particle contributes
        q1_cell = (q1[i] - bins_start_1) * inv_spacing_1
        q2_cell = (q2[i] - bins_start_2) * inv_spacing_2
        i1_low_bin = int(math.floor(q1_cell))
        i2_low_bin = int(math.floor(q2_cell))

        # Calculate corresponding CIC shape and deposit the weight
        S1_low = 1. - (q1_cell - i1_low_bin)
        S2_low = 1. - (q2_cell - i2_low_bin)
        if (i1_low_bin >= 0) and (i1_low_bin < nbins_1):
            if (i2_low_bin >= 0) and (i2_low_bin < nbins_2):
                hist_data[i1_low_bin, i2_low_bin] += w[i] * S1_low * S2_low
            if (i2_low_bin + 1 >= 0) and (i2_low_bin + 1 < nbins_2):
                hist_data[i1_low_bin, i2_low_bin + 1] += w[i] * S1_low * (1. - S2_low)
        if (i1_low_bin + 1 >= 0) and (i1_low_bin + 1 < nbins_1):
            if (i2_low_bin >= 0) and (i2_low_bin < nbins_2):
                hist_data[i1_low_bin + 1, i2_low_bin] += w[i] * (1. - S1_low) * S2_low
            if (i2_low_bin + 1 >= 0) and (i2_low_bin + 1 < nbins_2):
                hist_data[i1_low_bin + 1, i2_low_bin + 1] += w[i] * (1. - S1_low) * (1. - S2_low)

    return (hist_data)


class DF_tracker:
    """
    The class for the 'distribution function' (DF) of a beam
    Doesn't handle the discrete data of the DF, rather it stores and compute the beam characteristics (ie tracker)
    """
    def __init__(self, input_dic={}):
        """
        Initializes the DF tracker
        Parameters:
            input_dic: optional, a dictionary mapping various optional parameters (keys) to
                       their values that can be specified if desired
        Returns:
            Instance of DF_Tracker
        """

        # Initalize the optional parameters
        self.configure_params(**input_dic)

        # Params for current DF, basically what the distribution function looks like 'now'
        self.sigma_x = None
        self.sigma_z = None
        self.density = None
        self.density_x = None
        self.density_z = None

        # Velocity and velcoity graident histograms
        self.vx = None
        self.vx_x = None

        # Integration mesh coordinates
        self.x_grids = None
        self.z_grids = None

        # Some timing variables
        self.start_time = 0.0
        self.t = 0.0

        # Params for DF log
        # The DF log stores all the data about the DF in the past
        self.slope_log = deque([])
        self.DF_log = deque([])
        self.sigma_x_log = deque([])
        self.sigma_z_log = deque([])
        self.time_log = deque([])

        # Params for interpolant
        self.sigma_x_interp = None
        self.sigma_z_interp = None
        # TODO: Add chirp
        self.time_interp = deque([])
        self.density_interp = deque([])
        self.density_x_interp = deque([])
        self.density_z_interp = deque([])
        self.vx_interp = deque([])
        self.vx_x_interp = deque([])
        self.interp_start = 0
        self.interp_end = 0
        self.x_grid_interp = None
        self.z_grid_interp = None

    def configure_params(self, xbins=100, zbins=100, xlim=5, zlim=5,
                         filter_order=0, filter_window=0,
                         velocity_threhold=5, upper_limit = None):
        """
        Initalizes the optional parameters
        """

        # Number of bins allocated for the histogram in x and z direction
        self.xbins = xbins
        self.zbins = zbins

        # The number of standard deviations away from the mean (in both directions) for our histogram
        self.xlim = xlim
        self.zlim = zlim

        # 1/(proportion of the highest occupied bin)
        # TODO: make this a proportion
        self.velocity_threhold = velocity_threhold

        # Savitzky-Golay filter settings
        self.filter_order = filter_order # The order of the polynomial used to fit the samples
        self.filter_window = filter_window # The length of the filter_window, must be positive and odd

        # The maximum number of bins for
        self.upper_limit = upper_limit

    def get_DF(self, x, z, px, t):
        """
        Given the discrete data of the distribution function (in 2D), computes the beam characteristics
        Parameters:
            x: array, x position of each particle
            z: array, z position of ...
            px: array, px phase space position of ...
            t: int, the time at which the DF is shaped like this
        """
        # TODO: add filter, add different depositing type
        # Compute basic beam characteristics
        self.sigma_x = np.std(x)
        self.sigma_z = np.std(z)
        self.xmean = np.mean(x)
        self.zmean = np.mean(z)
        npart = len(x)

        # The boundaries of the mesh (integration space)
        x_min = self.xmean - self.xlim * self.sigma_x
        x_max = self.xmean + self.xlim * self.sigma_x
        z_min = self.zmean - self.zlim * self.sigma_z
        z_max = self.zmean + self.zlim * self.sigma_z

        # If the beam is highly tilted, we need to have a higer number of bins to have higher resolution
        # TODO: coordinates transform for highly chirped case
        # TODO: find the tilt of the beam a better way, use slide window?
        # Find indices where the z coordinate is much smaller than the standard deviation (this will be a small slice of the beam)
        slice_ind = np.argwhere(np.abs(z - self.xmean) < 0.1*self.sigma_z)

        # Gets the standard deviation of the slice
        slice_sigX = np.std(x[slice_ind])

        # The ratio of the std of the slice to the std of the overall beam distribution function
        frac = self.sigma_x/slice_sigX

        # If our fraction is large (high tilt) we need higher resolution
        if frac > 5:
            xbins_t = self.xbins
            zbins_t = self.zbins
            filter_window = self.filter_window

        # If not, smaller resolution will do
        else:
            xbins_t = 100
            zbins_t = 100
            filter_window = 5

        # The x and z position of each mesh point
        x_grids = np.linspace(x_min, x_max, xbins_t)
        z_grids = np.linspace(z_min, z_max, zbins_t)

        # Create a 2D DF position density histogram
        density = histogram_cic_2d(q1=x, q2=z, w=np.ones(x.shape),
                                    nbins_1 = xbins_t, bins_start_1 = x_min, bins_end_1 = x_max,
                                    nbins_2 = zbins_t, bins_start_2 = z_min, bins_end_2 = z_max)

        # Create a 2D DF velocity distribution histogram
        vx = histogram_cic_2d(q1=x, q2=z, w=px,
                                    nbins_1 = xbins_t, bins_start_1 = x_min, bins_end_1 = x_max,
                                    nbins_2 = zbins_t, bins_start_2 = z_min, bins_end_2 = z_max)

        # The minimum particle number of particles in a bin for that bin to have non zero vx value
        threshold = np.max(density) / self.velocity_threhold

        # Make each mesh element value be equal to the AVERAGE velocity of particles in said element
        # Only for bins with particle density above the threshold
        vx[density > threshold] /= density[density > threshold]

        # Apply 2D Savitzky-Golay to both position density and velocity distribution histogram
        #TODO: Consider other 2D sgolay filter
        #TODO: consider using the derivative in sgoaly filter
        #density = sgolay2d(density, self.filter_window, self.filter_order, derivative=None)
        density = savgol_filter(x= savgol_filter(x = density, window_length=filter_window, polyorder=self.filter_order, axis = 0),
                                window_length=filter_window, polyorder=self.filter_order, axis = 1)

        #vx = sgolay2d(vx, self.filter_window, self.filter_order, derivative=None)  # adding this seems to be wrong
        vx = savgol_filter(x= savgol_filter(x = vx, window_length=filter_window, polyorder=self.filter_order, axis = 0),
                                window_length=filter_window, polyorder=self.filter_order, axis = 1)

        # Integrate the density function over the integration space using trapezoidal rule
        dsum = np.trapz(np.trapz(density, x_grids, axis=0), z_grids)

        # Normalize the density distirbution historgram
        density /= dsum

        # Set all bins in velocity distribution histogram with low particle count to zero
        vx[density <= threshold] = 0

        # TODO: how to do it if apply coordiante tranformation?
        # TODO: reanme density_x and other variables to gradient related name
        # Calculate rate of change (gradient) values at each mesh point for both position and velocity histograms
        density_x, density_z = np.gradient(density, x_grids, z_grids)
        vx_x, vx_z = np.gradient(vx, x_grids, z_grids)

        # Apply 2D Savitzky-Golay to position and velocity gradient histograms
        density_x = savgol_filter(
            x=savgol_filter(x=density_x, window_length=filter_window, polyorder=self.filter_order, axis=0),
            window_length=filter_window, polyorder=self.filter_order, axis=1)

        density_z = savgol_filter(
            x=savgol_filter(x=density_z, window_length=filter_window, polyorder=self.filter_order, axis=0),
            window_length=filter_window, polyorder=self.filter_order, axis=1)

        vx_x = savgol_filter(
            x=savgol_filter(x=vx_x, window_length=filter_window, polyorder=self.filter_order, axis=0),
            window_length=filter_window, polyorder=self.filter_order, axis=1)

        #density_x, density_z = sgolay2d(density, self.filter_window, self.filter_order, derivative='both')
        #density_x /= np.mean(np.diff(x_grids))
        #density_z /= np.mean(np.diff(z_grids))

        # Again filter out the low populated bins
        # TODO: set input for velocity filter
        #vx_x, _ = sgolay2d(vx, self.filter_window, self.filter_order, derivative='both')
        threshold = np.max(density) / self.velocity_threhold * 8
        #vx_x[density < threshold] = 0
        vx_x[density < threshold] =  np.mean(vx_x[density > threshold])
        #vx_x /= np.mean(np.diff(x_grids))

        # Update the current DF characteristics
        self.x_grids = x_grids
        self.z_grids = z_grids
        self.density = density
        self.vx = vx
        self.density_x = density_x
        self.density_z = density_z
        self.vx_x = vx_x
        self.t = t

    def append_DF(self):
        """
        append current DF to the log
        :return:
        """
        self.DF_log.append((self.x_grids, self.z_grids, self.density, self.vx, self.density_x, self.density_z, self.vx_x))
        self.time_log.append(self.t)
        self.sigma_x_log.append(self.sigma_x)
        self.sigma_z_log.append(self.sigma_z)
        self.end_time = self.t

    def pop_left_DF(self, new_start_time):
        """
        Pop history of DFs until new_start_time. Used to remove information about beam that is outside of the formation legnth
        relative to the new beam position.
        Parameters:
            new_start_time: the point in time at which we begin to use past beam distribution functions from to compute CSR wake
            at a later point in time
        """
        # remove outdated DF log
        while self.start_time < new_start_time:
            self.DF_log.popleft()
            self.time_log.popleft()
            self.sigma_x_log.popleft()
            self.sigma_z_log.popleft()
            self.start_time = self.time_log[0]

        #  remove outdated interpolant
        while self.interp_start < new_start_time:
            self.density_interp.popleft()
            self.density_x_interp.popleft()
            self.density_z_interp.popleft()
            self.vx_interp.popleft()
            self.vx_x_interp.popleft()
            self.time_interp.popleft()
            self.interp_start = self.time_interp[0]

    def pop_right_DF(self):
        """
        Pop the newest DF from the log
        """
        self.DF_log.pop()
        self.time_log.pop()
        self.sigma_x_log.pop()
        self.sigma_z_log.pop()
        self.end_time = self.time_log[-1]

    def DF_interp(self, DF, x_grid_interp = None, z_grid_interp = None, x_grids = None, z_grids = None, fill_value = 0.0):
        """
        Creates a new scipy interpolator for a distribution function on a specified mesh
        """
        # If the mesh is not specified, then we assume that the interpolator should follow the same mesh as the interpolators for
        # the other slices
        if x_grids is None:
            x_grids = self.x_grids
        if z_grids is None:
            z_grids = self.z_grids
        if x_grid_interp is None:
            x_grid_interp = self.x_grid_interp
        if z_grid_interp is None:
            z_grid_interp = self.z_grid_interp

        # Create meshgrid
        X, Z = np.meshgrid(x_grid_interp, z_grid_interp, indexing = 'ij')
        #TODO: check this 2D interpolation
        # Create interpolator using scipy
        interp = RegularGridInterpolator((x_grids, z_grids), DF, method='linear', fill_value = fill_value, bounds_error=False)
        return interp((X,Z))

    def append_interpolant(self, formation_length, n_formation_length):
        """
        In addition to the DF slices (which have their own mesh with not necesarily equal parameters such as bin size), we make an
        'interpolation' mesh for each slice. The purpose of these meshes is so that the integration of the space between slices
        can be done in O(1) time using a simple linear interpolation. Hence why all interpolation meshes need to have identical
        meshes. This method creates a new interpolation mesh for the lastest slice.
        Parameters:
            formation_length: the formation length of the beam's distribution on the nominal path on the current lattice element
            n_formation_length: the number of formation length that we need to "look" back in time
        """
        # The furthest point back in time from which we will compute the CSR wake
        start_point = np.amax(a=(0, self.end_time - n_formation_length * formation_length))

        # Remove all data from the DF logs earlier than start_point
        self.pop_left_DF(new_start_time=start_point)

        # This conditionial checks to see if the newest slice in the DF log has a beam profile that is significantly different from
        # the other slices. If so, then we will need to reshape all interpolation meshes
        if self.sigma_x_interp and self.sigma_z_interp and \
                2 > self.sigma_x/self.sigma_x_interp > 1/2 and \
                    2 > self.sigma_z/self.sigma_z_interp > 1 / 2:

            # Not too much change in beam size (and chirp in the future), just interp with current interp configuration
            self.time_interp.append(self.t)

            # Create the scipy interpolator instances for the new slice
            current_density_interp = self.DF_interp(DF = self.density)
            current_density_x_interp = self.DF_interp(DF = self.density_x)
            current_density_z_interp = self.DF_interp(DF = self.density_z)
            current_vx_interp = self.DF_interp(DF = self.vx)
            current_vx_x_interp = self.DF_interp(DF = self.vx_x, fill_value=np.mean(self.vx_x))

            # Add the interpolator instances to the various lists of interpolator slices
            self.density_interp.append(current_density_interp)
            self.density_x_interp.append(current_density_x_interp)
            self.density_z_interp.append(current_density_z_interp)
            self.vx_interp.append(current_vx_interp)
            self.vx_x_interp.append(current_vx_x_interp)

        # If current DF slice is too different from the slices in the log, reshape all interpolate meshes
        else:
            #TODO: hard code from matlab. Consider change in the future
            print('start reinterpolation. number of slice', str(len(self.time_log)))

            # Find the maximum and minimum standard deviation of the distribution function at each time slice since the start_point
            max_sigma_x = np.max(self.sigma_x_log)
            min_sigma_x = np.min(self.sigma_x_log)
            max_sigma_z = np.max(self.sigma_z_log)
            min_sigma_z = np.min(self.sigma_z_log)

            # Compute the number of bins for the new mesh
            t_x = max_sigma_x / min_sigma_x
            t_z = max_sigma_z / min_sigma_z
            xbins = int(500 * t_x)
            zbins = int(500 * t_z)

            # If there is an upper limit on the bin number defined by the user implement it here
            if isinstance(self.upper_limit, int):
                xbins = min(xbins, self.upper_limit)
                zbins = min(zbins, self.upper_limit)

            print("xbins =", xbins, " zbins = ", zbins)

            # Redefine the max std for the meshes
            self.sigma_x_interp = max_sigma_x
            self.sigma_z_interp = max_sigma_z

            # Make the new mesh
            self.x_grid_interp = np.linspace(self.xmean-5*self.sigma_x_interp, self.xmean + 5*self.sigma_x_interp, xbins)
            self.z_grid_interp = np.linspace(self.zmean -5* self.sigma_z_interp, self.zmean + 5* self.sigma_z_interp, zbins)

            # Clear interpolant meshes from pervious slices
            self.density_interp = deque([])
            self.density_x_interp = deque([])
            self.density_z_interp = deque([])
            self.vx_interp = deque([])
            self.vx_x_interp = deque([])
            self.time_interp = self.time_log.copy()

            # Loop through all the existing mesh slices and reconstruct thier mesh interpolants
            for x_grids, z_grids, density, vx, density_x, density_z, vx_x in self.DF_log:
                # Create mesh interpolants
                current_density_interp = self.DF_interp(DF=density, x_grids = x_grids, z_grids = z_grids)
                current_density_x_interp = self.DF_interp(DF=density_x, x_grids = x_grids, z_grids = z_grids)
                current_density_z_interp = self.DF_interp(DF=density_z, x_grids = x_grids, z_grids = z_grids)
                current_vx_interp = self.DF_interp(DF=vx, x_grids = x_grids, z_grids = z_grids)
                current_vx_x_interp = self.DF_interp(DF=vx_x, x_grids = x_grids, z_grids = z_grids, fill_value=np.mean(vx_x))

                # Add them to the list of slice mesh interpolants
                self.density_interp.append(current_density_interp)
                self.density_x_interp.append(current_density_x_interp)
                self.density_z_interp.append(current_density_z_interp)
                self.vx_interp.append(current_vx_interp)
                self.vx_x_interp.append(current_vx_x_interp)

    def build_interpolant(self):
        """
        build interpolant for CSR intergration with the 3D matrix self.*_interp
        :return:
        """
        #TODO: check fill value
        #TODO: Important! consider faster 3D interpolation (Cython and parellel with prange, GIL release) https://github.com/jglaser/interp3d, https://ndsplines.readthedocs.io/en/latest/compare.html
        #TODO: Probably accelerate trapz with jit (parallel) https://berkeley-stat159-f17.github.io/stat159-f17/lectures/09-intro-numpy/trapezoid..html
        #TODO: Parallel cic with jit?
        #TODO: Also think about accelerate all numpy with numba.https://towardsdatascience.com/supercharging-numpy-with-numba-77ed5b169240
        #TODO: Also consider GPU acceleration of trapz and scipy regulargridinterpolant with Cupy
        #self.F_density = RegularGridInterpolator((self.time_interp, self.x_grid_interp, self.z_grid_interp),
        #                                         self.density_interp, fill_value= 0.0,bounds_error=False)
        #self.F_density_x = RegularGridInterpolator((self.time_interp, self.x_grid_interp, self.z_grid_interp),
        #                                         self.density_x_interp, fill_value= 0.0,bounds_error=False)
        #self.F_density_z = RegularGridInterpolator((self.time_interp, self.x_grid_interp, self.z_grid_interp),
        #                                           self.density_z_interp, fill_value= 0.0,bounds_error=False)
        #self.F_vx = RegularGridInterpolator((self.time_interp, self.x_grid_interp, self.z_grid_interp),
        #                                           self.vx_interp, fill_value= 0.0,bounds_error=False)
        #self.F_vx_x= RegularGridInterpolator((self.time_interp, self.x_grid_interp, self.z_grid_interp),
        #                                           self.vx_x_interp, fill_value= 0.0,bounds_error=False)

        # TODO: rename these variables to make more intutive sense

        # In our 3D matrix 'x' is time 'y' is x and 'z' is z
        # The time range of interpolation meshes
        self.min_x, self.max_x = self.time_interp[0], self.time_interp[-1]

        # The x and z range of interpolation meshes
        self.min_y, self.max_y = self.x_grid_interp[0], self.x_grid_interp[-1]
        self.min_z, self.max_z = self.z_grid_interp[0], self.z_grid_interp[-1]

        # The change in t between each slice
        self.delta_x = (self.max_x - self.min_x) / (len(self.time_interp) - 1)

        # The change in x and z between each mesh element
        self.delta_y = (self.max_y - self.min_y) / (self.x_grid_interp.shape[0] - 1)
        self.delta_z =  (self.max_z - self.min_z) / (self.z_grid_interp.shape[0] - 1)

        # Convert the array type of interpolant meshes from deque to np array
        self.data_density_interp = np.array(self.density_interp)
        self.data_density_z_interp = np.array(self.density_z_interp)
        self.data_density_x_interp = np.array(self.density_x_interp)
        self.data_vx_interp = np.array(self.vx_interp)
        self.data_vx_x_interp = np.array(self.vx_x_interp)
