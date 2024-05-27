from numba import jit
import math
import numpy as np
from collections import deque
#from SGolay_filter import *
from params import *
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
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
    def __init__(self, input_dic={}):


        self.configure_params(**input_dic)
        #params for current DF
        self.sigma_x = None
        self.sigma_z = None
        self.density = None
        self.density_x = None
        self.density_z = None
        self.vx = None
        self.vx_x = None
        self.x_grids = None
        self.z_grids = None
        self.start_time = 0.0
        self.t = 0.0

        #params for DF log
        self.slope_log = deque([])
        self.DF_log = deque([])
        self.time_log = deque([])
        self.log_start_time = 0
        self.log_end_time = 0

        #params for interpolant
        self.sigma_x_interp = None
        self.sigma_z_interp = None
        # Todo: Add chirp
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
                         velocity_threhold=5):
        self.xbins = xbins
        self.zbins = zbins
        self.xlim = xlim
        self.zlim = zlim
        self.velocity_threhold = velocity_threhold

        self.filter_order = filter_order
        self.filter_window = filter_window

    def get_DF(self, x, z, xp, t):
        # Todo: add filter, add different depositing type
        sigma_x = np.std(x)
        sigma_z = np.std(z)
        self.sigma_x = sigma_x
        self.sigma_z = sigma_z
        self.xmean = np.mean(x)
        self.zmean = np.mean(z)
        npart = len(x)

        #########test#######################
        #Todo: coordinates transform for highly chirped case
        slice_ind = np.argwhere(np.abs(z) < 0.1*sigma_z)
        slice_sigX = np.std(x[slice_ind])
        frac = sigma_x/slice_sigX
        if frac > 5:
            self.xbins = 100
            self.zbins = 300
        else:
            self.xbins = 100
            self.zbins = 100
            #print('frac', frac)

        x_grids = np.linspace(self.xmean - self.xlim * sigma_x, self.xmean + self.xlim * sigma_x, self.xbins)
        z_grids = np.linspace(self.zmean - self.zlim * sigma_z, self.zmean + self.zlim * sigma_z, self.zbins)
        density = histogram_cic_2d(q1=x, q2=z, w=np.ones(x.shape),
                                   nbins_1=self.xbins, bins_start_1=self.xmean - self.xlim * sigma_x,
                                   bins_end_1=self.xmean + self.xlim * sigma_x,
                                   nbins_2=self.zbins, bins_start_2=self.zmean - self.zlim * sigma_z,
                                   bins_end_2=self.zmean + self.zlim * sigma_z)

        vx = histogram_cic_2d(q1=x, q2=z, w=xp,
                              nbins_1=self.xbins, bins_start_1=self.xmean - self.xlim * sigma_x,
                              bins_end_1=self.xmean + self.xlim * sigma_x,
                              nbins_2=self.zbins, bins_start_2=self.zmean - self.zlim * sigma_z,
                              bins_end_2=self.zmean + self.zlim * sigma_z)
        threshold = np.max(density) / self.velocity_threhold
        vx[density > threshold] /= density[density > threshold]

        # Add filter to density and vx
        #vx = sgolay2d(vx, self.filter_window, self.filter_order, derivative=None)  # adding this seems to be wrong

        vx[density <= threshold] = 0

        # Add filter to density and vx
        #Todo: Consider other 2D sgolay filter
        #Todo: consider using the derivative in sgoaly filter
        #density = sgolay2d(density, self.filter_window, self.filter_order, derivative=None)
        density = savgol_filter(x= savgol_filter(x = density, window_length=self.filter_window, polyorder=self.filter_order, axis = 0),
                                window_length=self.filter_window, polyorder=self.filter_order, axis = 1)

        vx = savgol_filter(x= savgol_filter(x = vx, window_length=self.filter_window, polyorder=self.filter_order, axis = 0),
                                window_length=self.filter_window, polyorder=self.filter_order, axis = 1)

        dsum = np.trapz(np.trapz(density, x_grids, axis=0), z_grids)
        density /= dsum





        # Todo: how to do it if apply coordiante tranformation?

        density_x, density_z = np.gradient(density, x_grids, z_grids)
        vx_x, vx_z = np.gradient(vx, x_grids, z_grids)

        density_x = savgol_filter(
            x=savgol_filter(x=density_x, window_length=self.filter_window, polyorder=self.filter_order, axis=0),
            window_length=self.filter_window, polyorder=self.filter_order, axis=1)
        density_z = savgol_filter(
            x=savgol_filter(x=density_z, window_length=self.filter_window, polyorder=self.filter_order, axis=0),
            window_length=self.filter_window, polyorder=self.filter_order, axis=1)

        vx_x = savgol_filter(
            x=savgol_filter(x=vx_x, window_length=self.filter_window, polyorder=self.filter_order, axis=0),
            window_length=self.filter_window, polyorder=self.filter_order, axis=1)


        #density_x, density_z = sgolay2d(density, self.filter_window, self.filter_order, derivative='both')
        #density_x /= np.mean(np.diff(x_grids))
        #density_z /= np.mean(np.diff(z_grids))

        # Todo: set input for velocity filter
        #vx_x, _ = sgolay2d(vx, self.filter_window, self.filter_order, derivative='both')
        threshold = np.max(density) / self.velocity_threhold * 4
        #vx_x[density < threshold] = 0
        vx_x[density < threshold] =  np.mean(vx_x[density > threshold])
        #vx_x /= np.mean(np.diff(x_grids))

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
        self.end_time = self.t

    def pop_left_DF(self, new_start_time):
        """
        pop history of DFs until new_start_time
        :param new_start_time:
        :return:
        """
        # remove outdated DF log
        while self.start_time < new_start_time:
            self.DF_log.popleft()
            self.time_log.popleft()
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
        pop the newest DF from the log
        :return:
        """
        self.DF_log.pop()
        self.time_log.pop()
        self.end_time = self.time_log[-1]



    def DF_interp(self, DF, x_grid_interp = None, z_grid_interp = None, x_grids = None, z_grids = None, fill_value = 0.0):
        if x_grids is None:
            x_grids = self.x_grids
        if z_grids is None:
            z_grids = self.z_grids
        if x_grid_interp is None:
            x_grid_interp = self.x_grid_interp
        if z_grid_interp is None:
            z_grid_interp = self.z_grid_interp

        X, Z = np.meshgrid(x_grid_interp, z_grid_interp, indexing = 'ij')
        #Todo: check this 2D interpolation
        interp = RegularGridInterpolator((x_grids, z_grids), DF, method='linear', fill_value = fill_value, bounds_error=False)
        return interp((X,Z))


    def append_interpolant(self, formation_length, n_formation_length, interpolation):
        start_point = np.amax(a=(0, self.end_time - n_formation_length * formation_length))
        self.pop_left_DF(new_start_time=start_point)
        xlim_interp = interpolation.xlim
        zlim_interp = interpolation.zlim
        xbins = interpolation.xbins
        zbins = interpolation.zbins

        if self.sigma_x_interp and self.sigma_z_interp and \
                interpolation.re_interpolate_threshold > self.sigma_x/self.sigma_x_interp > 1/interpolation.re_interpolate_threshold and \
                    interpolation.re_interpolate_threshold > self.sigma_z/self.sigma_z_interp > 1 / interpolation.re_interpolate_threshold:
            # Not too much change in beam size (and chirp in the future), just interp with current interp configuration
            self.time_interp.append(self.t)
            self.x_grid_interp = np.linspace(self.xmean-xlim_interp*self.sigma_x_interp, self.xmean + xlim_interp*self.sigma_x_interp, xbins)
            self.z_grid_interp = np.linspace(self.zmean-zlim_interp*self.sigma_z_interp, self.zmean + zlim_interp*self.sigma_z_interp, zbins)
            current_density_interp = self.DF_interp(DF = self.density)
            current_density_x_interp = self.DF_interp(DF = self.density_x)
            current_density_z_interp = self.DF_interp(DF = self.density_z)
            current_vx_interp = self.DF_interp(DF = self.vx)
            current_vx_x_interp = self.DF_interp(DF = self.vx_x, fill_value=np.mean(self.vx_x))
            self.density_interp.append(current_density_interp)
            self.density_x_interp.append(current_density_x_interp)
            self.density_z_interp.append(current_density_z_interp)
            self.vx_interp.append(current_vx_interp)
            self.vx_x_interp.append(current_vx_x_interp)

        else:
            #Todo: hard code from matlab. Consider change in the future
            #print('start reinterpolation. number of slice', str(len(self.time_log)))
            if self.sigma_x_interp:
                if self.sigma_x >= 0.9*self.sigma_x_interp:  # if the transverse size increase
                    xlim_interp = 5
                else:
                    xlim_interp = 10


            self.sigma_x_interp = self.sigma_x
            self.sigma_z_interp = self.sigma_z

            self.x_grid_interp = np.linspace(self.xmean-xlim_interp*self.sigma_x_interp, self.xmean + xlim_interp*self.sigma_x_interp, xbins)
            self.z_grid_interp = np.linspace(self.zmean -zlim_interp * self.sigma_z_interp, self.zmean + zlim_interp * self.sigma_z_interp, zbins)

            #clear interpolant and redo interpolation
            self.density_interp = deque([])
            self.density_x_interp = deque([])
            self.density_z_interp = deque([])
            self.vx_interp = deque([])
            self.vx_x_interp = deque([])
            self.time_interp = self.time_log.copy()

            for x_grids, z_grids, density, vx, density_x, density_z, vx_x in self.DF_log:
                current_density_interp = self.DF_interp(DF=density, x_grids = x_grids, z_grids = z_grids)
                current_density_x_interp = self.DF_interp(DF=density_x, x_grids = x_grids, z_grids = z_grids)
                current_density_z_interp = self.DF_interp(DF=density_z, x_grids = x_grids, z_grids = z_grids)
                current_vx_interp = self.DF_interp(DF=vx, x_grids = x_grids, z_grids = z_grids)
                current_vx_x_interp = self.DF_interp(DF=vx_x, x_grids = x_grids, z_grids = z_grids, fill_value=np.mean(vx_x))

                self.density_interp.append(current_density_interp)
                self.density_x_interp.append(current_density_x_interp)
                self.density_z_interp.append(current_density_z_interp)
                self.vx_interp.append(current_vx_interp)
                self.vx_x_interp.append(current_vx_x_interp)

            #print('Re-interpolation finished!')


    def build_interpolant(self):
        """
        build interpolant for CSR intergration with the 3D matrix self.*_interp
        :return:
        """
        #Todo: check fill value
        #Todo: Important! consider faster 3D interpolation (Cython and parellel with prange, GIL release) https://github.com/jglaser/interp3d, https://ndsplines.readthedocs.io/en/latest/compare.html
        #Todo: Probably accelerate trapz with jit (parallel) https://berkeley-stat159-f17.github.io/stat159-f17/lectures/09-intro-numpy/trapezoid..html
        #Todo: Parallel cic with jit?
        #Todo: Also think about accelerate all numpy with numba.https://towardsdatascience.com/supercharging-numpy-with-numba-77ed5b169240
        #Todo: Also consider GPU acceleration of trapz and scipy regulargridinterpolant with Cupy
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
        self.min_x, self.max_x = self.time_interp[0], self.time_interp[-1]
        self.min_y, self.max_y = self.x_grid_interp[0], self.x_grid_interp[-1]
        self.min_z, self.max_z = self.z_grid_interp[0], self.z_grid_interp[-1]
        self.delta_x = (self.max_x - self.min_x) / (len(self.time_interp) - 1)
        self.delta_y = (self.max_y - self.min_y) / (self.x_grid_interp.shape[0] - 1)
        self.delta_z =  (self.max_z - self.min_z) / (self.z_grid_interp.shape[0] - 1)
        self.data_density_interp = np.array(self.density_interp)
        self.data_density_z_interp = np.array(self.density_z_interp)
        self.data_density_x_interp = np.array(self.density_x_interp)
        self.data_vx_interp = np.array(self.vx_interp)
        self.data_vx_x_interp = np.array(self.vx_x_interp)






