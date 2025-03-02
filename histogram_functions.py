from numba import jit
import math
import numpy as np

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
    bin_spacing_1 = (bins_end_1 - bins_start_1) / (nbins_1 - 1)
    inv_spacing_1 = 1. / bin_spacing_1
    bin_spacing_2 = (bins_end_2 - bins_start_2) / (nbins_2 -1)
    inv_spacing_2 = 1. / bin_spacing_2
    n_ptcl = len(w)

    # Allocate array for histogrammed data
    hist_data = np.zeros((nbins_1, nbins_2), dtype=np.float64)

    # Go through particle array and bin the data
    for i in range(n_ptcl):

        # Calculate the index of lower bin to which this particle contributes
        q1_cell = (q1[i] - bins_start_1) * inv_spacing_1
        q2_cell = (q2[i] - bins_start_2) * inv_spacing_2
        #i1_low_bin = int(math.floor(q1_cell))
        i1_low_bin = int(q1_cell)
        #i2_low_bin = int(math.floor(q2_cell))
        i2_low_bin = int(q2_cell)

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