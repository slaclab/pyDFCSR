import numpy as np

"""
Module Name: twiss.py

Contains helper functions involving computing twiss
"""

def twiss_from_bmadx_particles(p):
    """ Gets the twiss from bmadx particle object"""
    # Actual calc
    twiss = twiss_dispersion_calc(np.cov([p.x, p.px, p.pz]))

    # Add norm
    twiss['norm_emit'] = twiss['emit'] * p.p0c / p.mc2
    out = {}
    for k in twiss:
        out[k + f'_x'] = twiss[k]

    # Actual calc
    twiss = twiss_dispersion_calc(np.cov([p.y, p.py, p.pz]))

    # Add norm
    twiss['norm_emit'] = twiss['emit'] * p.p0c / p.mc2

    out2 = {}
    for k in twiss:
        out2[k + f'_y'] = twiss[k]

    out.update(out2)

    return out


def twiss_dispersion_calc(sigma3):
    """
    Twiss and Dispersion calculation from a 3x3 sigma (covariance) matrix from particles
    x, p, delta

    From https://github.com/ChristopherMayes/openPMD-beamphysics/blob/master/pmd_beamphysics/statistics.py

    Formulas from:
        https://uspas.fnal.gov/materials/19Knoxville/g-2/creation-and-analysis-of-beam-distributions.html

    Returns a dict with:
        alpha
        beta
        gamma
        emit
        eta
        etap

    """

    # Collect terms

    delta2 = sigma3[2, 2]
    xd = sigma3[0, 2]
    pd = sigma3[1, 2]

    eb = sigma3[0, 0] - xd ** 2 / delta2
    eg = sigma3[1, 1] - pd ** 2 / delta2
    ea = -sigma3[0, 1] + xd * pd / delta2

    emit = np.sqrt(eb * eg - ea ** 2)

    # Form the output dict
    d = {}

    d['alpha'] = ea / emit
    d['beta'] = eb / emit
    d['gamma'] = eg / emit
    d['emit'] = emit
    d['eta'] = xd / delta2
    d['etap'] = pd / delta2

    return d