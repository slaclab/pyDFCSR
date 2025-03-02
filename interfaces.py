import numpy as np
from bmadx.constants import C_LIGHT, M_ELECTRON
from bmadx.structures import Particle
import numpy as np
import torch
import sys

# Modified from https://github.com/bmad-sim/Bmad-X

from pmd_beamphysics import ParticleGroup
def openpmd_to_bmadx_coords(
        pmd_particle: ParticleGroup,
        p0c
    ):
    """
    Transforms openPMD-beamphysics ParticleGroup to
    bmad phase-space coordinates.

        Parameters:
            pmd_particle (pmd_beamphysics.ParticleGroup): openPMD-beamphysics ParticleGroup
            p0c (float): reference momentum in eV

        Returns:
            bmad_coods (list): list of bmad coords (x, px, y, py, z, pz)
    """

    x = pmd_particle.x
    px = pmd_particle.px / p0c
    y = pmd_particle.y
    py = pmd_particle.py / p0c
    z =  pmd_particle.z
    pz = pmd_particle.p / p0c - 1.0

    bmad_coords = (x, px, y, py, z, pz)

    return bmad_coords

def openpmd_to_bmadx_particles(
        pmd_particle: ParticleGroup,
        p0c: float,
        s : float = 0.0,
        mc2 : float = M_ELECTRON
    ):
    """
    Transforms openPMD-beamphysics ParticleGroup to
    bmad phase-space Particle named tuple.
        Parameters:
            pmd_particle (pmd_beamphysics.ParticleGroup): openPMD-beamphysics ParticleGroup
            p0c (float): reference momentum in eV

        Returns:
            Bmadx Particle
    """
    coords = openpmd_to_bmadx_coords(pmd_particle, p0c)
    particle = Particle(
        *coords,
        s = s,
        p0c = p0c,
        mc2 = mc2)
    return particle

def bmadx_particles_to_openpmd(particle: Particle, charge):
    """
    Transforms bmadx Particle to openPMD-beamphysics ParticleGroup.
    Parameters:
        particle: bmax Particle particle to transform
    Returns:
        pmd_beamphysics.ParticleGroup
    """
    lib = sys.modules[type(particle.x).__module__]
    if lib == np:
        x = particle.x
        px = particle.px
        y = particle.y
        py = particle.py
        z = particle.z
        pz = particle.pz

    elif lib == torch:
        x = particle.x.detach().numpy()
        px = particle.px.detach().numpy()
        y = particle.y.detach().numpy()
        py = particle.py.detach().numpy()
        z = particle.z.detach().numpy()
        pz = particle.pz.detach().numpy()

    else:
        raise ValueError('Only numpy and torch Particles are supported as of now')

    dat = {}

    dat['x'] = x
    dat['px'] = px * particle.p0c
    dat['y'] = y
    dat['py'] = py * particle.p0c
    dat['z'] = z
    dat['pz'] = particle.p0c * ((pz + 1.0) ** 2 - px ** 2 - py ** 2) ** 0.5

    p = (1 + pz) * particle.p0c
    beta = (
                   (p / M_ELECTRON) ** 2 /
                   (1 + (p / M_ELECTRON) ** 2)
           ) ** 0.5

    dat['t'] = (particle.s - z) / (C_LIGHT * beta)

    dat['status'] = np.ones_like(x, dtype=int)
    dat['weight'] = np.ones_like(x) * charge/len(x)

    if np.isclose(particle.mc2, M_ELECTRON):
        dat['species'] = 'electron'
    else:
        raise ValueError('only electrons are supported as of now')

    return ParticleGroup(data=dat)
