"""
Element tracking that bmad-x does not do the way Bmad does.

bmad-x's sextupole (bmadx/tracking_routines/track_a_sextupole.py) advances x by `step * px`
and divides the kick by (1 + pz). With canonical px both are wrong: the drift is
x += L px / sqrt((1+pz)^2 - px^2 - py^2), and the field kick to the canonical momentum does not
depend on pz. Measured on the FACET-II W chicane sextupoles (sigma_delta ~ 1 %), bmad-x was up to
5e-5 m off Tao in x per element; the kick-drift-kick below agrees with Tao to 1e-19
(progress.md, FACET-II BC20).

Everything else is forwarded to bmadx.track_element.
"""
import sys
from collections import namedtuple

from bmadx import Particle, Drift, track_element
from bmadx.low_level.offset_particle import make_offset_particle

SextupoleKDK = namedtuple(
    'SextupoleKDK',
    ['L', 'K2', 'NUM_STEPS', 'X_OFFSET', 'Y_OFFSET', 'TILT'],
    defaults=(None, None, 1, 0.0, 0.0, 0.0))


def track_sextupole_kdk(p_in, sext):
    """
    Bmad's sextupole: NUM_STEPS of [kick, exact drift(h), kick], h = L/NUM_STEPS, each kick
    applied to the canonical momenta and independent of pz:
        px -= k (x^2 - y^2) / 2,   py += k x y,   k = K2 h / 2
    Tao with num_steps = n is reproduced to round-off by NUM_STEPS = n.
    """
    lib = sys.modules[type(p_in.x).__module__]
    offset_set = make_offset_particle(lib, 'set')
    offset_unset = make_offset_particle(lib, 'unset')

    n = int(sext.NUM_STEPS)
    h = sext.L / n
    k = sext.K2 * h / 2

    p = offset_set(sext.X_OFFSET, sext.Y_OFFSET, sext.TILT, p_in)
    x, px, y, py = p.x, p.px, p.y, p.py
    for _ in range(n):
        px = px - k * (x ** 2 - y ** 2) / 2
        py = py + k * x * y
        p = track_element(Particle(x, px, y, py, p.z, p.pz, p.s, p.p0c, p.mc2), Drift(L=h))
        x, px, y, py = p.x, p.px, p.y, p.py
        px = px - k * (x ** 2 - y ** 2) / 2
        py = py + k * x * y
    p = Particle(x, px, y, py, p.z, p.pz, p.s, p.p0c, p.mc2)
    return offset_unset(sext.X_OFFSET, sext.Y_OFFSET, sext.TILT, p)


def track(p_in, element):
    if isinstance(element, SextupoleKDK):
        return track_sextupole_kdk(p_in, element)
    return track_element(p_in, element)
