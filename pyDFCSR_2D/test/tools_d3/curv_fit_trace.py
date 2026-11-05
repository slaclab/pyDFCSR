"""Track the fitted curvature coefficients (c2, c3) along the chicane in a CSR-off run, with the core u-range they
imply at |w| = 3. Large or jumping coefficients mark where the shear may be ill-conditioned."""
import sys, os, numpy as np, yaml
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from d3_wake_diagnostic import build
sim = build(13.3, {'particle_deposition': dict(curvature=True), 'CSR_integration': dict(n_formation_length=1.0, drift_reach=False)})
