"""
PyTorch backend of the 2D CSR wake (branch gpu_diff): the same quadrature as the numba path, with
the integrand evaluation batched over all wake-mesh points on a device (CPU or CUDA), float64
throughout. The node layout (s' regions, x' bands, refinement decisions) stays the numpy code's
(WakeMixin.wake_layout); only the values are computed here, so they can later carry gradients.

Selected by CSR_computation.backend: torch (device: CSR_computation.device). Importing
pyDFCSR_2D does not import torch; this package is imported only when the backend is used.
"""
