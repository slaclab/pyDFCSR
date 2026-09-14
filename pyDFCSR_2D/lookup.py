"""
O(1) bracketing lookup for the density history, for uniform AND non-uniform snapshot times.

The problem
-----------
The CSR integral needs the density at a RETARDED time q, which falls between two stored
snapshots. So every integrand evaluation -- measured at ~1e8 per wake mesh, in query arrays of
median 34 200 entries -- must return the bracketing index k and the blend weight a, after which
the caller forms (1-a)*f[k] + a*f[k+1].

With equally spaced snapshots that is one division. Once the schedule refines locally (around
waists and bend edges) the times are no longer equally spaced and the division is simply wrong,
while the obvious replacement -- binary search -- costs 7-38 ns against 0.74 ns.

The design: HYBRID
------------------
One branch on a precomputed `is_uniform` flag.

  uniform      -> the exact historical expression, so results are BIT-FOR-BIT unchanged
  non-uniform  -> a bucket table, flat at 1.5-3.5 ns regardless of how the spacing is shaped

Benchmarked in progress Step 9 (prototype in test/prototype_nonuniform_lookup.py):

    node set               S order         uniform   segment    bucket    hybrid
    uniform  n=20000       1 sorted           0.74      1.03      2.42      0.73
    piecewise S=8          6 random           0.74      6.72      2.98      2.97
    random ratio 1e3    1499 random           0.75    196.58      3.26      3.32

A per-segment scheme was the original plan and lost: its linear segment scan mispredicts on
random queries and collapses to 196 ns when the times are not cleanly segmentable, which a
scheduler driven by waists and bend edges cannot guarantee. The hybrid's uniform branch gets the
bit-for-bit guarantee for free, at 0.73 ns against the 0.74 ns baseline.

Cost of the non-uniform path, measured end to end: **+1.84 %** of a wake mesh, because the lookup
is only 0.49 % of the work -- 32 million lookups come to 24 ms, while the 160-tap B-spline
evaluation that follows each one is ~50x more expensive than finding the index.
"""
import math

import numpy as np
from numba import jit

UNIFORM_RTOL = 1e-9


def is_uniform_times(times, rtol=UNIFORM_RTOL):
    """
    Is this time array uniformly spaced, to within floating-point noise?

    The tolerance is essential, not cosmetic. Snapshot times come from
    `np.arange(0, L + h/2, h)`, whose successive differences are NOT exactly h -- measured, 35 of
    37 differ by ~1e-16. An exact test would call the historical uniform grid "non-uniform", send
    it down the bucket path, and silently destroy the bit-for-bit guarantee.
    """
    t = np.asarray(times, dtype=np.float64)
    if t.size < 3:
        return True
    d = np.diff(t)
    m = d.mean()
    if m <= 0.0:
        return False
    return bool((d.max() - d.min()) <= rtol * m)


def build_bucket(times, cap_factor=4):
    """
    Uniform auxiliary index grid over the span, strict-predecessor construction.

    Returns (bucket, inv_h, t0, M, max_iters).

    Two non-obvious requirements, both established by the prototype's overshoot tests:

    * `j_k` is built with the EXACT expression used at lookup time. If the build and the runtime
      rounded differently, `j` could come out one too high, `t[bucket[j]] > q`, and an
      increment-only correction loop could never recover -- wrong index, negative weight. The
      strict-predecessor form plus IEEE division monotonicity guarantees `t[bucket[j]] <= q`.
    * `M` is capped so the table stays cache-resident. A 40 MB table would be an LLC miss on
      every lookup, costing more than the few extra iterations it saves over a node array that is
      hot in L1. The real worst-case iteration count is then MEASURED here and returned, rather
      than asserted in a comment.
    """
    t = np.asarray(times, dtype=np.float64)
    n = t.size
    if n < 2:
        return np.zeros(2, dtype=np.int64), 1.0, float(t[0] if n else 0.0), 1, 0
    d = np.diff(t)
    h_min = float(d.min())
    span = float(t[-1] - t[0])
    if h_min <= 0.0 or span <= 0.0:
        raise ValueError('snapshot times must be strictly increasing')

    M = max(min(int(math.ceil(span / h_min)), cap_factor * n), 1)
    inv_h = M / span

    j_k = ((t - t[0]) * inv_h).astype(np.int64)
    np.clip(j_k, 0, M, out=j_k)

    bucket = np.zeros(M + 2, dtype=np.int64)
    k = 0
    for j in range(M + 2):
        while k + 1 < n and j_k[k + 1] < j:
            k += 1
        bucket[j] = k

    worst = 0
    for j in range(M + 1):
        kk = bucket[j]
        c = 0
        while kk + 1 < n and j_k[kk + 1] <= j:
            kk += 1
            c += 1
        if c > worst:
            worst = c
    return bucket, float(inv_h), float(t[0]), int(M), int(worst)


def empty_bucket():
    """Placeholder table for the uniform path, so the numba signature stays fixed."""
    return np.zeros(2, dtype=np.int64), 1.0, 0.0, 1


@jit(nopython=True, cache=True)
def lookup_hybrid(q, times, bucket, inv_h, bt0, M, n, is_uniform, min_t, delta_t):
    """
    Return (k, a) with k in [0, n-2] and a in [0, 1], so the caller can form
    (1-a)*f[k] + a*f[k+1]. Queries outside the range clamp to the end nodes.

    The uniform branch is the historical expression character for character, which is what makes
    existing results reproduce bit-for-bit.
    """
    if is_uniform:
        t_idx = (q - min_t) / delta_t
        k = int(math.floor(t_idx))
        if k < 0:
            k = 0
        if k >= n - 1:
            k = n - 2
        a = t_idx - k
        if a < 0.0:
            a = 0.0
        if a > 1.0:
            a = 1.0
        return k, a

    if q <= times[0]:
        return 0, 0.0
    if q >= times[n - 1]:
        return n - 2, 1.0
    j = int((q - bt0) * inv_h)
    if j < 0:
        j = 0
    if j > M:
        j = M
    k = bucket[j]
    while k + 1 < n - 1 and times[k + 1] <= q:
        k += 1
    a = (q - times[k]) / (times[k + 1] - times[k])
    if a < 0.0:
        a = 0.0
    if a > 1.0:
        a = 1.0
    return k, a


def lookup_vec(q, times, is_uniform, min_t, delta_t, n):
    """
    Vectorized twin of lookup_hybrid, for the numpy mirrors in CSR.py.

    `_comoving_frame_at` and the band construction reproduce the interpolant's blend on whole
    arrays, and they MUST agree with it -- a band located with a different blend than the
    interpolant uses will not sit where the density is. So the two must share this arithmetic.

    Uses searchsorted rather than the bucket table on purpose: these are called a few thousand
    times per wake mesh on modest arrays and account for ~9 % of runtime between them, so the
    O(log n) is irrelevant here, whereas the scalar path runs ~1e8 times and is not.
    """
    q = np.asarray(q, dtype=np.float64)
    if is_uniform:
        t_idx = (q - min_t) / delta_t
        k = np.clip(np.floor(t_idx).astype(int), 0, max(n - 2, 0))
        a = np.clip(t_idx - k, 0.0, 1.0)
        return k, a
    t = np.asarray(times, dtype=np.float64)
    k = np.clip(np.searchsorted(t, q, side='right') - 1, 0, max(n - 2, 0))
    k1 = np.minimum(k + 1, n - 1)
    denom = t[k1] - t[k]
    a = np.where(denom > 0.0, (q - t[k]) / np.where(denom > 0.0, denom, 1.0), 0.0)
    return k, np.clip(a, 0.0, 1.0)
