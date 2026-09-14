"""
PROTOTYPE ONLY -- nothing in pyDFCSR_2D imports this.

Goal: locate the bracketing snapshot index for a retarded time, in O(1), when the snapshot
times are NON-uniform. Today they are uniform and the lookup is one division
(interp3D.py:499-510):

    t_idx = (q - min_t) / delta_t
    k = int(floor(t_idx));  k = 0 if k < 0 else (n-2 if k >= n-1 else k)
    a = t_idx - k;          a = 0.0 if a < 0 else (1.0 if a > 1 else a)

and the caller forms (1-a)*f[k] + a*f[k+1]. This sits in the innermost loop of the CSR
integrand and runs of order 1e8 times per run, so the replacement has to be O(1) with a small
constant, not merely asymptotically fine.

Two candidate structures are implemented and measured:

  SEGMENT  piecewise-uniform. Our schedules are piecewise uniform BY CONSTRUCTION -- a few
           refined windows (bend edges, waists) in an otherwise coarse lattice -- so store
           (t0, dt, first, n) per segment and do one division inside the located segment.
           At S == 1 this reduces algebraically to the current expression, so bit-for-bit
           reproduction of every existing result is free rather than something to fight for.

  BUCKET   general. A uniform auxiliary index grid over the whole span; bucket[j] gives a
           starting node index, then a bounded correction loop. Works for arbitrary times,
           costs auxiliary memory, and CANNOT be bit-identical because
           a = (q-t[k])/(t[k+1]-t[k]) is not the same floating-point expression as t_idx - k.

The subtle failure mode for BUCKET is not the loop bound -- in exact arithmetic a bucket of
width <= h_min contains at most one interior node. It is that `j` can come out one too HIGH if
the build-time and run-time expressions round differently; then t[bucket[j]] > q and an
increment-only loop can never recover, giving a wrong k and a negative weight. The table is
therefore built with the strict-predecessor construction using the exact runtime expression,
and the real max-nodes-per-bucket is measured at build time and stored as the verified bound.

Run:  python prototype_nonuniform_lookup.py          (correctness + benchmark)
"""
import math
import time

import numpy as np
from numba import jit

# ----------------------------------------------------------------------------------------
# (a) the CURRENT production contract, uniform only -- the reference and the speed baseline
# ----------------------------------------------------------------------------------------


@jit(nopython=True, cache=True)
def lookup_uniform(q, min_t, delta_t, n):
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


@jit(nopython=True, cache=True)
def run_uniform(qs, min_t, delta_t, n):
    acc_k = 0
    acc_a = 0.0
    for i in range(qs.shape[0]):
        k, a = lookup_uniform(qs[i], min_t, delta_t, n)
        acc_k += k
        acc_a += a
    return acc_k, acc_a


# ----------------------------------------------------------------------------------------
# (b) SEGMENT: piecewise uniform
# ----------------------------------------------------------------------------------------


def build_segments(times, rtol=1e-9):
    """
    Split a sorted time array into maximal runs of (near-)constant spacing.

    Returns seg_t0, seg_dt, seg_first, seg_n. Segment s owns node indices
    [seg_first[s], seg_first[s] + seg_n[s] - 1]; the node shared with the next segment
    belongs to exactly one of them, so index ranges never overlap.
    """
    times = np.asarray(times, dtype=np.float64)
    n = times.size
    if n < 2:
        raise ValueError('need at least 2 nodes')
    d = np.diff(times)
    if np.any(d <= 0.0):
        raise ValueError('times must be strictly increasing')

    starts = [0]
    for i in range(1, d.size):
        if abs(d[i] - d[starts[-1]]) > rtol * max(abs(d[i]), abs(d[starts[-1]])):
            starts.append(i)
    starts.append(d.size)

    seg_t0, seg_dt, seg_first, seg_n = [], [], [], []
    for j in range(len(starts) - 1):
        i0, i1 = starts[j], starts[j + 1]          # intervals i0..i1-1
        seg_first.append(i0)
        seg_n.append(i1 - i0 + 1)                  # nodes i0..i1
        seg_t0.append(times[i0])
        seg_dt.append(d[i0])
    return (np.array(seg_t0), np.array(seg_dt),
            np.array(seg_first, dtype=np.int64), np.array(seg_n, dtype=np.int64))


@jit(nopython=True, cache=True)
def lookup_segment(q, seg_t0, seg_dt, seg_first, seg_n, n):
    S = seg_t0.shape[0]
    s = 0
    while s < S - 1 and q >= seg_t0[s + 1]:
        s += 1
    loc = (q - seg_t0[s]) / seg_dt[s]
    f = math.floor(loc)
    k = seg_first[s] + int(f)
    a = loc - f
    # clamp into this segment's own index range, then into the global bracket range
    lo = seg_first[s]
    hi = seg_first[s] + seg_n[s] - 2          # last index that has a right neighbour
    if hi < lo:
        hi = lo
    if k < lo:
        k = lo
        a = 0.0
    if k > hi:
        k = hi
        a = 1.0
    if k < 0:
        k = 0
    if k >= n - 1:
        k = n - 2
    if a < 0.0:
        a = 0.0
    if a > 1.0:
        a = 1.0
    return k, a


@jit(nopython=True, cache=True)
def run_segment(qs, seg_t0, seg_dt, seg_first, seg_n, n):
    acc_k = 0
    acc_a = 0.0
    for i in range(qs.shape[0]):
        k, a = lookup_segment(qs[i], seg_t0, seg_dt, seg_first, seg_n, n)
        acc_k += k
        acc_a += a
    return acc_k, acc_a


# ----------------------------------------------------------------------------------------
# (c) BUCKET: general non-uniform
# ----------------------------------------------------------------------------------------


def build_bucket(times, cap_factor=4):
    """
    Uniform auxiliary index grid, strict-predecessor construction.

    bucket[j] = max{k : j_k < j}, where j_k = int((t[k]-t0) * inv_h) uses the EXACT runtime
    expression. Building from the runtime expression (rather than from t0 + j*h) is what
    guarantees t[bucket[j]] <= q for every q landing in bucket j: if the two disagreed by one
    ulp, j could come out too high and an increment-only correction loop could never recover.

    M is capped at cap_factor*n so the table stays cache-resident: a 40 MB table would be an
    LLC miss on every lookup, which costs more than a few extra iterations over a node array
    that is hot in L1. The real max nodes-per-bucket is then measured and returned as the
    verified loop bound.
    """
    times = np.asarray(times, dtype=np.float64)
    n = times.size
    d = np.diff(times)
    h_min = float(d.min())
    span = float(times[-1] - times[0])
    M_ideal = int(math.ceil(span / h_min))
    M = min(M_ideal, cap_factor * n)
    M = max(M, 1)
    inv_h = M / span

    j_k = ((times - times[0]) * inv_h).astype(np.int64)
    np.clip(j_k, 0, M, out=j_k)

    bucket = np.zeros(M + 2, dtype=np.int64)
    k = 0
    for j in range(M + 2):
        while k + 1 < n and j_k[k + 1] < j:
            k += 1
        bucket[j] = k

    # verified loop bound: worst-case corrections actually needed
    worst = 0
    for j in range(M + 1):
        lo = bucket[j]
        cnt = 0
        kk = lo
        while kk + 1 < n and j_k[kk + 1] <= j:
            kk += 1
            cnt += 1
        worst = max(worst, cnt)
    return bucket, inv_h, float(times[0]), M, worst, M_ideal


@jit(nopython=True, cache=True)
def lookup_bucket(q, times, bucket, inv_h, t0, M, n):
    if q <= times[0]:
        return 0, 0.0
    if q >= times[n - 1]:
        return n - 2, 1.0
    j = int((q - t0) * inv_h)
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


@jit(nopython=True, cache=True)
def run_bucket(qs, times, bucket, inv_h, t0, M, n):
    acc_k = 0
    acc_a = 0.0
    for i in range(qs.shape[0]):
        k, a = lookup_bucket(qs[i], times, bucket, inv_h, t0, M, n)
        acc_k += k
        acc_a += a
    return acc_k, acc_a


@jit(nopython=True, cache=True)
def bucket_iters(qs, times, bucket, inv_h, t0, M, n):
    """Correction-loop iteration histogram, for asserting the bound rather than claiming it."""
    hist = np.zeros(16, dtype=np.int64)
    for i in range(qs.shape[0]):
        q = qs[i]
        if q <= times[0] or q >= times[n - 1]:
            hist[0] += 1
            continue
        j = int((q - t0) * inv_h)
        if j < 0:
            j = 0
        if j > M:
            j = M
        k = bucket[j]
        c = 0
        while k + 1 < n - 1 and times[k + 1] <= q:
            k += 1
            c += 1
        hist[min(c, 15)] += 1
    return hist


# ----------------------------------------------------------------------------------------
# (c2) HYBRID: the variant that would actually ship.
#
# One branch on a precomputed is_uniform flag. Uniform histories (every run today) take the
# exact current expression and are therefore bit-for-bit unchanged; non-uniform histories take
# the bucket table. Costs one predictable, perfectly-predicted branch.
# ----------------------------------------------------------------------------------------


@jit(nopython=True, cache=True)
def lookup_hybrid(q, times, bucket, inv_h, t0, M, n, is_uniform, min_t, delta_t):
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
    return lookup_bucket(q, times, bucket, inv_h, t0, M, n)


@jit(nopython=True, cache=True)
def run_hybrid(qs, times, bucket, inv_h, t0, M, n, is_uniform, min_t, delta_t):
    acc_k = 0
    acc_a = 0.0
    for i in range(qs.shape[0]):
        k, a = lookup_hybrid(qs[i], times, bucket, inv_h, t0, M, n,
                             is_uniform, min_t, delta_t)
        acc_k += k
        acc_a += a
    return acc_k, acc_a


# ----------------------------------------------------------------------------------------
# (d) reference oracles -- correctness only, never the hot path
# ----------------------------------------------------------------------------------------


@jit(nopython=True, cache=True)
def lookup_binary(q, times, n):
    if q <= times[0]:
        return 0, 0.0
    if q >= times[n - 1]:
        return n - 2, 1.0
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if times[mid] <= q:
            lo = mid
        else:
            hi = mid
    a = (q - times[lo]) / (times[lo + 1] - times[lo])
    return lo, a


@jit(nopython=True, cache=True)
def run_binary(qs, times, n):
    acc_k = 0
    acc_a = 0.0
    for i in range(qs.shape[0]):
        k, a = lookup_binary(qs[i], times, n)
        acc_k += k
        acc_a += a
    return acc_k, acc_a


@jit(nopython=True, cache=True)
def run_searchsorted(qs, times, n):
    acc_k = 0
    acc_a = 0.0
    for i in range(qs.shape[0]):
        k = np.searchsorted(times, qs[i], 'right') - 1
        if k < 0:
            k = 0
        if k >= n - 1:
            k = n - 2
        a = (qs[i] - times[k]) / (times[k + 1] - times[k])
        if a < 0.0:
            a = 0.0
        if a > 1.0:
            a = 1.0
        acc_k += k
        acc_a += a
    return acc_k, acc_a


def oracle(q, times):
    n = times.size
    k = int(np.clip(np.searchsorted(times, q, 'right') - 1, 0, n - 2))
    a = (q - times[k]) / (times[k + 1] - times[k])
    return k, float(np.clip(a, 0.0, 1.0))


# ----------------------------------------------------------------------------------------
# node-set generators
# ----------------------------------------------------------------------------------------


def nodes_uniform(n, t0=0.0, h=6e-4):
    return t0 + h * np.arange(n, dtype=np.float64)


def nodes_piecewise(n_coarse=40, n_fine=60, h_coarse=5e-2, h_fine=6e-4, n_tail=30):
    """Coarse, then a refined window, then coarse again -- the schedule shape we will emit."""
    a = nodes_uniform(n_coarse, 0.0, h_coarse)
    b = a[-1] + h_fine * np.arange(1, n_fine + 1)
    c = b[-1] + h_coarse * np.arange(1, n_tail + 1)
    return np.concatenate([a, b, c])


def nodes_one_tiny_gap(n=200, h=5e-2, tiny=1e-6):
    t = nodes_uniform(n, 0.0, h)
    t[n // 2:] += tiny - h            # one tiny interval in the middle
    return np.sort(np.unique(t))


def nodes_random(n=500, seed=0, ratio=1e3):
    rng = np.random.default_rng(seed)
    d = 10.0 ** rng.uniform(0, math.log10(ratio), n - 1)
    return np.concatenate([[0.0], np.cumsum(d)]) * 1e-4


# ----------------------------------------------------------------------------------------
# correctness
# ----------------------------------------------------------------------------------------


def check(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL':<5} {name}" + (f'   {detail}' if detail else ''))
    return bool(cond)


def test_correctness():
    print('=' * 92)
    print('CORRECTNESS')
    print('=' * 92)
    ok = True
    rng = np.random.default_rng(1)

    # --- 1. S == 1 must be BIT-FOR-BIT identical to the production contract -------------
    print('\n1. uniform nodes: segment path vs the current production expression')
    for n, h in ((50, 6e-4), (1500, 6e-4), (20000, 5e-2)):
        t = nodes_uniform(n, 0.0, h)
        sg = build_segments(t)
        qs = np.concatenate([
            rng.uniform(t[0] - 3 * h, t[-1] + 3 * h, 20000),
            t.copy(), t + 0.5 * h, t - 1e-18, np.array([t[0], t[-1]]),
        ])
        bad = 0
        for q in qs:
            k1, a1 = lookup_uniform(q, t[0], h, n)
            k2, a2 = lookup_segment(q, *sg, n)
            # bit-for-bit: identical index AND identical float bits for the weight
            if k1 != k2 or a1.hex() != a2.hex():
                bad += 1
        ok &= check(f'n={n:<6} S=1 bit-for-bit over {qs.size} queries', bad == 0,
                    f'{bad} mismatches')

    # --- 2. oracle equivalence on genuinely non-uniform nodes ---------------------------
    print('\n2. oracle equivalence (value, not index) on non-uniform nodes')
    for label, t in (('piecewise', nodes_piecewise()),
                     ('one tiny gap', nodes_one_tiny_gap()),
                     ('random ratio 1e3', nodes_random())):
        n = t.size
        sg = build_segments(t)
        bk = build_bucket(t)
        qs = np.concatenate([
            rng.uniform(t[0], t[-1], 200000),
            t.copy(), 0.5 * (t[:-1] + t[1:]),
            np.array([t[0] - 1.0, t[-1] + 1.0, t[0], t[-1]]),
        ])
        f = 3.0 * t - 1.7                      # a linear function: must be reproduced exactly
        worst_seg = 0.0
        worst_bkt = 0.0
        for q in qs:
            ko, ao = oracle(q, t)
            vo = (1 - ao) * f[ko] + ao * f[ko + 1]
            ks, as_ = lookup_segment(q, *sg, n)
            vs = (1 - as_) * f[ks] + as_ * f[ks + 1]
            kb, ab = lookup_bucket(q, t, bk[0], bk[1], bk[2], bk[3], n)
            vb = (1 - ab) * f[kb] + ab * f[kb + 1]
            worst_seg = max(worst_seg, abs(vs - vo))
            worst_bkt = max(worst_bkt, abs(vb - vo))
        scale = max(abs(f).max(), 1.0)
        ok &= check(f'{label:<18} segment  max |value - oracle|',
                    worst_seg <= 1e-11 * scale, f'{worst_seg:.3e}')
        ok &= check(f'{label:<18} bucket   max |value - oracle|',
                    worst_bkt <= 1e-11 * scale, f'{worst_bkt:.3e}')

    # --- 3. overshoot regression at every bucket edge -----------------------------------
    print('\n3. overshoot: q one ulp either side of EVERY bucket edge')
    for label, t in (('piecewise', nodes_piecewise()),
                     ('random ratio 1e3', nodes_random(n=300))):
        n = t.size
        bucket, inv_h, t0, M, worst, M_ideal = build_bucket(t)
        edges = t0 + np.arange(M + 1) / inv_h
        qs = np.concatenate([np.nextafter(edges, -np.inf), edges,
                             np.nextafter(edges, np.inf)])
        qs = qs[(qs > t[0]) & (qs < t[-1])]
        viol = 0
        for q in qs:
            k, a = lookup_bucket(q, t, bucket, inv_h, t0, M, n)
            if not (t[k] <= q <= t[k + 1]) or not (0.0 <= a <= 1.0):
                viol += 1
        ok &= check(f'{label:<18} bracket holds at {qs.size} edge-adjacent queries',
                    viol == 0, f'{viol} violations')

    # --- 4. the loop bound, asserted rather than claimed --------------------------------
    print('\n4. correction-loop iteration bound vs spacing ratio')
    for ratio in (1e1, 1e2, 1e3, 1e4):
        t = nodes_random(n=400, seed=2, ratio=ratio)
        n = t.size
        bucket, inv_h, t0, M, worst, M_ideal = build_bucket(t)
        qs = rng.uniform(t[0], t[-1], 200000)
        hist = bucket_iters(qs, t, bucket, inv_h, t0, M, n)
        obs = int(np.max(np.nonzero(hist)[0])) if hist.any() else 0
        capped = M < M_ideal
        # Two separate claims:
        #  (i)  the build-time measured bound is PREDICTIVE of runtime -- this is the
        #       property the implementation relies on, and it must hold always;
        #  (ii) the theoretical bound of <=1 interior node per bucket holds only when the
        #       table is NOT capped. Capping to stay cache-resident deliberately buys a
        #       longer loop in exchange for fewer cache misses, so a larger bound there is
        #       the designed behaviour, not a defect.
        ok &= check(f'ratio {ratio:>7.0e}  M={M:<6} (ideal {M_ideal:<9}) '
                    f'build-bound={worst}  observed={obs}'
                    + ('  [capped]' if capped else '  [uncapped]'),
                    obs <= worst and (capped or worst <= 1))

    # --- 5. edge cases ------------------------------------------------------------------
    print('\n5. edge cases')
    t = nodes_uniform(2, 0.0, 1e-3)
    sg = build_segments(t)
    k, a = lookup_segment(t[-1], *sg, 2)
    ok &= check('n=2, q=t[-1] gives (k=0, a=1)', k == 0 and abs(a - 1.0) < 1e-15,
                f'k={k}, a={a}')
    t = nodes_piecewise()
    sg = build_segments(t)
    k, a = lookup_segment(t[-1], *sg, t.size)
    ok &= check('q=t[n-1] gives k=n-2 (never n-1)', k == t.size - 2, f'k={k}, a={a}')
    k, a = lookup_segment(t[0], *sg, t.size)
    ok &= check('q=t[0] gives (k=0, a=0)', k == 0 and a == 0.0, f'k={k}, a={a}')
    seg_t0, seg_dt, seg_first, seg_n = sg
    ok &= check(f'segmentation found S={seg_t0.size} segments for the 3-window schedule',
                seg_t0.size == 3, f'S={seg_t0.size}')
    # a 1-ulp gap must not divide by zero
    t = nodes_uniform(50, 0.0, 1e-3)
    t[25] = np.nextafter(t[24], np.inf)
    t = np.sort(t)
    bk = build_bucket(t)
    kk, aa = lookup_bucket(0.5 * (t[24] + t[26]), t, bk[0], bk[1], bk[2], bk[3], t.size)
    ok &= check('1-ulp gap: finite weight, no division blow-up',
                np.isfinite(aa) and 0.0 <= aa <= 1.0, f'a={aa}')

    print('\n  ' + ('ALL CORRECTNESS CHECKS PASSED' if ok else 'SOME CHECKS FAILED'))
    return ok


# ----------------------------------------------------------------------------------------
# benchmark
# ----------------------------------------------------------------------------------------


def bench_one(fn, args, reps=3):
    fn(*args)                                  # JIT warm-up
    best = math.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def test_benchmark(NQ=2_000_000):
    print('\n' + '=' * 92)
    print(f'BENCHMARK  ({NQ:,} lookups per cell, ns per lookup, best of 3)')
    print('=' * 92)
    rng = np.random.default_rng(7)
    cases = [
        ('uniform  n=50',     nodes_uniform(50, 0.0, 6e-4)),
        ('uniform  n=1500',   nodes_uniform(1500, 0.0, 6e-4)),
        ('uniform  n=20000',  nodes_uniform(20000, 0.0, 5e-2)),
        ('piecewise S=3',     nodes_piecewise()),
        ('piecewise S=8',     np.unique(np.concatenate(
            [nodes_piecewise(20, 30, 5e-2, 6e-4, 10),
             nodes_piecewise(20, 30, 5e-2, 6e-4, 10)[-1]
             + nodes_piecewise(20, 30, 2e-2, 3e-4, 10)]))),
        ('random ratio 1e3',  nodes_random(n=1500)),
    ]
    print(f"{'node set':<20} {'S':>3} {'order':<11} {'uniform':>9} {'segment':>9} "
          f"{'bucket':>9} {'hybrid':>9} {'binary':>9} {'srchsrt':>9}")
    for label, t in cases:
        n = t.size
        try:
            sg = build_segments(t)
        except ValueError:
            continue
        S = sg[0].size
        bucket, inv_h, t0, M, worst, M_ideal = build_bucket(t)
        h_uni = (t[-1] - t[0]) / (n - 1)
        for order in ('sorted', 'clustered', 'random'):
            if order == 'sorted':
                qs = np.linspace(t[0], t[-1], NQ)
            elif order == 'clustered':
                c = 0.5 * (t[0] + t[-1])
                qs = np.clip(c + rng.normal(0, 0.02 * (t[-1] - t[0]), NQ), t[0], t[-1])
            else:
                qs = rng.uniform(t[0], t[-1], NQ)
            tu = bench_one(run_uniform, (qs, t[0], h_uni, n)) / NQ * 1e9
            ts = bench_one(run_segment, (qs, *sg, n)) / NQ * 1e9
            tb = bench_one(run_bucket, (qs, t, bucket, inv_h, t0, M, n)) / NQ * 1e9
            tn = bench_one(run_binary, (qs, t, n)) / NQ * 1e9
            tss = bench_one(run_searchsorted, (qs, t, n)) / NQ * 1e9
            uni = S == 1
            th = bench_one(run_hybrid, (qs, t, bucket, inv_h, t0, M, n, uni,
                                        t[0], h_uni)) / NQ * 1e9
            print(f'{label:<20} {S:>3} {order:<11} {tu:>9.2f} {ts:>9.2f} {tb:>9.2f} '
                  f'{th:>9.2f} {tn:>9.2f} {tss:>9.2f}')
    print('\n  acceptance: segment within 1.3x of uniform at S <= 8')


if __name__ == '__main__':
    good = test_correctness()
    test_benchmark()
    raise SystemExit(0 if good else 1)
