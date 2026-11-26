"""Per-element mean-energy change [MeV] of two pyDFCSR runs from their statistics.h5, and the difference.
    python fl_per_element.py <run_a> <run_b>     (newest <run>-2*-statistics.h5 in benchmark_results/bc20_indranil/runs)"""
import glob, os, sys
import h5py, numpy as np, yaml
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'benchmark_results', 'bc20_indranil', 'runs')
LAT = os.path.join(HERE, '..', '..', 'example', 'BC20_Indranil', 'lattice_bc20w_c_0.060000_000000.yaml')
def load(run):
    f = sorted(glob.glob(os.path.join(RUNS, f'{run}-2*-statistics.h5')), key=os.path.getmtime)[-1]
    h = h5py.File(f, 'r'); return np.asarray(h['step_positions']), np.asarray(h['mean_energy']) / 1e6
lat = yaml.safe_load(open(LAT)); els = []; s = 0.0
for k, v in lat.items():
    if isinstance(v, dict) and 'L' in v:
        els.append((k, s, s + float(v['L']))); s += float(v['L'])
def per_element(run):
    sp, E = load(run)
    at = lambda x: E[int(np.argmin(np.abs(sp - x)))]
    return {k: at(b) - at(a) for k, a, b in els}, E[-1] - E[0]
a, b = sys.argv[1], sys.argv[2]
A, tA = per_element(a); B, tB = per_element(b)
groups = [('B1LE', ['b1le_1']), ('D1 (after B1LE)', ['D1']), ('B1LE exit -> B2LE (rest)', None), ('B2LE', ['b2le']),
          ('B2LE exit -> B3LE', None), ('B3LE + gap + B3RE', ['b3le', 'D16', 'q6e', 'D17', 'b3re']),
          ('after B3RE: D18 .. D24', None), ('D24 -> B2RE', None), ('B2RE', ['b2re']), ('D30', ['D30']),
          ('D30 -> D31 (q1er ..)', None), ('D31', ['D31']), ('B1RE', ['b1re']), ('D32', ['D32'])]
names = [e[0] for e in els]
def span(n1, n2):           # elements strictly between n1 and n2 (by order)
    i, j = names.index(n1), names.index(n2); return names[i + 1:j]
fill = {'B1LE exit -> B2LE (rest)': [n for n in span('b1le_1', 'b2le') if n != 'D1'], 'B2LE exit -> B3LE': span('b2le', 'b3le'),
        'after B3RE: D18 .. D24': span('b3re', 'D24') + ['D24'], 'D24 -> B2RE': span('D24', 'b2re'),
        'D30 -> D31 (q1er ..)': span('D30', 'D31')}
print(f'{"section":28s} {a:>14s} {b:>14s} {"b - a":>9s}   [MeV]')
for lab, mem in groups:
    mem = mem if mem is not None else fill[lab]
    va, vb = sum(A[m] for m in mem), sum(B[m] for m in mem)
    print(f'{lab:28s} {va:+14.3f} {vb:+14.3f} {vb - va:+9.3f}')
print(f'{"TOTAL":28s} {tA:+14.3f} {tB:+14.3f} {tB - tA:+9.3f}   ({(tB / tA - 1) * 100:+.2f}%)')
big = sorted(names, key=lambda n: -abs(B[n] - A[n]))[:6]
print('largest element changes:', ', '.join(f'{n} {B[n] - A[n]:+.3f}' for n in big))
