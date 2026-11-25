import sys, numpy as np
beam = sys.argv[1]
off = np.load(f'{beam}_off.npz'); p0c = float(off['p0c']); names = list(off['names'])
def cum(run):
    d = np.load(f'{run}.npz'); return (d['exit_mean'][:, 5] - off['exit_mean'][:, 5]) * p0c / 1e3   # keV
# element ends in order; pick b1le end, the end of the element before b2le, b2le end
iB1 = max(i for i, n in enumerate(names) if n.startswith('b1le'))
iB2s = min(i for i, n in enumerate(names) if n.startswith('b2le')); iB2e = max(i for i, n in enumerate(names) if n.startswith('b2le'))
print(f'{beam}: energy change vs CSR off [keV] collected in  b1le | b1le exit -> b2le entrance | b2le')
for run in (f'{beam}_v2', beam, f'{beam}_1d'):
    try:
        c = cum(run)
    except FileNotFoundError:
        continue
    print(f'   {run:9s} {c[iB1]:+9.1f} | {c[iB2s - 1] - c[iB1]:+9.1f} | {c[iB2e] - c[iB2s - 1]:+9.1f}')
