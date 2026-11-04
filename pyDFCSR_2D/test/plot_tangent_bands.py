"""
Figure for integration_region.md section 8: why _retarded_xi_bands dropped near columns (tangent_bands).
Uses the D3 numbers at s = 12.30 m (tau = -6.91, sigma_xi = 34.7 um) and the saved coulomb_/vxparts_ results.
Writes benchmark_results/d3_diagnostic/tangent_bands_explained.png
"""
import os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'benchmark_results', 'd3_diagnostic')

SURF, INK, INK2, MUTED, GRID, BASE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'axes.edgecolor': BASE, 'axes.labelcolor': INK2,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.facecolor': SURF, 'figure.facecolor': SURF,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlecolor': INK,
                     'axes.titlelocation': 'left'})

TAU, SXI = -6.91, 34.7          # D3, s = 12.30 m;  um
D = 100.0                        # column distance s - s' [um]
XI_OBS = 0.37 * SXI              # row 6 observer [um]

fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.9), gridspec_kw=dict(width_ratios=[1.15, 1, 1.1], wspace=0.32))

# ---------------- (a) one column, two observers ----------------
ax = axs[0]
y = np.linspace(-45, 25, 400)
xi_grid = np.linspace(-3, 3, 300)
ax.imshow(np.exp(-xi_grid[:, None] ** 2 / 2) * np.ones((1, 2)), extent=(y[0], y[-1], -3, 3), origin='lower',
          aspect='auto', cmap='Greys', vmin=0, vmax=4.0, zorder=0)
ax.axhline(0, color=INK2, lw=1.2, ls=(0, (5, 3)), zorder=2)
ax.text(24, -0.08, 'beam centre line  ξ = 0', ha='right', va='top', color=INK2, fontsize=9)
ax.text(y[0] + 1, 2.55, 'grey = beam density at the retarded time', color=MUTED, fontsize=8.5, va='top')
for xo, col, lab in ((+XI_OBS, ORANGE, 'observer ABOVE the axis'), (-XI_OBS, BLUE, 'observer BELOW the axis')):
    xr = (xo + y - TAU * y ** 2 / (2 * D)) / SXI
    ax.plot(y, xr, color=col, lw=2, zorder=3)
    ax.plot(0, xo / SXI, 'o', ms=8, color=col, mec=SURF, mew=2, zorder=4)
    disc = 1 + 2 * TAU * xo / D
    if disc >= 0:
        for r in ((-1 + np.sqrt(disc)) / (-TAU / D), (-1 - np.sqrt(disc)) / (-TAU / D)):
            ax.plot(r, 0, 'X', ms=10, color=col, mec=SURF, mew=1.5, zorder=4)
    ym = D / TAU; xm = (xo + D / (2 * TAU)) / SXI
    if xo > 0:
        ax.annotate('closest approach: 0.16 σξ from the axis,\nin the beam core — but no crossing,\nso the column was DROPPED',
                    xy=(ym, xm), xytext=(-44, 1.55), color=INK, fontsize=9, arrowprops=dict(arrowstyle='->', color=INK2, lw=1))
        ax.text(24, 2.25, lab, ha='right', color=ORANGE, fontsize=9.5, fontweight='bold')
    else:
        ax.text(-44, -1.75, 'crosses the axis twice (×):\nbands placed, column kept', color=INK, fontsize=9)
        ax.text(24, -0.75, lab, ha='right', color=BLUE, fontsize=9.5, fontweight='bold')
ax.plot([], [], 'o', color=MUTED, label='observer’s own position (x′ = x)')
ax.plot([], [], 'X', color=MUTED, label='centre-line crossing (Eq 4.24 root)')
ax.legend(loc='lower right', frameon=False, fontsize=8.5, labelcolor=INK2)
ax.set_xlim(y[0], y[-1]); ax.set_ylim(-2.6, 2.6)
ax.set_xlabel('position along the s′ column,  x′ − x  [µm]')
ax.set_ylabel('retarded offset from the beam axis,  ξ_ret / σξ')
ax.set_title(f'(a) One s′ column, {D:.0f} µm behind the observer')

# ---------------- (b) which columns are dropped ----------------
ax = axs[1]
xo = np.linspace(-1.5, 1.5, 301)
Dc = 2 * abs(TAU) * np.clip(xo, 0, None) * SXI
ax.fill_between(xo, 0, Dc, color=ORANGE, alpha=0.18, lw=0, zorder=1)
ax.plot(xo[xo >= 0], Dc[xo >= 0], color=ORANGE, lw=2, zorder=3)
ax.text(1.45, 40, 'columns DROPPED\nby the default code', ha='right', color=INK, fontsize=9, fontweight='bold')
ax.text(0.80, 520, 'boundary\ns − s′ = 2|τ|·ξ_obs', ha='right', color=INK2, fontsize=9)
ax.text(-1.4, 330, 'ξ_obs < 0: every column\ncrosses the centre line,\nnothing dropped\n(measured at ξ_obs = −0.36)', color=INK2, fontsize=9)
# measured last-dead / first-alive columns from the band diagnostics at s = 12.30 m
for xi_o, dead, alive in ((0.37, 138.7, 242.6), (1.10, 442.4, 592.9)):
    ax.plot([xi_o, xi_o], [dead, alive], color=INK2, lw=1, zorder=4)
    ax.plot(xi_o, dead, 'o', ms=8, color=ORANGE, mec=SURF, mew=2, zorder=5)
    ax.plot(xi_o, alive, 'o', ms=8, color=BLUE, mec=SURF, mew=2, zorder=5)
ax.plot([], [], 'o', color=ORANGE, label='measured: last dropped column')
ax.plot([], [], 'o', color=BLUE, label='measured: first kept column')
ax.legend(loc='upper left', frameon=False, fontsize=8.5, labelcolor=INK2)
ax.axvline(0, color=BASE, lw=1)
ax.set_xlim(-1.5, 1.5); ax.set_ylim(0, 800)
ax.set_xlabel('observer offset from the axis,  ξ_obs / σξ')
ax.set_ylabel('column distance behind the observer,  s − s′  [µm]')
ax.set_title('(b) Which columns are dropped  (τ = −6.9)')

# ---------------- (c) what it does to the wake ----------------
ax = axs[2]
xi_rows = np.array([float(m) for m in re.findall(r'xi/sigma_xi ([+-]\d+\.\d+)',
                                                 open(os.path.join(OUT, 'vxparts_12.3000_c20.txt')).read())])
off = np.load(os.path.join(OUT, 'coulomb_12.3000.npz')); on = np.load(os.path.join(OUT, 'coulomb_12.3000_tangent_bands1.npz'))
c = 20
ax.axhline(0, color=BASE, lw=1)
ax.axvline(0, color=BASE, lw=1)
ax.plot(xi_rows, off['P'][:, c], color=AQUA, lw=2, ls=(0, (4, 2)), zorder=2)
ax.plot(xi_rows, off['dW'][:, c], color=ORANGE, lw=2, marker='o', ms=6, mec=SURF, mew=1.5, zorder=3)
ax.plot(xi_rows, on['dW'][:, c], color=BLUE, lw=2, marker='o', ms=6, mec=SURF, mew=1.5, zorder=4)
ax.text(-4.1, 0.40, 'default code: jumps by\n+0.5–1.3 MeV/m as soon as\nthe observer is above the axis', color=INK, fontsize=9)
ax.set_xlim(-4.3, 4.3); ax.set_ylim(-0.5, 1.2)
ax.set_xlabel('observer offset from the axis,  ξ_obs / σξ')
ax.set_ylabel('vₓ part of the wake at bunch centre  [MeV/m]')
ax.set_title('(c) Effect on the D3 wake, s = 12.30 m')
ax.plot([], [], color=ORANGE, marker='o', label='default (columns dropped)')
ax.plot([], [], color=BLUE, marker='o', label='tangent_bands: true')
ax.plot([], [], color=AQUA, ls=(0, (4, 2)), label='Coulomb work e·vₓ·Eₓ, computed\ndirectly from the particles')
ax.legend(loc='upper left', frameon=False, fontsize=8.5, labelcolor=INK2)

fig.suptitle('Dropped integration columns: the light cone misses the beam CENTRE LINE but still passes through the beam',
             x=0.06, ha='left', fontsize=12.5, fontweight='bold', color=INK, y=1.0)
fig.savefig(os.path.join(OUT, 'tangent_bands_explained.png'), dpi=160, bbox_inches='tight')
print('wrote', os.path.join(OUT, 'tangent_bands_explained.png'))
