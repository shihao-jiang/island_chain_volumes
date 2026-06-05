"""
IHotVol_AgeVolPlot.py

Interpolates ages along the hotspot profile and saves a 4-panel figure
(total, edifice, infill, underplating) to results/volume_components.png.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def IHotVol_AgeVolPlot(VOL, Crosses, p):
    """
    Assign interpolated ages to each cross-section and save a 4-panel
    volume plot.

    Parameters
    ----------
    VOL      : ndarray, shape (N, >=7)
    Crosses  : list of ndarrays
    p        : 1-D array-like – polynomial coefficients (age vs. along-track distance)

    Returns
    -------
    VOL : ndarray
    """

    os.makedirs('results', exist_ok=True)

    VOL[:, 6] = np.polyval(p, VOL[:, 5])
    age = VOL[:, 6]

    panels = [
        ('Total',        (VOL[:, 2] + VOL[:, 3] + VOL[:, 4]) * 4, 'k'),
        ('Edifice',      VOL[:, 2] * 4,                            'r'),
        ('Infill',       VOL[:, 3] * 4,                            'b'),
        ('Underplating', VOL[:, 4] * 4,                            'g'),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(10, 16))

    for ax, (name, data, colour) in zip(axes, panels):
        ax.plot(age, data, color=colour, linewidth=2)
        ax.set_ylabel(r'Volume, km$^3$', fontsize=14)
        ax.set_title(name, fontsize=15)
        ax.tick_params(labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)

    axes[-1].set_xlabel('Age (Ma)', fontsize=14)

    plt.tight_layout()
    fname = 'results/volume_components.png'
    fig.savefig(fname, dpi=150)
    print(f'Saved {fname}')
    try:
        plt.close(fig)
    except Exception:
        pass

    return VOL
