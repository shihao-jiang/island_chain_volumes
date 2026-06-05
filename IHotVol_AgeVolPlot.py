"""
IHotVol_AgeVolPlot.py

Interpolates ages along the hotspot profile and plots volume components
vs. age. Equivalent to the MATLAB function IHotVol_AgeVolPlot.
"""

import numpy as np
import matplotlib.pyplot as plt


def IHotVol_AgeVolPlot(VOL, Crosses, p):
    """
    Assign interpolated ages to each cross-section and plot volume vs. age.

    Parameters
    ----------
    VOL      : ndarray, shape (N, >=7)  – volume table (columns 1-indexed as in MATLAB)
    Crosses  : list of ndarrays         – cross-section profiles (unused here but kept for API)
    p        : 1-D array-like           – polynomial coefficients from np.polyfit
                                          (age as a function of along-track distance)

    Returns
    -------
    VOL : ndarray  – same table with column 7 (index 6) filled with interpolated ages
    """

    # Interpolate age at each cross-section location (column 6, 0-indexed = col 5 in MATLAB)
    # VOL[:, 6] = polyval(p, VOL[:, 5])   [MATLAB 1-indexed col 7 = Python index 6]
    VOL[:, 6] = np.polyval(p, VOL[:, 5])

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # --- subplot 1: total volume ---
    ax1 = axes[0]
    total_vol = (VOL[:, 2] + VOL[:, 3] + VOL[:, 4]) * 4
    ax1.plot(VOL[:, 6], total_vol, 'k-', linewidth=4)
    ax1.set_ylabel(r'Volume, km$^3$', fontsize=18)
    ax1.set_title('Total volume', fontsize=18)
    ax1.tick_params(labelsize=18)
    ax1.grid(False)
    for spine in ax1.spines.values():
        spine.set_visible(True)

    # --- subplot 2: volume components ---
    ax2 = axes[1]
    ax2.plot(VOL[:, 6], VOL[:, 2] * 4, 'r-', linewidth=2, label='Edifice')
    ax2.plot(VOL[:, 6], VOL[:, 3] * 4, 'b-', linewidth=2, label='Infill')
    ax2.plot(VOL[:, 6], VOL[:, 4] * 4, 'k-', linewidth=2, label='Underplating')
    ax2.set_ylabel(r'Volume, km$^3$', fontsize=18)
    ax2.set_xlabel('Age (Ma)', fontsize=18)
    ax2.set_title('Volume components', fontsize=18)
    ax2.legend(fontsize=14)
    ax2.tick_params(labelsize=18)
    for spine in ax2.spines.values():
        spine.set_visible(True)

    plt.tight_layout()
    plt.show()

    return VOL
