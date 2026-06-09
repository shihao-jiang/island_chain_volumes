"""
IHotVol_Spectral.py

Lomb-Scargle spectral analysis of melt volume components.
Equivalent to the MATLAB function IHotVol_Spectral.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

try:
    from astropy.timeseries import LombScargle
    _ASTROPY = True
except ImportError:
    _ASTROPY = False
    from scipy.signal import lombscargle as _scipy_lombscargle


def _lombscargle_normalized(t, y, Pfa=None):
    """
    Compute normalised Lomb-Scargle periodogram and (optionally) detection thresholds.

    Parameters
    ----------
    t   : 1-D array – time / distance axis (irregularly sampled OK)
    y   : 1-D array – signal values
    Pfa : array-like of floats, optional – false-alarm probabilities (e.g. [0.5,0.1,0.01,0.0001])

    Returns
    -------
    f   : 1-D array – frequencies
    pxx : 1-D array – normalised power
    pth : 1-D array or None – detection thresholds for each Pfa level
    """
    if _ASTROPY:
        ls  = LombScargle(t, y, normalization='psd')
        f, pxx = ls.autopower(minimum_frequency=None, maximum_frequency=None)
        # Normalise so that white-noise expectation is 1
        pxx = pxx / np.var(y)
        if Pfa is not None:
            pth = np.array([ls.false_alarm_level(p) / np.var(y) for p in Pfa])
        else:
            pth = None
    else:
        # scipy fallback (less feature-complete)
        N  = len(t)
        T  = t[-1] - t[0]
        f  = np.linspace(1 / T, N / (2 * T), 5 * N)
        ang_freqs = 2 * np.pi * f
        pxx = _scipy_lombscargle(t.astype(float), y.astype(float),
                                  ang_freqs, normalize=True)
        pth = None

    return f, pxx, pth


def IHotVol_Spectral(VOL, grdfile):
    """
    Run spectral analysis on the volume components.

    Parameters
    ----------
    VOL     : ndarray  – volume table (N x >=7 columns; col indices 2,3,4 are
                         edifice, infill, underplating volumes; col 6 is age)
    grdfile : str      – base grid filename (unused here, kept for API)

    Returns
    -------
    PEAKS : dict – spectral peaks for each component
    """

    os.makedirs('results', exist_ok=True)

    # Add tiny jitter to ages to avoid duplicate sample positions
    VOL = VOL.copy()
    VOL[:, 6] = VOL[:, 6] + np.random.rand(len(VOL)) * 1e-4

    posVOL = VOL[:, 6] - VOL[:, 6].min()

    # Ensure increasing age axis
    if posVOL[-1] == 0:
        posVOL = posVOL[::-1]
        VOL    = VOL[::-1]

    Pfa = np.array([0.50, 0.10, 0.01, 0.0001])
    Pd  = 1 - Pfa
    flim = 1.0

    components = {
        'Edifice':     VOL[:, 2],
        'Infill':      VOL[:, 3],
        'Underplating': VOL[:, 4],
    }

    Pow   = {}
    PEAKS = {}

    # --- Component power plot ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.set_size_inches(12, 14)

    for idx, (name, signal) in enumerate(components.items()):
        ax = axes[idx]

        f, pxx, pth = _lombscargle_normalized(posVOL, signal, Pfa=Pfa)

        ax.plot(f, pxx, linewidth=2)
        if pth is not None:
            for th in pth:
                ax.axhline(th, linewidth=2, linestyle='--')

        ax.set_title(f'{name} power', fontsize=18)
        ax.set_xlim([0, flim])
        ax.grid(True, which='both')
        ax.tick_params(labelsize=18)

        # Find peaks above the 1% false-alarm threshold
        min_height = 2.0
        pk_idx, _ = find_peaks(pxx, height=min_height)
        pk_f   = f[pk_idx]
        pk_pow = pxx[pk_idx]

        if pth is not None:
            # Keep peaks above 1% level (index 2 in Pfa array)
            above_1pct = pk_pow > pth[2]
            pk_f_1pct  = pk_f[above_1pct]
            pk_pow_1pct = pk_pow[above_1pct]

            above_01pct = pk_pow > pth[3]
            pk_f_01pct  = pk_f[above_01pct]
            pk_pow_01pct = pk_pow[above_01pct]
        else:
            pk_f_1pct = pk_f_01pct = pk_f
            pk_pow_1pct = pk_pow_01pct = pk_pow

        ax.plot(pk_f_1pct, pk_pow_1pct, 'ko', markersize=8)

        Pow[name]          = np.column_stack([pk_f_1pct,  pk_pow_1pct])  if len(pk_f_1pct)  else np.empty((0, 2))
        PEAKS[f'{name}99'] = np.column_stack([pk_f_01pct, pk_pow_01pct]) if len(pk_f_01pct) else np.empty((0, 2))

    plt.tight_layout()
    fig.savefig('results/ComponentPower.eps')
    fig.savefig('results/ComponentPower.png', dpi=150)
    print('Saved results/ComponentPower.png')
    try:
        plt.close(fig)
    except Exception:
        pass

    # --- Total power plot ---
    total_signal = VOL[:, 2] + VOL[:, 3] + VOL[:, 4]
    f, pxx, pth = _lombscargle_normalized(posVOL, total_signal, Pfa=Pfa)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(f, pxx, linewidth=2)
    if pth is not None:
        for th in pth:
            ax2.axhline(th, linewidth=2, linestyle='--')

    ax2.set_title('Total power', fontsize=18)
    ax2.set_xlim([0, flim])
    ax2.grid(True, which='both')
    ax2.tick_params(labelsize=18)

    pk_idx, _ = find_peaks(pxx, height=2.0)
    pk_f   = f[pk_idx]
    pk_pow = pxx[pk_idx]

    if pth is not None:
        above_1pct  = pk_pow > pth[2]
        above_01pct = pk_pow > pth[3]
        pk_f_1pct   = pk_f[above_1pct]
        pk_pow_1pct = pk_pow[above_1pct]
        pk_f_01pct  = pk_f[above_01pct]
        pk_pow_01pct = pk_pow[above_01pct]
    else:
        pk_f_1pct = pk_f_01pct = pk_f
        pk_pow_1pct = pk_pow_01pct = pk_pow

    ax2.plot(pk_f_1pct, pk_pow_1pct, 'ko', markersize=8)

    Pow['Total']       = np.column_stack([pk_f_1pct,  pk_pow_1pct])  if len(pk_f_1pct)  else np.empty((0, 2))
    PEAKS['Total99']   = np.column_stack([pk_f_01pct, pk_pow_01pct]) if len(pk_f_01pct) else np.empty((0, 2))

    plt.tight_layout()
    fig2.savefig('results/TotalPower.eps')
    fig2.savefig('results/TotalPower.png', dpi=150)
    print('Saved results/TotalPower.png')
    try:
        plt.close(fig2)
    except Exception:
        pass

    PEAKS.update(Pow)

    # Write peak periods (1/frequency) to text file
    keys = ['Edifice99', 'Infill99', 'Underplating99', 'Total99']
    max_len = max(len(PEAKS.get(k, np.empty((0, 2)))) for k in keys)
    OUT = np.zeros((max(max_len, 1), 4))
    for col, key in enumerate(keys):
        arr = PEAKS.get(key, np.empty((0, 2)))
        if len(arr) > 0:
            periods = 1.0 / arr[:, 0]
            OUT[:len(periods), col] = periods

    np.savetxt('peaks.txt', OUT)

    return PEAKS
