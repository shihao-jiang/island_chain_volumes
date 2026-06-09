"""
IHotVol_VolumeSlices.py

Integrate cross-section data to obtain along-track volume estimates.
Equivalent to the MATLAB function IHotVol_VolumeSlices.
"""

import numpy as np
from scipy.signal import windows as sig_windows
from IHotVol_Track import _deg2km   # reuse great-circle helper
import os

def IHotVol_VolumeSlices(Xes, HSPT_TRK):
    """
    Compute volumetric contributions (edifice, infill, underplating) from
    cross-section profiles exported by GMT grdtrack.

    Parameters
    ----------
    Xes      : ndarray  – cross-section data loaded from *Xes.txt.
                          NaN rows act as separators between profiles.
                          Columns (0-indexed):
                            0: lon  1: lat  2: along-distance  3: depth
                            4: edifice  5: flexure  6: underplating
                            7: underplating INP
    HSPT_TRK : ndarray  – hotspot track array

    Returns
    -------
    VOL     : ndarray  – (N_sections x 7) volume table:
                          cols: lon, lat, VolEd, VolFill, VolUpl, dist_km, age
    Crosses : list     – list of per-section profile ndarrays
    """

    # Split Xes into individual profiles at NaN rows
    # Add sentinel NaN rows at start and end if not already present
    if not np.isnan(Xes[0, 0]):
        Xes = np.vstack([np.full((1, Xes.shape[1]), np.nan), Xes])
    if not np.isnan(Xes[-1, 0]):
        Xes = np.vstack([Xes, np.full((1, Xes.shape[1]), np.nan)])

    nan_rows = np.where(np.isnan(Xes[:, 0]))[0]

    Crosses = []
    for idx in range(len(nan_rows) - 1):
        start = nan_rows[idx] + 1
        end   = nan_rows[idx + 1]
        Crosses.append(Xes[start:end, :])

    VOL  = np.zeros((len(Crosses), 7))
    DIST = 0.0
    OldPt = None

    for ii, PRFL in enumerate(Crosses):

        PRFL = PRFL.copy()
        PRFL[np.isnan(PRFL)] = 0.0

        # ---------- Edifice contribution ----------
        VolEd = np.trapz(PRFL[:, 4] / 1e3, PRFL[:, 2])

        # ---------- Infill (flexure-fill) contribution ----------
        FLX = np.column_stack([PRFL[:, 2], -PRFL[:, 5]])

        # De-trend flexure profile
        p_flx     = np.polyfit(FLX[:, 0], FLX[:, 1], 1)
        FLX[:, 1] -= np.polyval(p_flx, FLX[:, 0])

        FLX[:, 1] -= FLX[:, 1].min()
        PEAK = FLX[:, 1].max()

        # Fit a Gaussian window to the flexure profile
        best_tw  = _fit_gausswin(FLX[:, 1], PEAK)
        GUSFLX   = sig_windows.gaussian(len(FLX), best_tw) * PEAK  # noqa (stored, unused in integral)

        VolFill = np.trapz(FLX[:, 1] / 1e3, FLX[:, 0])

        # ---------- Underplating contribution ----------
        UPL = np.column_stack([PRFL[:, 2], PRFL[:, 6], PRFL[:, 7]])

        # De-trend
        p_upl     = np.polyfit(UPL[:, 0], UPL[:, 1], 1)
        UPL[:, 1] -= np.polyval(p_upl, UPL[:, 0])

        UPL[:, 1] -= UPL[:, 1].min()
        PEAK_upl  = UPL[:, 2].max()

        best_tw_upl = _fit_gausswin(UPL[:, 1], PEAK_upl)
        GUSUPL      = sig_windows.gaussian(len(UPL), best_tw_upl) * UPL[:, 1].max()

        VolUpl = 0.683 * np.trapz(GUSUPL, UPL[:, 0])

        # ---------- Location and cumulative distance ----------
        zero_idx = np.where(PRFL[:, 2] == 0)[0]
        if len(zero_idx) > 0:
            VOL[ii, 0:2] = PRFL[zero_idx[0], 0:2]
        else:
            VOL[ii, 0:2] = PRFL[0, 0:2]

        if ii == 0:
            DIST = 0.0
            VOL[ii, 5] = 0.0
        else:
            zero_mask = PRFL[:, 2] == 0
            CurPt     = PRFL[zero_mask, 0:2][0] if zero_mask.any() else PRFL[0, 0:2]
            DIST     += _deg2km(OldPt[1], OldPt[0], CurPt[1], CurPt[0])
            VOL[ii, 5] = DIST

        zero_mask = PRFL[:, 2] == 0
        OldPt = PRFL[zero_mask, 0:2][0] if zero_mask.any() else PRFL[0, 0:2]

        VOL[ii, 2] = VolEd
        VOL[ii, 3] = VolFill
        VOL[ii, 4] = VolUpl
        # col 6 (age) filled later by IHotVol_AgeVolPlot

    # Wrap negative longitudes
    VOL[VOL[:, 0] < 0, 0] += 360

    # Ensure distance increases monotonically from the active end of the track
    if (VOL[0, 5] == 0 and HSPT_TRK[0, 4] > 0) or \
       (VOL[-1, 5] == 0 and HSPT_TRK[-1, 4] > 0):
        VOL[:, 5] = VOL[:, 5][::-1]

    return VOL, Crosses


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _fit_gausswin(signal, peak, tw_range=None):
    """
    Find the Gaussian window width (sigma parameter for scipy.signal.gaussian)
    that minimises the RMS between the window (scaled to peak) and the signal.

    Returns the best-fit width parameter.
    """
    if tw_range is None:
        tw_range = np.arange(0.1, 100.01, 0.01)

    n   = len(signal)
    rms = np.array([
        np.sqrt(np.nanmean(
            (signal - sig_windows.gaussian(n, tw) * peak) ** 2))
        for tw in tw_range
    ])
    return tw_range[np.argmin(rms)]
