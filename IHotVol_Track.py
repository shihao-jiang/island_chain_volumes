"""
IHotVol_Track.py

Upload hotspot age track and determine least-squares polynomial fit for
both age and location.
Equivalent to the MATLAB function IHotVol_Track.
"""

import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from grd_utils import grdread2


# ---------------------------------------------------------------------------
# Helper: closest point on polynomial curve (replaces MATLAB closepoint)
# ---------------------------------------------------------------------------

def _closepoint(x0, y0, p):
    """
    Find the point on the polynomial curve y = polyval(p, x) that is
    closest (in Euclidean distance) to (x0, y0).

    Returns
    -------
    xc, yc : floats – coordinates of the closest point
    """
    def dist2(x):
        yc = np.polyval(p, x)
        return (x - x0) ** 2 + (yc - y0) ** 2

    res = minimize_scalar(dist2, bounds=(x0 - 30, x0 + 30), method='bounded')
    xc  = res.x
    yc  = np.polyval(p, xc)
    return xc, yc


# ---------------------------------------------------------------------------
# Helper: approximate haversine / great-circle distance (deg → km)
# ---------------------------------------------------------------------------

def _deg2km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points (degrees) → km."""
    R    = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a    = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def IHotVol_Track(AGES, X, Y, Z, grdfile, SFLagegrd):
    """
    Build the hotspot track from radiometric age data, fit a polynomial,
    and compute weighted-average seafloor loading ages.

    Parameters
    ----------
    AGES     : ndarray  – age data  (columns: lon, lat, age_Ma)
    X        : 1-D ndarray – bathymetry x coords
    Y        : 1-D ndarray – bathymetry y coords
    Z        : 2-D ndarray – bathymetry values
    grdfile  : str      – base grid filename
    SFLagegrd: str      – seafloor age grid path

    Returns
    -------
    HSPT_TRK : ndarray  – hotspot track (cols: lon, lat, seamount_age,
                          litho_age, along_dist_km, Te)
    AGES     : ndarray  – sorted age data
    pA       : 1-D array – polynomial coefficients (age vs. along-track distance)
    TMPTRK   : ndarray  – temporary track array used internally
    """

    def run(cmd):
        env = os.environ.copy()
        env['GMT_VERBOSE'] = 'e'
        subprocess.run(cmd + ' 2>/dev/null', shell=True, check=True, env=env)

    # Sort age data by longitude
    AGES = AGES[np.argsort(AGES[:, 0])]

    Xtrk = AGES[:, 0]
    Ytrk = AGES[:, 1]
    Lage = AGES[:, 2]

    # Max polynomial order
    pmax = min(len(AGES) - 1, 6)

    try:
        plt.close('all')
    except Exception:
        pass

    # Inspection map
    Xmesh, Ymesh = np.meshgrid(X, Y)
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.contourf(Xmesh, Ymesh, Z, levels=20)
    ax.contour(Xmesh, Ymesh, Z, levels=20, colors='k', linewidths=0.5)
    ax.plot(AGES[:, 0], AGES[:, 1], 'ko', markersize=6)
    ax.set_aspect('equal')

    x_plot = np.linspace(Xtrk.min(), Xtrk.max(), 300)
    for n in range(1, pmax + 1):
        p, _ = np.polyfit(Xtrk, Ytrk, n), None
        p    = np.polyfit(Xtrk, Ytrk, n)
        ax.plot(x_plot, np.polyval(p, x_plot), linewidth=2, label=str(n))

    ax.plot(AGES[:, 0], AGES[:, 1], 'ko', markersize=6)
    ax.legend()
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

    # numFit = int(input('What order polynomial best follows the hotspot?\n> '))
    numFit = 3

    p = np.polyfit(Xtrk, Ytrk, numFit)

    # Orthogonal relocation of age data onto polynomial
    x_track = []
    y_track = []
    for ii in range(len(Xtrk)):
        xc, yc = _closepoint(AGES[ii, 0], AGES[ii, 1], p)
        x_track.append(xc)
        y_track.append(yc)

    TMPTRK = np.column_stack([x_track, y_track, AGES[:, 2]])
    TMPTRK = TMPTRK[np.argsort(TMPTRK[:, 0])]

    # Uncomment for hotspots crossing ±180°:
    # TMPTRK[TMPTRK[:, 0] < 0, 0] += 360

    # Write track to file
    np.savetxt('track.txt', TMPTRK)

    # Generate track cross-sections with GMT
    run(f"cat track.txt | awk '{{print $1,$2}}' | grdtrack -G{grdfile} -C2k/1/20 -Ar > track_cross.txt")
    run("awk '$3==0 {print $1,$2}' track_cross.txt > track20k.txt")

    TRK = np.loadtxt('track20k.txt')

    # Uncomment for hotspots crossing ±180°:
    # TRK[TRK[:, 0] < 0, 0] += 360

    TMPTRK = TMPTRK[np.argsort(TMPTRK[:, 2])]

    # Along-track distance
    DIST = np.sqrt((TRK[:, 0] - TMPTRK[0, 0]) ** 2 +
                   (TRK[:, 1] - TMPTRK[0, 1]) ** 2)

    if np.argmin(DIST) < 0.5 * len(TMPTRK):
        TRK = np.column_stack([TRK, 20.0 * np.arange(len(TRK))])
    else:
        TRK  = TRK[::-1]
        TRK  = np.column_stack([TRK, 20.0 * np.arange(len(TRK))])

    # Polynomial fit: distance along track → age
    pDA = np.polyfit(TRK[:, 0], TRK[:, 2], numFit)
    TMPTRK = np.column_stack([TMPTRK, np.polyval(pDA, TMPTRK[:, 0])])

    try:
        plt.close('all')
    except Exception:
        pass

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(TMPTRK[:, 3], TMPTRK[:, 2], 'rs', markersize=8)

    d_plot = np.linspace(TMPTRK[:, 3].min(), TMPTRK[:, 3].max(), 300)
    for n in range(1, 6):
        pA_tmp = np.polyfit(TMPTRK[:, 3], TMPTRK[:, 2], n)
        ax2.plot(d_plot, np.polyval(pA_tmp, d_plot), linewidth=2, label=str(n))

    ax2.plot(TMPTRK[:, 3], TMPTRK[:, 2], 'rs', markersize=8)
    ax2.legend()
    ax2.set_xlabel('Along-track distance')
    ax2.set_ylabel('Age (Ma)')
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

    # numFit2 = int(input('What order polynomial best follows the age/distance relationship?\n> '))
    numFit2 = 2
    pA = np.polyfit(TMPTRK[:, 3], TMPTRK[:, 2], numFit2)

    TRK_age = np.polyval(pA, TRK[:, 2])  # age from along-track distance

    # Build HSPT_TRK
    HSPT_X = TRK[:, 0]
    HSPT_Y = TRK[:, 1]
    HSPT_A = TRK_age
    HSPT_L = TRK[:, 2]

    HSPT_TRK = np.column_stack([HSPT_X, HSPT_Y, HSPT_A])

    # Directory for age grid sub-samples
    os.makedirs('cutagetmp', exist_ok=True)

    # Read and clip seafloor age grid
    run(f'grdsample {SFLagegrd} -Gcutagetmp/age.{grdfile} -R{grdfile}')

    # Weighted-average seafloor age at each track point
    litho_age = np.zeros(len(HSPT_TRK))

    for ii in range(len(HSPT_TRK)):
        cut_file = f'cutagetmp/cut.age.{ii + 1}.grd'
        run((f"grdcut cutagetmp/age.{grdfile} "
             f"-Sn{HSPT_TRK[ii, 0]}/{HSPT_TRK[ii, 1]}/5d "
             f"-G{cut_file}"))

        tmpX, tmpY, tmpZ = grdread2(cut_file)
        tmpXm, tmpYm = np.meshgrid(tmpX, tmpY)

        # Distance weighting (1/distance from track point)
        dist_arr = _deg2km(HSPT_TRK[ii, 1], HSPT_TRK[ii, 0],
                           tmpYm, tmpXm)
        dist_arr[dist_arr == 0] = 1e-10   # avoid division by zero
        WTG = 1.0 / dist_arr

        z_flat   = tmpZ.ravel()
        wtg_flat = WTG.ravel()
        valid    = ~np.isnan(z_flat)
        if valid.any():
            litho_age[ii] = (np.nansum(wtg_flat * z_flat) /
                             np.nansum(wtg_flat))

    # Append loading age and finalise
    HSPT_TRK = np.column_stack([HSPT_X, HSPT_Y, HSPT_A, litho_age, HSPT_L,
                                 np.zeros(len(HSPT_X))])   # col 5 reserved for Te
    HSPT_TRK[:, 2] -= HSPT_TRK[:, 2].min()

    # Remove duplicate rows
    HSPT_TRK = np.unique(HSPT_TRK, axis=0)

    try:
        plt.close('all')
    except Exception:
        pass

    return HSPT_TRK, AGES, pA, TMPTRK
