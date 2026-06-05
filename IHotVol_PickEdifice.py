"""
IHotVol_PickEdifice.py

Hand-pick the volcanic edifice outline from the residual-separated bathymetry.
Equivalent to the MATLAB function IHotVol_PickEdifice.
"""

import subprocess
import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # change backend if needed (e.g. 'Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from grd_utils import grdread2, grdwrite2


def IHotVol_PickEdifice(grdfile, AGES):
    """
    Display the residual bathymetry, ask the user to draw a polygon around
    the edifice, and write the edifice and in-polygon grids to disk.

    Parameters
    ----------
    grdfile : str      – base grid filename
    AGES    : ndarray  – age data (columns: lon, lat, age_Ma)

    Returns
    -------
    Xpoly : 1-D ndarray – polygon x vertices (user-picked)
    Ypoly : 1-D ndarray – polygon y vertices (user-picked)
    INP   : 2-D bool ndarray – True inside the edifice polygon
    """

    def run(cmd):
        subprocess.run(cmd, shell=True, check=True)

    # Load residual grid
    Xres, Yres, Zres = grdread2(f'{grdfile}_residual.grd')

    # Zero out negative anomalies
    Zres[Zres <= 0] = 0.0

    plt.close('all')

    # --- PICKING ---
    fig, ax = plt.subplots(figsize=(10, 14))
    Xmesh, Ymesh = np.meshgrid(Xres, Yres)
    ax.contourf(Xmesh, Ymesh, Zres, levels=20)
    ax.set_title('Residual bathymetry')
    ax.set_aspect('equal')
    ax.plot(AGES[:, 0], AGES[:, 1], 'ro-')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.set_size_inches(10, 14)
    plt.tight_layout()

    print("Please click to draw a polygon around the edifice of interest.")
    print("Press ENTER when finished.")
    pts = plt.ginput(n=-1, timeout=0)
    plt.close('all')

    Xpoly = np.array([p[0] for p in pts])
    Ypoly = np.array([p[1] for p in pts])

    # --- PROCESS ---

    # Find grid cells inside the polygon
    poly_path = Path(np.column_stack([Xpoly, Ypoly]))
    flat_pts  = np.column_stack([Xmesh.ravel(), Ymesh.ravel()])
    INP       = poly_path.contains_points(flat_pts).reshape(Xmesh.shape)

    Zed     = Zres * INP.astype(float)
    Edifice = Zed.copy()
    Zed[Zed == 0] = np.nan

    # Write edifice and in-polygon grids
    grdwrite2(Xres, Yres, Edifice, 'ED.grd')
    grdwrite2(Xres, Yres, INP.astype(float), 'INP.grd')

    # Reformat grid with GMT (grdwrite2 output not compatible with grdflexure)
    run(f'grd2xyz ED.grd | xyz2grd -G{grdfile}_edifice.grd -R{grdfile}.trim')

    # De-NaN the edifice grid (set 0 outside)
    run(f'grdmath {grdfile}_edifice.grd 0 DENAN = {grdfile}_edifice.grd')

    # Read back in and re-write INP at same resolution
    Xg, Yg, Zg = grdread2(f'{grdfile}_edifice.grd')
    Xgmesh, Ygmesh = np.meshgrid(Xg, Yg)
    flat_pts2 = np.column_stack([Xgmesh.ravel(), Ygmesh.ravel()])
    INP_new   = poly_path.contains_points(flat_pts2).reshape(Xgmesh.shape)
    grdwrite2(Xg, Yg, INP_new.astype(float), 'INP.grd')

    # --- DISPLAY ---
    fig2, ax2 = plt.subplots(figsize=(10, 14))
    ax2.contourf(Xgmesh, Ygmesh, Zg, levels=20)
    ax2.set_title('Edifice')
    ax2.set_aspect('equal')
    plt.tight_layout()
    plt.show()

    print('You picked this material as your volcanic edifice.')

    return Xpoly, Ypoly, INP_new
