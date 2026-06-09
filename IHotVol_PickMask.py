"""
IHotVol_PickMask.py

Hand-pick a mask polygon to blank out interfering regions of the map
before the regional/residual separation and gravity modelling.
Equivalent to the MATLAB function IHotVol_PickMask.
"""

import os
import subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from grd_utils import grdread2, grdwrite2
from gmt_utils import run

def IHotVol_PickMask(grdfile):
    """
    Display the bathymetry grid and let the user draw a mask polygon.
    Writes MASK.grd (1 inside, 0 outside) to disk.

    Parameters
    ----------
    grdfile : str  – grid filename (with .grd extension)

    Returns
    -------
    XpolyM  : 1-D ndarray – polygon x vertices
    YpolyM  : 1-D ndarray – polygon y vertices
    INPmask : 2-D bool ndarray – True inside the mask polygon
    """

    # Load the bathymetry grid
    X, Y, Z = grdread2(grdfile)
    Xmesh, Ymesh = np.meshgrid(X, Y)

    try:
        plt.close('all')
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(10, 14))
    ax.contourf(Xmesh, Ymesh, Z, levels=20)
    ax.set_title('Bathymetry – pick mask polygon')
    ax.set_aspect('equal')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.tight_layout()

    print("Please click to draw a polygon around the region to MASK OUT.")
    print("Press ENTER when finished.")
    pts = plt.ginput(n=-1, timeout=0)
    try:
        plt.close('all')
    except Exception:
        pass

    XpolyM = np.array([p[0] for p in pts])
    YpolyM = np.array([p[1] for p in pts])

    # Build in-polygon mask
    poly_path = Path(np.column_stack([XpolyM, YpolyM]))
    flat_pts  = np.column_stack([Xmesh.ravel(), Ymesh.ravel()])
    INPmask   = poly_path.contains_points(flat_pts).reshape(Xmesh.shape)

    # MASK grid: 0 inside the mask polygon, 1 outside
    mask_grid = (~INPmask).astype(float)
    grdwrite2(X, Y, mask_grid, 'MASK.grd')

    print('Mask written to MASK.grd')

    return XpolyM, YpolyM, INPmask

def IHotVol_PickMask2(refgrdfile):
    """
    Secondary mask pick (used in IHotVol_SampleVolGrids when further masking
    is required). Writes MASK2.grd.

    Parameters
    ----------
    refgrdfile : str  – reference grid to display

    Returns
    -------
    XpolyM  : 1-D ndarray
    YpolyM  : 1-D ndarray
    INPmask : 2-D bool ndarray
    """

    X, Y, Z = grdread2(refgrdfile)
    Xmesh, Ymesh = np.meshgrid(X, Y)

    try:
        plt.close('all')
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.contourf(Xmesh, Ymesh, Z, levels=20)
    ax.set_title('Pick secondary mask polygon')
    ax.set_aspect('equal')
    plt.tight_layout()

    print("Please click to draw a secondary mask polygon. Press ENTER when finished.")
    pts = plt.ginput(n=-1, timeout=0)
    try:
        plt.close('all')
    except Exception:
        pass

    XpolyM = np.array([p[0] for p in pts])
    YpolyM = np.array([p[1] for p in pts])

    poly_path = Path(np.column_stack([XpolyM, YpolyM]))
    flat_pts  = np.column_stack([Xmesh.ravel(), Ymesh.ravel()])
    INPmask   = poly_path.contains_points(flat_pts).reshape(Xmesh.shape)

    mask_grid = (~INPmask).astype(float)
    grdwrite2(X, Y, mask_grid, 'MASK2.grd')

    print('Secondary mask written to MASK2.grd')

    return XpolyM, YpolyM, INPmask
