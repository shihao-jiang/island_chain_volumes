"""
IHotVol_ORS.py

Optimised Residual Separation (Wessel, 2016).
Equivalent to the MATLAB function IHotVol_ORS.
"""

import os
import subprocess
import shutil
import numpy as np
import platform
from gmt_utils import run

def IHotVol_ORS(grdfile, X, Y, Z, minW, maxW, intW, level, mask):
    """
    Run the Optimised Residual Separation on the bathymetry grid.

    Parameters
    ----------
    grdfile : str    – grid filename (with .grd extension)
    X       : 1-D ndarray – x (longitude) coordinates
    Y       : 1-D ndarray – y (latitude)  coordinates
    Z       : 2-D ndarray – bathymetry values
    minW    : float  – minimum filter width candidate (km)
    maxW    : float  – maximum filter width candidate (km)
    intW    : float  – filter width step (km)
    level   : float  – step for base contour calculations
    mask    : int    – 1 if masking is applied, 0 otherwise

    Returns
    -------
    ORS_L  : 1-D ndarray – row of ORS table with the optimal filter wavelength
    region : str         – GMT -R string for the sub-region
    """

    # Sub-region of map (slightly inset to avoid edge effects)
    region = (f'-R{np.ceil(X.min()) + 0.05:.4f}/{np.floor(X.max()) - 0.05:.4f}'
              f'/{np.ceil(Y.min()) + 0.05:.4f}/{np.floor(Y.max()) - 0.05:.4f}')

    # If masked, apply mask before ORS then restore
    if mask == 1:
        shutil.copy(grdfile, 'backupgrdfile_ORS.grd')
        run(f'gmt grdmath MASK.grd {grdfile} MUL 0 DENAN = {grdfile}')
        cmd = (f'bash ./RR-Sep_mask.sh {grdfile} {region} '
               f'{minW} {maxW} {intW} {level}')
        run(cmd)
    else:
        cmd = (f'bash ./RR-Sep.sh {grdfile} {region} '
               f'{minW} {maxW} {intW} {level}')
        run(cmd)

    # Read ORS table and find the row with the maximum score (column 5, index 4)
    ORS = np.loadtxt('ORStable.txt')
    ORS_L = ORS[ORS[:, 4] == ORS[:, 4].max(), :]
    if ORS_L.ndim == 2:
        ORS_L = ORS_L[0]   # take first match if tie

    # Restore original grid if masked
    if mask == 1:
        shutil.copy('backupgrdfile_ORS.grd', grdfile)

    # Final filtering with the optimal wavelength
    print('Filter wavelength found, performing final separation')

    if mask == 1:
        cmd = (f'bash ./RR-Sep-single_mask.sh {grdfile} {region} '
               f'{ORS_L[0]} {ORS_L[0]} {intW} {level}')
        run(cmd)
    else:
        cmd = (f'bash ./RR-Sep-single.sh {grdfile} {region} '
               f'{ORS_L[0]} {ORS_L[0]} {intW} {level}')
        run(cmd)

    # Subtract ORS regional from observed to get residual
    shutil.copy('./finalRR/resid.grd', f'{grdfile}_residual.grd')
    run(f'gmt grdsample {grdfile} -R{grdfile}_residual.grd -Gtmpregion.grd')
    run(f'gmt grdmath tmpregion.grd {grdfile}_residual.grd SUB = {grdfile}_regional.grd')

    return ORS_L, region
