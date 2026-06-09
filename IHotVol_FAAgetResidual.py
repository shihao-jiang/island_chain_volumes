"""
IHotVol_FAAgetResidual.py

Determines FAA anomaly and residual from the synthetic gravity approximation.
Equivalent to the MATLAB function IHotVol_FAAgetResidual.
"""

import os
import subprocess
import shutil
import numpy as np
from grd_utils import grdread2
from gmt_utils import run

def IHotVol_FAAgetResidual(ORS_L, WGMFAAgrd, mask, subaq):
    """
    Compute the gravity residual (observed FAA minus synthetic forward model).

    Parameters
    ----------
    ORS_L    : array-like  – ORS result row; ORS_L[0] is the optimal filter wavelength (km)
    WGMFAAgrd: str         – path to WGM FAA grid file
    mask     : int         – 1 if masking is used, 0 otherwise
    subaq    : int         – 1 if hotspot is fully submarine (unused here, kept for API)

    Returns
    -------
    XResG : 1-D ndarray – x coordinates of residual grid
    YResG : 1-D ndarray – y coordinates of residual grid
    ZResG : 2-D ndarray – residual gravity values
    """

    # Sample WGM data to match synthetic grid extent
    run(f'gmt grdsample {WGMFAAgrd} -RSYNTH.grav.grd -Ggravmodel/faa.cut.grd')

    # Subtract forward model to get residual
    run('gmt grdmath gravmodel/faa.cut.grd SYNTH.grav.grd SUB = gravmodel/UPLATE.grav.grd')

    # Remove mean from observed FAA
    run('gmt grdmath gravmodel/faa.cut.grd gravmodel/faa.cut.grd MEAN SUB = gravmodel/faa.cut.grd')

    # Remove mean from forward model
    run('gmt grdmath SYNTH.grav.grd SYNTH.grav.grd MEAN SUB = SYNTH.grav.grd')

    # Low-pass filter underplating grid at 20 km
    run('gmt grdfilter gravmodel/UPLATE.grav.grd -Fc20k -D1 -Ggravmodel/UPLATE.grav.grd')
    shutil.copy('gravmodel/UPLATE.grav.grd', 'gravmodel/UPLATE.lowpass.grav.grd')

    # If masked, also mask the result
    if mask == 1:
        run('gmt grdsample MASK.grd -Rgravmodel/UPLATE.grav.grd -GMASK.grav.grd')
        shutil.copy('gravmodel/UPLATE.grav.grd', 'gravmodel/backup.UPLATE.grav.grd')
        run('gmt grdmath MASK.grav.grd gravmodel/UPLATE.grav.grd MUL = gravmodel/UPLATE.grav.grd')

    # Read in results
    XResG, YResG, ZResG = grdread2('gravmodel/UPLATE.grav.grd')

    # Pad edges to help with inversion (replicate edge rows/columns)
    ZResG[:, 0]  = ZResG[:, 1]
    ZResG[:, -1] = ZResG[:, -2]
    ZResG[-1, :] = ZResG[-2, :]
    ZResG[0, :]  = ZResG[1, :]

    return XResG, YResG, ZResG
