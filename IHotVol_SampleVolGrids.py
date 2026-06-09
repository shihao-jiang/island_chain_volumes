"""
IHotVol_SampleVolGrids.py

Formats and samples the final output grids along the hotspot track.
Equivalent to the MATLAB function IHotVol_SampleVolGrids.
"""

import os
import subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from grd_utils import grdread2
from IHotVol_PickMask import IHotVol_PickMask2
from gmt_utils import run

def quickcontgrd(grdfile, nlevels=20):
    """Quick contour plot of a grd file (replaces MATLAB quickcontgrd)."""
    X, Y, Z = grdread2(grdfile)
    Xm, Ym = np.meshgrid(X, Y)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.contourf(Xm, Ym, Z, levels=nlevels)
    ax.set_title(grdfile)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

def IHotVol_SampleVolGrids(grdfile, ii, HSPT_TRK, mask):
    """
    Reformat grids, optionally apply secondary masking, then sample along track.

    Parameters
    ----------
    grdfile  : str     – base grid filename
    ii       : int     – final iteration number
    HSPT_TRK : ndarray – hotspot track array
    mask     : int     – 1 if masking was applied (unused here but kept for API)
    """

    # Reformat grids
    run(f'gmt grd2xyz Uplate.{ii}.grd | gmt xyz2grd -GUplate.{ii}.grd -R{grdfile}')
    run(f'gmt grd2xyz flexure.DENAN.{ii}.grd | gmt xyz2grd -Gflexure.DENAN.{ii}.grd -R{grdfile}')
    run(f'gmt grd2xyz {grdfile}_edifice.grd | gmt xyz2grd -G{grdfile}_edifice.grd -R{grdfile}')
    run(f'gmt grd2xyz INP.grd | gmt xyz2grd -GINP.grd -R{grdfile}')

    # Sample the in-polygon grid at underplating resolution
    run(f'gmt grdsample INP.grd -RUplate.{ii}.grd -GINP.sampled.grd')

    # Determine underplating within the polygon region
    run(f'gmt grdmath INP.sampled.grd Uplate.{ii}.grd MUL = Uplate.INP.{ii}.grd')

    # Quick visual inspection
    quickcontgrd(f'Uplate.{ii}.grd', 20)
    quickcontgrd(f'flexure.DENAN.{ii}.grd', 20)

    mask2 = int(input('Do the results need another masking? 1=yes, 0=no\n> '))

    if mask2 == 1:
        XpolyM, YpolyM, INPmask = IHotVol_PickMask2(f'Uplate.{ii}.grd')

        # Resample MASK2.grd to each target grid's resolution before multiplying
        # (grids may differ in size/spacing, grdmath requires identical registration)
        run(f'gmt grdsample MASK2.grd -RUplate.{ii}.grd -GMASK2_uplate.grd')
        run(f'gmt grdsample MASK2.grd -Rflexure.DENAN.{ii}.grd -GMASK2_flex.grd')

        run(f'gmt grdmath Uplate.{ii}.grd MASK2_uplate.grd MUL 0 NAN = Uplate.{ii}.grd')
        run(f'gmt grdmath flexure.DENAN.{ii}.grd MASK2_flex.grd MUL 0 NAN = flexure.DENAN.{ii}.grd')

    # Remove any conflicting cross-section file
    xes_file = f'{grdfile}Xes.txt'
    if os.path.exists(xes_file):
        os.remove(xes_file)

    # Sample resultant grids along track with cross-sections
    run((f"cat track.txt | awk '{{print $1,$2}}' | "
         f"gmt grdtrack -R{grdfile} -V "
         f"-G{grdfile}_edifice.grd "
         f"-Gflexure.DENAN.{ii}.grd "
         f"-GUplate.{ii}.grd "
         f"-GUplate.INP.{ii}.grd "
         f"-C300k/4k/4k -Ar > {xes_file}"))
