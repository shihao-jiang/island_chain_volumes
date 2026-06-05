"""
IHotVol_SampleVolGrids.py

Formats and samples the final output grids along the hotspot track.
Equivalent to the MATLAB function IHotVol_SampleVolGrids.
"""

import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from grd_utils import grdread2
from IHotVol_PickMask import IHotVol_PickMask2


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

    def run(cmd):
        subprocess.run(cmd, shell=True, check=True)

    # Reformat grids
    run(f'grd2xyz Uplate.{ii}.grd | xyz2grd -GUplate.{ii}.grd -R{grdfile}')
    run(f'grd2xyz flexure.DENAN.{ii}.grd | xyz2grd -Gflexure.DENAN.{ii}.grd -R{grdfile}')
    run(f'grd2xyz {grdfile}_edifice.grd | xyz2grd -G{grdfile}_edifice.grd -R{grdfile}')
    run(f'grd2xyz INP.grd | xyz2grd -GINP.grd -R{grdfile}')

    # Sample the in-polygon grid at underplating resolution
    run(f'grdsample INP.grd -RUplate.{ii}.grd -GINP.sampled.grd')

    # Determine underplating within the polygon region
    run(f'grdmath INP.sampled.grd Uplate.{ii}.grd MUL = Uplate.INP.{ii}.grd')

    # Quick visual inspection
    quickcontgrd(f'Uplate.{ii}.grd', 20)
    quickcontgrd(f'flexure.DENAN.{ii}.grd', 20)

    mask2 = int(input('Do the results need another masking? 1=yes, 0=no\n> '))

    if mask2 == 1:
        XpolyM, YpolyM, INPmask = IHotVol_PickMask2(f'Uplate.{ii}.grd')

        run(f'grdmath flexure.DENAN.{ii}.grd MASK2.grd MUL 0 NAN = flexure.DENAN.{ii}.grd')
        run(f'grdmath Uplate.{ii}.grd MASK2.grd MUL 0 NAN = Uplate.{ii}.grd')

    # Remove any conflicting cross-section file
    xes_file = f'{grdfile}Xes.txt'
    if os.path.exists(xes_file):
        os.remove(xes_file)

    # Sample resultant grids along track with cross-sections
    run((f"cat track.txt | awk '{{print $1,$2}}' | "
         f"grdtrack -R{grdfile} -V "
         f"-G{grdfile}_edifice.grd "
         f"-Gflexure.DENAN.{ii}.grd "
         f"-GUplate.{ii}.grd "
         f"-GUplate.INP.{ii}.grd "
         f"-C1000k/4k/4k -Ar > {xes_file}"))
