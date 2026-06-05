"""
IHotVol_Flexure.py

Calculates a blended flexure profile using loading ages from HSPT_TRK.
Equivalent to the MATLAB function IHotVol_Flexure.
"""

import os
import subprocess
import shutil
import numpy as np
from scipy.special import erfinv
from grd_utils import grdread2


def IHotVol_Flexure(loadfile, rho_c, rho_w, rho_m, rho_i,
                    T_elas, T_mant, kappa, HSPT_TRK, grdfile, ii):
    """
    Compute a blended flexure grid for the hotspot load.

    Parameters
    ----------
    loadfile  : str      – edifice/load grid file
    rho_c     : float    – crust density (kg/m3)
    rho_w     : float    – water density (kg/m3)
    rho_m     : float    – mantle density (kg/m3)
    rho_i     : float    – infill density (kg/m3)
    T_elas    : float    – elastic isotherm (°C)
    T_mant    : float    – mantle temperature (°C)
    kappa     : float    – thermal diffusivity (m2/s)
    HSPT_TRK  : ndarray  – hotspot track array (N x >=4 columns:
                           lon, lat, seamount_age, litho_age, ...)
    grdfile   : str      – base grid filename
    ii        : int      – iteration number

    Returns
    -------
    Xflx : 1-D ndarray – x coordinates
    Yflx : 1-D ndarray – y coordinates
    Zflx : 2-D ndarray – flexure grid values
    """

    def run(cmd):
        env = os.environ.copy()
        env['GMT_VERBOSE'] = 'e'
        subprocess.run(cmd + ' 2>/dev/null', shell=True, check=True, env=env)

    # Seconds per year
    sec_per_yr = 365.25 * 24 * 3600

    # Generate blend file
    if os.path.exists('Blendfile.txt'):
        os.remove('Blendfile.txt')
    os.makedirs('FLXcomptmp', exist_ok=True)

    blend_lines = []

    for jj in range(len(HSPT_TRK)):

        # Loading age = lithosphere age - seamount age
        Agediff = HSPT_TRK[jj, 3] - HSPT_TRK[jj, 2]

        if Agediff < 0:
            Agediff = 0.001   # litho min age is that of seamount (Te -> 0)

        Agediff_s = Agediff * 1e6 * sec_per_yr   # convert to seconds

        # Elastic thickness (km)
        Te = erfinv(T_elas / T_mant) * 2 * np.sqrt(kappa * Agediff_s) / 1e3

        HSPT_TRK[jj, 5] = Te   # store Te in track array

        # GMT grdflexure for this Te
        run((f'grdflexure {loadfile} '
             f'-D{rho_m}/{rho_c}/{rho_i}/{rho_w} '
             f'-E{Te}k -fg -N+a '
             f'-GFLX_comp_Te_{round(10 * Te) / 10:.1f}km.grd'))

        # Append to blend file with blending region
        lon  = HSPT_TRK[jj, 0]
        lat  = HSPT_TRK[jj, 1]
        Te_r = round(10 * Te) / 10
        blend_lines.append(
            f'FLX_comp_Te_{Te_r:.1f}km.grd '
            f'-R{lon - 2}/{lon + 2}/{lat - 2}/{lat + 2} 1'
        )

    # Write blend file
    with open('Blendfile.txt', 'w') as fid:
        fid.write('\n'.join(blend_lines) + '\n')

    # Generate blended flexure grid
    run(f'grdblend Blendfile.txt -Gflexure.{ii}.grd -R{grdfile}')

    # Stash pre-blend grids (overwrite if already exists)
    for fname in os.listdir('.'):
        if fname.startswith('FLX_comp'):
            dst = os.path.join('FLXcomptmp', fname)
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(fname, 'FLXcomptmp/')

    # DENAN flexure grid (replace NaN with 0)
    run(f'grdmath flexure.{ii}.grd 0 DENAN = flexure.DENAN.{ii}.grd')

    # Read in flexure grid
    Xflx, Yflx, Zflx = grdread2(f'flexure.DENAN.{ii}.grd')

    return Xflx, Yflx, Zflx
