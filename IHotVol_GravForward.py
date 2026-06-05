"""
IHotVol_GravForward.py

Forward gravity model from topographic interfaces.
Equivalent to the MATLAB function IHotVol_GravForward.
"""

import os
import subprocess
import numpy as np
from scipy.special import erfinv
from grd_utils import grdread2


def IHotVol_GravForward(grdfile, denangrdfile, edificegrdfile,
                         sedcutgrdfile, subaerialgrdfile, mohoflexgrdfile,
                         rho_c, rho_w, rho_m, rho_i, kappa,
                         INP, ORS_L, subaq):
    """
    Compute the synthetic (forward) gravity from multiple interfaces.

    Parameters
    ----------
    grdfile         : str   – base grid filename
    denangrdfile    : str   – DENAN'd bathymetry grid
    edificegrdfile  : str   – edifice load grid
    sedcutgrdfile   : str   – sediment thickness grid
    subaerialgrdfile: str   – subaerial topography grid
    mohoflexgrdfile : str   – flexed Moho grid
    rho_c           : float – crust density (kg/m3)
    rho_w           : float – water density (kg/m3)
    rho_m           : float – mantle density (kg/m3)
    rho_i           : float – infill density (kg/m3)
    kappa           : float – thermal diffusivity
    INP             : ndarray – in-polygon mask
    ORS_L           : array-like – ORS result row
    subaq           : int   – 1 = fully submarine

    Returns
    -------
    XgMod : 1-D ndarray
    YgMod : 1-D ndarray
    ZgMod : 2-D ndarray  – synthetic gravity (mGal)
    """

    def run(cmd):
        env = os.environ.copy()
        env['GMT_VERBOSE'] = 'e'
        subprocess.run(cmd, shell=True, check=True, env=env)

    # Gravity contribution from sediment/water interface
    run(f'gravfft {denangrdfile} -D{rho_i - rho_w} -E5 -N+a -fg -Ggravmodel/sed.grav.grd')

    # Gravity from rock/sediment & rock/air interface
    if subaq == 0:
        # Variable density for subaerial/subaqueous portions
        run(f'grdmath {subaerialgrdfile} 2400 MUL = subaerial_rho.grd')
        run((f'grdmath {subaerialgrdfile} 1 SUB -1 DIV '
             f'{rho_c - rho_i} MUL subaerial_rho.grd ADD = rho_var.grd'))
        run(f'gravfft {denangrdfile} -Drho_var.grd -E4 -N+a -fg -Ggravmodel/rock.grav.grd')
        run(f'grdmath {denangrdfile} {subaerialgrdfile} MUL = subaerialtopo.grd')
        run('grdmath subaerialtopo.grd 0 LE gravmodel/sed.grav.grd MUL = gravmodel/sed.grav.grd')

    if subaq == 1:
        run(f'grdmath {denangrdfile} {sedcutgrdfile} SUB = gravmodel/rockoffset.grd')
        run(f'gravfft gravmodel/rockoffset.grd -D{rho_c - rho_i} -E4 -N+a -fg -Ggravmodel/rock.grav.grd')

    # Moho gravity (bottom of 7 km crust, 5 km water, plus flexure)
    run(f'grdmath {mohoflexgrdfile} 12000 SUB = {mohoflexgrdfile}')
    run((f'gravfft {mohoflexgrdfile} -D{rho_m - rho_c} '
         f'-E4 -fg -N+a -Ggravmodel/flex.moho.grav.grd'))

    # Get age and regional bathymetry
    run(f'grdsample cutagetmp/age.{grdfile} -R{grdfile}_regional.grd '
        f'-Ggravmodel/age.trim.{grdfile}')

    # Gravity from thermal structure (50°C slices up to 600°C isotherm)
    age_conv = 1e6 * 365.25 * 24 * 3600   # Ma to seconds

    for iso_T in range(50, 600, 50):   # 50, 100, ..., 550
        COEF  = erfinv(iso_T / 600) * 2 * np.sqrt(kappa)
        delRho = 50 * 3e-5 * 3300

        run((f'grdmath {grdfile}_regional.grd gravmodel/age.trim.{grdfile} '
             f'{age_conv} MUL SQRT '
             f'{COEF} MUL SUB = gravmodel/depth.{iso_T}C.grd'))
        run(f'grdmath gravmodel/depth.{iso_T}C.grd 0 DENAN = gravmodel/depth.{iso_T}C.grd')

        if iso_T == 50:
            import os
            if os.path.exists('thermal.grav.grd'):
                os.remove('thermal.grav.grd')
            run((f'gravfft gravmodel/depth.{iso_T}C.grd -D{-delRho} '
                 f'-E4 -fg -N+a -Ggravmodel/thermal.grav.grd'))
        else:
            run((f'gravfft gravmodel/depth.{iso_T}C.grd -D{-delRho} '
                 f'-E4 -fg -N+a -Ggravmodel/Grav.tmp.grd'))
            run('grdmath gravmodel/Grav.tmp.grd gravmodel/thermal.grav.grd ADD = gravmodel/thermal.grav.grd')

    # Resize grids to common extent
    run('grdsample -Rgravmodel/thermal.grav.grd gravmodel/flex.moho.grav.grd '
        '-Ggravmodel/flex.moho.trim.grav.grd')
    run('grdsample -Rgravmodel/thermal.grav.grd gravmodel/rock.grav.grd '
        '-Ggravmodel/rock.trim.grav.grd')
    run('grdsample -Rgravmodel/thermal.grav.grd gravmodel/sed.grav.grd '
        '-Ggravmodel/sed.trim.grav.grd')

    # Combine interfaces
    run(('grdmath gravmodel/flex.moho.trim.grav.grd gravmodel/rock.trim.grav.grd ADD '
         'gravmodel/thermal.grav.grd ADD gravmodel/sed.trim.grav.grd ADD = SYNTH.grav.grd'))

    # Read back in
    XgMod, YgMod, ZgMod = grdread2('SYNTH.grav.grd')

    return XgMod, YgMod, ZgMod
