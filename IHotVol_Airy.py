"""
IHotVol_Airy.py

Airy isostatic underplating estimate.
Equivalent to the MATLAB function IHotVol_Airy.
"""

import subprocess
from gmt_utils import run

def IHotVol_Airy(grdfile, rho_c, rho_w, rho_m):
    """
    Compute Airy compensation directly from the edifice grid using GMT grdmath.

    Parameters
    ----------
    grdfile : str   – base grid filename (without extension where appropriate,
                      but the edifice grid is  grdfile + '_edifice.grd')
    rho_c   : float – crust density (kg/m3)
    rho_w   : float – water density (kg/m3)
    rho_m   : float – mantle density (kg/m3)
    """

    cmd = (
        f'gmt grdmath {rho_c} {rho_w} SUB {rho_m} {rho_c} SUB DIV '
        f'{grdfile}_edifice.grd MUL NEG = '
        f'{grdfile}_airy_compensation.grd'
    )
    run(cmd)
