"""
IHotspotVolume.py

Calculates along-track volumetric contributions from a hotspot trace
using bathymetry, age data, and gravity model inputs.

Bathymetry bounds should provide an adequate buffer around the hotspot
track such that flexure calculations and gravity models (FFT: subject to
ringing noise) do not influence the near-hotspot result. Suggested bounds
are at least 10 degrees lon/lat from the edges of the hotspot expression.

Age data should be placed in AGES.txt following the format in the header file.

Remaining inputs are clipped from global grids using the boundaries of
the chosen topo file.

Developed with GMT version 6.1.0 (uses GMT shell commands via subprocess).

Revision history (from original MATLAB):
  First draft - TMorrow - Aug 03 2018
  Orthogonal interp added - TMorrow Sept 08 2019
  Separating out functions - TMorrow 25 Sep 2019
  Code cleanup - TMorrow 14 Sept 2020
  Python conversion - 2026
"""

import os
import shutil
import subprocess
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # change backend if needed
import matplotlib.pyplot as plt

from grd_utils import grdread2
from IHotVol_PickMask import IHotVol_PickMask
from IHotVol_ORS import IHotVol_ORS
from IHotVol_Track import IHotVol_Track
from IHotVol_PickEdifice import IHotVol_PickEdifice
from IHotVol_Airy import IHotVol_Airy
from IHotVol_Flexure import IHotVol_Flexure
from IHotVol_GravForward import IHotVol_GravForward
from IHotVol_FAAgetResidual import IHotVol_FAAgetResidual
from IHotVol_Underplating import IHotVol_Underplating
from IHotVol_SampleVolGrids import IHotVol_SampleVolGrids
from IHotVol_VolumeSlices import IHotVol_VolumeSlices
from IHotVol_AgeVolPlot import IHotVol_AgeVolPlot
from IHotVol_Spectral import IHotVol_Spectral


def run(cmd):
    """Run a shell (GMT) command, suppressing GMT warnings."""
    env = os.environ.copy()
    env['GMT_VERBOSE'] = 'e'   # e = errors only, suppress warnings
    subprocess.run(cmd, shell=True, check=True, env=env)


# =============================================================================
# -- Inputs and paths
# =============================================================================

# grid file (name without extension)
grdname = 'Balleny'
grdfile = grdname + '.grd'

# AGES file  [lon, lat, age_Ma]
AGES = np.loadtxt('AGES.txt')

# seafloor age file
SFLagegrd = './global/infl.age.3.6.grd'

# sediment thickness file
SEDthckgrd = './global/sedthick_world_v2.grd'

# WGM FAA file
WGMFAAgrd = './global/WGM2012_Freeair_ponc_2min_360.grd'

# adjust AGES data – uncomment if needed
AGES[AGES[:, 0] < 0, 0] += 360

# northeast bathy corner  [north, east]
NE = [np.max(AGES[:, 1]) + 5, np.max(AGES[:, 0]) + 5]
# southwest bathy corner  [south, west]
SW = [np.min(AGES[:, 1]) - 5, np.min(AGES[:, 0]) - 5]

# filter parameters for RR separation
minW  = 100   # minimum filter width candidate for ORS (km)
maxW  = 600   # maximum filter width candidate for ORS (km)
intW  = 100    # filter width step (km)
level = 300   # step for base contour calculations
subaq = 1     # 1 = all hotspot is underwater
mask  = 0     # 1 = prominent regions need masking

# Flexure model inputs
rho_c   = 2800    # crust density (kg/m3)
rho_w   = 1035    # water density (kg/m3)
rho_m   = 3300    # mantle density (kg/m3)
rho_i   = 2400    # infill density (kg/m3)
rho_u   = 3000    # underplating density (kg/m3)
E       = 2e23    # Young's modulus
v       = 0.25    # Poisson's ratio
g       = 9.8     # gravity accel (m/s2)
cr_thck = 7000    # crust thickness (m)
T_elas  = 400     # elastic isotherm (°C)
T_mant  = 1300    # mantle temperature (°C)
kappa   = 1e-6    # thermal diffusivity

# =============================================================================
# -- (1) Get grid file
# =============================================================================

# create CORNERS file for WGET_BATHY
if os.path.exists('CORNERS.xy'):
    os.remove('CORNERS.xy')
with open('CORNERS.xy', 'w') as f:
    cornerline = f"{grdname},{NE[1]},{NE[0]},{SW[1]},{SW[0]}\n"
    f.write(cornerline)

# retrieve WGET script and download grid only if not already present
if not os.path.exists(grdfile):
    shutil.copy('./dependencies/WGET_BATHY.sh', './')
    run('bash WGET_BATHY.sh')
else:
    print(f'{grdfile} already exists, skipping download.')

# resample high-res to 2 arc-minutes
run(f'grdsample {grdfile} -R{grdfile} -I2m+e -G{grdfile}')

# read in final grid
X, Y, Z = grdread2(grdfile)

# =============================================================================
# -- (2) Mask interfering regions of the map
# =============================================================================

if mask == 1:
    XpolyM, YpolyM, INPmask = IHotVol_PickMask(grdfile)

# =============================================================================
# -- (3) Optimised Residual Separation
# =============================================================================

if mask == 1:
    shutil.copy('./dependencies/RR-Sep_mask.sh', './')
    shutil.copy('./dependencies/RR-Sep-single_mask.sh', './')
else:
    shutil.copy('./dependencies/RR-Sep.sh', './')
    shutil.copy('./dependencies/RR-Sep-single.sh', './')

if not os.path.exists('ORStable.txt'):
    ORS_L, region = IHotVol_ORS(grdfile, X, Y, Z, minW, maxW, intW, level, mask)
    print(f'Regional/residual separation complete! ORS optimal filter wavelength {ORS_L[0]} km')
else:
    print('ORStable.txt already exists, skipping ORS.')
    ORS = np.loadtxt('ORStable.txt')
    ORS_L = ORS[ORS[:, 4] == ORS[:, 4].max(), :]
    if ORS_L.ndim == 2:
        ORS_L = ORS_L[0]
    region = (f'-R{np.ceil(X.min()) + 0.05:.4f}/{np.floor(X.max()) - 0.05:.4f}'
              f'/{np.ceil(Y.min()) + 0.05:.4f}/{np.floor(Y.max()) - 0.05:.4f}')
    print(f'Loaded ORS optimal filter wavelength {ORS_L[0]} km')

# =============================================================================
# -- (4) Generate hotspot age track and clipped age sub-grids
# =============================================================================

HSPT_TRK, AGES, pA, TMPTRK = IHotVol_Track(AGES, X, Y, Z, grdfile, SFLagegrd)

# =============================================================================
# -- (5) Generate initial topographic load grid
# =============================================================================

Xpoly, Ypoly, INP = IHotVol_PickEdifice(grdfile, AGES)

# =============================================================================
# -- (6) Initial flexure solve (Airy)
# =============================================================================

IHotVol_Airy(grdfile, rho_c, rho_w, rho_m)

# =============================================================================
# -- (7) Flexure calculation
# =============================================================================

ii = 1
Xflx, Yflx, Zflx = IHotVol_Flexure(
    f'{grdfile}_edifice.grd', rho_c, rho_w, rho_m, rho_i,
    T_elas, T_mant, kappa, HSPT_TRK, grdfile, ii)

# =============================================================================
# -- (8) Gravity forward model
# =============================================================================

os.makedirs('gravmodel', exist_ok=True)
sedcutgrdfile   = 'gravmodel/sedcut.DENAN.plusone.grd'
subaerialgrdfile= 'gravmodel/subair.DENAN.grd'
mohoflexgrdfile = f'flexure.DENAN.{ii}.grd'
denangrdfile    = f'gravmodel/{grdfile}.DENAN.grd'
edificegrdfile  = f'{grdfile}_edifice.grd'

# make zero grid for subbing NaNs
run(f'grdmath {grdfile} 0 MUL = zerogrdfile.grd')
run(f'grdsample {grdfile} -R{edificegrdfile} -G{denangrdfile}')

# DENAN bathy
run(f'grdmath {denangrdfile} 0 DENAN = {denangrdfile}')

# subaerial part of grid
run(f'grdmath {grdfile} 0 GT {grdfile} MUL = gravmodel/subair.grd')
run(f'grdmath gravmodel/subair.grd 0 DENAN = {subaerialgrdfile}')

# trim and DENAN sediment grid for gravity
run(f'grdsample {SEDthckgrd} -R{grdfile} -Ggravmodel/sedcut.{grdfile}')
run(f'grdmath gravmodel/sedcut.{grdfile} 0 DENAN 1 {grdfile} ADD  = {sedcutgrdfile}')
run(f'grdmath {sedcutgrdfile} 0 DENAN = {sedcutgrdfile}')

# generate forward gravity model
XgMod, YgMod, ZgMod = IHotVol_GravForward(
    grdfile, denangrdfile, edificegrdfile, sedcutgrdfile,
    subaerialgrdfile, mohoflexgrdfile,
    rho_c, rho_w, rho_m, rho_i, kappa, INP, ORS_L, subaq)

# =============================================================================
# -- (9) FAA
# =============================================================================

XResG, YResG, ZResG = IHotVol_FAAgetResidual(ORS_L, WGMFAAgrd, mask, subaq)

# polygon select for residual determination
INPold = INP.copy()
XResGmesh, YResGmesh = np.meshgrid(XResG, YResG)
from matplotlib.path import Path
poly_path = Path(np.column_stack([Xpoly, Ypoly]))
pts = np.column_stack([XResGmesh.ravel(), YResGmesh.ravel()])
INP = poly_path.contains_points(pts).reshape(XResGmesh.shape)

# gravity residual along HSPT_TRK
grav_resid = np.sqrt(float(np.sum((INP * ZResG) ** 2)))

# =============================================================================
# -- (10) Underplating
# =============================================================================

Xg, Yg, finaltopoinverse = IHotVol_Underplating(
    Xflx, Yflx, Zflx, XResG, YResG, ZResG, ii, grdfile, ORS_L, 1e-5)

# =============================================================================
# -- (11) Iterative flexure/underplating revisions
# =============================================================================

residDiff = [1e10]  # index 0 corresponds to ii=1

while residDiff[ii - 1] > 0.0001:

    ii += 1
    print(f'Iteration {ii}')

    # resample/fit
    run(f'grdsample Uplate.{ii-1}.grd -R{grdfile}_edifice.grd -GUplate.{ii-1}.grd')
    run(f'grdmath Uplate.{ii-1}.grd  Uplate.{ii-1}.grd LOWER SUB = Uplate.{ii-1}.grd')

    # load reduction correction
    run((f'grdmath {grdfile}_edifice.grd {rho_c - rho_w} MUL '
         f'Uplate.{ii-1}.grd 1000 MUL {rho_u - rho_m} MUL ADD '
         f'{rho_c - rho_w} DIV 0 DENAN = {grdfile}_edifice.{ii}.grd'))

    # new flexure profile
    Xflx, Yflx, Zflx = IHotVol_Flexure(
        f'{grdfile}_edifice.{ii}.grd', rho_c, rho_w, rho_m, rho_i,
        T_elas, T_mant, kappa, HSPT_TRK, grdfile, ii)
    mohoflexgrdfile = f'flexure.DENAN.{ii}.grd'

    if ii == 2:
        run(f'grdsample {grdfile} -R{grdfile}_edifice.{ii}.grd -G{grdfile}sampleiter.grd')
        run((f'grdmath {grdfile}_edifice.{ii}.grd {grdfile}_edifice.{ii}.grd MEAN '
             f'{grdfile}sampleiter.grd MEAN SUB SUB 0 DENAN = {grdfile}_edifice.{ii}.grd'))
    else:
        run((f'grdmath {grdfile}_edifice.{ii}.grd {grdfile}_edifice.{ii}.grd MEAN '
             f'{grdfile}_edifice.{ii-1}.grd MEAN SUB SUB 0 DENAN = {grdfile}_edifice.{ii}.grd'))

    # forward gravity model
    XgMod, YgMod, ZgMod = IHotVol_GravForward(
        grdfile, denangrdfile, edificegrdfile, sedcutgrdfile,
        subaerialgrdfile, mohoflexgrdfile,
        rho_c, rho_w, rho_m, rho_i, kappa, INP, ORS_L, subaq)

    # new residual
    XResG, YResG, ZResGNEW = IHotVol_FAAgetResidual(ORS_L, WGMFAAgrd, mask, subaq)

    # gravity residual along HSPT_TRK
    grav_residNEW = np.sqrt(float(np.sum((INP * ZResGNEW) ** 2)))

    # assess convergence
    new_diff = abs(grav_resid - grav_residNEW)
    residDiff.append(new_diff)
    grav_resid = grav_residNEW
    print(f'current iteration residual change: {new_diff}')

    # calculate underplating
    Xg, Yg, finaltopoinverse = IHotVol_Underplating(
        Xflx, Yflx, Zflx, XResG, YResG, ZResG, ii, grdfile, ORS_L, 1e-5)

# final flexure calculation using only the compensated edifice
run(f'grdsample {grdfile}_edifice.grd -R{grdfile}_edifice.{ii}.grd -G{grdfile}_edifice.flexsample.grd')
run((f'grdmath {grdfile}_edifice.flexsample.grd 0 GT '
     f'{grdfile}_edifice.{ii}.grd MUL 0 DENAN = {grdfile}_edifice.FINAL.grd'))
Xflx, Yflx, Zflx = IHotVol_Flexure(
    f'{grdfile}_edifice.{ii}.grd', rho_c, rho_w, rho_m, rho_i,
    T_elas, T_mant, kappa, HSPT_TRK, grdfile, ii)

# =============================================================================
# -- (12) Sample final output grids
# =============================================================================

IHotVol_SampleVolGrids(grdfile, ii, HSPT_TRK, mask)

# =============================================================================
# -- (13) Import cross sections
# =============================================================================
# Import the cross-section text file as a numpy array.
# The variable name should match what is passed to IHotVol_VolumeSlices below.
# e.g.:  Xes = np.loadtxt(f'{grdname}Xes.txt')

# =============================================================================
# -- (14) Calculate volumes
# =============================================================================

try:
    plt.close('all')
except Exception:
    pass
VOL = np.array([])

# volume calculation  (Xes must be loaded from step 13)
Xes = np.loadtxt(f'{grdname}Xes.txt')
VOL, Crosses = IHotVol_VolumeSlices(Xes, HSPT_TRK)

# generate age-volume plot
VOL = IHotVol_AgeVolPlot(VOL, Crosses, pA)

# =============================================================================
# -- (15) Spectral analysis
# =============================================================================

PEAKS = IHotVol_Spectral(VOL, grdfile)

# =============================================================================
# -- (16) Save workspace
# =============================================================================

np.savez('completed.npz',
         VOL=VOL, HSPT_TRK=HSPT_TRK, AGES=AGES, ORS_L=ORS_L,
         PEAKS_total=PEAKS.get('Total', np.array([])))

print('Done. Results saved to completed.npz')
