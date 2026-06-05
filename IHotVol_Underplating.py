"""
IHotVol_Underplating.py

Parker-Oldenburg gravity inversion for underplating thickness.
Adapted from 3DINVER.M (Gomez-Ortiz & Agarwal).
Equivalent to the MATLAB function IHotVol_Underplating.
"""

import math
import subprocess
import numpy as np
from scipy.signal import windows as sig_windows
from grd_utils import grdwrite2


def IHotVol_Underplating(Xfl, Yfl, Zfl, Xg, Yg, Zg,
                          ii, grdfile, ORS_L, criterio):
    """
    Invert gravity for underplating topography using the Parker-Oldenburg
    iterative algorithm (up to 10 iterations).

    Parameters
    ----------
    Xfl      : 1-D ndarray – flexure x coords (used for z0 via Zfl)
    Yfl      : 1-D ndarray – flexure y coords
    Zfl      : 2-D ndarray – flexure values (used to set reference depth)
    Xg       : 1-D ndarray – gravity x coords
    Yg       : 1-D ndarray – gravity y coords
    Zg       : 2-D ndarray – residual gravity (mGal)
    ii       : int          – iteration number
    grdfile  : str          – base grid filename
    ORS_L    : array-like   – ORS result; ORS_L[0] = filter wavelength (km)
    criterio : float        – convergence criterion (km)

    Returns
    -------
    Xg               : 1-D ndarray – (possibly padded) x coords
    Yg               : 1-D ndarray – (possibly padded) y coords
    finaltopoinverse : 2-D ndarray – estimated underplating topography (km)
    """

    def run(cmd):
        subprocess.run(cmd, shell=True, check=True)

    # ------------------------------------------------------------------
    # Mirror and extend the gravity grid to reduce edge effects
    # ------------------------------------------------------------------
    Zg1 = Zg[::-1, :]           # flipud
    Zg2 = Zg[:, ::-1]           # fliplr
    Zg3 = Zg[::-1, ::-1]        # rot90 by 180

    Zstitch = np.block([
        [Zg3, Zg1, Zg3],
        [Zg2, Zg,  Zg2],
        [Zg3, Zg1, Zg3],
    ])

    dx_start = abs(Xg[0] - Xg[-1]) + abs(Xg[0]  - Xg[1])
    dy_start = abs(Yg[0] - Yg[-1]) + abs(Yg[0]  - Yg[1])
    dx_end   = abs(Xg[0] - Xg[-1]) + abs(Xg[-2] - Xg[-1])
    dy_end   = abs(Yg[0] - Yg[-1]) + abs(Yg[-2] - Yg[-1])

    Xstitch = np.concatenate([Xg - dx_start, Xg, Xg + dx_end])
    Ystitch = np.concatenate([Yg - dy_start, Yg, Yg + dy_end])

    # ------------------------------------------------------------------
    # Square power-of-2 sub-grid
    # ------------------------------------------------------------------
    nx = Xg.size
    ny = Yg.size

    boxEl = max(int(2 ** np.ceil(np.log2(nx))),
                int(2 ** np.ceil(np.log2(ny))))

    # Avoid grossly over-sizing one dimension
    def _half_pow2(n):
        p = int(2 ** np.ceil(np.log2(n)))
        return int(p // 2 + (p - p // 2) // 2)

    if boxEl == int(2 ** np.ceil(np.log2(nx))) and boxEl > 3 * ny:
        boxEl = max(_half_pow2(nx), _half_pow2(ny))
    if boxEl == int(2 ** np.ceil(np.log2(ny))) and boxEl > 3 * nx:
        boxEl = max(_half_pow2(nx), _half_pow2(ny))

    Xind = round(nx / 2) + nx
    Yind = round(ny / 2) + ny

    half = boxEl // 2
    Xsq = Xstitch[Xind - half: Xind + half]
    Ysq = Ystitch[Yind - half: Yind + half]
    Zsq = Zstitch[Yind - half: Yind + half,
                  Xind - half: Xind + half]

    Xg = Xsq
    Yg = Ysq
    Zg = Zsq.astype(float)

    numrows    = len(Yg)
    numcolumns = len(Xg)

    # ------------------------------------------------------------------
    # Physical dimensions of the box
    # ------------------------------------------------------------------
    mid_row = numrows // 2
    mid_col = numcolumns // 2

    from IHotVol_Track import _deg2km   # reuse great-circle helper
    longx = _deg2km(Yg[mid_row], Xg[0],    Yg[mid_row], Xg[-1])
    longy = _deg2km(Yg[0],       Xg[mid_col], Yg[-1],   Xg[mid_col])

    contrast = 0.3          # density contrast (g/cm3)
    z0       = 11 + abs(np.nanmin(Zfl)) / 1e3   # mean reference depth (km)
    WH       = 1 / 100      # smaller cut-off freq (1/km)
    SH       = 1 / 80       # larger  cut-off freq (1/km)
    truncation = 0.1        # Tukey window truncation fraction

    # ------------------------------------------------------------------
    # Demean and window the input gravity
    # ------------------------------------------------------------------
    bou = Zg - np.nanmean(Zg)

    wrows    = sig_windows.tukey(numrows,    truncation)
    wcols    = sig_windows.tukey(numcolumns, truncation)
    w2       = np.outer(wrows, wcols)
    bou      = bou * w2

    fftbou = np.fft.fft2(bou)

    # ------------------------------------------------------------------
    # Build frequency matrix (1/wavelength in km)
    # ------------------------------------------------------------------
    fh = numrows    // 2 + 1
    fw = numcolumns // 2 + 1

    f_arr = np.zeros((fh, fw))
    for f_idx in range(fh):
        for g_idx in range(fw):
            f_arr[f_idx, g_idx] = np.sqrt(
                ((f_idx) / longx) ** 2 + ((g_idx) / longy) ** 2)

    # Mirror to full size
    f2 = np.fliplr(f_arr)
    f3 = np.flipud(f_arr)
    f4 = np.flipud(np.fliplr(f_arr))

    even_cols = (numcolumns % 2 == 0)
    even_rows = (numrows    % 2 == 0)

    if even_cols:
        f2 = f2[:, 1:]
        f4 = f4[:, 1:]
    if even_rows:
        f3 = f3[1:, :]
        f4 = f4[1:, :]

    ftot = np.block([[f_arr, f2], [f3, f4]])
    if ftot.shape[0] > numrows:
        ftot = ftot[:numrows, :]
    if ftot.shape[1] > numcolumns:
        ftot = ftot[:, :numcolumns]

    ftot_wn = ftot * (2 * np.pi)   # convert to wavenumber

    # ------------------------------------------------------------------
    # High-cut filter
    # ------------------------------------------------------------------
    filt = np.zeros_like(ftot)
    for r in range(numrows):
        for c in range(numcolumns):
            fv = ftot[r, c]
            if fv < WH:
                filt[r, c] = 1.0
            elif fv < SH:
                filt[r, c] = 0.5 * (1 + np.cos(
                    (2 * np.pi * fv - 2 * np.pi * WH) / (2 * (SH - WH))))
            # else 0 (already zero)

    # ------------------------------------------------------------------
    # First term of the Parker series
    # ------------------------------------------------------------------
    up       = -(fftbou * np.exp(z0 * ftot_wn))
    down     = 2 * np.pi * 6.67e-3 * contrast   # G in mGal·km·g⁻¹·cm³
    constant = (up / down) * filt

    topoinverse = np.real(np.fft.ifft2(constant))

    # ------------------------------------------------------------------
    # Iterative Parker-Oldenburg refinement (up to 10 iterations)
    # ------------------------------------------------------------------
    def _rms(a, b, nr, nc):
        diff = (a - b) ** 2
        return np.sqrt(np.sum(diff) / (2 * nr * nc))

    def _next_topo(prev, n_terms, ftot_wn, constant, filt):
        """Compute topo estimate using n_terms terms of the Parker series."""
        series_sum = np.zeros_like(fftbou)
        for k in range(1, n_terms):
            series_sum += (
                (ftot_wn ** (k - 1)) / math.factorial(k)
            ) * np.fft.fft2(prev ** k)
        series_sum *= filt
        result = constant - series_sum
        return np.real(np.fft.ifft2(result))

    finaltopoinverse = topoinverse.copy()
    prev             = topoinverse.copy()
    rms_val          = _rms(finaltopoinverse, prev * 0, numrows, numcolumns)

    max_iter = 10
    it       = 1

    for n in range(2, max_iter + 1):
        curr = _next_topo(finaltopoinverse, n, ftot_wn, constant, filt)
        rms_val = _rms(curr, finaltopoinverse, numrows, numcolumns)
        print(f'  Underplating iter {n}, rms = {rms_val:.6f} km')
        finaltopoinverse = curr
        it = n
        if rms_val < criterio:
            break

    print(f'Underplating converged at iteration {it}, rms = {rms_val:.6f} km')

    # ------------------------------------------------------------------
    # Write output grid
    # ------------------------------------------------------------------
    grdwrite2(Xg, Yg, finaltopoinverse, f'Uplate.{ii}.grd')
    run(f'grd2xyz Uplate.{ii}.grd | xyz2grd -GUplate.{ii}.grd -R{grdfile}')

    return Xg, Yg, finaltopoinverse
