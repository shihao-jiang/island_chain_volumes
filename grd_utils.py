"""
grd_utils.py
Utility functions for reading and writing GMT .grd (NetCDF) grid files,
replacing MATLAB's grdread2 / grdwrite2.
"""

import numpy as np

try:
    import netCDF4 as nc
    _BACKEND = 'netCDF4'
except ImportError:
    try:
        import scipy.io.netcdf as _scipy_nc
        _BACKEND = 'scipy'
    except ImportError:
        _BACKEND = None


def grdread2(filename):
    """
    Read a GMT .grd (NetCDF) file.

    Returns
    -------
    X : 1-D ndarray  – longitude / x coordinates
    Y : 1-D ndarray  – latitude  / y coordinates
    Z : 2-D ndarray  – data values  (shape: len(Y) x len(X))
    """
    if _BACKEND == 'netCDF4':
        with nc.Dataset(filename, 'r') as ds:
            # GMT grids use 'x'/'y' or 'lon'/'lat'
            xname = 'x' if 'x' in ds.variables else 'lon'
            yname = 'y' if 'y' in ds.variables else 'lat'
            X = np.array(ds.variables[xname][:], dtype=float)
            Y = np.array(ds.variables[yname][:], dtype=float)
            # data variable is whatever is left
            skip = {xname, yname, 'x_range', 'y_range', 'z_range',
                    'spacing', 'dimension'}
            zname = next(k for k in ds.variables if k not in skip)
            raw = ds.variables[zname][:]
            Z = np.ma.filled(raw, np.nan).astype(float)
            if Z.ndim == 1:
                # some GMT files store z as a flat array
                Z = Z.reshape(len(Y), len(X))
    elif _BACKEND == 'scipy':
        with _scipy_nc.netcdf_file(filename, 'r') as f:
            xname = 'x' if 'x' in f.variables else 'lon'
            yname = 'y' if 'y' in f.variables else 'lat'
            X = f.variables[xname][:].copy().astype(float)
            Y = f.variables[yname][:].copy().astype(float)
            skip = {xname, yname}
            zname = next(k for k in f.variables if k not in skip)
            Z = f.variables[zname][:].copy().astype(float)
            if Z.ndim == 1:
                Z = Z.reshape(len(Y), len(X))
    else:
        raise ImportError("Install netCDF4 or scipy to read .grd files.")
    return X, Y, Z


def grdwrite2(X, Y, Z, filename):
    """
    Write a GMT-compatible .grd (NetCDF) file.

    Parameters
    ----------
    X : 1-D array-like  – x / longitude coordinates
    Y : 1-D array-like  – y / latitude  coordinates
    Z : 2-D array-like  – data  (shape: len(Y) x len(X))
    filename : str
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if _BACKEND == 'netCDF4':
        with nc.Dataset(filename, 'w', format='NETCDF3_CLASSIC') as ds:
            ds.createDimension('x', len(X))
            ds.createDimension('y', len(Y))
            xv = ds.createVariable('x', 'f8', ('x',))
            yv = ds.createVariable('y', 'f8', ('y',))
            zv = ds.createVariable('z', 'f4', ('y', 'x'))
            zv.missing_value = np.float32(np.nan)
            xv[:] = X
            yv[:] = Y
            zv[:] = Z.astype(np.float32)
    elif _BACKEND == 'scipy':
        with _scipy_nc.netcdf_file(filename, 'w') as f:
            f.createDimension('x', len(X))
            f.createDimension('y', len(Y))
            xv = f.createVariable('x', 'f8', ('x',))
            yv = f.createVariable('y', 'f8', ('y',))
            zv = f.createVariable('z', 'f4', ('y', 'x'))
            xv[:] = X
            yv[:] = Y
            zv[:] = Z.astype(np.float32)
    else:
        raise ImportError("Install netCDF4 or scipy to write .grd files.")
