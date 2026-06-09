"""
gmt_utils.py

Cross-platform GMT helper utilities.

Provides a single `run()` function that:
  - Auto-detects the GMT binary location on macOS, Linux, and Windows WSL
  - Injects the correct PATH so subprocesses can always find GMT
  - Suppresses GMT warnings via GMT_VERBOSE=e

Usage:
    from gmt_utils import run
    run('gmt grdsample input.grd -Routput.grd -I2m+e -Goutput.grd')
"""

import os
import shutil
import subprocess
import sys


# ---------------------------------------------------------------------------
# GMT path detection
# ---------------------------------------------------------------------------

# Common GMT install locations per platform
_GMT_SEARCH_PATHS = [
    '/opt/homebrew/bin',       # macOS Apple Silicon (Homebrew)
    '/usr/local/bin',          # macOS Intel (Homebrew) / Linux
    '/usr/bin',                # Linux / WSL (apt install)
    '/usr/local/gmt/bin',      # manual installs
    '/opt/local/bin',          # macOS MacPorts
    'C:/Program Files/GMT/bin',# Windows native
]


def _find_gmt_dir():
    """
    Return the directory containing the gmt binary.
    Tries shutil.which() first (respects the current PATH),
    then falls back to checking known locations.
    Raises RuntimeError if GMT cannot be found.
    """
    # 1. Check if gmt is already on PATH
    gmt_exe = shutil.which('gmt')
    if gmt_exe:
        return os.path.dirname(gmt_exe)

    # 2. Check known locations
    for path in _GMT_SEARCH_PATHS:
        candidate = os.path.join(path, 'gmt')
        if sys.platform == 'win32':
            candidate += '.exe'
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return path

    raise RuntimeError(
        "GMT not found. Please install GMT (https://www.generic-mapping-tools.org) "
        "and ensure it is on your PATH."
    )


# Detect once at import time
try:
    GMT_BIN_DIR = _find_gmt_dir()
except RuntimeError as _e:
    GMT_BIN_DIR = None
    print(f"[gmt_utils] WARNING: {_e}")


# ---------------------------------------------------------------------------
# Shared run() function
# ---------------------------------------------------------------------------

def run(cmd):
    """
    Run a GMT shell command, suppressing GMT warnings.

    Automatically injects the detected GMT binary directory into PATH
    so the command works regardless of how Jupyter/Python was launched.

    Parameters
    ----------
    cmd : str
        Shell command to execute (e.g. 'gmt grdsample ...')

    Raises
    ------
    RuntimeError
        If GMT binary directory could not be detected.
    subprocess.CalledProcessError
        If the command returns a non-zero exit status.
    """
    if GMT_BIN_DIR is None:
        raise RuntimeError(
            "GMT not found. Cannot run command: " + cmd
        )

    env = os.environ.copy()
    env['GMT_VERBOSE'] = 'e'
    env['PATH'] = GMT_BIN_DIR + os.pathsep + env.get('PATH', '')

    # Auto-prepend 'gmt' for bare GMT module commands (e.g. 'grdsample' → 'gmt grdsample')
    _GMT_MODULES = {
        'grdmath', 'grdsample', 'grdtrack', 'grdcut', 'grdflexure',
        'grdblend', 'grdfilter', 'grdvolume', 'grdinfo', 'grdproject',
        'grd2xyz', 'xyz2grd', 'grd2cpt', 'dimfilter', 'gmtset',
        'blockmean', 'surface', 'nearneighbor', 'triangulate',
        'gravfft', 'talwani2d', 'talwani3d', 'grdgravmag3d',
        'grdfft', 'grdlandmask', 'grdclip', 'grdedit', 'grdpaste',
    }
    first_word = cmd.split()[0]
    if first_word in _GMT_MODULES:
        cmd = 'gmt ' + cmd

    subprocess.run(
        cmd + ' 2>/dev/null',
        shell=True,
        check=True,
        env=env
    )
