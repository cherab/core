# Copyright 2016-2018 Euratom
# Copyright 2016-2018 United Kingdom Atomic Energy Authority
# Copyright 2016-2018 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.


import re
import warnings
import numpy as np

from raysect.optical import Point2D
from .efit import EFITEquilibrium


def _eqdsk_file_numbers(fp):
    """Generator to get numbers from a text file"""
    toklist = []
    while True:
        line = fp.readline()
        if not line: break
        # Match numbers in the line using regular expression
        pattern = r'[+-]?\d*[\.]?\d+(?:[Ee][+-]?\d+)?|\w*NaN'
        toklist = re.findall(pattern, line)
        for tok in toklist:
            yield tok


def _process_eqdsk_polygon(poly_r, poly_z):

    if poly_r.shape != poly_z.shape:
        raise ValueError("EFIT polygon coordinate arrays are inconsistent in length.")

    n = poly_r.shape[0]
    if n < 2:
        raise ValueError("EFIT polygon coordinates contain less than 2 points.")

    # boundary polygon contains redundant points that must be removed
    unique = (poly_r != poly_r[0]) | (poly_z != poly_z[0])
    unique[0] = True  # first point must be included!
    poly_r = poly_r[unique]
    poly_z = poly_z[unique]

    # generate single array containing coordinates
    polygon = np.empty((2, poly_r.shape[0]))
    polygon[0, :] = poly_r
    polygon[1, :] = poly_z
    return polygon


def import_eqdsk(file_path, drop_nan=False):
    """
    Imports equilibrium data from an EFIT G EQDSK file.

    The file format is described at https://w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf.

    .. WARNING::
       The G EQDSK file format is unstable and unreliable. Use with caution.

    :param str file_path: Path to the EFIT eqdsk file.
    :param bool drop_nan: Drop NaN values in the f and q profiles.
    :rtype: EFITEquilibrium

    .. code-block:: pycon

       >>> from cherab.tools.equilibrium import import_eqdsk
       >>> equilibrium = import_eqdsk("equilibrium.eqdsk")
    """

    fh = open(file_path, 'r')
    # Read the first line, which should contain the mesh sizes
    desc = fh.readline()
    if not desc:
        raise IOError("Cannot read from input file")

    s = desc.split()
    # Split by whitespace
    if len(s) < 3:
        raise IOError("First line must contain at least 3 numbers")

    idum = int(s[-3])
    nx = int(s[-2])  # number of horizontal grid points
    ny = int(s[-1])  # number of vertical grid points

    # Use a generator to read numbers
    token = _eqdsk_file_numbers(fh)

    rdim = float(next(token))                # Horizontal dimension in meter of computational box
    zdim = float(next(token))                # Vertical dimension in meter of computational box
    b_vacuum_radius = float(next(token))     # R in meter of vacuum toroidal magnetic field BCENTR
    rleft = float(next(token))               # Minimum R in meter of rectangular computational box
    zmid = float(next(token))                # Z of center of computational box in meter

    rmaxis = float(next(token))              # R of magnetic axis in meter
    zmaxis = float(next(token))              # Z of magnetic axis in meter
    psi_axis = float(next(token))            # poloidal flux at magnetic axis in Weber /rad
    psi_lcfs = float(next(token))            # poloidal flux at the plasma boundary in Weber /rad
    b_vacuum_magnitude = float(next(token))  # Reference vacuum toroidal field (T) ???

    current = float(next(token))             # Plasma current in Ampere
    psi_axis = float(next(token))            # poloidal flux at magnetic axis in Weber /rad
    next(token)
    rmaxis = float(next(token))              # R of magnetic axis in meter
    next(token)

    zmaxis = float(next(token))              # Z of magnetic axis in meter
    next(token)
    psi_lcfs = float(next(token))            # poloidal flux at the plasma boundary in Weber /rad
    next(token)
    next(token)

    # Read arrays
    def read_array(n, name="Unknown"):
        data = np.zeros([n])
        try:
            for i in np.arange(n):
                data[i] = float(next(token))
        except:
            raise IOError("Failed reading array '"+name+"' of size ", n)
        return data

    def read_2d(nx, ny, name="Unknown"):
        data = np.zeros([nx, ny])
        for i in np.arange(ny):
            data[:, i] = read_array(nx, name+"["+str(i)+"]")
        return data

    f_profile_magnitude = read_array(nx, "fpol")  # Poloidal current function in m-T, F = RB T on flux grid
    pres = read_array(nx, "pres")                 # Plasma pressure in nt / m 2 on uniform flux grid
    ffprim = read_array(nx, "ffprim")             # FF’(ψ) in (mT) 2 / (Weber /rad) on uniform flux grid
    pprime = read_array(nx, "pprime")             # P’(ψ) in (nt /m 2 ) / (Weber /rad) on uniform flux grid
    psi_grid = read_2d(nx, ny, "psi")             # Poloidal flux in Weber / rad on the rectangular grid points
    qpsi = read_array(nx, "qpsi")                 # q values on uniform flux grid from axis to boundary

    # Read boundary and limiters, if present
    nbdry = int(next(token))  # Number of boundary points
    nlim = int(next(token))   # Number of limiter points

    r_z_bdry = np.zeros([2, nbdry])
    for i in range(nbdry):
        r_z_bdry[0, i] = float(next(token))
        r_z_bdry[1, i] = float(next(token))

    r_z_lim = np.zeros([2,nlim])
    for i in range(nlim):
        r_z_lim[0, i] = float(next(token))
        r_z_lim[1, i] = float(next(token))

    r = np.linspace(rleft, rleft + rdim, nx)
    z = np.linspace(zmid - zdim/2, zmid + zdim/2, ny)

    # psi_grid = np.swapaxes(psi_grid, 0, 1)

    magnetic_axis = Point2D(rmaxis, zmaxis)

    # generate uniform flux grid
    f_profile_psin = np.linspace(0, 1, len(f_profile_magnitude))
    f_profile = np.zeros((2, len(f_profile_magnitude)))
    f_profile[0, :] = f_profile_psin
    f_profile[1, :] = f_profile_magnitude

    q_profile_psin = np.linspace(0, 1, len(qpsi))
    q_profile = np.zeros((2, len(qpsi)))
    q_profile[0, :] = q_profile_psin
    q_profile[1, :] = qpsi

    # NaN values mess up the interpolation in 1D profiles
    warning_template = (
        "The {} profile contains NaN values, which will cause interpolation to"
        "fail. Use drop_nan=True to remove these."
    )
    if np.isnan(f_profile).any():
        if not drop_nan:
            warnings.warn(warning_template.format("f"))
        else:
            f_profile = f_profile[:, ~np.isnan(f_profile[1, :])]

    if np.isnan(q_profile).any():
        if not drop_nan:
            warnings.warn(warning_template.format("q"))
        else:
            q_profile = q_profile[:, ~np.isnan(q_profile[1, :])]

    poly_r = r_z_bdry[0, :]
    poly_z = r_z_bdry[1, :]
    try:
        lcfs_polygon = _process_eqdsk_polygon(poly_r, poly_z)
    except ValueError:  # No LCFS in GEQDSK
        lcfs_polygon = None

    lim_r = r_z_lim[0, :]
    lim_z = r_z_lim[1, :]
    try:
        limiter_polygon = _process_eqdsk_polygon(lim_r, lim_z)
    except ValueError:  # No limiter in GEQDSK
        limiter_polygon = None

    # No x point or strike point data in GEQDSK
    x_points = []
    strike_points = []

    time = 0

    return EFITEquilibrium(r, z, psi_grid, psi_axis, psi_lcfs, magnetic_axis,
                           x_points, strike_points, f_profile, q_profile,
                           b_vacuum_radius, b_vacuum_magnitude, lcfs_polygon,
                           limiter_polygon, time)
