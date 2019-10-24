
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


import numpy as np
import scipy.io as sio
from raysect.core import Point2D

from cherab.tools.equilibrium import EFITEquilibrium


def import_fiesta(file_path):
    """
    Imports equilibrium data from a FIESTA .mat MATLAB file.

    The MATLAB .mat file structure is described in the FIESTA code documentation.

    :param str file_path: Path to the FIESTA .mat file.
    :rtype: EFITEquilibrium

    .. code-block:: pycon

       >>> from cherab.tools.equilibrium import import_fiesta
       >>> equilibrium = import_fiesta("my_equilibrium.mat")
    """

    mat = sio.loadmat(file_path, mat_dtype=True, squeeze_me=True)

    r = mat['r']
    z = mat['z']
    psi_grid = np.swapaxes(mat['psi'], 0, 1)

    psi_axis = mat['psi_a']
    psi_lcfs = mat['psi_b']
    mag_axis = mat['mag_axis']  # (r0, z0)
    magnetic_axis = Point2D(mag_axis[0], mag_axis[1])

    xpoints = mat['xpoints']  # Number xpoints x 2
    x_points = []
    for point in xpoints:
        x_points.append(Point2D(point[0], point[1]))

    strike_points = []  # not available

    f_profile = mat['f_profile']  # 2 x 100(1st row psi_n)
    q_profile = mat['q_profile']  # 2 x 100(1st row psi_n)

    b_vacuum_radius = mat['b_vacuum_radius']
    b_vacuum_magnitude = mat['b_vacuum_magnitude']

    lcfs_polygon = mat['lcfs_polygon']  # shape 2xM, indexing to remove duplicated point
    if np.all(lcfs_polygon[:, 0] == lcfs_polygon[:, -1]):
        lcfs_polygon = lcfs_polygon[:, 0:-1]

    limiter_polygon = np.array([mat['R_limits'], mat['Z_limits']])  # 2xM
    # limiter_polygon = None  # 2xM

    time = 0.0

    return EFITEquilibrium(r, z, psi_grid, psi_axis, psi_lcfs, magnetic_axis,
                           x_points, strike_points, f_profile, q_profile,
                           b_vacuum_radius, b_vacuum_magnitude, lcfs_polygon,
                           limiter_polygon, time)
