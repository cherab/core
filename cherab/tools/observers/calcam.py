
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
from scipy.io.netcdf import netcdf_file
from raysect.core import Point3D, Vector3D


def load_calcam_calibration(cal_file_path):
    """
    Extract camera calibration information from a calcam netCDF file.

    :param cal_file_path: path to calcam calibration netCDF file.
    :return: tuple of (pixels_shape, pixel_origins, pixel_directions).
    """

    camera_config = netcdf_file(cal_file_path)

    ray_start_coords = camera_config.variables['RayStartCoords']
    ray_end_coords = camera_config.variables['RayEndCoords']

    pixels_shape = ray_start_coords.shape[0:2][::-1]
    pixel_origins = np.empty(shape=pixels_shape, dtype=np.dtype(object))
    pixel_directions = np.empty(shape=pixels_shape, dtype=np.dtype(object))

    for (x, y), _ in np.ndenumerate(pixel_origins):
        xi, yi, zi = ray_start_coords[y, x]
        pixel_origins[x, y] = Point3D(xi, yi, zi)

    for (x, y), _ in np.ndenumerate(pixel_directions):
        xi, yi, zi = ray_start_coords[y, x]
        xj, yj, zj = ray_end_coords[y, x]
        pixel_directions[x, y] = Vector3D(xj-xi, yj-yi, zj-zi).normalise()

    return pixels_shape, pixel_origins, pixel_directions

