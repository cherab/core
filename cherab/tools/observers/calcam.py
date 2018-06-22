
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


def load_calcam_calibration(cal_file_path, reduction_factor=1):
    """
    Extract camera calibration information from a calcam netCDF file.

    :param cal_file_path: path to calcam calibration netCDF file.
    :param reduction_factor: number of pixels to skip when reading the netCDF file.
    :return: tuple of (pixels_shape, pixel_origins, pixel_directions).
    """

    camera_config = netcdf_file(cal_file_path)

    try:
        ray_start_coords = camera_config.variables['RayStartCoords']
        ray_end_coords = camera_config.variables['RayEndCoords']

        pixels_shape = ray_start_coords.shape[0:2][::-1]
        pixels_shape = (int(np.ceil(pixels_shape[0]/reduction_factor)), int(np.ceil(pixels_shape[1]/reduction_factor)))
        nx, ny = pixels_shape
        pixel_origins = np.empty(shape=pixels_shape, dtype=np.dtype(object))
        pixel_directions = np.empty(shape=pixels_shape, dtype=np.dtype(object))

        for ix in range(0, nx, reduction_factor):
            for iy in range(0, ny, reduction_factor):
                xi, yi, zi = ray_start_coords[iy, ix]
                pixel_origins[ix, iy] = Point3D(xi, yi, zi)

        for ix in range(0, nx, reduction_factor):
            for iy in range(0, ny, reduction_factor):
                xi, yi, zi = ray_start_coords[iy, ix]
                xj, yj, zj = ray_end_coords[iy, ix]
                pixel_directions[ix, iy] = Vector3D(xj-xi, yj-yi, zj-zi).normalise()

    # catch older calcam format, note this will need to be removed at some point
    except KeyError:
        ray_origin = camera_config.variables['ray_origin']
        ray_origin = Point3D(*ray_origin)

        ray_direction = camera_config.variables['ray_direction']

        ny, nx, _ = ray_direction.shape

        pixel_origins = []
        pixel_directions = []
        for ix in range(0, nx, reduction_factor):

            pixel_origins_row = []
            pixel_directions_row = []

            for iy in range(0, ny, reduction_factor):
                pixel_origins_row.append(ray_origin)
                pixel_directions_row.append(Vector3D(*ray_direction[iy, ix, :]).normalise())

            pixel_origins.append(pixel_origins_row)
            pixel_directions.append(pixel_directions_row)

        pixel_origins = np.array(pixel_origins, dtype=object)
        pixel_directions = np.array(pixel_directions, dtype=object)
        pixels_shape = pixel_origins.shape

    return pixels_shape, pixel_origins, pixel_directions
