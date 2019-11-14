# -*- coding: utf-8 -*-
#
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
#
# The following code is created by Vladislav Neverov (NRC "Kurchatov Institute") for CHERAB Spectroscopy Modelling Framework

import numpy as np
from raysect.optical.material import VolumeIntegrator, InhomogeneousVolumeEmitter


class RayTransferRPhiZIntegrator(VolumeIntegrator):
    """
    Calculates the distances traveled by the ray through the voxels defined on a regular grid
    in cylindrical coordinate system: :math:`(R, \phi, Z)`. This integrator can be used along with
    the `RayTransferRPhiZEmitter` material class to calculate ray transfer matrices (geometry matrices).
    The value for each voxel is stored in respective bin of the spectral array.
    Custom geometry may be applied to the voxels via the material.map array,
    which can map multiple cells in :math:`(R, \phi, Z)` space to a single spectral bin (and thus to a single voxel).
    It is assumed that the emitter is periodic in :math: `\phi` direction with a period equal to `material.period`.
    The distances traveled by the ray through the voxel is calculated approximately and the accuracy depends on the integration step.

    :param float step: Integration step (in meters), defaults to 0.001.
    """

    def __init__(self, step=0.001):
        self.step = step
        self._rad2deg = 180. / np.pi

    def integrate(self, spectrum, world, ray, primitive, material, start_point, end_point, world_to_primitive, primitive_to_world):
        start = start_point.transform(world_to_primitive)  # start point in local coordinates
        end = end_point.transform(world_to_primitive)  # end point in local coordinates
        direction = start.vector_to(end)  # direction of integration
        length = direction.length  # integration length
        if length < 0.1 * self.step:  # return if ray's path is too short
            return spectrum
        direction = direction.normalise()  # normalized direction
        n = max(2, int(length / self.step))  # number of points along ray's trajectory
        t, dt = np.linspace(0, length, n, retstep=True)  # regulary scattered points along ray's trajectory and integration step
        t = t[:-1] + 0.5 * dt  # moving them into the centers of the intervals (and removing the last point)
        x = start.x + direction.x * t  # x coordinates of the points
        y = start.y + direction.y * t  # y coordinates of the points
        z = start.z + direction.z * t  # z coordinates of the points
        iz = (z / material._dz).astype(int)  # Z-indices of grid cells, in which the points are located
        r = np.sqrt(x * x + y * y)  # R coordinates of the points
        ir = ((r - material._rmin) / material._dr).astype(int)  # R-indices of grid cell, in which the points are located
        if material._map.ndim > 2:  # 3D grid
            phi = self._rad2deg * np.arctan2(y, x)  # phi coordinates of the points (in degrees)
            phi[phi < 0] += 360.  # making them all in [0, 360) interval
            phi = phi % material._period  # moving into the [0, period) sector (periodic emitter)
            iphi = (phi / material._dphi).astype(int)  # phi-indices of grid cell, in which the points are located
            i0 = material._map[ir, iphi, iz]  # light source indeces in spectral array
        else:  # 2D grid (RZ-plane)
            i0 = material._map[ir, iz]  # light source indeces in spectral array
        i, counts = np.unique(i0[i0 > -1], return_counts=True)  # exclude voxels for which i0 == -1
        spectrum.samples[i] += counts * dt

        return spectrum


class RayTransferXYZIntegrator(VolumeIntegrator):
    """
    Calculates the distances traveled by the ray through the voxels defined on a regular grid
    in Cartesian coordinate system: :math:`(X, Y, Z)`. This integrator can be used to calculate ray transfer matrices
    (geometry matrices). The value for each voxel is stored in respective bin of the spectral array.
    Custom geometry may be applied to the voxels via the material.map array (see `RayTransferXYZEmitter` class),
    which can map multiple cells in :math:`(X, Y, Z)` space to a single spectral bin.
    The distances traveled by the ray through the voxel is calculated approximately and the accuracy depends on the integration step.

    :param float step: Integration step (in meters), defaults to 0.001.
    """

    def __init__(self, step=0.001):
        self.step = step

    def integrate(self, spectrum, world, ray, primitive, material, start_point, end_point, world_to_primitive, primitive_to_world):
        start = start_point.transform(world_to_primitive)  # start point in local coordinates
        end = end_point.transform(world_to_primitive)  # end point in local coordinates
        direction = start.vector_to(end)  # direction of integration
        length = direction.length  # integration length
        if length < 0.1 * self.step:  # return if ray's path is too short
            return spectrum
        direction = direction.normalise()  # normalized direction
        n = max(2, int(length / self.step))  # number of points along ray's trajectory
        t, dt = np.linspace(0, length, n, retstep=True)  # regulary scattered points along ray's trajectory and integration step
        t = t[:-1] + 0.5 * dt  # moving them into the centers of the intervals (and removing the last point)
        x = start.x + direction.x * t  # x coordinates of the points
        y = start.y + direction.y * t  # y coordinates of the points
        z = start.z + direction.z * t  # z coordinates of the points
        ix = (x / material._dx).astype(int)  # X-indices of grid cells, in which the points are located
        iy = (y / material._dy).astype(int)  # Y-indices of grid cells, in which the points are located
        iz = (z / material._dz).astype(int)  # Z-indices of grid cells, in which the points are located
        i0 = material._map[ix, iy, iz]  # light source indeces in spectral array
        i, counts = np.unique(i0[i0 > -1], return_counts=True)  # exclude voxels for which i0 == -1
        spectrum.samples[i] += counts * dt

        return spectrum


class RayTransferRPhiZEmitter(InhomogeneousVolumeEmitter):
    """
    A dummy emitter defined on a regular 2D (RZ plane) or 3D :math:`(R, \phi, Z)` grid, which can be used
    (along with `RayTransferRPhiZIntegrator`) to calculate ray transfer matrices (geometry matrices).
    In case of 3D grid this emitter is periodic in :math: `\phi` direction.

    :param int nr: Number of grid points in `R` direction.
    :param int nz: Number of grid points in `Z` direction.
    :param float dr: Grid step in `R` direction (in meters).
    :param float dz: Grid step in `Z` direction (in meters).
    :param float rmin: Lower bound of grid in `R` direction (in meters), defaults to `rmin=0`.
    :param int nphi: Number of grid points in :math: `\phi` direction, defaults to nphi=0 (2D grid).
    :param float dphi: Grid step in :math: `\phi` direction (in degree). Used only if `nphi > 0`, defaults to `dphi=0.1`.
    :param float period: A period in :math: `\phi` direction (in degree). Used only if `nphi > 0`, defaults to `period=360`.
    :param np.ndarray voxel_map: An array with shape :math: `(nr, nz)` (2D case) or :math: `(nr, nphi, nz)` (3D case)
        containing the indeces of the light sources. This array maps the cells in :math:`(R, \phi, Z)` space to
        the respective voxels (light sources). The cells with identical indeces in voxel_map array form a single voxel (light source).
        If `voxel_map[ir, iphi, iz] == -1`, the cell with indeces `(ir, iphi, iz)` will not be mapped to any light source.
        This parameters allows to apply a custom geometry (pixelated though) to the light sources. Default value: `voxel_map=None`.
    :param np.ndarray mask: A boolean mask array with shape :math: `(nr, nz)` (2D case) or :math: `(nr, nphi, nz)` (3D case).
        Allows to include (mask is True) or exclude (mask is False) the cells from the calculation.
        The ray tranfer matrix will be calculated only for those cells for which mask is True.
        This parameter is ignored if `voxel_map` is provided, defaults to `mask=None` (all cells are included).
    :param raysect.optical.material.VolumeIntegrator integrator: Volume integrator,
        defaults to `integrator=RayTransferRPhiZIntegrator(step=0.1 * min(dr, dz))`
    """

    def __init__(self, nr, nz, dr, dz, rmin=0, nphi=0, dphi=1., period=360., voxel_map=None, mask=None, integrator=None):
        integrator = integrator or RayTransferRPhiZIntegrator(0.1 * min(dr, dz))
        super().__init__(integrator)
        if nphi and not dphi * period:
            raise ValueError('period and dphi must be non-zero for 3D grid')
        self._rmin = rmin
        self._period = period
        self._dr = dr
        self._dphi = dphi
        self._dz = dz
        self._nr = nr
        self._nphi = nphi
        self._nz = nz
        self._map = None
        self._bins = None
        if voxel_map is None:
            self.mask = mask
        else:
            self.voxel_map = voxel_map

    def _map_from_mask(self, mask):
        if mask is not None:
            if self._nphi and mask.shape != (self._nr, self._nphi, self._nz):
                raise ValueError('Mask array must be of shape: %d, %d, %d' % (self._nr, self._nphi, self._nz))
            if (not self._nphi) and mask.shape != (self._nr, self._nz):
                raise ValueError('Mask array must be of shape: %d, %d' % (self._nr, self._nz))
            if mask.dtype != np.bool:
                raise ValueError('Mask array must be of numpy.bool type')
        else:
            mask = np.ones((self._nr, self._nphi, self._nz), dtype=np.bool) if self._nphi else np.ones((self._nr, self._nz), dtype=np.bool)
        voxel_map = -1 * np.ones(mask.shape, dtype=int)
        voxel_map[mask] = np.arange(mask.sum(), dtype=int)

        return voxel_map

    @property
    def bins(self):
        return self._bins

    @property
    def voxel_map(self):
        return self._map

    @voxel_map.setter
    def voxel_map(self, value):
        if self._nphi and value.shape != (self._nr, self._nphi, self._nz):
            raise ValueError('Voxel_map array must be of shape: %d, %d, %d' % (self._nr, self._nphi, self._nz))
        if (not self._nphi) and value.shape != (self._nr, self._nz):
            raise ValueError('Voxel_map array must be of shape: %d, %d' % (self._nr, self._nz))
        if value.dtype != np.int:
            raise ValueError('Voxel_map array must be of numpy.int type')
        self._map = value
        self._bins = self._map.max() + 1

    @property
    def mask(self):
        return self._map > -1

    @mask.setter
    def mask(self, value):
        self._map = self._map_from_mask(value)
        self._bins = self._map.max() + 1


class RayTransferXYZEmitter(InhomogeneousVolumeEmitter):
    """
    A dummy emitter defined on a regular 3D :math:`(X, Y, Z)` grid, which can be used
    (along with `RayTransferXYZIntegrator`) to calculate ray transfer matrices (geometry matrices).

    :param int nx: Number of grid points in `X` direction.
    :param int ny: Number of grid points in `Y` direction.
    :param int nz: Number of grid points in `Z` direction.
    :param float dx: Grid step in `X` direction (in meters).
    :param float dy: Grid step in `Y` direction (in meters).
    :param float dz: Grid step in `Z` direction (in meters).
    :param np.ndarray voxel_map: An array with shape :math: `(nx, ny, nz)` containing the indeces of the light sources.
        This array maps the cells in :math:`(X, Y, Z)` space to the respective voxels (light sources).
        The cells with identical indeces in voxel_map array form a single voxel (light source).
        If `voxel_map[ix, iy, iz] == -1`, the cell with indeces `(ix, iy, iz)` will not be mapped to any light source.
        This parameters allows to apply a custom geometry (pixelated though) to the light sources. Default value: `voxel_map=None`.
    :param np.ndarray mask: A boolean mask array with shape :math: `(nx, ny, nz)`.
        Allows to include (mask is True) or exclude (mask is False) the cells from the calculation.
        The ray tranfer matrix will be calculated only for those cells for which mask is True.
        This parameter is ignored if `voxel_map` is provided, defaults to `mask=None` (all cells are included).
    :param raysect.optical.material.VolumeIntegrator integrator: Volume integrator,
        defaults to `integrator=RayTransferXYZIntegrator(step=0.1 * min(dx, dy, dz))`
    """

    def __init__(self, nx, ny, nz, dx, dy, dz, voxel_map=None, mask=None, integrator=None):
        integrator = integrator or RayTransferXYZIntegrator(0.1 * min(dx, dy, dz))
        super().__init__(integrator)
        self._nx = nx
        self._ny = ny
        self._nz = nz
        self._dx = dx
        self._dy = dy
        self._dz = dz
        self._map = None
        self._bins = None
        if voxel_map is None:
            self.mask = mask
        else:
            self.voxel_map = voxel_map

    def _map_from_mask(self, mask):
        if mask is not None:
            if mask.shape != (self._nx, self._ny, self._nz):
                raise ValueError('Mask array must be of shape: %d, %d, %d' % (self._nx, self._ny, self._nz))
            if mask.dtype != np.bool:
                raise ValueError('Mask array must be of numpy.bool type')
        else:
            mask = np.ones((self._nx, self._ny, self._nz), dtype=np.bool)
        voxel_map = -1 * np.ones(mask.shape, dtype=int)
        voxel_map[mask] = np.arange(mask.sum(), dtype=int)

        return voxel_map

    @property
    def bins(self):
        return self._bins

    @property
    def voxel_map(self):
        return self._map

    @voxel_map.setter
    def voxel_map(self, value):
        if value.shape != (self._nx, self._ny, self._nz):
            raise ValueError('Voxel_map array must be of shape: %d, %d, %d' % (self._nx, self._ny, self._nz))
        if value.dtype != np.int:
            raise ValueError('Voxel_map array must be of numpy.int type')
        self._map = value
        self._bins = self._map.max() + 1

    @property
    def mask(self):
        return self._map > -1

    @mask.setter
    def mask(self, value):
        self._map = self._map_from_mask(value)
        self._bins = self._map.max() + 1
