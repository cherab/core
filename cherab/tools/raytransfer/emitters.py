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

"""
The following emitters and integrators are used in ray transfer objects.
Note that these emitters support other integrators as well, however high performance
with other integrators is not guaranteed.
"""

import numpy as np
from raysect.optical.material import VolumeIntegrator, InhomogeneousVolumeEmitter


class RayTransferIntegrator(VolumeIntegrator):
    """
    Basic class for ray transfer integrators that calculate distances traveled by the ray
    through the voxels defined on a regular grid.

    :param float step: Integration step (in meters), defaults to `step=0.001`.
    :param int min_samples: The minimum number of samples to use over integration range,
        defaults to `min_samples=2`.

    :ivar float step: Integration step.
    :ivar int min_samples: The minimum number of samples to use over integration range.
    """

    def __init__(self, step=0.001, min_samples=2):
        self.step = step
        self.min_samples = min_samples

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        if value <= 0:
            raise ValueError("Numerical integration step size can not be less than or equal to zero")
        self._step = value

    @property
    def min_samples(self):
        return self._min_samples

    @min_samples.setter
    def min_samples(self, value):
        if value < 2:
            raise ValueError("At least two samples are required to perform the numerical integration.")
        self._min_samples = value


class CylindricalRayTransferIntegrator(RayTransferIntegrator):
    """
    Calculates the distances traveled by the ray through the voxels defined on a regular grid
    in cylindrical coordinate system: :math:`(R, \phi, Z)`. This integrator is used
    with the `CylindricalRayTransferEmitter` material class to calculate ray transfer matrices
    (geometry matrices). The value for each voxel is stored in respective bin of the spectral
    array. It is assumed that the emitter is periodic in :math:`\phi` direction with a period
    equal to `material.period`. The distances traveled by the ray through the voxel is calculated
    approximately and the accuracy depends on the integration step.
    """

    def integrate(self, spectrum, world, ray, primitive, material, start_point, end_point, world_to_primitive, primitive_to_world):
        if not isinstance(material, CylindricalRayTransferEmitter):
            raise ValueError('Only CylindricalRayTransferEmitter material is supported by CylindricalRayTransferIntegrator')
        start = start_point.transform(world_to_primitive)  # start point in local coordinates
        end = end_point.transform(world_to_primitive)  # end point in local coordinates
        direction = start.vector_to(end)  # direction of integration
        length = direction.length  # integration length
        if length < 0.1 * self.step:  # return if ray's path is too short
            return spectrum
        direction = direction.normalise()  # normalized direction
        n = max(self.min_samples, int(length / self.step))  # number of points along ray's trajectory
        t, dt = np.linspace(0, length, n, retstep=True)  # regulary scattered points along ray's trajectory and integration step
        t = t[:-1] + 0.5 * dt  # moving them into the centers of the intervals (and removing the last point)
        x = start.x + direction.x * t  # x coordinates of the points
        y = start.y + direction.y * t  # y coordinates of the points
        z = start.z + direction.z * t  # z coordinates of the points
        iz = (z / material.dz).astype(int)  # Z-indices of grid cells, in which the points are located
        r = np.sqrt(x * x + y * y)  # R coordinates of the points
        ir = ((r - material.rmin) / material.dr).astype(int)  # R-indices of grid cells, in which the points are located
        if material.voxel_map.ndim > 2:  # 3D grid
            phi = (180. / np.pi) * np.arctan2(y, x)  # phi coordinates of the points (in degrees)
            phi[phi < 0] += 360.  # making them all in [0, 360) interval
            phi = phi % material.period  # moving into the [0, period) sector (periodic emitter)
            iphi = (phi / material.dphi).astype(int)  # phi-indices of grid cells, in which the points are located
            i0 = material.voxel_map[ir, iphi, iz]  # light source indices in spectral array
        else:  # 2D grid (RZ-plane)
            i0 = material.voxel_map[ir, iz]  # light source indices in spectral array
        i, counts = np.unique(i0[i0 > -1], return_counts=True)  # exclude voxels for which i0 == -1
        spectrum.samples[i] += counts * dt

        return spectrum


class CartesianRayTransferIntegrator(RayTransferIntegrator):
    """
    Calculates the distances traveled by the ray through the voxels defined on a regular grid
    in Cartesian coordinate system: :math:`(X, Y, Z)`. This integrator is used with
    the `CartesianRayTransferEmitter` material to calculate ray transfer matrices (geometry
    matrices). The value for each voxel is stored in respective bin of the spectral array.
    The distances traveled by the ray through the voxel is calculated approximately and
    the accuracy depends on the integration step.
    """

    def integrate(self, spectrum, world, ray, primitive, material, start_point, end_point, world_to_primitive, primitive_to_world):
        if not isinstance(material, CartesianRayTransferEmitter):
            raise ValueError('Only CartesianRayTransferEmitter material is supported by CartesianRayTransferIntegrator')
        start = start_point.transform(world_to_primitive)  # start point in local coordinates
        end = end_point.transform(world_to_primitive)  # end point in local coordinates
        direction = start.vector_to(end)  # direction of integration
        length = direction.length  # integration length
        if length < 0.1 * self.step:  # return if ray's path is too short
            return spectrum
        direction = direction.normalise()  # normalized direction
        n = max(self.min_samples, int(length / self.step))  # number of points along ray's trajectory
        t, dt = np.linspace(0, length, n, retstep=True)  # regulary scattered points along ray's trajectory and integration step
        t = t[:-1] + 0.5 * dt  # moving them into the centers of the intervals (and removing the last point)
        x = start.x + direction.x * t  # x coordinates of the points
        y = start.y + direction.y * t  # y coordinates of the points
        z = start.z + direction.z * t  # z coordinates of the points
        ix = (x / material.dx).astype(int)  # X-indices of grid cells, in which the points are located
        iy = (y / material.dy).astype(int)  # Y-indices of grid cells, in which the points are located
        iz = (z / material.dz).astype(int)  # Z-indices of grid cells, in which the points are located
        i0 = material.voxel_map[ix, iy, iz]  # light source indices in spectral array
        i, counts = np.unique(i0[i0 > -1], return_counts=True)  # exclude voxels for which i0 == -1
        spectrum.samples[i] += counts * dt

        return spectrum


class RayTransferEmitter(InhomogeneousVolumeEmitter):
    """
    Basic class for ray transfer emitters defined on a regular grid. Ray transfer emitters
    are used to calculate ray transfer matrices (geometry matrices) for a single value
    of wavelength.

    :param tuple grid_shape: The shape of regular grid (the number of grid cells
        along each direction).
    :param tuple grid_steps: The sizes of grid cells along each direction.
    :param np.ndarray voxel_map: An array with shape `grid_shape` containing the indices of
        the light sources. This array maps the cells of regular grid to the respective voxels
        (light sources). The cells with identical indices in `voxel_map` array form a single
        voxel (light source). If `voxel_map[i1, i2, ...] == -1`, the cell with indices
        `(i1, i2, ...)` will not be mapped to any light source. This parameters allows to
        apply a custom geometry (pixelated though) to the light sources.
        Default value: `voxel_map=None`.
    :param np.ndarray mask: A boolean mask array with shape `grid_shape`.
        Allows to include (`mask[...] == True`) or exclude (`mask[...] == False`) the cells
        from the calculation. The ray tranfer matrix will be calculated only for those cells
        for which mask is True. This parameter is ignored if `voxel_map` is provided,
        defaults to `mask=None` (all cells are included).
    :param raysect.optical.material.VolumeIntegrator integrator: Volume integrator,
        defaults to `integrator=NumericalVolumeIntegrator()`

    :ivar tuple grid_shape: The shape of regular grid.
    :ivar tuple grid_steps: The sizes of grid cells along each direction.
    :ivar np.ndarray voxel_map: An array containing the indices of the light sources.
    :ivar np.ndarray ~.mask: A boolean mask array showing active (True) and inactive
        (False) gird cells.
    :ivar int bins: Number of light sources (the size of spectral array must be equal to this value).
    """

    def __init__(self, grid_shape, grid_steps, voxel_map=None, mask=None, integrator=None):
        if len(grid_shape) != len(grid_steps):
            raise ValueError('Grid dimension %d is not equal to the number of grid steps given: %d' % (len(grid_shape), len(grid_steps)))
        for i in grid_shape:
            if type(i) != int:
                raise ValueError('grid_shape must be a tuple of integers')
            if i < 1:
                raise ValueError('Number of grid cells must be > 0')
        for step in grid_steps:
            if step <= 0:
                raise ValueError('Grid steps must be > 0')
        # grid_shape and grid_steps are defined on initialisation and must not be changed after that
        self._grid_shape = grid_shape
        self._grid_steps = grid_steps
        if voxel_map is None:
            self.mask = mask
        else:
            self.voxel_map = voxel_map
        super().__init__(integrator)

    @property
    def grid_shape(self):
        return self._grid_shape

    @property
    def grid_steps(self):
        return self._grid_steps

    def _map_from_mask(self, mask):
        if mask is not None:
            if mask.shape != self._grid_shape:
                raise ValueError('Mask array must be of shape: %s' % (' '.join(['%d' % i for i in self._grid_shape])))
            if mask.dtype != np.bool:
                raise ValueError('Mask array must be of numpy.bool type')
        else:
            mask = np.ones(self._grid_shape, dtype=np.bool)
        voxel_map = -1 * np.ones(mask.shape, dtype=int)
        voxel_map[mask] = np.arange(mask.sum(), dtype=int)

        return voxel_map

    @property
    def bins(self):
        return self._bins

    @property
    def voxel_map(self):
        return self._voxel_map

    @voxel_map.setter
    def voxel_map(self, value):
        if value.shape != self._grid_shape:
            raise ValueError('Voxel_map array must be of shape: %s' % (' '.join(['%d' % i for i in self._grid_shape])))
        if value.dtype != np.int:
            raise ValueError('Voxel_map array must be of numpy.int type')
        self._voxel_map = value
        self._bins = self._voxel_map.max() + 1

    @property
    def mask(self):
        return self._voxel_map > -1

    @mask.setter
    def mask(self, value):
        self._voxel_map = self._map_from_mask(value)
        self._bins = self._voxel_map.max() + 1


class CylindricalRayTransferEmitter(RayTransferEmitter):
    """
    A unit emitter defined on a regular 2D (RZ plane) or 3D :math:`(R, \phi, Z)` grid, which
    can be used to calculate ray transfer matrices (geometry matrices) for a single value
    of wavelength.
    In case of 3D grid this emitter is periodic in :math:`\phi` direction.
    Note that for performance reason there are no boundary checks in `emission_function()`,
    or in `CylindricalRayTranferIntegrator`, so this emitter must be placed between a couple
    of coaxial cylinders that act like a bounding box.

    :param tuple grid_shape: The shape of regular :math:`(R, \phi, Z)` (3D case)
        or :math:`(R, Z)` (axisymmetric case) grid.
    :param tuple grid_steps: The sizes of grid cells in `R`, :math:`\phi` and `Z`
        (3D case) or `R` and `Z` (axisymmetric case) directions. The size in :math:`\phi`
        must be provided in degrees (sizes in `R` and `Z` are provided in meters).
    :param np.ndarray voxel_map: An array with shape `grid_shape` containing the indices of
        the light sources. This array maps the cells in :math:`(R, \phi, Z)` space to
        the respective voxels (light sources). The cells with identical indices in `voxel_map`
        array form a single voxel (light source). If `voxel_map[ir, iphi, iz] == -1`, the
        cell with indices `(ir, iphi, iz)` will not be mapped to any light source.
        This parameters allows to apply a custom geometry (pixelated though) to the light
        sources. Default value: `voxel_map=None`.
    :param np.ndarray mask: A boolean mask array with shape `grid_shape`.
        Allows to include (mask[ir, iphi, iz] == True) or exclude (mask[ir, iphi, iz] == False)
        the cells from the calculation. The ray tranfer matrix will be calculated only for
        those cells for which mask is True. This parameter is ignored if `voxel_map` is provided,
        defaults to `mask=None` (all cells are included).
    :param raysect.optical.material.VolumeIntegrator integrator: Volume integrator, defaults to
        `integrator=CylindricalRayTransferIntegrator(step=0.1*min(grid_shape[0], grid_shape[-1]))`.
    :param float rmin: Lower bound of grid in `R` direction (in meters), defaults to `rmin=0`.
    :param float period: A period in :math:`\phi` direction (in degree).
        Used only in 3D case, defaults to `period=360`.

    :ivar float period: The period in :math:`\phi` direction in 3D case or `None` in
        axisymmetric case.
    :ivar float rmin: Lower bound of grid in `R` direction.
    :ivar float dr: The size of grid cell in `R` direction (equals to `grid_shape[0]`).
    :ivar float dz: The size of grid cell in `Z` direction (equals to `grid_shape[-1]`).
    :ivar float dphi: The size of grid cell in :math:`\phi` direction
        (equals to None in axisymmetric case or to `grid_shape[1]` in 3D case).

    .. code-block:: pycon

        >>> from raysect.optical import World, translate
        >>> from raysect.primitive import Cylinder, Subtract
        >>> from cherab.tools.raytransfer import CylindricalRayTransferEmitter
        >>> world = World()
        >>> grid_shape = (10, 10)
        >>> grid_steps = (0.5, 0.5)
        >>> rmin = 2.5
        >>> material = CylindricalRayTransferEmitter(grid_shape, grid_steps, rmin=rmin)
        >>> eps = 1.e-6  # ray must never leave the grid when passing through the volume
        >>> radius_outer = grid_shape[0] * grid_steps[0] - eps
        >>> height = grid_shape[1] * grid_steps[1] - eps
        >>> radius_inner = rmin + eps
        >>> bounding_box = Subtract(Cylinder(radius_outer, height), Cylinder(radius_inner, height),
                                    material=material, parent=world)  # bounding primitive
        >>> bounding_box.transform = translate(0, 0, -2.5)
        ...
        >>> camera.spectral_bins = material.bins
        >>> # ray transfer matrix will be calculated for 600.5 nm
        >>> camera.min_wavelength = 600.
        >>> camera.max_wavelength = 601.
    """

    def __init__(self, grid_shape, grid_steps, voxel_map=None, mask=None, integrator=None, rmin=0, period=360.):
        if not 1 < len(grid_shape) < 4:
            raise ValueError('grid_shape must contain 2 or 3 elements')
        if not 1 < len(grid_steps) < 4:
            raise ValueError('grid_steps must contain 2 or 3 elements')
        def_integration_step = 0.1 * min(grid_steps[0], grid_steps[-1])
        integrator = integrator or CylindricalRayTransferIntegrator(def_integration_step)
        super().__init__(grid_shape, grid_steps, voxel_map=voxel_map, mask=mask, integrator=integrator)
        self.period = period
        self.rmin = rmin
        self._dr = self.grid_steps[0]
        self._dphi = self.grid_steps[1] if len(self.grid_steps) == 3 else None
        self._dz = self.grid_steps[-1]

    @property
    def rmin(self):
        return self._rmin

    @rmin.setter
    def rmin(self, value):
        if value < 0:
            raise ValueError('rmin must be >= 0')
        self._rmin = value

    @property
    def period(self):
        return self._period

    @period.setter
    def period(self, value):
        if len(self._grid_shape) < 3:
            self._period = None
            return
        if not 0 < value <= 360.:
            raise ValueError('period must be > 0 and <= 360')
        self._period = value

    @property
    def dr(self):
        return self._dr

    @property
    def dphi(self):
        return self._dphi

    @property
    def dz(self):
        return self._dz

    def emission_function(self, point, direction, spectrum, world, ray, primitive, world_to_primitive, primitive_to_world):
        iz = int(point.z / self._dz)  # Z-index of grid cell, in which the point is located
        r = np.sqrt(point.x * point.x + point.y * point.y)  # R coordinates of the points
        ir = int((r - self._rmin) / self._dr)  # R-index of grid cell, in which the points is located
        if self.voxel_map.ndim > 2:  # 3D grid
            phi = (180. / np.pi) * np.arctan2(point.y, point.x)  # phi coordinate of the point (in degrees)
            if phi < 0:
                phi += 360.  # moving to [0, 360) interval
            phi = phi % self._period  # moving into the [0, period) sector (periodic emitter)
            iphi = int(phi / self._dphi)  # phi-index of grid cell, in which the point is located
            i = self.voxel_map[ir, iphi, iz]  # index of the light source in spectral array
        else:  # 2D grid (RZ-plane)
            i = self.voxel_map[ir, iz]  # index of the light source in spectral array
        if i < 0:  # grid cell is not mapped to any light source
            return spectrum
        spectrum.samples[i] += 1.  # unit emissivity
        return spectrum


class CartesianRayTransferEmitter(RayTransferEmitter):
    """
    A unit emitter defined on a regular 3D :math:`(X, Y, Z)` grid, which can be used
    to calculate ray transfer matrices (geometry matrices).
    Note that for performance reason there are no boundary checks in `emission_function()`,
    or in `CartesianRayTranferIntegrator`, so this emitter must be placed inside a bounding box.

    :param tuple grid_shape: The shape of regular :math:`(X, Y, Z)` grid.
        The number of points in `X`, `Y` and `Z` directions.
    :param tuple grid_steps: The sizes of grid cells in `X`, `Y` and `Z`
        directions (in meters).
    :param np.ndarray voxel_map: An array with shape `grid_shape` containing the indices
        of the light sources. This array maps the cells in :math:`(X, Y, Z)` space to the
        respective voxels (light sources). The cells with identical indices in `voxel_map`
        array form a single voxel (light source). If `voxel_map[ix, iy, iz] == -1`,
        the cell with indices `(ix, iy, iz)` will not be mapped to any light source.
        This parameters allows to apply a custom geometry (pixelated though) to the
        light sources. Default value: `voxel_map=None`.
    :param np.ndarray mask: A boolean mask array with shape `grid_shape`.
        Allows to include (`mask[ix, iy, iz] == True`) or exclude (`mask[ix, iy, iz] == False`)
        the cells from the calculation. The ray tranfer matrix will be calculated only for
        those cells for which mask is True. This parameter is ignored if `voxel_map` is
        provided, defaults to `mask=None` (all cells are included).
    :param raysect.optical.material.VolumeIntegrator integrator: Volume integrator,
        defaults to `integrator=CartesianRayTransferIntegrator(step=0.1 * min(grid_steps))`

    :ivar float dx: The size of grid cell in `X` direction (equals to `grid_shape[0]`).
    :ivar float dy: The size of grid cell in `Y` direction (equals to `grid_shape[1]`).
    :ivar float dz: The size of grid cell in `Z` direction (equals to `grid_shape[2]`).

     .. code-block:: pycon

        >>> from raysect.optical import World, translate, Point3D
        >>> from raysect.primitive import Box
        >>> from cherab.tools.raytransfer import CartesianRayTransferEmitter
        >>> world = World()
        >>> grid_shape = (10, 10, 10)
        >>> grid_steps = (0.5, 0.5, 0.5)
        >>> material = CartesianRayTransferEmitter(grid_shape, grid_steps)
        >>> eps = 1.e-6  # ray must never leave the grid when passing through the volume
        >>> upper = Point3D(grid_shape[0] * grid_steps[0] - eps,
                            grid_shape[1] * grid_steps[1] - eps,
                            grid_shape[2] * grid_steps[2] - eps)
        >>> bounding_box = Box(lower=Point3D(0, 0, 0), upper=upper, material=material,
                               parent=world)
        >>> bounding_box.transform = translate(-2.5, -2.5, -2.5)
        ...
        >>> camera.spectral_bins = material.bins
        >>> # ray transfer matrix will be calculated for 600.5 nm
        >>> camera.min_wavelength = 600.
        >>> camera.max_wavelength = 601.
    """

    def __init__(self, grid_shape, grid_steps, voxel_map=None, mask=None, integrator=None):
        if len(grid_shape) != 3:
            raise ValueError('grid_shape must contain 3 elements')
        if len(grid_steps) != 3:
            raise ValueError('grid_steps must contain 3 elements')
        def_integration_step = 0.1 * min(grid_steps)
        integrator = integrator or CartesianRayTransferIntegrator(def_integration_step)
        super().__init__(grid_shape, grid_steps, voxel_map=voxel_map, mask=mask, integrator=integrator)
        self._dx = self.grid_steps[0]
        self._dy = self.grid_steps[1]
        self._dz = self.grid_steps[2]

    @property
    def dx(self):
        return self._dx

    @property
    def dy(self):
        return self._dy

    @property
    def dz(self):
        return self._dz

    def emission_function(self, point, direction, spectrum, world, ray, primitive, world_to_primitive, primitive_to_world):
        ix = int(point.x / self._dx)  # X-index of grid cell, in which the point is located
        iy = int(point.y / self._dy)  # Y-index of grid cell, in which the point is located
        iz = int(point.z / self._dz)  # Z-index of grid cell, in which the point is located
        i = self.voxel_map[ix, iy, iz]  # index of the light source in spectral array
        if i < 0:  # grid cell is not mapped to any light source
            return spectrum
        spectrum.samples[i] += 1.  # unit emissivity
        return spectrum
