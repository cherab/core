# cython: language_level=3

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

"""
The following emitters and integrators are used in ray transfer objects.
Note that these emitters support other integrators as well, however high performance
with other integrators is not guaranteed.
"""

import numpy as np
from raysect.optical cimport World, Primitive, Ray, Spectrum, Point3D, Vector3D, AffineMatrix3D
from raysect.optical.material cimport VolumeIntegrator, InhomogeneousVolumeEmitter
from libc.math cimport sqrt, atan2, M_PI as pi
cimport numpy as np
cimport cython


cdef class RayTransferIntegrator(VolumeIntegrator):
    """
    Basic class for ray transfer integrators that calculate distances traveled by the ray
    through the voxels defined on a regular grid.

    :param float step: Integration step (in meters), defaults to `step=0.001`.
    :param int min_samples: The minimum number of samples to use over integration range,
        defaults to `min_samples=2`.

    :ivar float step: Integration step.
    :ivar int min_samples: The minimum number of samples to use over integration range.
    """

    def __init__(self, double step=0.001, int min_samples=2):
        self.step = step
        self.min_samples = min_samples

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        if value <= 0:
            raise ValueError("Numerical integration step size can not be less than or equal to zero.")
        self._step = value

    @property
    def min_samples(self):
        return self._min_samples

    @min_samples.setter
    def min_samples(self, value):
        if value < 2:
            raise ValueError("At least two samples are required to perform the numerical integration.")
        self._min_samples = value


cdef class CylindricalRayTransferIntegrator(RayTransferIntegrator):
    """
    Calculates the distances traveled by the ray through the voxels defined on a regular grid
    in cylindrical coordinate system: :math:`(R, \phi, Z)`. This integrator is used
    with the `CylindricalRayTransferEmitter` material class to calculate ray transfer matrices
    (geometry matrices). The value for each voxel is stored in respective bin of the spectral
    array. It is assumed that the emitter is periodic in :math:`\phi` direction with a period
    equal to `material.period`. The distances traveled by the ray through the voxel is calculated
    approximately and the accuracy depends on the integration step.
    """

    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cpdef Spectrum integrate(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                             InhomogeneousVolumeEmitter material, Point3D start_point, Point3D end_point,
                             AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            Point3D start, end
            Vector3D direction
            int isource, isource_current, it, ir, iphi, iz, ir_current, iphi_current, iz_current, n, nphi
            double length, t, dt, x, y, z, r, phi, dr, dz, dphi, rmin, period, res
            int[:, :, ::1] voxel_map_mv

        if not isinstance(material, CylindricalRayTransferEmitter):
            raise TypeError('Only CylindricalRayTransferEmitter material is supported by CylindricalRayTransferIntegrator.')
        start = start_point.transform(world_to_primitive)  # start point in local coordinates
        end = end_point.transform(world_to_primitive)  # end point in local coordinates
        direction = start.vector_to(end)  # direction of integration
        length = direction.get_length()  # integration length
        if length < 0.1 * self._step:  # return if ray's path is too short
            return spectrum
        direction = direction.normalise()  # normalized direction
        n = max(self._min_samples, <int>(length / self._step))  # number of points along ray's trajectory
        dt = length / n  # integration step
        # cython performs checks on attributes of external class, so it's better to do the checks before the loop
        voxel_map_mv = material.voxel_map_mv
        nphi = material.grid_shape[1]
        dz = material.dz
        dr = material.dr
        dphi = material.dphi
        period = material.period
        rmin = material.rmin
        ir_current = -1
        iphi_current = -1
        iz_current = -1
        isource_current = -1
        res = 0
        for it in range(n):
            t = (it + 0.5) * dt
            x = start.x + direction.x * t  # x coordinates of the points
            y = start.y + direction.y * t  # y coordinates of the points
            z = start.z + direction.z * t  # z coordinates of the points
            iz = <int>(z / dz)  # Z-indices of grid cells, in which the points are located
            r = sqrt(x * x + y * y)  # R coordinates of the points
            ir = <int>((r - rmin) / dr)  # R-indices of grid cells, in which the points are located
            if nphi == 1:  # axisymmetric case
                iphi = 0
            else:
                phi = (180. / pi) * atan2(y, x)  # phi coordinates of the points (in degrees)
                phi = (phi + 360.) % period  # moving into the [0, period) sector (periodic emitter)
                iphi = <int>(phi / dphi)  # phi-indices of grid cells, in which the points are located
            if ir != ir_current or iphi != iphi_current or iz != iz_current:  # we moved to the next cell
                ir_current = ir
                iphi_current = iphi
                iz_current = iz
                isource = voxel_map_mv[ir, iphi, iz]  # light source indices in spectral array
                if isource != isource_current:  # we moved to the next source
                    if isource_current > -1:
                        spectrum.samples_mv[isource_current] += res  # writing results for the current source
                    isource_current = isource
                    res = 0
            if isource_current > -1:
                res += dt
        if isource_current > -1:
            spectrum.samples_mv[isource_current] += res

        return spectrum


cdef class CartesianRayTransferIntegrator(RayTransferIntegrator):
    """
    Calculates the distances traveled by the ray through the voxels defined on a regular grid
    in Cartesian coordinate system: :math:`(X, Y, Z)`. This integrator is used with
    the `CartesianRayTransferEmitter` material to calculate ray transfer matrices (geometry
    matrices). The value for each voxel is stored in respective bin of the spectral array.
    The distances traveled by the ray through the voxel is calculated approximately and
    the accuracy depends on the integration step.
    """

    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cpdef Spectrum integrate(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                             InhomogeneousVolumeEmitter material, Point3D start_point, Point3D end_point,
                             AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            Point3D start, end
            Vector3D direction
            int isource, isource_current, it, ix, iy, iz, ix_current, iy_current, iz_current, n
            double length, t, dt, x, y, z, dx, dy, dz, res
            int[:, :, ::1] voxel_map_mv

        if not isinstance(material, CartesianRayTransferEmitter):
            raise TypeError('Only CartesianRayTransferEmitter material is supported by CartesianRayTransferIntegrator')
        start = start_point.transform(world_to_primitive)  # start point in local coordinates
        end = end_point.transform(world_to_primitive)  # end point in local coordinates
        direction = start.vector_to(end)  # direction of integration
        length = direction.get_length()  # integration length
        if length < 0.1 * self._step:  # return if ray's path is too short
            return spectrum
        direction = direction.normalise()  # normalized direction
        n = max(self._min_samples, <int>(length / self._step))  # number of points along ray's trajectory
        dt = length / n  # integration step
        # cython performs checks on attributes of external class, so it's better to do the checks before the loop
        voxel_map_mv = material.voxel_map_mv
        dx = material.dx
        dy = material.dy
        dz = material.dz
        ix_current = -1
        iy_current = -1
        iz_current = -1
        isource_current = -1
        res = 0
        for it in range(n):
            t = (it + 0.5) * dt
            x = start.x + direction.x * t  # x coordinates of the points
            y = start.y + direction.y * t  # y coordinates of the points
            z = start.z + direction.z * t  # z coordinates of the points
            ix = <int>(x / dx)  # X-indices of grid cells, in which the points are located
            iy = <int>(y / dy)  # Y-indices of grid cells, in which the points are located
            iz = <int>(z / dz)  # Z-indices of grid cells, in which the points are located
            if ix != ix_current or iy != iy_current or iz != iz_current:  # we moved to the next cell
                ix_current = ix
                iy_current = iy
                iz_current = iz
                isource = voxel_map_mv[ix, iy, iz]  # light source indices in spectral array
                if isource != isource_current:  # we moved to the next source
                    if isource_current > -1:
                        spectrum.samples_mv[isource_current] += res  # writing results for the current source
                    isource_current = isource
                    res = 0
            if isource_current > -1:
                res += dt
        if isource_current > -1:
            spectrum.samples_mv[isource_current] += res

        return spectrum


cdef class RayTransferEmitter(InhomogeneousVolumeEmitter):
    """
    Basic class for ray transfer emitters defined on a regular 3D grid. Ray transfer emitters
    are used to calculate ray transfer matrices (geometry matrices) for a single value
    of wavelength.

    :param tuple grid_shape: The shape of regular grid (the number of grid cells
        along each direction).
    :param tuple grid_steps: The sizes of grid cells along each direction.
    :param np.ndarray voxel_map: An array with shape `grid_shape` containing the indices of
        the light sources. This array maps the cells of regular grid to the respective voxels
        (light sources). The cells with identical indices in `voxel_map` array form a single
        voxel (light source). If `voxel_map[i1, i2, i3] == -1`, the cell with indices
        `(i1, i2, i3)` will not be mapped to any light source. This parameters allows to
        apply a custom geometry (pixelated though) to the light sources.
        Default value: `voxel_map=None`.
    :param np.ndarray mask: A boolean mask array with shape `grid_shape`.
        Allows to include (`mask[i1, i2, i3] == True`) or exclude (`mask[i1, i2, i3] == False`)
        the cells from the calculation. The ray tranfer matrix will be calculated only for those
        cells for which mask is True. This parameter is ignored if `voxel_map` is provided,
        defaults to `mask=None` (all cells are included).
    :param raysect.optical.material.VolumeIntegrator integrator: Volume integrator,
        defaults to `integrator=NumericalVolumeIntegrator()`

    :ivar tuple grid_shape: The shape of regular 3D grid.
    :ivar tuple grid_steps: The sizes of grid cells along each direction.
    :ivar np.ndarray voxel_map: An array containing the indices of the light sources.
    :ivar np.ndarray ~.mask: A boolean mask array showing active (True) and inactive
        (False) gird cells.
    :ivar int bins: Number of light sources (the size of spectral array must be equal to this value).
    """

    def __init__(self, tuple grid_shape, tuple grid_steps, np.ndarray voxel_map=None, np.ndarray mask=None, VolumeIntegrator integrator=None):

        cdef:
            int i
            double step

        if len(grid_shape) != 3:
            raise ValueError("Attribute 'grid_shape' must contain 3 elements.")
        if len(grid_steps) != 3:
            raise ValueError("Attribute 'grid_steps' must contain 3 elements.")
        for i in grid_shape:
            if i < 1:
                raise ValueError('Number of grid cells must be > 0.')
        for step in grid_steps:
            if step <= 0:
                raise ValueError('Grid steps must be > 0.')
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
        return <tuple>self._grid_shape

    @property
    def grid_steps(self):
         return <tuple>self._grid_steps

    cdef np.ndarray _map_from_mask(self, mask):

        cdef:
            int i
            np.ndarray voxel_map

        if mask is not None:
            if mask.shape != self.grid_shape:
                raise ValueError('Mask array must be of shape: %s.' % (' '.join(['%d' % i for i in self._grid_shape])))
            mask = mask.astype(bool)
        else:
            mask = np.ones(self.grid_shape, dtype=bool)
        voxel_map = -1 * np.ones(mask.shape, dtype=np.int32)
        voxel_map[mask] = np.arange(mask.sum(), dtype=np.int32)

        return voxel_map

    @property
    def bins(self):
        return self._bins

    @property
    def voxel_map(self):
        return self._voxel_map

    @voxel_map.setter
    def voxel_map(self, value):

        cdef:
            int i

        if value.shape != self.grid_shape:
            raise ValueError('Voxel_map array must be of shape: %s.' % (' '.join(['%d' % i for i in self._grid_shape])))
        self._voxel_map = value.astype(np.int32)
        self.voxel_map_mv = self._voxel_map
        self._bins = self._voxel_map.max() + 1

    @property
    def mask(self):
        return self._voxel_map > -1

    @mask.setter
    def mask(self, np.ndarray value):
        self._voxel_map = self._map_from_mask(value)
        self.voxel_map_mv = self._voxel_map
        self._bins = self._voxel_map.max() + 1


cdef class CylindricalRayTransferEmitter(RayTransferEmitter):
    """
    A unit emitter defined on a regular 3D :math:`(R, \phi, Z)` grid, which
    can be used to calculate ray transfer matrices (geometry matrices) for a single value
    of wavelength.
    This emitter is periodic in :math:`\phi` direction.
    Note that for performance reason there are no boundary checks in `emission_function()`,
    or in `CylindricalRayTranferIntegrator`, so this emitter must be placed between a couple
    of coaxial cylinders that act like a bounding box.

    :param tuple grid_shape: The shape of regular :math:`(R, \phi, Z)` 3D grid.
        If `grid_shape[1] = 1`, the emitter is axisymmetric.
    :param tuple grid_steps: The sizes of grid cells in `R`, :math:`\phi` and `Z`
        directions. The size in :math:`\phi` must be provided in degrees (sizes in `R` and `Z`
        are provided in meters). The period in :math:`\phi` direction is defined as
        `grid_shape[1] * grid_steps[1]`. Note that the period must be a multiple of 360.
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

    :ivar float period: The period in :math:`\phi` direction (equals to
        `grid_shape[1] * grid_steps[1]`).
    :ivar float rmin: Lower bound of grid in `R` direction.
    :ivar float dr: The size of grid cell in `R` direction (equals to `grid_shape[0]`).
    :ivar float dphi: The size of grid cell in :math:`\phi` direction (equals to `grid_shape[1]`).
    :ivar float dz: The size of grid cell in `Z` direction (equals to `grid_shape[2]`).

    .. code-block:: pycon

        >>> from raysect.optical import World, translate
        >>> from raysect.primitive import Cylinder, Subtract
        >>> from cherab.tools.raytransfer import CylindricalRayTransferEmitter
        >>> world = World()
        >>> grid_shape = (10, 1, 10)  # axisymmetric case
        >>> grid_steps = (0.5, 360, 0.5)
        >>> rmin = 2.5
        >>> material = CylindricalRayTransferEmitter(grid_shape, grid_steps, rmin=rmin)
        >>> eps = 1.e-6  # ray must never leave the grid when passing through the volume
        >>> radius_outer = grid_shape[0] * grid_steps[0] - eps
        >>> height = grid_shape[2] * grid_steps[2] - eps
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

    def __init__(self, tuple grid_shape, tuple grid_steps, np.ndarray voxel_map=None, np.ndarray mask=None, VolumeIntegrator integrator=None,
                 double rmin=0):

        cdef:
            double def_integration_step, period, num_sectors

        def_integration_step = 0.1 * min(grid_steps[0], grid_steps[-1])
        integrator = integrator or CylindricalRayTransferIntegrator(def_integration_step)
        super().__init__(grid_shape, grid_steps, voxel_map=voxel_map, mask=mask, integrator=integrator)
        self.rmin = rmin
        self._dr = self._grid_steps[0]
        self._dphi = self._grid_steps[1]
        self._dz = self._grid_steps[2]
        period = self._grid_shape[1] * self._grid_steps[1]
        num_sectors = 360. / period
        if abs(round(num_sectors) - num_sectors) > 1.e-3:
            raise ValueError("The period %.3f (grid_shape[1] * grid_steps[1]) is not a multiple of 360." % period)
        self._period = period

    @property
    def rmin(self):
        return self._rmin

    @rmin.setter
    def rmin(self, value):
        if value < 0:
            raise ValueError("Attribute 'rmin' must be >= 0.")
        self._rmin = value

    @property
    def period(self):
        return self._period

    @property
    def dr(self):
        return self._dr

    @property
    def dphi(self):
        return self._dphi

    @property
    def dz(self):
        return self._dz

    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            int isource, ir, iphi, iz
            double r, phi

        iz = <int>(point.z / self._dz)  # Z-index of grid cell, in which the point is located
        r = sqrt(point.x * point.x + point.y * point.y)  # R coordinates of the points
        ir = <int>((r - self._rmin) / self._dr)  # R-index of grid cell, in which the points is located
        if self._grid_shape[1] == 1:  # axisymmetric case
            iphi = 0
        else:
            phi = (180. / pi) * atan2(point.y, point.x)  # phi coordinate of the point (in degrees)
            phi = (phi + 360) % self._period  # moving into the [0, period) sector (periodic emitter)
            iphi = <int>(phi / self._dphi)  # phi-index of grid cell, in which the point is located
        isource = self.voxel_map_mv[ir, iphi, iz]  # index of the light source in spectral array
        if isource < 0:  # grid cell is not mapped to any light source
            return spectrum
        spectrum.samples_mv[isource] += 1.  # unit emissivity
        return spectrum


cdef class CartesianRayTransferEmitter(RayTransferEmitter):
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

    def __init__(self, tuple grid_shape, tuple grid_steps, np.ndarray voxel_map=None, np.ndarray mask=None, VolumeIntegrator integrator=None):

        cdef:
            double def_integration_step

        def_integration_step = 0.1 * min(grid_steps)
        integrator = integrator or CartesianRayTransferIntegrator(def_integration_step)
        super().__init__(grid_shape, grid_steps, voxel_map=voxel_map, mask=mask, integrator=integrator)
        self._dx = self._grid_steps[0]
        self._dy = self._grid_steps[1]
        self._dz = self._grid_steps[2]

    @property
    def dx(self):
        return self._dx

    @property
    def dy(self):
        return self._dy

    @property
    def dz(self):
        return self._dz

    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            int isource, ix, iy, iz

        ix = <int>(point.x / self._dx)  # X-index of grid cell, in which the point is located
        iy = <int>(point.y / self._dy)  # Y-index of grid cell, in which the point is located
        iz = <int>(point.z / self._dz)  # Z-index of grid cell, in which the point is located
        isource = self.voxel_map_mv[ix, iy, iz]  # index of the light source in spectral array
        if isource < 0:  # grid cell is not mapped to any light source
            return spectrum
        spectrum.samples_mv[isource] += 1.  # unit emissivity
        return spectrum
