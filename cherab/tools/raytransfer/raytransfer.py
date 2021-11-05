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

"""
Ray transfer objects accelerate the calculation of geometry matrices (or Ray Transfer Matrices as
they were called in `S. Kajita, et al. Contrib. Plasma Phys., 2016, 1-9
<https://onlinelibrary.wiley.com/doi/abs/10.1002/ctpp.201500124>`_)
in the case of regular spatial grids. As in the case of Voxels, the spectral array is used to store
the data for individual light sources (in this case the grid cells or their unions), however
no voxels are created at all. Instead, a custom integration along the ray is implemented.
Ray transfer objects allow to calculate geometry matrices for a single value of wavelength.

Use `RayTransferBox` class for Cartesian grids and `RayTransferCylinder` class for cylindrical grids
(3D or axisymmetrical).

Performance tips:

The best performance is achieved when Ray Transfer Objects are used with special pipelines and
optimised materials (currently only rough metals are optimised, see the demos).

When the number of individual light sources and respective bins in the spectral array is higher
than ~50-70 thousands, the lack of CPU cache memory becomes a serious factor affecting performance.
Therefore, it is not recommended to use hyper-threading when calculating geometry matrices for
a large number of light sources. It is also recommended to divide the calculation into several
parts and to calculate partial geometry matrices for not more than ~50-70 thousands of light
sources in a single run. Partial geometry matrices can easily be combined into one when all
computations are complete.
"""

import numpy as np
from raysect.primitive import Cylinder, Subtract, Box
from raysect.optical import Point3D
from .emitters import CylindricalRayTransferIntegrator, CartesianRayTransferIntegrator, CylindricalRayTransferEmitter, CartesianRayTransferEmitter


class RayTransferObject:
    """
    Basic class for ray transfer objects.

    :ivar np.ndarray voxel_map: An array containing the indices of the light sources.
    :ivar np.ndarray ~.mask: A boolean mask array showing active (True) and inactive (False) gird cells.
    :ivar Node parent: Scene-graph parent node.
    :ivar AffineMatrix3D transform: An AffineMatrix3D defining the local co-ordinate system
        relative to the scene-graph parent.
    :ivar float step: Integration step of volume integrator.
    :ivar int bins: Number of light sources (the size of spectral array must be equal to this value).
    """

    def __init__(self, primitive):
        self._primitive = primitive

    @property
    def parent(self):
        return self._primitive.parent

    @parent.setter
    def parent(self, value):
        self._primitive.parent = value

    @property
    def transform(self):
        return self._primitive.transform

    @transform.setter
    def transform(self, value):
        self._primitive.transform = value

    @property
    def step(self):
        return self._primitive.material.integrator.step

    @step.setter
    def step(self, value):
        self._primitive.material.integrator.step = value

    @property
    def voxel_map(self):
        return self._primitive.material.voxel_map

    @voxel_map.setter
    def voxel_map(self, value):
        self._primitive.material.voxel_map = value

    @property
    def mask(self):
        return self._primitive.material.mask

    @mask.setter
    def mask(self, value):
        self._primitive.material.mask = value

    @property
    def bins(self):
        return self._primitive.material.bins

    @property
    def material(self):
        return self._primitive.material

    def invert_voxel_map(self):
        """
        Returns a list of arrays of cell indices belonging to each light source.
        This list is an inversion of `voxel_map` array.
        """
        inverted_voxel_map = []
        for i in range(self._primitive.material.bins):
            inverted_voxel_map.append(np.where(self._primitive.material.voxel_map == i))

        return inverted_voxel_map


class RayTransferCylinder(RayTransferObject):
    """
    Ray transfer object for cylindrical emitter defined on a regular 3D :math:`(R, \phi, Z)` grid.
    This emitter is periodic in :math:`\phi` direction.
    The base of the cylinder is located at `Z = 0` plane. Use `transform`
    parameter to move it.

    :param float radius_outer: Radius of the outer cylinder and the upper bound of grid in
        `R` direction (in meters).
    :param float height: Height of the cylinder and the length of grid in `Z` direction
        (in meters).
    :param int n_radius: Number of grid points in `R` direction.
    :param int n_height: Number of grid points in `Z` direction.
    :param float radius_inner: Radius of the inner cylinder and the lower bound of grid in
        `R` direction (in meters), defaults to `radius_inner=0`.
    :param int n_polar: Number of grid points in :math:`\phi` direction, defaults to
        `n_polar=1` (axisymmetric case).
    :param float period: A period in :math:`\phi` direction (in degree), defaults to `period=360`.
    :param float step: The step of integration along the ray (in meters),
        defaults to `step = 0.1 * min((radius_outer - radius_inner)/n_radius, height/n_height)`.
    :param np.ndarray voxel_map: An array with shape `(n_radius, n_polar, n_height)`
        containing the indices of the light sources.
        This array maps the cells in :math:`(R, \phi, Z)` space to the respective voxels
        (light sources). The cells with identical indices in `voxel_map` array form a single voxel
        (light source). If `voxel_map[ir, iphi, iz] == -1`, the cell with index `(ir, iphi, iz)`
        will not be mapped to any light source. This parameters allows to apply a custom geometry
        (pixelated though) to the light sources. Default value: `voxel_map=None`.
        Convert 2D (axisymmetric) `voxel_map` to 3D with `voxel_map = voxel_map[:, None, :]`.
    :param np.ndarray mask: A boolean mask array with shape `(n_radius, n_polar, n_height)`.
        Allows to include (`mask[ir, iphi, iz] == True`) or exclude (`mask[ir, iphi, iz] == False`)
        the cells from the calculation. The ray tranfer matrix will be calculated only for those
        cells for which mask is True. This parameter is ignored if `voxel_map` is provided,
        defaults to `mask=None` (all cells are included).
        Convert 2D (axisymmetric) `mask` to 3D with `mask = mask[:, None, :]`.
    :param Node parent: Scene-graph parent node or None (default = None).
    :param AffineMatrix3D transform: An AffineMatrix3D defining the local co-ordinate system
        relative to the scene-graph parent (default = identity matrix).

    .. code-block:: pycon

        >>> from raysect.optical import World, translate
        >>> from cherab.tools.raytransfer import RayTransferCylinder
        >>> world = World()
        >>> rtc = RayTransferCylinder(radius_outer=8., height=10., n_radius=400, n_height=1000,
                                      radius_inner=4.)
        >>> rtc.parent = world
        >>> rtc.transform = translate(0, 0, -5.)
        ...
        >>> camera.spectral_bins = rtc.bins
        >>> # ray transfer matrix will be calculated for 600.5 nm
        >>> camera.min_wavelength = 600.
        >>> camera.max_wavelength = 601.
    """

    def __init__(self, radius_outer, height, n_radius, n_height, radius_inner=0, n_polar=1, period=360., step=None, voxel_map=None, mask=None,
                 parent=None, transform=None):
        num_sectors = 360. / period
        if abs(round(num_sectors) - num_sectors) > 1.e-3:
            raise ValueError("The period %.3f is not a multiple of 360." % period)
        grid_shape = (n_radius, n_polar, n_height)
        dr = (radius_outer - radius_inner) / n_radius
        dz = height / n_height
        dphi = period / n_polar
        grid_steps = (dr, dphi, dz)
        eps_r = 1.e-5 * dr
        eps_z = 1.e-5 * dz
        step = step or 0.1 * min(dr, dz)
        material = CylindricalRayTransferEmitter(grid_shape, grid_steps, mask=mask, voxel_map=voxel_map,
                                                 integrator=CylindricalRayTransferIntegrator(step), rmin=radius_inner)
        primitive = Subtract(Cylinder(radius_outer - eps_r, height - eps_z), Cylinder(radius_inner + eps_r, height - eps_z),
                             material=material, parent=parent, transform=transform)
        super().__init__(primitive)


class RayTransferBox(RayTransferObject):
    """
    Ray transfer object for rectangular emitter defined on a regular 3D :math:`(X, Y, Z)` grid.
    The grid starts at (0, 0, 0). Use `transform` parameter to move it.

    :param float xmax: Upper bound of grid in `X` direction (in meters).
    :param float ymax: Upper bound of grid in `Y` direction (in meters).
    :param float zmax: Upper bound of grid in `Z` direction (in meters).
    :param int nx: Number of grid points in `X` direction.
    :param int ny: Number of grid points in `Y` direction.
    :param int nz: Number of grid points in `Z` direction.
    :param float step: The step of integration along the ray (in meters), defaults to
        `step = 0.1 * min(xmax / nx, ymax / ny, zmax / nz)`.
    :param np.ndarray voxel_map: An array with shape `(nx, ny, nz)`
        containing the indices of the light sources. This array maps the cells in
        :math:`(X, Y, Z)` space to the respective voxels (light sources). The cells with
        identical indices in `voxel_map` array form a single voxel (light source).
        If `voxel_map[ix, iy, iz] == -1`, the cell with index `(ix, iy, iz)` will not be mapped
        to any light source. This parameters allows to apply a custom geometry (pixelated though)
        to the light sources. Default value: `voxel_map=None`.
    :param np.ndarray mask: A boolean mask array with shape `(nx, ny, nz)`.
        Allows to include (`mask[ix, iy, iz] == True`) or exclude (`mask[ix, iy, iz] == False`)
        the cells from the calculation. The ray tranfer matrix will be calculated only for those
        cells for which mask is True. This parameter is ignored if `voxel_map` is provided,
        defaults to `mask=None` (all cells are included).
    :param Node parent: Scene-graph parent node or None (default = None).
    :param AffineMatrix3D transform: An AffineMatrix3D defining the local co-ordinate system
        relative to the scene-graph parent (default = identity matrix).

    .. code-block:: pycon

        >>> from raysect.optical import World, translate
        >>> from cherab.tools.raytransfer import RayTransferBox
        >>> world = World()
        >>> rtb = RayTransferBox(xmax=1., ymax=1., zmax=1., nx=100, ny=100, nz=100)
        >>> rtb.parent = world
        >>> rtb.transform = translate(-0.5, -0.5, -0.5)
        >>> ### cutting out a sphere of radius 0.5 ###
        >>> x = np.linspace(-0.495, 0.495, 100)
        >>> xsqr = x * x
        >>> ### mask is a bollean array of shape (100, 100, 100) ###
        >>> mask = xsqr[:, None, None] + xsqr[None, :, None] + xsqr[None, None, :] < 0.25
        >>> rtb.mask = mask  # all cells outside this sphere are excluded
        ...
        >>> camera.spectral_bins = rtb.bins
        >>> # ray transfer matrix will be calculated for 600.5 nm
        >>> camera.min_wavelength = 600.
        >>> camera.max_wavelength = 601.
    """

    def __init__(self, xmax, ymax, zmax, nx, ny, nz, step=None, voxel_map=None, mask=None,
                 parent=None, transform=None):
        grid_shape = (nx, ny, nz)
        dx = xmax / nx
        dy = ymax / ny
        dz = zmax / nz
        grid_steps = (dx, dy, dz)
        eps_x = 1.e-5 * dx
        eps_y = 1.e-5 * dy
        eps_z = 1.e-5 * dz
        step = step or 0.1 * min(dx, dy, dz)
        material = CartesianRayTransferEmitter(grid_shape, grid_steps, mask=mask, voxel_map=voxel_map,
                                               integrator=CartesianRayTransferIntegrator(step))
        primitive = Box(lower=Point3D(0, 0, 0), upper=Point3D(xmax - eps_x, ymax - eps_y, zmax - eps_z),
                        material=material, parent=parent, transform=transform)
        super().__init__(primitive)
