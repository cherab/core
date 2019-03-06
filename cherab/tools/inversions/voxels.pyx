
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
cimport numpy as np
from libc.math cimport abs as cabs, floor
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from raysect.core cimport (Node, Point2D, Vector2D, Point3D, Vector3D,
                           rotate_z, AffineMatrix3D, new_point3d)
from raysect.core.math cimport triangulate2d, translate, rotate_basis
from raysect.core.math.function cimport Function3D
from raysect.core.math.cython.utility cimport winding2d, find_index
from raysect.core.math.random cimport uniform, point_triangle
from raysect.primitive import Mesh, Cylinder, Cone, Intersect, Subtract, Union
from raysect.optical import UnityVolumeEmitter
from raysect.optical cimport Spectrum, World, Primitive, Ray
from raysect.optical.material.emitter.homogeneous cimport HomogeneousVolumeEmitter


PI = 3.141592653589793


class Voxel(Node):
    """
    A Voxel base class.

    Each Voxel is a Node in the scenegraph. Each Voxel type that
    inherits from this class defines its own geometry.

    :ivar float volume: The geometric volume of this voxel.
    """

    @property
    def volume(self):
        raise NotImplementedError()


class AxisymmetricVoxel(Voxel):
    """
    An axis-symmetric Voxel.

    This Voxel is symmetric about the vertical z-axis. The cross section
    of the voxel can be arbitrarily defined by a polygon in the r-z plane.
    The type of geometric primitive used to define the geometric extent of
    this Voxel can be selected by the user and either of type Mesh or CSG.
    The two representations should approximately the same geometry but have
    different performance goals. The CSG representation uses lower memory and
    is a better choice when large numbers of Voxels will be present in a single
    scene. The Mesh representation is split into smaller components and better
    for cases where multiple importance sampling is important, such as weight
    matrices including reflection effects.


    :param vertices: A list/tuple of Point2D objects specifying the voxel's
      polygon outline in the r-z plane.
    :param Node parent: The scenegraph to which this Voxel is attached.
    :param Material material: The emission material of this Voxel, defaults
      to a UnityVolumeEmitter() for weight matrix calculations.
    :param str primitive_type: Specifies the primitive type, can be either
      'mesh' or 'csg'. Defaults to the mesh representation.

    :ivar float volume: The geometric volume of this voxel.
    :ivar float cross_sectional_area: The cross sectional area of the voxel in
      the r-z plane.
    """

    def __init__(self, vertices, parent=None, material=None, primitive_type='mesh'):

        super().__init__(parent=parent)

        self._material = material or UnityVolumeEmitter()

        num_vertices = len(vertices)
        if not num_vertices >= 3:
            raise TypeError('The AxisSymmetricVoxel can only be specified by a polygon with at least 3 Point2D objects.')

        self._vertices = np.zeros((num_vertices, 2))
        for i, vertex in enumerate(vertices):
            if not isinstance(vertex, Point2D):
                raise TypeError('The AxisSymmetricVoxel can only be specified with a list/tuple of Point2D objects.')
            self._vertices[i, :] = vertex.x, vertex.y

        if any(self._vertices[:, 0] < 0):
            raise ValueError('The polygon vertices must be in the r-z plane.')

        # Check the polygon is clockwise, if not => reverse it.
        if not winding2d(self._vertices):
            self._vertices = self._vertices[::-1]

        self._triangles = triangulate2d(self._vertices)

        # Generate summary statistics
        self.radius = self._vertices[:, 0].sum()/num_vertices

        if primitive_type == 'mesh':
            self._build_mesh()
        elif primitive_type == 'csg':
            if self._has_rectangular_cross_section():
                self._build_csg_from_rectangle()
            else:
                for triangle in self._triangles:
                    self._build_csg_from_triangle(self._vertices[triangle])
        else:
            raise ValueError("primitive_type should be 'mesh' or 'csg'")

    def _build_mesh(self):
        """Build the Voxel out of triangular mesh elements."""
        num_vertices = len(self._vertices)
        radial_width = self._vertices[:, 0].max() - self._vertices[:, 0].min()

        number_segments = int(floor(2 * PI * self.radius / radial_width))
        theta_adjusted = 360 / number_segments

        # Construct 3D outline of polygon in x-z plane and the rotated plane
        xz_points = []  # Set of points in x-z plane
        rotated_points = []  # Set of points rotated away from x-z plane
        for i in range(num_vertices):
            xz_point = Point3D(self._vertices[i, 0], 0, self._vertices[i, 1])
            xz_points.append(xz_point)
            rotated_point = xz_point.transform(rotate_z(theta_adjusted))
            rotated_points.append(rotated_point)

        # assemble mesh vertices
        vertices = []
        for p in xz_points:
            vertices.append([p.x, p.y, p.z])
        for p in rotated_points:
            vertices.append([p.x, p.y, p.z])

        # assemble mesh triangles
        triangles = []
        # front face triangles
        for i in range(self._triangles.shape[0]):
            triangles.append([self._triangles[i, 2], self._triangles[i, 1], self._triangles[i, 0]])
        # rear face triangles
        for i in range(self._triangles.shape[0]):
            triangles.append([self._triangles[i, 0]+num_vertices, self._triangles[i, 1]+num_vertices, self._triangles[i, 2]+num_vertices])
        # Assemble side triangles
        for i in range(num_vertices):
            if i == num_vertices-1:
                triangles.append([i+1, i+num_vertices, i])
                triangles.append([0, i+1, i])
            else:
                triangles.append([i+num_vertices+1, i+num_vertices, i])
                triangles.append([i, i+1, i+num_vertices+1])

        base_segment = Mesh(vertices=vertices, triangles=triangles, smoothing=False)

        # Construct annulus by duplicating and rotating base segment.
        for i in range(number_segments):
            theta_rotation = theta_adjusted * i
            segment = base_segment.instance(transform=rotate_z(theta_rotation), material=self._material, parent=self)

    def _has_rectangular_cross_section(self):
        """
        Test if the voxel has a rectangular cross section, and is aligned with
        the coordinate axes.
        """
        if len(self.vertices) != 4:
            return False
        # A rectangle (including a square, which is considered to have a
        # rectangular cross section too) is defined by having equal length
        # diagonals.
        distance_13 = self.vertices[0].distance_to(self.vertices[2])
        distance_24 = self.vertices[1].distance_to(self.vertices[3])
        if distance_13 != distance_24:
            return False
        # The rectangle should be aligned with the coordinate axes, i.e. the
        # edge from vertex 1 to 2 should be parallel to either the x or z axes.
        side_12 = self.vertices[0].vector_to(self.vertices[1])
        if side_12.dot(Vector2D(1, 0)) != 0 and side_12.dot(Vector2D(0, 1)) != 0:
            return False
        return True

    def _build_csg_from_triangle(self, vertices):
        if vertices.shape != (3, 2):
            raise ValueError("Vertices must be an array of 3 (x, z) coordinates")
        # Sort the vertices of the triangle in decreasing x
        # Vertex 1 is at largest x, vertex 2 is middle x, vertex 3 is smallest x
        sort_inds = np.argsort(vertices[:, 0])[::-1]
        vertices = vertices[sort_inds]
        vertex_rs = vertices[:, 0]
        vertex_zs = vertices[:, 1]
        # Create a bounding ring around the vertices, with rectangular cross section
        box_rmax = max(vertex_rs)
        box_rmin = min(vertex_rs)
        box_zmax = max(vertex_zs)
        box_zmin = min(vertex_zs)
        cylinder_height = box_zmax - box_zmin
        outer_cylinder = Cylinder(radius=box_rmax, height=cylinder_height,
                                  transform=translate(0, 0, box_zmin))
        inner_cylinder = Cylinder(radius=box_rmin, height=cylinder_height,
                                  transform=translate(0, 0, box_zmin))
        bounding_ring = Subtract(outer_cylinder, inner_cylinder)

        def create_cone(rx, zx, ry, zy):
            if zx == zy:
                return None
            if rx == ry:
                return Cylinder(radius=rx, height=abs(zx - zy),
                                transform=translate(0, 0, min(zx, zy)))
            if rx < ry:
                raise ValueError('rx must be larger than ry')
            if zx > zy:
                transform = translate(0, 0, zx) * rotate_basis(Vector3D(0, 0, -1),
                                                               Vector3D(0, 1, 0))
            else:
                transform = translate(0, 0, zx)
            rcone = rx
            hcone = abs(zx - zy) * rx / (rx - ry)
            cone = Cone(radius=rcone, height=hcone, transform=transform)
            return cone

        r1, r2, r3 = vertex_rs
        z1, z2, z3 = vertex_zs
        cone13 = create_cone(r1, z1, r3, z3)
        cone12 = create_cone(r1, z1, r2, z2)
        cone23 = create_cone(r2, z2, r3, z3)
        if z2 <= z1 <= z3 or z3 <= z1 <= z2:
            voxel_element = Intersect(Subtract(Union(cone12, cone13), cone23), bounding_ring)
        elif z1 <= z2 <= z3 and r1 != r2 != r3:
            # Requires slightly different treatment, with a different bounding ring
            outer_cylinder = Cylinder(radius=r1, height=z2 - z1, transform=translate(0, 0, z1))
            inner_cylinder = Cylinder(radius=r3, height=z2 - z1, transform=translate(0, 0, z1))
            bounding_ring = Subtract(outer_cylinder, inner_cylinder)
            voxel_element = Intersect(Subtract(cone12, cone13), Union(bounding_ring, cone23))
        elif r1 == r2 and z3 >= z2 and z3 >= z1:
            if z1 >= z2:
                voxel_element = Intersect(Subtract(Union(cone12, cone13), cone23), bounding_ring)
            else:
                voxel_element = Intersect(Subtract(Union(cone12, cone23), cone13), bounding_ring)
        else:
            if abs(z1 - z2) >= abs(z1 - z3):
                voxel_element = Intersect(Subtract(Subtract(cone12, cone13), cone23), bounding_ring)
            else:
                voxel_element = Intersect(Subtract(Subtract(cone13, cone12), cone23), bounding_ring)

        voxel_element.parent = self
        voxel_element.material = self._material

    def _build_csg_from_rectangle(self):
        rmax = max(self._vertices[:, 0])
        rmin = min(self._vertices[:, 0])
        zmax = max(self._vertices[:, 1])
        zmin = min(self._vertices[:, 1])
        cylinder_height = zmax - zmin
        cylinder_transform = translate(0, 0, zmin)
        outer_cylinder = Cylinder(radius=rmax, height=cylinder_height,
                                  transform=cylinder_transform)
        inner_cylinder = Cylinder(radius=rmin, height=cylinder_height,
                                  transform=cylinder_transform)
        voxel = Subtract(outer_cylinder, inner_cylinder)
        voxel.parent = self
        voxel.material = self._material

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, value):
        self._material = value
        for mesh_segment in self.children:
            mesh_segment.material = value

    @property
    def vertices(self):

        vertices = []
        for i in range(self._vertices.shape[0]):
            vertices.append(Point2D(self._vertices[i, 0], self._vertices[i, 1]))

        return vertices

    @property
    def cross_sectional_area(self):

        # Simple calculation of the polygon area using the shoelace algorithm
        # https://en.wikipedia.org/wiki/Shoelace_formula

        num_vertices = self._vertices.shape[0]

        area = 0
        for i in range(num_vertices - 1):
            area += self._vertices[i, 0] * self._vertices[i+1, 1]
        area += self._vertices[num_vertices - 1, 0] * self._vertices[0, 1]
        for i in range(num_vertices - 1):
            area -= self._vertices[i, 1] * self._vertices[i+1, 0]
        area -= self._vertices[num_vertices - 1, 1] * self._vertices[0, 0]

        return abs(area) / 2

    @property
    def volume(self):

        # return approximate cell volume
        return 2 * PI * self.radius * self.cross_sectional_area


class VoxelCollection(Node):
    """
    The base class for collections of voxels.

    Used for managing a collection of voxels when calculating a weight
    matrix for example.

    .. warning:
       No checks are performed by the base class to ensure that the voxel
       volumes don't overlap. This is the responsibility of the user.

    :ivar float count: The number of voxels in this collection.
    :ivar float total_volume: The total volume of all voxels.
    """

    def __len__(self):
        return len(self._voxels)

    def __getitem__(self, item):

        if not isinstance(item, int):
            raise TypeError("VoxelCollection can only be indexed with an integer.")

        if not (0 <= item < self.count):
            raise IndexError("The specified voxel index is out of range.")

        return self._voxels[item]

    def __iter__(self):

        for voxel in self._voxels:
            yield voxel

    @property
    def count(self):
        return len(self._voxels)

    @property
    def total_volume(self):

        total_volume = 0
        for voxel in self._voxels:
            total_volume += voxel.volume

        return total_volume

    def set_active(self, item):
        """
        Set the ith voxel as an active emitter.

        :param item: If item is an int, the ith voxel will be configured as an active emitter,
          all the others will be turned off. If item is the string 'all', all voxels will be
          active emitters.
        """
        raise NotImplementedError()

    def parent_all_voxels(self):
        """Add all voxels in this collection to the scenegraph."""

        for voxel in self._voxels:
            voxel.parent = self

    def unparent_all_voxels(self):
        """Remove all voxels in this collection from the scenegraph."""

        for voxel in self._voxels:
            voxel.parent = None

    def emissivities_from_function(self, emission_function, grid_samples=10):
        """
        Returns an array of sampled emissivities at each voxel location.

        This is a virtual method and must be implemented in the derived
        VoxelCollection class.

        Note that the results will be nonsense if you mix an emission function
        and VoxelCollection with incompatible symmetries.

        :param Function3D emission_function: Emission function to sample over.
        :param int grid_samples: Number of emission samples to average over.
        :rtype: np.ndarray
        """
        raise NotImplementedError()


class ToroidalVoxelGrid(VoxelCollection):
    """
    A collection of axis-symmetric toroidal voxels.

    This object manages a collection of voxels, where each voxel in the collection
    is an AxisymmetricVoxel object.

    :param voxel_coordinates: An array/list of voxels, where each voxel element
      is defined by a list of 2D points.
    :param str name: The name of this voxel collection.
    :param Node parent: The parent scenegraph to which these voxels belong.
    :param AffineMatrix3D transform: The coordinate transformation of this local
      coordinate system relative to the scenegraph parent, defaults to the identity
      transform.
    :param active: Selects which voxels are active emitters in the initialised state.
      If active is an int, the ith voxel will be configured as an active emitter, all
      the others will be turned off. If active is the string 'all', all voxels will be
      active emitters.
    :param str primitive_type: The geometry type to use for the AxisymmetricVoxel
      instances, can be ['mesh', 'csg']. See their documentation for more information.
      Defaults to `primitive_type='mesh'`.
    """

    def __init__(self, voxel_coordinates, name='', parent=None, transform=None,
                 active="all", primitive_type='mesh'):

        super().__init__(name=name, parent=parent, transform=transform)

        self._min_radius = 1E999
        self._max_radius = 0
        self._min_height = 1E999
        self._max_height = -1E999

        self._voxels = []
        for i, voxel_vertices in enumerate(voxel_coordinates):

            voxel = AxisymmetricVoxel(voxel_vertices, primitive_type=primitive_type)
            if active == "all" or active == i:
                voxel.parent = parent
            self._voxels.append(voxel)

            # Test and set extent values
            if voxel._vertices[:, 0].min() < self._min_radius:
                self._min_radius = voxel._vertices[:, 0].min()
            if voxel._vertices[:, 0].max() > self._max_radius:
                self._max_radius = voxel._vertices[:, 0].max()
            if voxel._vertices[:, 1].min() < self._min_height:
                self._min_height = voxel._vertices[:, 1].min()
            if voxel._vertices[:, 1].max() > self._max_height:
                self._max_height = voxel._vertices[:, 1].max()

    @property
    def min_radius(self):
        return self._min_radius

    @property
    def max_radius(self):
        return self._max_radius

    @property
    def min_height(self):
        return self._min_height

    @property
    def max_height(self):
        return self._max_height

    def set_active(self, item):

        if isinstance(item, int):

            if not (0 <= item < self.count):
                raise IndexError("The specified voxel index is out of range.")

            for i, voxel in enumerate(self._voxels):
                if i == item:
                    voxel.parent = self
                    voxel.material = UnityVolumeEmitter()
                else:
                    voxel.parent = None

        elif item == "all":
            for i, voxel in enumerate(self._voxels):
                voxel.parent = self
                voxel.material = UnityVoxelEmitter(i)

        else:
            raise ValueError("set_active() argument must be an index of type int or the string 'all'")

    def plot(self, title=None, voxel_values=None, ax=None, vmin=None, vmax=None, cmap=None):
        """
        Plots a voxel grid.

        If no voxel data values are provided, the plot is an outline of the grid in the r-z plane. If
        voxel values are provided, this method plots the voxel grid coloured by the voxel intensities.

        :param str title: The title of the plot.
        :param np.ndarray voxel_values: A 1D numpy array with length equal to the number of voxels
          in the collection.
        :param ax: The matplotlib Axes object on which the plot will be made. If None, this function
          generates a new plot.
        :param float vmin: The minimum value for the colour map.
        :param float vmax: The maximum value for the colour map.
        :param cmap: The matplotlib colour map to use for colouring the voxel intensities.
        """

        if voxel_values is not None:
            if not isinstance(voxel_values, (np.ndarray, list, tuple)):
                raise TypeError("Argument voxel_values should be a list/array of floats with length "
                                "equal to the number of voxels.")
            if not len(voxel_values) == self.count:
                raise TypeError("Argument voxel_values should be a list/array of floats with length "
                                "equal to the number of voxels.")

        patches = []
        for i in range(self.count):
            polygon = Polygon(self._voxels[i]._vertices, True)
            patches.append(polygon)

        p = PatchCollection(patches, cmap=cmap)
        if voxel_values is None:
            # Plot just the outlines of the grid cells
            p.set_edgecolor('black')
            p.set_facecolor('none')
        else:
            p.set_array(voxel_values)
            vmax = vmax or max(voxel_values)
            vmin = vmin or min(voxel_values)
            p.set_clim([vmin, vmax])

        if ax is None:
            fig, ax = plt.subplots()
        ax.add_collection(p)
        fig = plt.gcf()
        if voxel_values:
            fig.colorbar(p, ax=ax)
        ax.set_xlim(self.min_radius, self.max_radius)
        ax.set_ylim(self.min_height, self.max_height)
        ax.axis("equal")
        if title:
            pass
        elif self.name:
            title = self.name + " Voxel Grid"
        else:
            title = "Voxel Grid"
        ax.set_title(title)
        return ax

    def emissivities_from_function(self, emission_function, grid_samples=10):
        """
        Returns an array of sampled emissivities at each voxel location.

        Note that the results will be nonsense if you mix an emission function
        and VoxelCollection with incompatible symmetries.

        :param Function3D emission_function: Emission function to sample over.
        :param int grid_samples: Number of emission samples to average over.
        :rtype: np.ndarray
        """

        if not isinstance(emission_function, Function3D):
            raise TypeError("The emission_function argument must be of type Function2D.")

        emissivities = np.zeros(self.count)

        for i in range(self.count):

            voxel = self._voxels[i]
            num_triangles = voxel._triangles.shape[0]
            total_area = voxel.cross_sectional_area

            cumulative_areas = np.zeros(num_triangles)
            for triangle_j in range(num_triangles):
                u1 = voxel._vertices[1, 0] - voxel._vertices[0, 0]
                u2 = voxel._vertices[2, 0] - voxel._vertices[0, 0]
                v1 = voxel._vertices[1, 1] - voxel._vertices[0, 1]
                v2 = voxel._vertices[2, 1] - voxel._vertices[0, 1]
                triangle_area = cabs(u1*v2 - u2*v1)
                if triangle_j == 0:
                    cumulative_areas[triangle_j] = triangle_area
                else:
                    cumulative_areas[triangle_j] = cumulative_areas[triangle_j - 1] + triangle_area
            cumulative_areas /= total_area

            samples = 0
            for j in range(grid_samples):

                if num_triangles > 1:
                    tri_index = np.searchsorted(cumulative_areas, uniform())
                else:
                    tri_index = 0

                v1_i = voxel._triangles[tri_index, 0]
                v1 = new_point3d(voxel._vertices[v1_i, 0], 0.0, voxel._vertices[v1_i, 1])
                v2_i = voxel._triangles[tri_index, 1]
                v2 = new_point3d(voxel._vertices[v2_i, 0], 0.0, voxel._vertices[v2_i, 1])
                v3_i = voxel._triangles[tri_index, 2]
                v3 = new_point3d(voxel._vertices[v3_i, 0], 0.0, voxel._vertices[v3_i, 1])

                sample_point = point_triangle(v1, v2, v3)

                samples += emission_function(sample_point.x, 0, sample_point.z)

            samples /= grid_samples
            emissivities[i] = samples

        return emissivities


cdef class UnityVoxelEmitter(HomogeneousVolumeEmitter):

    cdef int voxel_id

    def __init__(self, int voxel_id):
        self.voxel_id = voxel_id
        super().__init__()

    cpdef Spectrum emission_function(self, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):
        spectrum.samples_mv[:] = 0.0
        spectrum.samples_mv[self.voxel_id] = 1.0
        return spectrum

