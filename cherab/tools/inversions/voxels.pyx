
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

cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport floor
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from raysect.core cimport (Node, Point2D, Vector2D, Point3D, Vector3D, Primitive,
                           rotate_z, AffineMatrix3D, new_point2d, new_point3d,
                           new_vector2d, new_vector3d)
from raysect.core.math cimport triangulate2d, translate, rotate_basis, AffineMatrix3D
from raysect.core.math.cython.utility cimport winding2d, find_index, maximum, minimum, peak_to_peak
from raysect.core.math.random cimport uniform, point_triangle
from raysect.primitive cimport Mesh, Cylinder, Cone, Intersect, Subtract, Union
from raysect.optical.material cimport UnityVolumeEmitter, HomogeneousVolumeEmitter, Material
from raysect.optical cimport Spectrum, World, Primitive, Ray
from cherab.core.math.function cimport Function3D, autowrap_function3d
from cherab.tools.primitives.axisymmetric_mesh cimport axisymmetric_mesh_from_polygon


cdef double PI = 3.141592653589793


cdef class Voxel(Node):
    """
    A Voxel base class.

    Each Voxel is a Node in the scenegraph. Each Voxel type that
    inherits from this class defines its own geometry.

    :ivar float volume: The geometric volume of this voxel.
    """

    @property
    def volume(self):
        raise NotImplementedError()

    cpdef double emissivity_from_function(self, emission_function, int grid_samples=10):
        raise NotImplementedError()


cdef class AxisymmetricVoxel(Voxel):
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


    :param vertices: An Nx2 array specifying the voxel's polygon outline in the
      r-z plane.
    :param Node parent: The scenegraph to which this Voxel is attached.
    :param Material material: The emission material of this Voxel, defaults
      to a UnityVolumeEmitter() for weight matrix calculations.
    :param str primitive_type: Specifies the primitive type, can be either
      'mesh' or 'csg'. Defaults to the CSG representation.

    :ivar float volume: The geometric volume of this voxel.
    :ivar float cross_sectional_area: The cross sectional area of the voxel in
      the r-z plane.
    :ivar Point2D cross_section_centroid: The centroid of the voxel in
      the r-z plane.
    """

    cdef:
        double[:, ::1] _vertices
        int[:, ::1] _triangles
        Material _material

    def __init__(self, vertices, parent=None, material=None, primitive_type='csg'):

        cdef:
            int i

        super().__init__(parent=parent)

        self._material = material or UnityVolumeEmitter()

        num_vertices = len(vertices)
        if not num_vertices >= 3:
            raise TypeError('The AxisymmetricVoxel can only be specified by a polygon with at least 3 vertices.')

        self._vertices = np.empty((num_vertices, 2))
        for i, vertex in enumerate(vertices):
            if not isinstance(vertex, Point2D) and len(vertex) != 2:
                raise TypeError('The polygon vertices must be an Nx2 array of coordinates')
            if vertex[0] < 0:
                raise ValueError('The polygon vertices must be in the r-z plane.')
            self._vertices[i, 0] = vertex[0]
            self._vertices[i, 1] = vertex[1]

        # Check the polygon is clockwise, if not => reverse it.
        if not winding2d(self._vertices):
            self._vertices[:] = self._vertices[::-1]

        self._triangles = triangulate2d(self._vertices.base)

        if primitive_type == 'mesh':
            self._build_mesh()
        elif primitive_type == 'csg':
            if self._has_rectangular_cross_section():
                self._build_csg_from_rectangle()
            else:
                for triangle in self._triangles:
                    self._build_csg_from_triangle(self._vertices.base[triangle])
        else:
            raise ValueError("primitive_type should be 'mesh' or 'csg'")

    cdef void _build_mesh(self):
        """Build the Voxel out of triangular mesh elements."""
        cdef:
            int number_segments
            double radial_width
            Mesh mesh

        radial_width = peak_to_peak(self._vertices[:, 0])
        number_segments = int(floor(2 * PI * self.cross_section_centroid.x / radial_width))
        mesh = axisymmetric_mesh_from_polygon(self._vertices.base, number_segments)
        mesh.parent = self
        mesh.material = self._material

    cdef bint _has_rectangular_cross_section(self):
        """
        Test if the voxel has a rectangular cross section, and is aligned with
        the coordinate axes.
        """
        cdef:
            double distance_13, distance_24
            Vector2D side_12, xaxis, yaxis
            Point2D v1, v2, v3, v4

        if self._vertices.shape[0] != 4:
            return False
        # A rectangle (including a square, which is considered to have a
        # rectangular cross section too) is defined by having equal length
        # diagonals.
        v1, v2, v3, v4 = self.vertices
        distance_13 = v1.distance_to(v3)
        distance_24 = v2.distance_to(v4)
        if distance_13 != distance_24:
            return False
        # The rectangle should be aligned with the coordinate axes, i.e. the
        # edge from vertex 1 to 2 should be parallel to either the x or z axes.
        side_12 = v1.vector_to(v2)
        xaxis = new_vector2d(1, 0)
        yaxis = new_vector2d(0, 1)
        if side_12.dot(xaxis) != 0 and side_12.dot(yaxis) != 0:
            return False
        return True

    @cython.boundscheck(False)
    cdef void _build_csg_from_triangle(self, vertices):
        cdef:
            double box_rmax, box_rmin, box_zmax, box_zmin, cylinder_height
            double r1, r2, r3, z1, z2, z3
            Primitive outer_cylinder, inner_cylinder, bounding_ring
            Primitive cone13, cone12, cone23, voxel_element
            double[:, :] vs
            double[:] vertex_rs, vertex_zs

        if vertices.shape != (3, 2):
            raise ValueError("Vertices must be an array of 3 (x, z) coordinates")
        # Sort the vertices of the triangle in decreasing x
        # Vertex 1 is at largest x, vertex 2 is middle x, vertex 3 is smallest x
        vs = vertices.copy()
        # Need an additional copy when swapping memoryviews which take a reference
        # N. B. Using <= rather than < reproduces the numpy.argsort result, but is faster
        if vs[0, 0] <= vs[1, 0]:
            vs[0], vs[1] = vs[1], vs[0].copy()
        if vs[0, 0] <= vs[2, 0]:
            vs[0], vs[2] = vs[2], vs[0].copy()
        if vs[1, 0] <= vs[2, 0]:
            vs[1], vs[2] = vs[2], vs[1].copy()
        vertex_rs = vs[:, 0]
        vertex_zs = vs[:, 1]
        # Create a bounding ring around the vertices, with rectangular cross section
        box_rmax = maximum(vertex_rs)
        box_rmin = minimum(vertex_rs)
        box_zmax = maximum(vertex_zs)
        box_zmin = minimum(vertex_zs)
        cylinder_height = box_zmax - box_zmin
        outer_cylinder = Cylinder(radius=box_rmax, height=cylinder_height,
                                  transform=translate(0, 0, box_zmin))
        inner_cylinder = Cylinder(radius=box_rmin, height=cylinder_height,
                                  transform=translate(0, 0, box_zmin))
        bounding_ring = Subtract(outer_cylinder, inner_cylinder)

        @cython.cdivision(True)
        def create_cone(double rx, double zx, double ry, double zy):
            cdef:
                AffineMatrix3D transform
                double rcone, hcone
                Primitive cone
                Vector3D minusz_axis, y_axis

            if zx == zy:
                return None
            if rx == ry:
                return Cylinder(radius=rx, height=abs(zx - zy),
                                transform=translate(0, 0, min(zx, zy)))
            if rx < ry:
                raise ValueError('rx must be larger than ry')
            if zx > zy:
                minusz_axis = new_vector3d(0, 0, -1)
                y_axis = new_vector3d(0, 1, 0)
                transform = translate(0, 0, zx) * rotate_basis(minusz_axis, y_axis)
            else:
                transform = translate(0, 0, zx)
            rcone = rx
            hcone = abs(zx - zy) * rx / (rx - ry)
            cone = Cone(radius=rcone, height=hcone, transform=transform)
            return cone

        r1 = vertex_rs[0]
        r2 = vertex_rs[1]
        r3 = vertex_rs[2]
        z1 = vertex_zs[0]
        z2 = vertex_zs[1]
        z3 = vertex_zs[2]
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

    @cython.boundscheck(False)
    cdef void _build_csg_from_rectangle(self):
        cdef:
            double rmax, rmin, zmax, zmin, cylinder_height
            AffineMatrix3D cylinder_transform
            Primitive outer_cylinder, inner_cylinder, voxel

        rmax = maximum(self._vertices[:, 0])
        rmin = minimum(self._vertices[:, 0])
        zmax = maximum(self._vertices[:, 1])
        zmin = minimum(self._vertices[:, 1])
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
            vertices.append(new_point2d(self._vertices[i, 0], self._vertices[i, 1]))

        return vertices

    @property
    def cross_sectional_area(self):

        # Simple calculation of the polygon area using the shoelace algorithm
        # https://en.wikipedia.org/wiki/Shoelace_formula

        cdef:
            int num_vertices, i
            double area
            double[:] x, y

        num_vertices = self._vertices.shape[0]
        x = self._vertices[:, 0]
        y = self._vertices[:, 1]
        area = 0
        with cython.boundscheck(False):
            for i in range(num_vertices - 1):
                area += x[i] * y[i + 1] - x[i + 1] * y[i]
            area += x[num_vertices - 1] * y[0] - x[0] * y[num_vertices - 1]
        return abs(area) / 2

    @property
    def cross_section_centroid(self):

        # Calculation of the centroid of the cross section using the formula
        # given in "Polygon Area and Centroid", P. Bourke, 1988
        cdef:
            int num_vertices, i
            double cx, cy, area
            double[:] x, y
        num_vertices = self._vertices.shape[0]
        x = self._vertices[:, 0]
        y = self._vertices[:, 1]
        cx = 0
        cy = 0
        # We need the signed area for this calculation, so can't re-use
        # self.cross_sectional_area
        area = 0
        with cython.boundscheck(False):
            for i in range(num_vertices - 1):
                cx += (x[i] + x[i + 1]) * (x[i] * y[i + 1] - x[i + 1] * y[i])
                cy += (y[i] + y[i + 1]) * (x[i] * y[i + 1] - x[i + 1] * y[i])
                area += x[i] * y[i + 1] - x[i + 1] * y[i]
            cx += ((x[num_vertices - 1] + x[0])
                   * (x[num_vertices - 1] * y[0] - x[0] * y[num_vertices - 1]))
            cy += ((y[num_vertices - 1] + y[0])
                   * (x[num_vertices - 1] * y[0] - x[0] * y[num_vertices - 1]))
            area += x[num_vertices - 1] * y[0] - x[0] * y[num_vertices - 1]
        area /= 2
        cx /= (6 * area)
        cy /= (6 * area)
        return new_point2d(cx, cy)

    @property
    def volume(self):

        # return approximate cell volume
        try:
            return 2 * PI * self.cross_section_centroid.x * self.cross_sectional_area
        except ZeroDivisionError:
            # Thown in self.cross_section_centroid if cross sectional area is 0
            return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef double emissivity_from_function(self, emission_function, int grid_samples=10):
        """
        Calculate the average emissivity in the voxel.

        :param callable emission_function: a function defining the emissivity
            in (r, ϕ, z) space
        :param int grid_samples: the number of samples of the emissivitiy to use
            to calculate the average

        :return float emissivity: the average emissivity in the voxel cross section

        Note that while the emissivity function is a 3D function, for
        Axisymmetric voxels the return value should be independent of
        toroidal angle ϕ.
        """
        cdef:
            double[::1] cumulative_areas
            double x1, y1, x2, y2, x3, y3, triangle_area, total_area, emissivity
            int num_triangles, triangle_j, v1_i, v2_i, v3_i, tri_index
            Point3D v1_p, v2_p, v3_p, sample_point
            Function3D emiss_function

        emiss_function = autowrap_function3d(emission_function)

        # Sample uniformly over the cross section.
        # Raysect already allows us to uniformly sample over a triangle,
        # but not an arbitrary cross section, so split the shape into
        # triangles and sample over these. In order to uniformly sample
        # over the entire voxel cross section, the number of samples in
        # each triangle must be weighted according to the fraction of the
        # total cross sectional area it occupies.

        num_triangles = self._triangles.shape[0]
        total_area = self.cross_sectional_area

        # Get the area of each triangle in the polygon using the Shoelace formula
        cumulative_areas = np.empty(num_triangles)
        for triangle_j in range(num_triangles):
            v1_i = self._triangles[triangle_j, 0]
            v2_i = self._triangles[triangle_j, 1]
            v3_i = self._triangles[triangle_j, 2]
            x1 = self._vertices[v1_i, 0]
            y1 = self._vertices[v1_i, 1]
            x2 = self._vertices[v2_i, 0]
            y2 = self._vertices[v2_i, 1]
            x3 = self._vertices[v3_i, 0]
            y3 = self._vertices[v3_i, 1]
            triangle_area = 0.5 * abs(x1 * y2 + x2 * y3 + x3 * y1
                                      - x2 * y1 - x3 * y2 - x1 * y3)
            if triangle_j == 0:
                cumulative_areas[triangle_j] = triangle_area
            else:
                cumulative_areas[triangle_j] = (cumulative_areas[triangle_j - 1] + triangle_area)

        emissivity = 0
        for _ in range(grid_samples):

            # Sample a random triangle, with the probability of picking each
            # triangle weighted by its area
            if num_triangles > 1:
                # find_index returns the left index in the interval. Since we
                # always have cumulative_areas < total_area (apart from the final
                # element), we want the index one up from the left index
                tri_index = find_index(cumulative_areas, total_area * uniform()) + 1
            else:
                tri_index = 0

            # Sample at a random point within the triangle
            v1_i = self._triangles[tri_index, 0]
            v1_p = new_point3d(self._vertices[v1_i, 0], 0.0, self._vertices[v1_i, 1])
            v2_i = self._triangles[tri_index, 1]
            v2_p = new_point3d(self._vertices[v2_i, 0], 0.0, self._vertices[v2_i, 1])
            v3_i = self._triangles[tri_index, 2]
            v3_p = new_point3d(self._vertices[v3_i, 0], 0.0, self._vertices[v3_i, 1])

            sample_point = point_triangle(v1_p, v2_p, v3_p)

            emissivity += emiss_function.evaluate(sample_point.x, 0, sample_point.z)

        emissivity /= grid_samples

        return emissivity


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

    def emissivities_from_function(self, emission_function, int grid_samples=10):
        """
        Returns an array of sampled emissivities at each voxel location.

        Note that the results will be nonsense if you mix an emission function
        and VoxelCollection with incompatible symmetries.

        :param Function3D emission_function: Emission function to sample over.
        :param int grid_samples: Number of emission samples to average over.
        :rtype: np.ndarray
        """
        cdef:
            double[::1] emissivities_mv
            int i
            Voxel voxel

        emission_function = autowrap_function3d(emission_function)

        emissivities = np.zeros(self.count)
        emissivities_mv = emissivities

        for i in range(self.count):
            voxel = self._voxels[i]
            emissivities_mv[i] = voxel.emissivity_from_function(
                emission_function, grid_samples
            )
        return emissivities


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
      Defaults to `primitive_type='csg'`.
    """

    def __init__(self, voxel_coordinates, name='', parent=None, transform=None,
                 active="all", primitive_type='csg'):

        cdef:
            AxisymmetricVoxel voxel
            double min_radius, max_radius, min_height, max_height

        super().__init__(name=name, parent=parent, transform=transform)

        min_radius = 1E999
        max_radius = 0
        min_height = 1E999
        max_height = -1E999

        self._voxels = []
        for voxel_vertices in voxel_coordinates:

            voxel = AxisymmetricVoxel(voxel_vertices, primitive_type=primitive_type)
            self._voxels.append(voxel)

            # Test and set extent values
            if minimum(voxel._vertices[:, 0]) < min_radius:
                min_radius = minimum(voxel._vertices[:, 0])
            if maximum(voxel._vertices[:, 0]) > max_radius:
                max_radius = maximum(voxel._vertices[:, 0])
            if minimum(voxel._vertices[:, 1]) < min_height:
                min_height = minimum(voxel._vertices[:, 1])
            if maximum(voxel._vertices[:, 1]) > max_height:
                max_height = maximum(voxel._vertices[:, 1])

        self._min_radius = min_radius
        self._max_radius = max_radius
        self._min_height = min_height
        self._max_height = max_height

        self.set_active(active)

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
        for voxel in self:
            polygon = Polygon([(v.x, v.y) for v in voxel.vertices], True)
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
        if p.get_array() is not None:
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
