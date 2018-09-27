
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

from raysect.core cimport Node, Point2D, Point3D, Vector3D, rotate_z, AffineMatrix3D
from raysect.core.math cimport triangulate2d
from raysect.core.math.cython.utility cimport winding2d
from raysect.primitive import Mesh
from raysect.optical import UnityVolumeEmitter
from raysect.optical cimport Spectrum, World, Primitive, Ray
from raysect.optical.material.emitter.homogeneous cimport HomogeneousVolumeEmitter


PI = 3.141592653589793


class Voxel(Node):

    @property
    def volume(self):
        raise NotImplementedError()


class AxisymmetricVoxel(Voxel):

    # cdef np.ndarray _vertices, _triangles

    def __init__(self, vertices, parent, material=None):

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

        # Check the polygon is clockwise, if not => reverse it.
        if not winding2d(self._vertices):
            self._vertices = self._vertices[::-1]

        # Generate summary statistics
        self.radius = self._vertices[:, 0].sum()/num_vertices
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

        self._triangles = triangulate2d(self._vertices)

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
        area += self._vertices[num_vertices, 0] * self._vertices[0, 1]
        for i in range(num_vertices - 1):
            area -= self._vertices[i, 1] * self._vertices[i+1, 0]
        area -= self._vertices[num_vertices, 1] * self._vertices[0, 0]

        return area / 2

    @property
    def volume(self):

        # return approximate cell volume
        return 2 * PI * self.radius * self.cross_sectional_area


class VoxelCollection(Node):

    # cdef:
    #     list _voxels

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
        raise NotImplementedError()

    def parent_all_voxels(self):

        for voxel in self._voxels:
            voxel.parent = self

    def unparent_all_voxels(self):

        for voxel in self._voxels:
            voxel.parent = None


class ToroidalVoxelGrid(VoxelCollection):

    # cdef:
    #     double _min_radius, _max_radius
    #     double _min_height, _max_height

    def __init__(self, voxel_coordinates, name='', parent=None, transform=None):

        super().__init__(name=name, parent=parent, transform=transform)

        self._min_radius = 1E999
        self._max_radius = 0
        self._min_height = 1E999
        self._max_height = -1E999

        self._voxels = []
        for voxel_vertices in voxel_coordinates:

            voxel = AxisymmetricVoxel(voxel_vertices, parent=self)
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

    def plot(self, title=None, voxel_values=None):

        if voxel_values is not None:
            if not isinstance(voxel_values, (np.ndarray, list, tuple)):
                raise TypeError("Argument voxel_values should be a list/array of floats with length "
                                "equal to the number of voxels.")
            if not len(voxel_values) == self.count:
                raise TypeError("Argument voxel_values should be a list/array of floats with length "
                                "equal to the number of voxels.")
        else:
            voxel_values = np.ones(self.count)

        patches = []
        for i in range(self.count):
            polygon = Polygon(self._voxels[i]._vertices, True)
            patches.append(polygon)

        p = PatchCollection(patches)
        p.set_array(voxel_values)

        fig, ax = plt.subplots()
        ax.add_collection(p)
        plt.xlim(self.min_radius, self.max_radius)
        plt.ylim(self.min_height, self.max_height)
        plt.axis("equal")
        title = title or self.name + " Voxel Grid"
        plt.title(title)


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

