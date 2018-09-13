
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

from libc.math cimport abs as cabs, floor

from raysect.core cimport Node, Point2D, Point3D, rotate_z
from raysect.optical import UnityVolumeEmitter
from raysect.primitive import Mesh


cdef double PI = 3.141592653589793


cdef class Voxel(Node):

    @property
    def volume(self):
        raise NotImplementedError()


cdef class AxisSymmetricVoxel(Voxel):

    @property
    def cross_sectional_area(self):
        raise NotImplementedError()


cdef class ToroidalAnnulusVoxel(AxisSymmetricVoxel):

    cdef:
        Point2D _lower_corner, _upper_corner
        list _voxel_primitives

    def __init__(self, Point2D lower_corner, Point2D upper_corner, material=None):

        if not isinstance(lower_corner, Point2D) or not isinstance(upper_corner, Point2D):
            raise TypeError('The ToroidalAnnulusVoxel can only be specified with two Point2D objects.')

        self._lower_corner = lower_corner
        self._upper_corner = upper_corner

        material = material or UnityVolumeEmitter()

        radius = (upper_corner.x + lower_corner.x)/2
        dr = upper_corner.x - lower_corner.x
        number_segments = floor(2 * PI * radius / dr)
        theta_adjusted = 360 / number_segments

        # Set of points in x-z plane
        p1a = Point3D(lower_corner.x, 0, lower_corner.y)  # corresponds to lower corner is x-z plane
        p2a = Point3D(lower_corner.x, 0, upper_corner.y)
        p3a = Point3D(upper_corner.x, 0, upper_corner.y)  # corresponds to upper corner in x-z plane
        p4a = Point3D(upper_corner.x, 0, lower_corner.y)

        # Set of points rotated away from x-z plane
        p1b = p1a.transform(rotate_z(theta_adjusted))
        p2b = p2a.transform(rotate_z(theta_adjusted))
        p3b = p3a.transform(rotate_z(theta_adjusted))
        p4b = p4a.transform(rotate_z(theta_adjusted))

        vertices = [[p1a.x, p1a.y, p1a.z], [p2a.x, p2a.y, p2a.z],
                    [p3a.x, p3a.y, p3a.z], [p4a.x, p4a.y, p4a.z],
                    [p1b.x, p1b.y, p1b.z], [p2b.x, p2b.y, p2b.z],
                    [p3b.x, p3b.y, p3b.z], [p4b.x, p4b.y, p4b.z]]

        triangles = [[1, 0, 3], [1, 3, 2],  # front face (x-z)
                     [7, 4, 5], [7, 5, 6],  # rear face (rotated out of x-z plane)
                     [5, 1, 2], [5, 2, 6],  # top face (x-y plane)
                     [3, 0, 4], [3, 4, 7],  # bottom face (x-y plane)
                     [4, 0, 5], [1, 5, 0],  # inner face (y-z plane)
                     [2, 3, 7], [2, 7, 6]]  # outer face (y-z plane)

        base_segment = Mesh(vertices=vertices, triangles=triangles, smoothing=False)

        # Construct annulus by duplicating and rotating base segment.
        for i in range(number_segments):
            theta_rotation = theta_adjusted * i
            segment = base_segment.instance(transform=rotate_z(theta_rotation), material=material, parent=self)
            self._voxel_primitives.append(segment)

    @property
    def lower_corner(self):
        return self._lower_corner

    @property
    def upper_corner(self):
        return self._upper_corner

    @property
    def cross_sectional_area(self):
        return cabs((self._upper_corner.x - self._lower_corner.x) * (self._upper_corner.y - self._lower_corner.y))

    @property
    def volume(self):

        cdef double voxel_area, voxel_radius

        voxel_area = self.cross_sectional_area
        voxel_radius = (self._upper_corner.x + self._lower_corner.x)/2

        # return approximate cell volume
        return 2 * PI * voxel_radius * voxel_area


cdef class VoxelCollection(Node):

    cdef:
        list _voxels

    def __getitem__(self, item):

        if not isinstance(item, int):
            raise TypeError("VoxelCollection can only be indexed with an integer.")

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


cdef class ToroidalVoxelGrid(VoxelCollection):

    cdef:
        Point2D _lower_point, _upper_point
        tuple _shape

    def __init__(self, Point2D lower_point, Point2D upper_point, tuple shape):

        self._lower_point = lower_point
        self._upper_point = upper_point
        self._shape = shape

    @property
    def lower_point(self):
        return self._lower_point

    @property
    def upper_point(self):
        return self._upper_point

    @property
    def shape(self):
        return self._shape


cdef class RectangularGrid:

    cdef:
        readonly str grid_id
        readonly int count
        readonly np.ndarray cell_data
        double[:,:,:] cell_data_mv

    def __init__(self, grid_id, cell_data):

        self.grid_id = grid_id
        self.cell_data = cell_data
        self.cell_data_mv = self.cell_data
        self.count = cell_data.shape[0]

    def __getitem__(self, item):

        cdef Point2D p1, p2, p3, p4

        if not isinstance(item, int):
            raise TypeError("Grid index must be of type int.")

        if not (0 <= item < self.count):
            raise IndexError("The specified grid index is out of range.")

        p1 = new_point2d(self.cell_data_mv[item, 0, 0], self.cell_data_mv[item, 0, 1])
        p2 = new_point2d(self.cell_data_mv[item, 1, 0], self.cell_data_mv[item, 1, 1])
        p3 = new_point2d(self.cell_data_mv[item, 2, 0], self.cell_data_mv[item, 2, 1])
        p4 = new_point2d(self.cell_data_mv[item, 3, 0], self.cell_data_mv[item, 3, 1])

        return p1, p2, p3, p4

    def __iter__(self):

        for i in range(self.count):
            yield self.__getitem__(i)

    def __getstate__(self):

        state = {
            'CHERAB_Object_Type': 'RectangularGrid',
            'Version': 1,
            'Count': self.count,
            'Grid_ID': self.grid_id,
        }

        cells = []
        for i in range(self.count):
            cell_description = {
                'p1': (self.cell_data_mv[i, 0, 0], self.cell_data_mv[i, 0, 1]),
                'p2': (self.cell_data_mv[i, 1, 0], self.cell_data_mv[i, 1, 1]),
                'p3': (self.cell_data_mv[i, 2, 0], self.cell_data_mv[i, 2, 1]),
                'p4': (self.cell_data_mv[i, 3, 0], self.cell_data_mv[i, 3, 1])
            }
            cells.append(cell_description)

        state['cells'] = cells

        return state

    def __setstate__(self, state):
        grid_id = state['Grid_ID']
        cell_data = np.asarray([[cell['p1'], cell['p2'], cell['p3'], cell['p4']]
                                for cell in state['cells']])
        self.__init__(grid_id, cell_data)

    def cell_area(self, cell_index):
        p1, p2, p3, p4 = self.__getitem__(cell_index)
        return cabs((p3.x - p2.x) * (p2.y - p1.y))

    def cell_volume(self, cell_index):
        p1, p2, p3, p4 = self.__getitem__(cell_index)

        cell_area = cabs((p3.x - p2.x) * (p2.y - p1.y))
        cell_radius = (p3.x + p2.x)/2

        # return approximate cell volume
        return 2 * np.pi * cell_radius * cell_area

    @property
    def grid_area(self):

        total_area = 0
        for i in range(self.count):
            total_area += self.cell_area(i)

        return total_area

    @property
    def grid_volume(self):

        total_volume = 0
        for i in range(self.count):
            total_volume += self.cell_volume(i)

        return total_volume

    def save(self, filename):

        name, extention = os.path.splitext(filename)

        if extention == '.json':
            file_handle = open(filename, 'w')
            json.dump(self.__getstate__(), file_handle, indent=2, sort_keys=True)

        else:
            raise NotImplementedError('Pickle serialisation has not been implemented yet.')


