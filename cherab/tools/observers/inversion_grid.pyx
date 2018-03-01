
# Copyright 2014-2017 United Kingdom Atomic Energy Authority
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
import os
import json
import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from raysect.core cimport translate
from raysect.core.math.point cimport new_point2d, Point2D
from raysect.primitive.cylinder cimport Cylinder
from raysect.core.math import Discrete2DMesh
from cherab.core.math import AxisymmetricMapper
from cherab.tools.emitters.simple_power_emitter import SimplePowerEmitter


PI_4 = 4 * np.pi


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

    def cell_area(self, cell_index):
        p1, p2, p3, p4 = self.__getitem__(cell_index)
        return (p3.x - p2.x) * (p2.y - p1.y)

    def cell_volume(self, cell_index):
        p1, p2, p3, p4 = self.__getitem__(cell_index)

        cell_area = (p3.x - p2.x) * (p2.y - p1.y)
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


def load_inversion_grid(filename):

    file_handle = open(filename, 'r')
    grid_state = json.load(file_handle)

    if not grid_state['CHERAB_Object_Type'] == 'RectangularGrid':
        raise ValueError("The selected json file does not contain a valid RectangularGrid description.")
    if not grid_state['Version'] == 1.0:
        raise ValueError("The RectangularGrid description in the selected json file is out of date, version = {}.".format(grid_state['Version']))
    if not grid_state['Count'] == len(grid_state['cells']):
        raise ValueError("The count attribute does not equal the number of cells. RectangularGrid file is corrupted.")

    cell_list = grid_state['cells']
    cell_array = np.empty((len(cell_list), 4, 2))

    for i, cell in enumerate(cell_list):
        cell_array[i, 0, :] = cell['p1'][0], cell['p1'][1]
        cell_array[i, 1, :] = cell['p2'][0], cell['p2'][1]
        cell_array[i, 2, :] = cell['p3'][0], cell['p3'][1]
        cell_array[i, 3, :] = cell['p4'][0], cell['p4'][1]

    return RectangularGrid(grid_state["Grid_ID"], cell_array)


cdef class SensitivityMatrix:

    cdef:
        readonly str detector_uid, description
        readonly RectangularGrid grid_geometry
        readonly int count
        readonly np.ndarray sensitivity
        double[:] _sensitivity_mv

    def __init__(self, grid, detector_uid, description='', sensitivity=None):

        self.detector_uid = detector_uid
        self.description = description
        self.grid_geometry = grid
        self.count = grid.count

        if sensitivity is not None and isinstance(sensitivity, np.ndarray) and sensitivity.shape[0] == grid.count:
            self.sensitivity = sensitivity
        else:
            self.sensitivity = np.zeros(grid.count)
        self._sensitivity_mv = self.sensitivity

    def __getstate__(self):

        state = {
            'CHERAB_Object_Type': 'SensitivityMatrix',
            'Version': 1,
            'detector_uid': self.detector_uid,
            'description': self.description,
            'grid_uid': self.grid_geometry.grid_id,
            'count': self.count,
            'sensitivity': self.sensitivity.tolist(),
        }

        return state

    def plot(self, title=None):

        patches = []
        for i in range(self.count):
            polygon = Polygon(self.grid_geometry.cell_data[i], True)
            patches.append(polygon)

        p = PatchCollection(patches)
        p.set_array(self.sensitivity)

        fig, ax = plt.subplots()
        ax.add_collection(p)
        plt.xlim(1, 2.5)
        plt.ylim(-1.5, 1.5)
        title = title or self.detector_uid + " - Sensitivity"
        plt.title(title)


cdef class EmissivityGrid:

    cdef:
        readonly str description, case_id
        readonly RectangularGrid grid_geometry
        readonly int count
        readonly np.ndarray emissivities
        double[:] _emissivities_mv

    def __init__(self, grid, case_id='', description='', emissivities=None):

        self.case_id = case_id
        self.description = description
        self.grid_geometry = grid
        self.count = grid.count

        if emissivities is not None:
            self.emissivities = np.array(emissivities)
            if not len(emissivities) == grid.count:
                raise ValueError("Emissivity array must be of shape (N) where N is the number of grid cells. "
                                 "N = {} values given while the inversion grid has N = {}."
                                 "".format(len(emissivities), grid.count))
        else:
            self.emissivities = np.zeros(grid.count)

        self._emissivities_mv = self.emissivities

    def __getstate__(self):

        state = {
            'CHERAB_Object_Type': 'EmissivityGrid',
            'Version': 1,
            'description': self.description,
            'grid_uid': self.grid_geometry.grid_id,
            'count': self.count,
            'sensitivity': self.emissivities.tolist(),
        }

        return state

    def total_radiated_power(self):

        total_radiated_power = 0
        for i in range(self.count):
            total_radiated_power += self.emissivities[i] * self.grid_geometry.cell_volume(i) * PI_4

        return total_radiated_power

    def create_emitter(self, parent=None):

        cell_data = self.grid_geometry.cell_data
        n_cells = cell_data.shape[0]

        # Iterate through the arrays from MDS plus to pull out unique vertices
        unique_vertices = {}
        vertex_id = 0
        for ith_cell in range(n_cells):
            for j in range(4):
                    vertex = (cell_data[ith_cell, j, 0], cell_data[ith_cell, j, 1])
                    try:
                        unique_vertices[vertex]
                    except KeyError:
                        unique_vertices[vertex] = vertex_id
                        vertex_id += 1

        # Load these unique vertices into a numpy array for later use in Raysect's mesh interpolator object.
        num_vertices = len(unique_vertices)
        vertex_coords = np.zeros((num_vertices, 2), dtype=np.float64)
        for vertex, vertex_id in unique_vertices.items():
            vertex_coords[vertex_id, :] = vertex

        # Work out the extent of the mesh.
        rmin = cell_data[:,:,0].min()
        rmax = cell_data[:,:,0].max()
        zmin = cell_data[:,:,1].min()
        zmax = cell_data[:,:,1].max()

        # Number of triangles must be equal to number of rectangle centre points times 2.
        num_tris = n_cells * 2
        triangles = np.zeros((num_tris, 3), dtype=np.int32)

        tri_index = 0
        for ith_cell in range(n_cells):
            # Pull out the index number for each unique vertex in this rectangular cell.
            v1_id = unique_vertices[(cell_data[ith_cell, 0, 0], cell_data[ith_cell, 0, 1])]
            v2_id = unique_vertices[(cell_data[ith_cell, 1, 0], cell_data[ith_cell, 1, 1])]
            v3_id = unique_vertices[(cell_data[ith_cell, 2, 0], cell_data[ith_cell, 2, 1])]
            v4_id = unique_vertices[(cell_data[ith_cell, 3, 0], cell_data[ith_cell, 3, 1])]

            # Split the quad cell into two triangular cells.
            # Each triangle cell is mapped to the tuple ID (ix, iy) of its parent mesh cell.
            triangles[tri_index, :] = (v1_id, v2_id, v3_id)
            tri_index += 1
            triangles[tri_index, :] = (v3_id, v4_id, v1_id)
            tri_index += 1

        emissivities = np.zeros(len(self.emissivities)*2)
        for i in range(len(self.emissivities)):
            j = i * 2
            emissivities[j] = self.emissivities[i]
            emissivities[j+1] = self.emissivities[i]

        emission_function = AxisymmetricMapper(Discrete2DMesh(vertex_coords, triangles, emissivities, limit=False))
        emitter = SimplePowerEmitter(emission_function)

        return Cylinder(radius=rmax, height=zmax-zmin, transform=translate(0, 0, zmin),
                        material=emitter, parent=parent)

    def plot(self, title=None):

        patches = []
        for i in range(self.count):
            polygon = Polygon(self.grid_geometry.cell_data[i], True)
            patches.append(polygon)

        p = PatchCollection(patches)
        p.set_array(self.emissivities)

        fig, ax = plt.subplots()
        ax.add_collection(p)
        plt.xlim(1, 2.5)
        plt.ylim(-1.5, 1.5)
        title = title or self.case_id + " - Emissivity"
        plt.title(title)

