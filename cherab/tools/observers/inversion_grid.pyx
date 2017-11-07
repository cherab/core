
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

from raysect.core cimport translate
from raysect.core.math.point cimport new_point2d, Point2D
from raysect.primitive.cylinder cimport Cylinder
from raysect.primitive.csg cimport Subtract
from raysect.optical.material.emitter cimport UnityVolumeEmitter


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

    def save(self, filename):

        name, extention = os.path.splitext(filename)

        if extention == '.json':
            file_handle = open(filename, 'w')
            json.dump(self.__getstate__(), file_handle, indent=2, sort_keys=True)

        else:
            raise NotImplementedError('Pickle serialisation has not been implemented yet.')

    def calculate_sensitivity(self, observer, pipeline, world):

        sensitivity_grid = SensitivityGrid(self, self.count)

        for i in range(self.count):

            p1, p2, p3, p4 = self.__getitem__(i)

            r_inner = p1.x
            r_outer = p3.x
            if r_inner > r_outer:
                t = r_inner
                r_inner = r_outer
                r_outer = t

            z_lower = p2.y
            z_upper = p1.y
            if z_lower > z_upper:
                t = z_lower
                z_lower = z_upper
                z_upper = t

            # TODO - switch to using CAD method such that reflections can be included automatically
            cylinder_height = z_upper - z_lower

            outer_cylinder = Cylinder(radius=r_outer, height=cylinder_height, transform=translate(0, 0, z_lower))
            inner_cylinder = Cylinder(radius=r_inner, height=cylinder_height, transform=translate(0, 0, z_lower))
            cell_emitter = Subtract(outer_cylinder, inner_cylinder, parent=world, material=UnityVolumeEmitter())

            observer.observe()

            sensitivity_grid.sensitivity[i] = pipeline.value.mean

            outer_cylinder.parent = None
            inner_cylinder.parent = None
            cell_emitter.parent = None

        return sensitivity_grid


cdef class SensitivityGrid:

    cdef:
        readonly RectangularGrid grid_geometry
        readonly int count
        readonly np.ndarray sensitivity
        double[:] _sensitivity_mv

    def __init__(self, grid, cell_count):

        self.grid_geometry = grid
        self.count = cell_count
        self.sensitivity = np.zeros(cell_count)
        self._sensitivity_mv = self.sensitivity


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
