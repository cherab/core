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

import numpy as np

from raysect.core.math.polygon cimport triangulate2d


cdef class PolygonMask2D(Function2D):
    """
    A 2D mask defined by a simple n-sided closed polygon.

    Inherits from Function2D, implements `__call__(x, y)`.

    This 2D function returns 1.0 if the (x, y) point lies inside the polygon
    and 0.0 outside.

    The mesh is specified as a set of 2D vertices supplied as an Nx2 numpy
    array or a suitably sized sequence that can be converted to a numpy array.

    The vertex list must define a closed polygon without self intersections -
    a mathematically "simple" polygon.

    .. code-block:: pycon

       >>> from cherab.core.math import PolygonMask2D
       >>>
       >>> fp = PolygonMask2D([[0, 0], [1, 0], [1, 1], [0, 1]])
       >>>
       >>> fp(0.5, 0.5)
       1.0
       >>> fp(-0.5, 0.5)
       0.0
    """

    def __init__(self, object vertices not None):

        # use numpy arrays to store data internally
        vertices = np.array(vertices, dtype=np.float64)

        # triangulate polygon
        triangles = triangulate2d(vertices)

        # generate data
        data = np.ones(triangles.shape[0])

        self._mesh = Discrete2DMesh(vertices, triangles, data, limit=False, default_value=0.0)

    cdef double evaluate(self, double x, double y) except? -1e999:
        return self._mesh.evaluate(x, y)



