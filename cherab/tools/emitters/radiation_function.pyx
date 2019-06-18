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

from raysect.optical cimport Point3D, Vector3D, Spectrum, World, Ray, Primitive, AffineMatrix3D
from raysect.optical.material.emitter cimport InhomogeneousVolumeEmitter, NumericalIntegrator
from raysect.core.math.function cimport Function3D
from libc.math cimport M_PI
import cython


cdef class RadiationFunction(InhomogeneousVolumeEmitter):
    """
    A general purpose radiation material.

    Radiates power over 4 pi according to the supplied 3D radiation function. Note that this model
    ignores the spectral range of the observer. The power specified will be spread of the entire
    observable spectral range. Useful for calculating total radiated power loads on reactor wall
    components.
    
    :param Function3D radiation_function: A 3D radiation function that specifies the amount of radiation
      to be radiated at a given point, :math:`\phi(x, y, z)` [W/m^2].
    :param float vertical_offset: The vertical offset that will be applied to the radiation function,
      defaults to 0.
    :param float step: The scale length for integration of the radiation function.

    .. code-block:: pycon

       >>> from cherab.tools.emitters import RadiationFunction
       >>>
       >>> # define your own 3D radiation function and insert it into this class
       >>> radiation_emitter = RadiationFunction(rad_function_3d)
    """

    cdef:
        readonly double vertical_offset
        readonly Function3D radiation_function

    def __init__(self, Function3D radiation_function, vertical_offset=0.0, step=0.1):

        super().__init__(NumericalIntegrator(step=step))
        self.vertical_offset = vertical_offset
        self.radiation_function = radiation_function

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_local, AffineMatrix3D local_to_world):

        cdef double wvl_range = ray.max_wavelength - ray.min_wavelength
        spectrum.samples_mv[:] = self.radiation_function(point.x, point.y, point.z + self.vertical_offset) / (4 * M_PI * wvl_range * spectrum.bins)
        return spectrum
