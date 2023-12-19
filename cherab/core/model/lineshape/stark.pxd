# cython: language_level=3

# Copyright 2016-2023 Euratom
# Copyright 2016-2023 United Kingdom Atomic Energy Authority
# Copyright 2016-2023 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

from raysect.optical cimport Spectrum
from cherab.core.math.integrators cimport Integrator1D
from cherab.core.model.lineshape.zeeman cimport ZeemanLineShapeModel


cpdef Spectrum add_lorentzian_line(double radiance, double wavelength, double lambda_1_2, Spectrum spectrum,
                                   Integrator1D integrator)


cdef class StarkBroadenedLine(ZeemanLineShapeModel):

    cdef:
        double _aij, _bij, _cij
        double _fwhm_poly_coeff_gauss[7]
        double _fwhm_poly_coeff_lorentz[7]
        double _weight_poly_coeff[6]
