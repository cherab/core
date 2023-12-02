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

import numpy as np


DEF PI_POLARISATION = 0
DEF SIGMA_PLUS_POLARISATION = 1
DEF SIGMA_MINUS_POLARISATION = -1


cdef class ZeemanStructure():
    r"""
    Provides wavelengths and ratios of
    :math:`\pi`-/:math:`\sigma`-polarised Zeeman components for any given value of
    magnetic field strength.
    """

    cdef double[:, :] evaluate(self, double b, int polarisation):

        raise NotImplementedError("The evaluate() virtual method must be implemented.")

    def __call__(self, double b, str polarisation):

        if polarisation.lower() == 'pi':
            return np.asarray(self.evaluate(b, PI_POLARISATION))

        if polarisation.lower() == 'sigma_plus':
            return np.asarray(self.evaluate(b, SIGMA_PLUS_POLARISATION))

        if polarisation.lower() == 'sigma_minus':
            return np.asarray(self.evaluate(b, SIGMA_MINUS_POLARISATION))

        raise ValueError('Argument "polarisation" must be "pi", "sigma_plus" or "sigma_minus", {} given.'.fotmat(polarisation))
