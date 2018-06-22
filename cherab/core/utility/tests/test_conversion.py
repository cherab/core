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

import unittest

import numpy as np

from cherab.core.utility import conversion as cv


class FactorConversion(cv.BaseFactorConversion):
    conversion_factor = 42.


class TestConversion(unittest.TestCase):

    def setUp(self):
        self.array = np.array([1., 0., 2.5e6, 4.7e-3, 6.3e19, 9.2e-15])

    def tearDown(self):
        del self.array

    def test_evamutoms_to_array(self):
        converted_array = cv.EvAmuToMS.to(self.array.copy())
        for i in range(len(self.array)):
            self.assertAlmostEqual(converted_array[i], cv.EvAmuToMS.to(self.array[i]), delta=1e-10,
                                   msg='EvAmuToMS forward conversion failed to convert values given in a numpy array (failure from index {})!'.format(i))

    def test_evamutoms_inv_array(self):
        converted_array = cv.EvAmuToMS.inv(self.array.copy())
        for i in range(len(self.array)):
            self.assertAlmostEqual(converted_array[i], cv.EvAmuToMS.inv(self.array[i]), delta=1e-10,
                                   msg='EvAmuToMS backward conversion failed to convert values given in a numpy array (failure from index {})!'.format(i))

    def test_photontoj_to_array(self):
        wvl = 529.
        converted_array = cv.PhotonToJ.to(self.array.copy(), wvl)
        for i in range(len(self.array)):
            self.assertAlmostEqual(converted_array[i], cv.PhotonToJ.to(self.array[i], wvl), delta=1e-10,
                                   msg='PhotonToJ forward conversion failed to convert values given in a numpy array (failure from index {})!'.format(i))

    def test_photontoj_inv_array(self):
        wvl = 529.
        converted_array = cv.PhotonToJ.inv(self.array.copy(), wvl)
        for i in range(len(self.array)):
            self.assertAlmostEqual(converted_array[i], cv.PhotonToJ.inv(self.array[i], wvl), delta=1e-10,
                                   msg='PhotonToJ backward conversion failed to convert values given in a numpy array (failure from index {})!'.format(i))

    def test_factorconversion_to_array(self):
        converted_array = FactorConversion.to(self.array.copy())
        for i in range(len(self.array)):
            self.assertAlmostEqual(converted_array[i], FactorConversion.to(self.array[i]), delta=1e-10,
                                   msg='Forward conversion derived from BaseFactorConversion failed to convert values given in a numpy array (failure from index {})!'.format(i))

    def test_factorconversion_inv_array(self):
        converted_array = FactorConversion.inv(self.array.copy())
        for i in range(len(self.array)):
            self.assertAlmostEqual(converted_array[i], FactorConversion.inv(self.array[i]), delta=1e-10,
                                   msg='Backward conversion derived from BaseFactorConversion failed to convert values given in a numpy array (failure from index {})!'.format(i))



if __name__ == '__main__':
    unittest.main()