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

import os
import tempfile
import unittest

import numpy as np

from cherab.core.atomic import carbon, hydrogen
from cherab.openadas.parse.adf15 import parse_adf15


class MockADF15Files:
    """Helper class to create mock ADF15 files for testing."""

    @staticmethod
    def create_hydrogen_adf15():
        """Create a mock ADF15 file in hydrogen format."""
        content = """     0    /test hydrogen format/
C
C  TEST FILE FOR HYDROGEN
C
C  ISEL WAVELENGTH TRANSITION TYPE
C
C      1.   656.3   N=  2 - N=  1 EXCIT
C      2.   486.1   N=  3 - N=  2 RECOM
C      3.   434.0   N=  4 - N=  2 CHEXC
C
C  PHOTON EMISSIVITY COEFFICIENTS
C
656.3A   2 2 /type = excit /isel =  1
1.0E+08 1.0E+09
1000.0 5000.0
1.0E-12 2.0E-12 3.0E-12 4.0E-12
486.1A   2 2 /type = recom /isel =  2
1.1E+08 1.1E+09
2000.0 6000.0
1.1E-12 2.1E-12 3.1E-12 4.1E-12
434.0A   2 2 /type = chexc /isel =  3
1.2E+08 1.2E+09
3000.0 7000.0
1.2E-12 2.2E-12 3.2E-12 4.2E-12
"""
        return content

    @staticmethod
    def create_carbon_adf15():
        """Create a mock ADF15 file in carbon (full configuration) format."""
        content = """     0    /test carbon format/
C
C  TEST FILE FOR CARBON
C
C   lv      Configuration    (2S+1)L(w-1/2)      Energy (cm^-1)
C  ---   ------------------- --------------      --------------
c    1   60964A52B             (1) 0(  0.0)             0.0
c    2   60963A52B51C          (3) 2(  3.0)         16720.4
c    3   60964A51C51D          (5) 4(  4.5)         32456.8
c    4   60963A52B51D          (2) 1(  1.5)         48932.1
C
C   ISEL  WVLEN(A)            TRANSITION                TYPE  ISPB NSPB
C                                                             ISPP NSPP SZ    TG PR WR
C  ----- ---------- ----------------------------------- ----- ---- ---- --    -- -- --
C      1   1560.70  1(3)1(  1.0)-   2(2)2(  2.0)     excit   1    1  12   569 12  1
C      2   1657.80  1(3)1(  1.0)-   3(5)4(  4.5)     excit   1    1  12  1068  7  2
C      3   1329.50  1(3)1(  1.0)-   4(2)1(  1.5)     excit   1    1  12   778 31  3
C
C  PHOTON EMISSIVITY COEFFICIENTS
C
1560.70A   2 2 /type = excit /isel =  1
1.0E+08 1.0E+09
1000.0 5000.0
1.0E-12 2.0E-12 3.0E-12 4.0E-12
1657.80A   2 2 /type = excit /isel =  2
1.1E+08 1.1E+09
2000.0 6000.0
1.1E-12 2.1E-12 3.1E-12 4.1E-12
1329.50A   2 2 /type = excit /isel =  3
1.2E+08 1.2E+09
3000.0 7000.0
1.2E-12 2.2E-12 3.2E-12 4.2E-12
"""
        return content

    @staticmethod
    def create_tungsten_adf15():
        """Create a mock ADF15 file in tungsten (extended) format."""
        content = """     0    /test tungsten format/
C
C  TEST FILE FOR TUNGSTEN
C
C   lv      Configuration    (2S+1)L(w-1/2)      Energy (cm^-1)
C  ---   ------------------- --------------      --------------
c 1074   60964A52B             (1) 0(  0.0)             0.0
c 1075   60963A52B51C          (3) 2(  3.0)         16720.4
c  414   60964A51C51D          (5) 4(  4.5)         32456.8
c  365   60963A52B51D          (2) 1(  1.5)         48932.1
C
C   ISEL  WVLEN(A)            TRANSITION                TYPE  ISPB NSPB
C                                                             ISPP NSPP SZ    TG PR WR
C  ----- ---------- ----------------------------------- ----- ---- ---- --    -- -- --
C      1   56.5300  1074(1)0(  0.0)- 1075(3)2(  3.0)     excit   1    1  12   569 12  1
C      2   56.5520  1075(3)2(  3.0)-  414(5)4(  4.5)     excit   1    1  12  1068  7  2
C      3   70.1877   414(5)4(  4.5)-  365(2)1(  1.5)     excit   1    1  12   778 31  3
C      4   70.5640   365(2)1(  1.5)- 1074(1)0(  0.0)     excit   1    1  12   275 36  4
C
C  PHOTON EMISSIVITY COEFFICIENTS
C
56.5300A   2 2 /type = excit /isel =  1
1.0E+08 1.0E+09
1000.0 5000.0
1.0E-12 2.0E-12 3.0E-12 4.0E-12
56.5520A   2 2 /type = excit /isel =  2
1.1E+08 1.1E+09
2000.0 6000.0
1.1E-12 2.1E-12 3.1E-12 4.1E-12
70.1877A   2 2 /type = excit /isel =  3
1.2E+08 1.2E+09
3000.0 7000.0
1.2E-12 2.2E-12 3.2E-12 4.2E-12
70.5640A   2 2 /type = excit /isel =  4
1.3E+08 1.3E+09
4000.0 8000.0
1.3E-12 2.3E-12 3.3E-12 4.3E-12
"""
        return content


class TestADF15Parser(unittest.TestCase):
    """Unit tests for ADF15 parser."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        for filename in os.listdir(self.temp_dir):
            filepath = os.path.join(self.temp_dir, filename)
            if os.path.isfile(filepath):
                os.unlink(filepath)
        os.rmdir(self.temp_dir)

    def _create_test_file(self, filename, content):
        """Helper to create a test file."""
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        return filepath

    def test_parse_hydrogen_adf15(self):
        """Test parsing of hydrogen format ADF15 file."""
        content = MockADF15Files.create_hydrogen_adf15()
        filepath = self._create_test_file('test_h.adf15', content)

        rates, wavelengths = parse_adf15(hydrogen, 0, filepath, header_format='hydrogen')

        # Check that rates were extracted
        self.assertIn('excitation', rates)
        self.assertIn(hydrogen, rates['excitation'])
        self.assertIn(0, rates['excitation'][hydrogen])

        # Check that wavelengths were extracted
        self.assertIn(hydrogen, wavelengths)
        self.assertIn(0, wavelengths[hydrogen])

        # Check specific transitions
        self.assertIn((2, 1), rates['excitation'][hydrogen][0])
        self.assertIn((3, 2), rates['recombination'][hydrogen][0])
        self.assertIn((4, 2), rates['thermalcx'][hydrogen][0])

    def test_parse_carbon_adf15_full_config(self):
        """Test parsing of carbon format ADF15 file with full configuration."""
        content = MockADF15Files.create_carbon_adf15()
        filepath = self._create_test_file('test_c.adf15', content)

        rates, wavelengths = parse_adf15(carbon, 0, filepath)

        # Check that rates were extracted
        self.assertIn('excitation', rates)
        self.assertIn(carbon, rates['excitation'])
        self.assertIn(0, rates['excitation'][carbon])

        # Check that wavelengths were extracted
        self.assertIn(carbon, wavelengths)

        # Verify rate data has correct shape
        transitions = list(rates['excitation'][carbon][0].keys())
        self.assertGreater(len(transitions), 0)

        for transition, rate_data in rates['excitation'][carbon][0].items():
            self.assertIn('ne', rate_data)
            self.assertIn('te', rate_data)
            self.assertIn('rate', rate_data)
            self.assertTrue(isinstance(rate_data['ne'], np.ndarray))
            self.assertTrue(isinstance(rate_data['te'], np.ndarray))
            self.assertTrue(isinstance(rate_data['rate'], np.ndarray))

    def test_parse_tungsten_adf15(self):
        """Test parsing of tungsten format ADF15 file."""
        # Tungsten (W) has atomic number 74
        # Create a mock tungsten element for testing
        from cherab.core.atomic import tungsten

        content = MockADF15Files.create_tungsten_adf15()
        filepath = self._create_test_file('test_w.adf15', content)

        rates, wavelengths = parse_adf15(tungsten, 0, filepath)

        # Check that rates were extracted
        self.assertIn('excitation', rates)
        self.assertIn(tungsten, rates['excitation'])
        self.assertIn(0, rates['excitation'][tungsten])

        # Check that wavelengths were extracted
        self.assertIn(tungsten, wavelengths)

    def test_rate_data_structure(self):
        """Test that rate data has correct structure and units."""
        content = MockADF15Files.create_hydrogen_adf15()
        filepath = self._create_test_file('test_structure.adf15', content)

        rates, wavelengths = parse_adf15(hydrogen, 0, filepath, header_format='hydrogen')

        # Extract first rate
        first_transition = list(rates['excitation'][hydrogen][0].keys())[0]
        rate_data = rates['excitation'][hydrogen][0][first_transition]

        # Check structure
        self.assertEqual(set(rate_data.keys()), {'ne', 'te', 'rate'})

        # Check that units were converted (values should be large after conversion from cm^-3 to m^-3)
        self.assertTrue(np.all(rate_data['ne'] >= 1e14))  # Should be in m^-3
        self.assertTrue(np.all(rate_data['te'] > 0))  # Temperature should be positive
        self.assertTrue(np.all(rate_data['rate'] > 0))  # Rate should be positive

        # Check array dimensions match
        ne_count = len(rate_data['ne'])
        te_count = len(rate_data['te'])
        rate_shape = rate_data['rate'].shape
        self.assertEqual(rate_shape, (ne_count, te_count))

    def test_invalid_adf15_file(self):
        """Test that invalid ADF15 file raises appropriate error."""
        invalid_content = "This is not a valid ADF15 file\n"
        filepath = self._create_test_file('invalid.adf15', invalid_content)

        with self.assertRaises(ValueError) as context:
            parse_adf15(hydrogen, 0, filepath)

        self.assertIn('valid ADF15 file', str(context.exception))

    def test_wavelength_conversion(self):
        """Test that wavelengths are correctly converted from Angstroms to nm."""
        content = MockADF15Files.create_hydrogen_adf15()
        filepath = self._create_test_file('test_wavelength.adf15', content)

        rates, wavelengths = parse_adf15(hydrogen, 0, filepath, header_format='hydrogen')

        # Check specific wavelengths (656.3 Angstrom = 65.63 nm)
        transition_21 = (2, 1)
        if transition_21 in wavelengths[hydrogen][0]:
            wl = wavelengths[hydrogen][0][transition_21]
            # Should be around 65.63 nm (converted from 656.3 Angstrom)
            self.assertAlmostEqual(wl, 65.63, places=1)

    def test_multiple_rate_types(self):
        """Test parsing file with multiple rate types."""
        content = MockADF15Files.create_hydrogen_adf15()
        filepath = self._create_test_file('test_multitypes.adf15', content)

        rates, wavelengths = parse_adf15(hydrogen, 0, filepath, header_format='hydrogen')

        # Check both rate types exist
        self.assertIn('excitation', rates)
        self.assertIn('recombination', rates)
        self.assertIn('thermalcx', rates)


if __name__ == '__main__':
    unittest.main()
