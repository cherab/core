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
from cherab.core.utility import RecursiveDict
from cherab.core.utility.conversion import Cm3ToM3, PerCm3ToPerM3
from .utility import readvalues


def parse_adf12(donor_ion, donor_metastable, receiver_ion, receiver_charge, adf_file_path):
    """
    Opens and parses ADAS ADF12 data files.

    :param donor_ion: The donor ion element described by the rate file.
    :param donor_metastable: The donor ion metastable level.
    :param receiver_ion: The receiver ion element described by the rate file.
    :param receiver_charge: The receiver ion charge state described by the rate file.
    :param adf_file_path: Path to ADF15 file from ADAS root.
    :return: Dictionary containing rates.
    """

    rates = RecursiveDict()

    with open(adf_file_path, 'r') as file:

        rate_count = int(file.readline()[3:5])
        for i in range(rate_count):

            # parse block
            transition, rate = _parse_block(file)

            # add to repository update dictionary, converting density from cm^-3 to m^-3
            rates[donor_ion][receiver_ion][receiver_charge][transition][donor_metastable] = {
                'eb': np.array(rate['ENER'], np.float64),
                'ti': np.array(rate['TIEV'], np.float64),
                'ni': PerCm3ToPerM3.to(np.array(rate['DENSI'], np.float64)),
                'z': np.array(rate['ZEFF'], np.float64),
                'b': np.array(rate['BMAG'], np.float64),

                'qeb': Cm3ToM3.to(np.array(rate['QENER'], np.float64)),
                'qti': Cm3ToM3.to(np.array(rate['QTIEV'], np.float64)),
                'qni': Cm3ToM3.to(np.array(rate['QDENSI'], np.float64)),
                'qz': Cm3ToM3.to(np.array(rate['QZEFF'], np.float64)),
                'qb': Cm3ToM3.to(np.array(rate['QBMAG'], np.float64)),

                'ebref': rate['EBREF'],
                'tiref': rate['TIREF'],
                'niref': PerCm3ToPerM3.to(rate['NIREF']),
                'zref': rate['ZEREF'],
                'bref': rate['BREF'],
                'qref': Cm3ToM3.to(rate['QEFREF'])
            }

    return rates


def _parse_block(file):
    """
    Reads and parses an ADF12 rate block from an IO stream.

    :param file: Text stream seeked to the start of the block.
    :return: Tuple containing (transition tuple, rate data dictionary).
    """

    # header
    line = file.readline()
    transition = (int(line[38:40]), int(line[41:43]))

    rate = {}

    # reference value section
    rate['QEFREF'] = readvalues(file, 1, 6)[0]
    ebref, tiref, niref, zeref, bref = readvalues(file, 5, 6)
    rate['EBREF'] = ebref
    rate['TIREF'] = tiref
    rate['NIREF'] = niref
    rate['ZEREF'] = zeref
    rate['BREF'] = bref

    # rate data section
    nbeam, nti, ndi, nze, nb = readvalues(file, 5, 6, type=int)
    rate['ENER'] = readvalues(file, 24, 6)[0:nbeam]
    rate['QENER'] = readvalues(file, 24, 6)[0:nbeam]
    rate['TIEV'] = readvalues(file, 12, 6)[0:nti]
    rate['QTIEV'] = readvalues(file, 12, 6)[0:nti]
    rate['DENSI'] = readvalues(file, 24, 6)[0:ndi]
    rate['QDENSI'] = readvalues(file, 24, 6)[0:ndi]
    rate['ZEFF'] = readvalues(file, 12, 6)[0:nze]
    rate['QZEFF'] = readvalues(file, 12, 6)[0:nze]
    rate['BMAG'] = readvalues(file, 12, 6)[0:nb]
    rate['QBMAG'] = readvalues(file, 12, 6)[0:nb]

    return transition, rate
