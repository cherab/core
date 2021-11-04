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
from cherab.core.utility.conversion import PerCm3ToPerM3


def parse_adas2x_rate(file, normalisation=1):
    """
    Read and parse data from the supplied adf21/22 file stream.

    :param file: A file stream.
    :param normalisation: Normalisation factor applied to rate coefficients. Equals to 1E-6
                          (cm3 to m3) for beam emission and beam stopping rates and 1 for beam
                          population coefficient.
    :return: A dictionary.
    """

    raw = {}

    line = file.readline()
    raw['ZT'] = int(line[3:5])                  # target ion charge (int)
    raw['SVREF'] = float(line[13:22])           # coefficient at reference conditions (float)
    raw['SPEC'] = line[29:31]                   # target element (string)
    raw['DATE'] = line[38:46]                   # date of calculation (string)
    raw['CODE'] = line[53:-1]                   # processing code (string)

    file.readline()  # jump the hyphen line

    line = file.readline()
    neb = int(line[1:5])
    ndt = int(line[6:10])
    raw['TREF'] = float(line[17:26])            # reference target temperature (float)

    file.readline()  # jump the hyphen line

    raw['EB'] = readvalues(file, neb, 8)        # beam energy coordinates (1D array)
    raw['DT'] = readvalues(file, ndt, 8)        # target density coordinates (1D array)

    file.readline()  # jump the hyphen line

    sv = np.zeros((neb, ndt))
    for index in range(ndt):
        sv[:, index] = readvalues(file, neb, 8)

    # coefficients at beam energy coordinates (first index) and target density coordinates (second index) (2D array)
    raw['SV'] = sv

    file.readline()  # jump the hyphen line

    line = file.readline()
    ntt = int(line[1:5])
    raw['EREF'] = float(line[12:21])            # reference beam energy (float)
    raw['DREF'] = float(line[28:37])            # reference target density (float)

    file.readline()  # jump the hyphen line

    raw['TT'] = readvalues(file, ntt, 8)        # target temperature coordinates (1D array)

    file.readline()  # jump the hyphen line

    raw['SVT'] = readvalues(file, ntt, 8)       # coefficients at target temperature coordinates (1D array)

    # return essential data and convert units from cm^3 to m^3
    return {
        'e': np.array(raw['EB'], np.float64),
        'n': PerCm3ToPerM3.to(np.array(raw['DT'], np.float64)),
        't': np.array(raw['TT'], np.float64),

        'sen': normalisation * np.array(raw['SV'], np.float64),
        'st': normalisation * np.array(raw['SVT'], np.float64),

        'eref': raw['EREF'],
        'nref': PerCm3ToPerM3.to(raw['DREF']),
        'tref': raw['TREF'],
        'sref': normalisation * raw['SVREF']
    }


def readvalues(file, nb_values, values_per_line, type=float):
    """
    Read and return a given number of values in a file, taking into account
    end of lines. The reading begins at the current read line of the file (which
    must be open to use this function). The read lines of the file are assumed
    to be shaped as following:
    a first useless character, then a given number of 10 characters values, and
    any other characters after (not read).

    :param file: file in which values have to be read
    :param nb_values: number of values to be read
    :param values_per_line: number of values per line on the file
    :param type: python type of the values to be returned
    :return: a numpy 1D array with the read values in the reading order.
    """
    nb_read = 0
    output = []

    while nb_read < nb_values:
        nb_read_line = nb_read % values_per_line
        if nb_read_line == 0:
            line = file.readline()

        output.append(type(line[1+nb_read_line*10:(nb_read_line+1)*10].replace('D', 'E')))
        nb_read += 1

    return np.array(output)