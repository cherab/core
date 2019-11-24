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

import re
import numpy as np

from cherab.core.atomic import Element
from cherab.core.utility import RecursiveDict, Cm3ToM3, PerCm3ToPerM3


def parse_adf11(element, adf_file_path):
    """
    Reads contents of open adas adf11 files

    :param element: Element described by ADF file.
    :param adf_file_path: Path to ADF11 file from ADAS root.
    :return: temperature, density, rates as numpy array
    """

    if not isinstance(element, Element):
        raise TypeError('The element must be an Element object.')

    with open(adf_file_path, "r") as source_file:

        lines = source_file.readlines()  # read file contents by lines
        tmp = re.split("\s{2,}", lines[0].strip())  # split into relevant variables
        # exctract variables
        z_nuclear = int(tmp[0])
        n_densities = int(tmp[1])
        n_temperatures = int(tmp[2])
        z_min = int(tmp[3])
        z_max = int(tmp[4])
        element_name = tmp[5].strip('/').lower()
        projectname = tmp[6]

        if element.atomic_number != z_nuclear or element.name != element_name:
            raise ValueError("The requested element '{}' does not match the element description in the"
                             "specified ADF11 file, '{}'.".format(element.name, element_name))

        # check if it is a resolved file
        if re.match("\s*[0-9]+", lines[3]):  # is it unresolved?
            startsearch = 2
        else:
            startsearch = 4  # skip vectors with info about resolved states

        # get temperature and density vectors
        for i in range(startsearch, len(lines)):
            if re.match("^\s*C{0}-{2,}", lines[i]):
                tmp = re.sub("\n*\s+", "\t",
                             "".join(lines[startsearch:i]).strip())  # replace unwanted chars
                tmp = np.fromstring(tmp, sep="\t", dtype=float)  # put into nunpy array
                densities = tmp[:n_densities]  # read density values
                temperatures = tmp[n_densities:]  # read temperature values
                startsearch = i
                break

        # process rate data
        rates = RecursiveDict()

        # get beginnig and end of requested rates data block and add it to xarray
        blockrates_start = None
        blockrates_stop = None
        for i in range(startsearch, len(lines)):

            if re.match("^\s*C*-{2,}", lines[i]):  # is it a rates block header?

                # is it a first data block found?
                if not blockrates_start is None:
                    blockrates_stop = i  # end of the requested block

                    rates_table = re.sub("\n*\s+", "\t",
                                         "".join(lines[
                                                 blockrates_start:blockrates_stop]).strip())  # replace unwanted chars
                    rates_table = np.fromstring(rates_table, sep="\t",
                                                dtype=float).reshape((n_temperatures,
                                                                      n_densities))  # transform into an array

                    rates[element][ion_charge]['ne'] = densities
                    rates[element][ion_charge]['te'] = temperatures
                    rates[element][ion_charge]['rates'] = np.swapaxes(rates_table, 0, 1)

                    # if end of data block beak the loop or reassign start of data block for next stage
                    if re.match("^\s*C{1}-{2,}", lines[i]) or re.match("^\s*C{0,1}-{2,}", lines[i]) and \
                            re.match("^\s*C\n", lines[i + 1]):
                        break

                z1_pos = re.search("Z1\s*=*\s*[0-9]+\s*", lines[i]).group()  # get Z1 part
                ion_charge = int(re.sub("Z1[\s*=]", "", z1_pos))  # remove Z1 to avoid getting 1  later
                if not re.search("IGRD\s*=*\s*[0-9]+\s*", lines[i]) is None:  # get the IGRD part
                    igrd_pos = re.search("IGRD\s*=*\s*[0-9]+\s*", lines[i]).group()  # get the IGRD part
                else:
                    igrd_pos = "No spec"
                if not re.search("IPRT\s*=*\s*[0-9]+\s*", lines[i]) is None:
                    iptr_pos = re.search("IPRT\s*=*\s*[0-9]+\s*", lines[i]).group()  # get the IPRT part
                else:
                    iptr_pos = "No spec"
                blockrates_start = i + 1  # if block start not known, check if we are at the right position

        return rates
