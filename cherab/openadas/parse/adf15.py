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
from cherab.core.atomic import hydrogen, Element
from cherab.core.utility import RecursiveDict
from cherab.core.utility.conversion import Cm3ToM3, PerCm3ToPerM3


_L_LOOKUP = {
    0: 'S',
    1: 'P',
    2: 'D',
    3: 'F',
    4: 'G',
    5: 'H',
    6: 'I',
    7: 'K',
    8: 'L',
    9: 'M',
    10: 'N',
    11: 'O',
    12: 'Q',
    13: 'R',
}


def parse_adf15(element, charge, adf_file_path, header_format=None):
    """
    Opens and parses ADAS ADF15 data files.

    :param element: Element described by ADF file.
    :param charge: Charge state described by ADF file.
    :param adf_file_path: Path to ADF15 file from ADAS root.
    :return: Dictionary containing rates.
    """

    if not isinstance(element, Element):
        raise TypeError('The element must be an Element object.')

    charge = int(charge)

    with open(adf_file_path, "r") as file:

        # for check header line
        header = file.readline()
        if not re.match('^\s*(\d*) {4}/(.*)/?\s*$', header):
            raise ValueError('The specified path does not point to a valid ADF15 file.')

        # scrape transition information and wavelength
        # use simple electron configuration structure for hydrogen-like ions
        if header_format == 'hydrogen' or element == hydrogen:
            config = _scrape_metadata_hydrogen(file, element, charge)
        elif header_format == 'hydrogen-like' or element.atomic_number - charge == 1:
            config = _scrape_metadata_hydrogen_like(file, element, charge)
        else:
            config = _scrape_metadata_full(file, element, charge)

        # process rate data
        rates = RecursiveDict()
        for cls in ('excitation', 'recombination', 'thermalcx'):
            for element, charge_states in config[cls].items():
                for charge, transitions in charge_states.items():
                    for transition in transitions.keys():
                        block_num = config[cls][element][charge][transition]
                        rates[cls][element][charge][transition] = _extract_rate(file, block_num)

    wavelengths = config['wavelength']
    return rates, wavelengths


def _scrape_metadata_hydrogen(file, element, charge):
    """
    Scrapes transition and block information from the comments.
    """

    config = RecursiveDict()

    # start parsing from the beginning
    file.seek(0)
    lines = file.readlines()

    pec_index_header_match = '^C\s*ISEL\s*WAVELENGTH\s*TRANSITION\s*TYPE'
    while not re.match(pec_index_header_match, lines[0], re.IGNORECASE):
        lines.pop(0)
    index_lines = lines

    for i in range(len(index_lines)):

        pec_hydrogen_transition_match = '^C\s*([0-9]*)\.\s*([0-9]*\.[0-9]*)\s*N=\s*([0-9]*) - N=\s*([0-9]*)\s*([A-Z]*)'
        match = re.match(pec_hydrogen_transition_match, index_lines[i], re.IGNORECASE)
        if not match:
            continue

        block_num = int(match.groups()[0])
        wavelength = float(match.groups()[1]) / 10  # convert Angstroms to nm
        upper_level = int(match.groups()[2])
        lower_level = int(match.groups()[3])
        rate_type_adas = match.groups()[4]
        if rate_type_adas == 'EXCIT':
            rate_type = 'excitation'
        elif rate_type_adas == 'RECOM':
            rate_type = 'recombination'
        elif rate_type_adas == 'CHEXC':
            rate_type = 'cx_thermal'
        else:
            raise ValueError("Unrecognised rate type - {}".format(rate_type_adas))

        config[rate_type][element][charge][(upper_level, lower_level)] = block_num
        config["wavelength"][element][charge][(upper_level, lower_level)] = wavelength

    return config


def _scrape_metadata_hydrogen_like(file, element, charge):
    """
    Scrapes transition and block information from the comments.
    """

    config = RecursiveDict()

    # start parsing from the beginning
    file.seek(0)
    lines = file.readlines()

    pec_index_header_match = '^C\s*ISEL\s*WAVELENGTH\s*TRANSITION\s*TYPE'
    while not re.match(pec_index_header_match, lines[0], re.IGNORECASE):
        lines.pop(0)
    index_lines = lines

    for i in range(len(index_lines)):

        pec_full_transition_match = '^C\s*([0-9]*)\.\s*([0-9]*\.[0-9]*)\s*([0-9]*)[\(\)\.0-9\s]*-\s*([0-9]*)[\(\)\.0-9\s]*([A-Z]*)'
        match = re.match(pec_full_transition_match, index_lines[i], re.IGNORECASE)
        if not match:
            continue

        block_num = int(match.groups()[0])
        wavelength = float(match.groups()[1]) / 10  # convert Angstroms to nm
        upper_level = int(match.groups()[2])
        lower_level = int(match.groups()[3])
        rate_type_adas = match.groups()[4]
        if rate_type_adas == 'EXCIT':
            rate_type = 'excitation'
        elif rate_type_adas == 'RECOM':
            rate_type = 'recombination'
        elif rate_type_adas == 'CHEXC':
            rate_type = 'cx_thermal'
        else:
            raise ValueError("Unrecognised rate type - {}".format(rate_type_adas))

        config[rate_type][element][charge][(upper_level, lower_level)] = block_num
        config["wavelength"][element][charge][(upper_level, lower_level)] = wavelength

    return config


def _scrape_metadata_full(file, element, charge):
    """
    Scrapes transition and block information from the comments.
    """

    config = RecursiveDict()

    # start parsing from the beginning
    file.seek(0)
    lines = file.readlines()

    configuration_lines = []
    configuration_dict = {}

    configuration_header_match = '^C\s*Configuration\s*\(2S\+1\)L\(w-1/2\)\s*Energy \(cm\*\*-1\)$'
    while not re.match(configuration_header_match, lines[0], re.IGNORECASE):
        lines.pop(0)
    pec_index_header_match = '^C\s*ISEL\s*WAVELENGTH\s*TRANSITION\s*TYPE'
    while not re.match(pec_index_header_match, lines[0], re.IGNORECASE):
        configuration_lines.append(lines[0])
        lines.pop(0)
    index_lines = lines

    for i in range(len(configuration_lines)):

        configuration_string_match = "^C\s*([0-9]*)\s*((?:[0-9][SPDFG][0-9]\s)*)\s*\(([0-9]*\.?[0-9]*)\)([0-9]*)\(\s*([0-9]*\.?[0-9]*)\)"
        match = re.match(configuration_string_match, configuration_lines[i], re.IGNORECASE)
        if not match:
            continue

        config_id = int(match.groups()[0])
        electron_configuration = match.groups()[1].rstrip().lower()
        spin_multiplicity = match.groups()[2]  # (2S+1)
        total_orbital_quantum_number = _L_LOOKUP[int(match.groups()[3])]  # L
        total_angular_momentum_quantum_number = match.groups()[4]  # J

        configuration_dict[config_id] = (electron_configuration + " " + spin_multiplicity +
                                         total_orbital_quantum_number + total_angular_momentum_quantum_number)

    for i in range(len(index_lines)):

        pec_full_transition_match = '^C\s*([0-9]*)\.?\s*([0-9]*\.[0-9]*)\s*([0-9]*)[\(\)\.0-9\s]*-\s*([0-9]*)[\(\)\.0-9\s]*([A-Z]*)'
        match = re.match(pec_full_transition_match, index_lines[i], re.IGNORECASE)
        if not match:
            continue

        block_num = int(match.groups()[0])
        wavelength = float(match.groups()[1]) / 10  # convert Angstroms to nm
        upper_level_id = int(match.groups()[2])
        upper_level = configuration_dict[upper_level_id]
        lower_level_id = int(match.groups()[3])
        lower_level = configuration_dict[lower_level_id]
        rate_type_adas = match.groups()[4]
        if rate_type_adas == 'EXCIT':
            rate_type = 'excitation'
        elif rate_type_adas == 'RECOM':
            rate_type = 'recombination'
        elif rate_type_adas == 'CHEXC':
            rate_type = 'cx_thermal'
        else:
            raise ValueError("Unrecognised rate type - {}".format(rate_type_adas))

        config[rate_type][element][charge][(upper_level, lower_level)] = block_num
        config["wavelength"][element][charge][(upper_level, lower_level)] = wavelength

    return config


def _extract_rate(file, block_num):
    """
    Reads and converts the rate data for the specified block.
    """

    # search from start of file
    file.seek(0)

    wavelength_match = "^\s*[0-9]*\.[0-9]* ?a? +.*$"
    block_id_match = "^\s*[0-9]*\.[0-9]* ?a?\s*([0-9]*)\s*([0-9]*).*/type *= *([a-zA-Z]*).*/isel *= * ([0-9]*)$"

    for block in _group_by_block(file, wavelength_match):
        match = re.match(block_id_match, block[0], re.IGNORECASE)

        if not match:
            continue

        if int(match.groups()[3]) == block_num:
            # get number of n, T and rate data points:
            num_n = int(match.groups()[0])
            num_t = int(match.groups()[1])
            num_r = num_n * num_t

            block.pop(0)

            # Load density values
            nn = 0
            density = []
            while nn != num_n:
                next_line = block.pop(0)
                components = next_line.split()
                for value in components:
                    nn += 1
                    density.append(float(value))

            # Load temperature values
            nt = 0
            temperature = []
            while nt != num_t:
                next_line = block.pop(0)
                components = next_line.split()
                for value in components:
                    nt += 1
                    temperature.append(float(value))

            # Load rate values
            nr = 0
            rates = []
            while nr != num_r:
                next_line = block.pop(0)
                components = next_line.split()
                for value in components:
                    nr += 1
                    rates.append(float(value))

            density = np.array(density)
            temperature = np.array(temperature)
            rates = np.array(rates)
            rates = rates.reshape((num_n, num_t))

            # convert units from cm^-3 to m^-3
            density = PerCm3ToPerM3.to(density)
            rates = Cm3ToM3.to(rates)

            return {'ne': density, 'te': temperature, 'rate': rates}

    # If code gets to here, block wasn't found.
    raise RuntimeError('Block number {} was not found in the ADF15 file.'.format(block_num))


def _group_by_block(source_file, match_string):
    """
    Generator the splits the ADF15 file into blocks.

    Groups lines of file into blocks based on precursor '  6561.9A   24...'

    Note: comment section not filtered out of last block, don't over-read!
    """

    buffer = []
    for line in source_file:
        if re.match(match_string, line, re.IGNORECASE):
            if buffer:
                yield buffer
            buffer = [line]
        else:
            buffer.append(line)
    yield buffer
