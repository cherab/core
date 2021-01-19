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

from cherab.core.utility import RecursiveDict
from cherab.core.utility.conversion import Cm3ToM3
from .utility import parse_adas2x_rate


def parse_adf21(beam_species, target_ion, target_charge, adf_file_path):
    """
    Opens and parses ADAS ADF21 data files.

    :param beam_species: Element object describing the beam species.
    :param target_ion: Element object describing the target ion species.
    :param target_charge: Ionisation level of the target species.
    :param adf_file_path: Path to ADF15 file from ADAS root.
    :return: Dictionary containing rates.
    """

    rate = RecursiveDict()
    with open(adf_file_path, 'r') as file:
        rate[beam_species][target_ion][target_charge] = parse_adas2x_rate(file, normalisation=Cm3ToM3.conversion_factor)
    return rate
