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

import os
import json
from cherab.core.utility import RecursiveDict
from cherab.core.atomic import Element
from .utility import DEFAULT_REPOSITORY_PATH, valid_charge, encode_transition

"""
Utilities for managing the local rate repository - wavelength section.
"""


def add_wavelength(element, charge, transition, wavelength, repository_path=None):
    """
    Adds a single wavelength to the repository.

    If adding multiple wavelengths, consider using the update_wavelengths()
    function instead. The update function avoid repeatedly opening and closing
    the rate files.

    :param element:
    :param charge:
    :param transition:
    :param wavelength:
    :param repository_path:
    """

    update_wavelengths({
        element: {
            charge: {
                transition: wavelength
            }
        }
    }, repository_path)


def update_wavelengths(wavelengths, repository_path=None):

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH

    for element, charge_states in wavelengths.items():
        for charge, transitions in charge_states.items():

            # sanitise and validate
            if not isinstance(element, Element):
                raise TypeError('The element must be an Element object.')

            if not valid_charge(element, charge):
                raise ValueError('The charge state is larger than the number of protons in the element.')

            path = os.path.join(repository_path, 'wavelength/{}/{}.json'.format(element.symbol.lower(), charge))

            # read in any existing wavelengths
            try:
                with open(path, 'r') as f:
                    content = RecursiveDict.from_dict(json.load(f))
            except FileNotFoundError:
                content = RecursiveDict()

            # add/replace data for a transition
            for transition in transitions:
                key = encode_transition(transition)
                content[key] = float(wavelengths[element][charge][transition])

            # create directory structure if missing
            directory = os.path.dirname(path)
            if not os.path.isdir(directory):
                os.makedirs(directory)

            # write new data
            with open(path, 'w') as f:
                json.dump(content, f, indent=2, sort_keys=True)


def get_wavelength(element, charge, transition, repository_path=None):

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH
    path = os.path.join(repository_path, 'wavelength/{}/{}.json'.format(element.symbol.lower(), charge))
    try:
        with open(path, 'r') as f:
            content = json.load(f)
        return content[encode_transition(transition)]
    except (FileNotFoundError, KeyError):
        raise RuntimeError('Requested wavelength (element={}, charge={}, transition={})'
                           ' is not available.'.format(element.symbol, charge, transition))

