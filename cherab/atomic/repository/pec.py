# Copyright 2016-2022 Euratom
# Copyright 2016-2022 United Kingdom Atomic Energy Authority
# Copyright 2016-2022 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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
import numpy as np
from cherab.core.utility import RecursiveDict
from cherab.core.atomic import Element
from .utility import DEFAULT_REPOSITORY_PATH, valid_charge, encode_transition

"""
Utilities for managing the local rate repository - PEC section.
"""


def add_pec_excitation_rate(element, charge, transition, rate, repository_path=None):
    """
    Adds a single PEC excitation rate to the repository.

    If adding multiple rate, consider using the update_pec_rates() function
    instead. The update function avoid repeatedly opening and closing the rate
    files.

    :param element: Plasma species (Element/Isotope).
    :param charge: Charge of the plasma species.
    :param transition: Tuple containing (initial level, final level).
    :param rate: Excitation PEC dictionary containing the following fields:

    |      'ne': array-like of size (N) with electron density in m^-3,
    |      'te': array-like of size (M) with electron temperature in eV,
    |      'rate': array-like of size (N, M) with excitation PEC in photon.m^3.s^-1.
    |      'reference': Optional data reference string.

    :param repository_path: Path to the atomic data repository.
    """

    update_pec_rates({
        'excitation': {
            element: {
                charge: {
                    transition: rate
                }
            }
        }
    }, repository_path)


def add_pec_recombination_rate(element, charge, transition, rate, repository_path=None):
    """
    Adds a single PEC recombination rate to the repository.

    If adding multiple rate, consider using the update_pec_rates() function
    instead. The update function avoid repeatedly opening and closing the rate
    files.

    :param element: Plasma species (Element/Isotope).
    :param charge: Charge of the plasma species.
    :param transition: Tuple containing (initial level, final level).
    :param rate: Recombination PEC dictionary containing the following fields:

    |      'ne': array-like of size (N) with electron density in m^-3,
    |      'te': array-like of size (M) with electron temperature in eV,
    |      'rate': array-like of size (N, M) with recombination PEC in photon.m^3.s^-1.
    |      'reference': Optional data reference string.

    :param repository_path: Path to the atomic data repository.
    """

    update_pec_rates({
        'recombination': {
            element: {
                charge: {
                    transition: rate
                }
            }
        }
    }, repository_path)


def add_pec_thermalcx_rate(element, charge, transition, rate, repository_path=None):
    """
    Adds a single PEC thermal charge exchange rate to the repository.

    If adding multiple rate, consider using the update_pec_rates() function
    instead. The update function avoid repeatedly opening and closing the rate
    files.

    :param element: Plasma species (Element/Isotope).
    :param charge: Charge of the plasma species.
    :param transition: Tuple containing (initial level, final level).
    :param rate: Thermal CX PEC dictionary containing the following fields:

    |      'ne': array-like of size (N) with electron density in m^-3,
    |      'te': array-like of size (M) with electron temperature in eV,
    |      'rate': array-like of size (N, M) with thermal CX PEC in photon.m^3.s^-1.
    |      'reference': Optional data reference string.

    :param repository_path: Path to the atomic data repository.
    """

    update_pec_rates({
        'thermalcx': {
            element: {
                charge: {
                    transition: rate
                }
            }
        }
    }, repository_path)


def update_pec_rates(rates, repository_path=None):
    """
    Updates the PEC files /pec/<class>/<element>/<charge>.json.
    in the atomic data repository.

    File contains multiple PECs, indexed by the transition.

    :param rates: Dictionary in the form:

    |  { <class>: { <element>: { <charge>: { <transition>: <pec> } } } }, where
    |      <class> is the one of the following PEC types: 'excitation', 'recombination', 'thermalcx'.
    |      <element> is the plasma species (Element/Isotope).
    |      <charge> is the charge of the plasma species.
    |      <transition> is the tuple containing (initial level, final level).
    |      <pec> is the PEC dictionary containing the following fields:
    |          'ne': array-like of size (N) with electron density in m^-3,
    |          'te': array-like of size (M) with electron temperature in eV,
    |          'rate': array-like of size (N, M) with PEC in photon.m^3.s^-1.
    |          'reference': Optional data reference string.

    :param repository_path: Path to the atomic data repository.
    """

    valid_classes = [
        'excitation',
        'recombination',
        'thermalcx'
    ]

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH

    for cls, elements in rates.items():
        for element, charge_states in elements.items():
            for charge, transitions in charge_states.items():

                # sanitise and validate
                cls = cls.lower()
                if cls not in valid_classes:
                    raise ValueError('Unrecognised pec rate class \'{}\'.'.format(cls))

                if not isinstance(element, Element):
                    raise TypeError('The element must be an Element object.')

                if not valid_charge(element, charge):
                    raise ValueError('Charge state is larger than the number of protons in the element.')

                path = os.path.join(repository_path, 'pec/{}/{}/{}.json'.format(cls, element.symbol.lower(), charge))

                # read in any existing rates
                try:
                    with open(path, 'r') as f:
                        content = RecursiveDict.from_dict(json.load(f))
                except FileNotFoundError:
                    content = RecursiveDict()

                # add/replace data for a transition
                for transition in transitions:
                    key = encode_transition(transition)
                    data = rates[cls][element][charge][transition]

                    # sanitise/validate data
                    data['ne'] = np.array(data['ne'], np.float64)
                    data['te'] = np.array(data['te'], np.float64)
                    data['rate'] = np.array(data['rate'], np.float64)

                    if data['ne'].ndim != 1:
                        raise ValueError('Density array must be a 1D array.')

                    if data['te'].ndim != 1:
                        raise ValueError('Temperature array must be a 1D array.')

                    if (data['ne'].shape[0], data['te'].shape[0]) != data['rate'].shape:
                        raise ValueError('Density, temperature and rate data arrays have inconsistent sizes.')

                    content[key] = {
                        'ne': data['ne'].tolist(),
                        'te': data['te'].tolist(),
                        'rate': data['rate'].tolist()
                    }
                    if 'reference' in data:
                        content[key]['reference'] = str(data['reference'])

                # create directory structure if missing
                directory = os.path.dirname(path)
                if not os.path.isdir(directory):
                    os.makedirs(directory)

                # write new data
                with open(path, 'w') as f:
                    json.dump(content, f, indent=2, sort_keys=True)


def get_pec_excitation_rate(element, charge, transition, repository_path=None):
    """
    Reads the excitation PEC from the repository for the given
    element, charge and transition.

    :param element: Plasma species (Element/Isotope).
    :param charge: Charge of the plasma species.
    :param transition: Tuple containing (initial level, final level).
    :param repository_path: Path to the atomic data repository.

    :return rate: Excitation PEC dictionary containing the following fields:

    |      'ne': 1D array of size (N) with electron density in m^-3,
    |      'te': 1D array of size (M) with electron temperature in eV,
    |      'rate': 2D array of size (N, M) with excitation PEC in photon.m^3.s^-1.
    |      'reference': Optional data reference string.
    """

    return _get_pec_rate('excitation', element, charge, transition, repository_path)


def get_pec_recombination_rate(element, charge, transition, repository_path=None):
    """
    Reads the recombination PEC from the repository for the given
    element, charge and transition.

    :param element: Plasma species (Element/Isotope).
    :param charge: Charge of the plasma species.
    :param transition: Tuple containing (initial level, final level).
    :param repository_path: Path to the atomic data repository.

    :return rate: Recombination PEC dictionary containing the following fields:

    |      'ne': 1D array of size (N) with electron density in m^-3,
    |      'te': 1D array of size (M) with electron temperature in eV,
    |      'rate': 2D array of size (N, M) with excitation PEC in photon.m^3.s^-1.
    |      'reference': Optional data reference string.
    """

    return _get_pec_rate('recombination', element, charge, transition, repository_path)


def get_pec_thermalcx_rate(element, charge, transition, repository_path=None):
    """
    Reads the thermal charge exchange PEC from the repository for the given
    element, charge and transition.

    :param element: Plasma species (Element/Isotope).
    :param charge: Charge of the plasma species.
    :param transition: Tuple containing (initial level, final level).
    :param repository_path: Path to the atomic data repository.

    :return rate: Thermal CX PEC dictionary containing the following fields:

    |      'ne': 1D array of size (N) with electron density in m^-3,
    |      'te': 1D array of size (M) with electron temperature in eV,
    |      'rate': 2D array of size (N, M) with excitation PEC in photon.m^3.s^-1.
    |      'reference': Optional data reference string.
    """

    return _get_pec_rate('thermalcx', element, charge, transition, repository_path)


def _get_pec_rate(cls, element, charge, transition, repository_path=None):

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH
    path = os.path.join(repository_path, 'pec/{}/{}/{}.json'.format(cls, element.symbol.lower(), charge))
    try:
        with open(path, 'r') as f:
            content = json.load(f)
        d = content[encode_transition(transition)]
    except (FileNotFoundError, KeyError):
        raise RuntimeError('Requested PEC rate (class={}, element={}, charge={}, transition={})'
                           ' is not available.'.format(cls, element.symbol, charge, transition))

    # convert to numpy arrays
    d['ne'] = np.array(d['ne'], np.float64)
    d['te'] = np.array(d['te'], np.float64)
    d['rate'] = np.array(d['rate'], np.float64)

    return d
