
# Copyright 2016-2024 Euratom
# Copyright 2016-2024 United Kingdom Atomic Energy Authority
# Copyright 2016-2024 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

from cherab.core.atomic import Element
from cherab.core.utility import RecursiveDict
from .utility import DEFAULT_REPOSITORY_PATH, valid_charge


def add_line_power_rate(species, charge, rate, repository_path=None):
    """
    Adds a single line radiated power rate to the repository.

    If adding multiple rates, consider using the update_line_power_rates()
    function instead. The update function avoids repeatedly opening and closing
    the rate files.

    :param species: Plasma species (Element/Isotope).
    :param charge: Charge of the plasma species.
    :param rate: Line radiated power rate dictionary containing the following entries:

    |      'ne': array-like of size (N) with electron density in m^-3,
    |      'te': array-like of size (M) with electron temperature in eV,
    |      'rate': array-like of size (N, M) with line radiated power rate in W.m^3.

    :param repository_path: Path to the atomic data repository.
    """

    update_line_power_rates({
        species: {
            charge: rate
        }
    }, repository_path)


def update_line_power_rates(rates, repository_path=None):
    """
    Update the files for the line radiated power rates:
    /radiated_power/line/<species>.json
    in the atomic data repository.

    File contains multiple rates, indexed by the ion's charge state.

    :param rates: Dictionary in the form {<species>: {<charge>: <rate>}}, where

    |      <species> is the plasma species (Element/Isotope),
    |      <charge> is the charge of the plasma species,
    |      <rate> is the line radiated rate dictionary containing the following entries:
    |          'ne': array-like of size (N) with electron density in m^-3,
    |          'te': array-like of size (M) with electron temperature in eV,
    |          'rate': array-like of size (N, M) with line radiated power rate in W.m^3.

    :param repository_path: Path to the atomic data repository.
    """

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH

    for species, rate_data in rates.items():

        # sanitise and validate arguments
        if not isinstance(species, Element):
            raise TypeError('The species must be an Element object.')

        path = os.path.join(repository_path, 'radiated_power/line/{}.json'.format(species.symbol.lower()))

        _update_and_write_adf11(species, rate_data, path)


def add_continuum_power_rate(species, charge, rate, repository_path=None):
    """
    Adds a single continuum power rate to the repository.

    If adding multiple rates, consider using the update_continuum_power_rates()
    function instead. The update function avoids repeatedly opening and closing
    the rate files.

    :param species: Plasma species (Element/Isotope).
    :param charge: Charge of the plasma species.
    :param rate: Continuum power rate dictionary containing the following entries:

    |      'ne': array-like of size (N) with electron density in m^-3,
    |      'te': array-like of size (M) with electron temperature in eV,
    |      'rate': array-like of size (N, M) with continuum power rate in W.m^3.

    :param repository_path: Path to the atomic data repository.
    """

    update_line_power_rates({
        species: {
            charge: rate
        }
    }, repository_path)


def update_continuum_power_rates(rates, repository_path=None):
    """
    Update the files for the continuum power rates:
    /radiated_power/continuum/<species>.json
    in the atomic data repository.

    File contains multiple rates, indexed by ion's charge state.

    :param rates: Dictionary in the form {<species>: {<charge>: <rate>}}, where

    |      <species> is the plasma species (Element/Isotope),
    |      <charge> is the charge of the plasma species,
    |      <rate> is the continuum power rate dictionary containing the following entries:
    |          'ne': array-like of size (N) with electron density in m^-3,
    |          'te': array-like of size (M) with electron temperature in eV,
    |          'rate': array-like of size (N, M) with continuum power rate in W.m^3.

    :param repository_path: Path to the atomic data repository.
    """

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH

    for species, rate_data in rates.items():

        # sanitise and validate arguments
        if not isinstance(species, Element):
            raise TypeError('The species must be an Element object.')

        path = os.path.join(repository_path, 'radiated_power/continuum/{}.json'.format(species.symbol.lower()))

        _update_and_write_adf11(species, rate_data, path)


def add_cx_power_rate(species, charge, rate, repository_path=None):
    """
    Adds a single CX radiation power rate to the repository
    (charge exchage with neutral hydrogen).

    If adding multiple rates, consider using the update_cx_power_rates()
    function instead. The update function avoids repeatedly opening and closing
    the rate files.

    :param species: Plasma species (Element/Isotope).
    :param charge: Charge of the plasma species.
    :param rate: CX power rate dictionary containing the following entries:

    |      'ne': array-like of size (N) with electron density in m^-3,
    |      'te': array-like of size (M) with electron temperature in eV,
    |      'rate': array-like of size (N, M) with CX power rate in W.m^3.

    :param repository_path: Path to the atomic data repository.
    """

    update_line_power_rates({
        species: {
            charge: rate
        }
    }, repository_path)


def update_cx_power_rates(rates, repository_path=None):
    """
    Update the files for the CX radiation power rates
    (charge exchage with neutral hydrogen):
    /radiated_power/cx/<species>.json
    in the atomic data repository.

    File contains multiple rates, indexed by ion's charge state.

    :param rates: Dictionary in the form {<species>: {<charge>: <rate>}}, where

    |      <species> is the plasma species (Element/Isotope),
    |      <charge> is the charge of the plasma species,
    |      <rate> is the thermal CX power rate dictionary containing the following entries:
    |          'ne': array-like of size (N) with electron density in m^-3,
    |          'te': array-like of size (M) with electron temperature in eV,
    |          'rate': array-like of size (N, M) with thermal CX power rate in W.m^3.

    :param repository_path: Path to the atomic data repository.    
    """

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH

    for species, rate_data in rates.items():

        # sanitise and validate arguments
        if not isinstance(species, Element):
            raise TypeError('The species must be an Element object.')

        path = os.path.join(repository_path, 'radiated_power/cx/{}.json'.format(species.symbol.lower()))

        _update_and_write_adf11(species, rate_data, path)


def _update_and_write_adf11(species, rate_data, path):

    # read in any existing rates
    try:
        with open(path, 'r') as f:
            content = RecursiveDict.from_dict(json.load(f))
    except FileNotFoundError:
        content = RecursiveDict()

    for charge, rates in rate_data.items():

        if not valid_charge(species, charge):
            raise ValueError('The charge state is larger than the number of protons in the specified species.')

        # sanitise and validate rate data
        te = np.array(rates['te'], np.float64)
        ne = np.array(rates['ne'], np.float64)
        rate_table = np.array(rates['rates'], np.float64)

        if ne.ndim != 1:
            raise ValueError('Density array must be a 1D array.')

        if te.ndim != 1:
            raise ValueError('Temperature array must be a 1D array.')

        if (ne.shape[0], te.shape[0]) != rate_table.shape:
            raise ValueError('Electron temperature, density and rate data arrays have inconsistent sizes.')

        # update file content with new rate
        content[str(charge)] = {
            'te': te.tolist(),
            'ne': ne.tolist(),
            'rate': rate_table.tolist(),
        }

        # create directory structure if missing
        directory = os.path.dirname(path)
        if not os.path.isdir(directory):
            os.makedirs(directory)

        # write new data
        with open(path, 'w') as f:
            json.dump(content, f, indent=2, sort_keys=True)


def get_line_radiated_power_rate(element, charge, repository_path=None):
    """
    Reads the line radiated power rate for the given species and charge
    from the atomic data repository.

    :param element: Plasma species (Element/Isotope).
    :param charge: Charge of the plasma species.
    :param repository_path: Path to the atomic data repository.

    :return rate: Line radiated power rate dictionary containing the following entries:

    |      'ne': 1D array of size (N) with electron density in m^-3,
    |      'te': 1D array of size (M) with electron temperature in eV,
    |      'rate': 2D array of size (N, M) with line radiated power rate in W.m^3.

    """

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH

    path = os.path.join(repository_path, 'radiated_power/line/{}.json'.format(element.symbol.lower()))
    try:
        with open(path, 'r') as f:
            content = json.load(f)
        d = content[str(charge)]
    except (FileNotFoundError, KeyError):
        raise RuntimeError('Requested radiated power rate (element={}, charge={})'
                           ' is not available.'.format(element.symbol, charge))

    # convert to numpy arrays
    d['ne'] = np.array(d['ne'], np.float64)
    d['te'] = np.array(d['te'], np.float64)
    d['rate'] = np.array(d['rate'], np.float64)

    return d


def get_continuum_radiated_power_rate(element, charge, repository_path=None):
    """
    Reads the continuum power rate for the given species and charge
    from the atomic data repository.

    :param element: Plasma species (Element/Isotope).
    :param charge: Charge of the plasma species.
    :param repository_path: Path to the atomic data repository.

    :return rate: Continuum power rate dictionary containing the following entries:

    |      'ne': 1D array of size (N) with electron density in m^-3,
    |      'te': 1D array of size (M) with electron temperature in eV,
    |      'rate': 2D array of size (N, M) with continuum power rate in W.m^3.

    """

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH

    path = os.path.join(repository_path, 'radiated_power/continuum/{}.json'.format(element.symbol.lower()))
    try:
        with open(path, 'r') as f:
            content = json.load(f)
        d = content[str(charge)]
    except (FileNotFoundError, KeyError):
        raise RuntimeError('Requested radiated power rate (element={}, charge={})'
                           ' is not available.'.format(element.symbol, charge))

    # convert to numpy arrays
    d['ne'] = np.array(d['ne'], np.float64)
    d['te'] = np.array(d['te'], np.float64)
    d['rate'] = np.array(d['rate'], np.float64)

    return d


def get_cx_radiated_power_rate(element, charge, repository_path=None):
    """
    Reads the CX radiation power rate for the given species and charge
    from the atomic data repository.

    :param element: Plasma species (Element/Isotope).
    :param charge: Charge of the plasma species.
    :param repository_path: Path to the atomic data repository.

    :return rate: CX radiation power rate dictionary containing the following entries:

    |      'ne': 1D array of size (N) with electron density in m^-3,
    |      'te': 1D array of size (M) with electron temperature in eV,
    |      'rate': 2D array of size (N, M) with CX radiation power rate in W.m^3.

    """

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH

    path = os.path.join(repository_path, 'radiated_power/cx/{}.json'.format(element.symbol.lower()))
    try:
        with open(path, 'r') as f:
            content = json.load(f)
        d = content[str(charge)]
    except (FileNotFoundError, KeyError):
        raise RuntimeError('Requested radiated power rate (element={}, charge={})'
                           ' is not available.'.format(element.symbol, charge))

    # convert to numpy arrays
    d['ne'] = np.array(d['ne'], np.float64)
    d['te'] = np.array(d['te'], np.float64)
    d['rate'] = np.array(d['rate'], np.float64)

    return d
