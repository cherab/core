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


def add_ionisation_rate(species, charge, rate, repository_path=None):
    """
    Adds a single ionisation rate to the repository.

    If adding multiple rates, consider using the update_ionisation_rates()
    function instead. The update function avoids repeatedly opening and closing
    the rate files.

    :param species: Plasma species (Element/Isotope).
    :param charge: Charge of the plasma species.
    :param rate: Ionisation rate dictionary containing the following entries:

    |      'ne': array-like of size (N) with electron density in m^-3,
    |      'te': array-like of size (M) with electron temperature in eV,
    |      'rate': array-like of size (N, M) with ionisation rate in m^3.s^-1.

    :param repository_path: Path to the atomic data repository.
    """

    update_ionisation_rates({
        species: {
            charge: rate
        }
    }, repository_path)


def update_ionisation_rates(rates, repository_path=None):
    """
    Updates the ionisation rate files `/ionisation/<species>.json`
    in atomic data repository.

    File contains multiple rates, indexed by the ion charge state.

    :param rates: Dictionary in the form {<species>: {<charge>: <rate>}}, where

    |      <species> is the plasma species (Element/Isotope),
    |      <charge> is the charge of the plasma species,
    |      <rate> is the ionisation rate dictionary containing the following entries:
    |          'ne': array-like of size (N) with electron density in m^-3,
    |          'te': array-like of size (M) with electron temperature in eV,
    |          'rate': array-like of size (N, M) with ionisation rate in m^3.s^-1.

    :param repository_path: Path to the atomic data repository.
    """

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH

    for species, rate_data in rates.items():

        # sanitise and validate arguments
        if not isinstance(species, Element):
            raise TypeError('The species must be an Element object.')

        path = os.path.join(repository_path, 'ionisation/{}.json'.format(species.symbol.lower()))

        _update_and_write_adf11(species, rate_data, path)


def add_recombination_rate(species, charge, rate, repository_path=None):
    """
    Adds a single recombination rate to the repository.

    If adding multiple rates, consider using the update_recombination_rates()
    function instead. The update function avoids repeatedly opening and closing
    the rate files.

    :param species: Plasma species (Element/Isotope).
    :param charge: Charge of the plasma species.
    :param rate: Recombination rate dictionary containing the following entries:

    |      'ne': array-like of size (N) with electron density in m^-3,
    |      'te': array-like of size (M) with electron temperature in eV,
    |      'rate': array-like of size (N, M) with recombination rate in m^3.s^-1.

    :param repository_path: Path to the atomic data repository.
    """

    update_recombination_rates({
        species: {
            charge: rate
        }
    }, repository_path)


def update_recombination_rates(rates, repository_path=None):
    """
    Updates the recombination rate files `/recombination/<species>.json`
    in the atomic data repository.

    File contains multiple rates, indexed by the ion charge state.

    :param rates: Dictionary in the form {<species>: {<charge>: <rate>}}, where

    |      <species> is the plasma species (Element/Isotope),
    |      <charge> is the charge of the plasma species,
    |      <rate> is the recombination rate dictionary containing the following entries:
    |          'ne': array-like of size (N) with electron density in m^-3,
    |          'te': array-like of size (M) with electron temperature in eV,
    |          'rate': array-like of size (N, M) with recombination rate in m^3.s^-1.

    :param repository_path: Path to the atomic data repository.
    """

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH

    for species, rate_data in rates.items():

        # sanitise and validate arguments
        if not isinstance(species, Element):
            raise TypeError('The species must be an Element object.')

        path = os.path.join(repository_path, 'recombination/{}.json'.format(species.symbol.lower()))

        _update_and_write_adf11(species, rate_data, path)


def add_thermal_cx_rate(donor_element, donor_charge, receiver_element, rate, repository_path=None):
    """
    Adds a single thermal charge exchange rate to the repository.

    If adding multiple rates, consider using the update_recombination_rates()
    function instead. The update function avoids repeatedly opening and closing
    the rate files.

    :param donor_element: Element donating the electron.
    :param donor_charge: Charge of the donating atom/ion.
    :param receiver_element: Element receiving the electron.
    :param receiver_charge: Charge of the receiving atom/ion.
    :param rate: Thermal CX rate dictionary containing the following entries:

    |      'ne': array-like of size (N) with electron density in m^-3,
    |      'te': array-like of size (M) with electron temperature in eV,
    |      'rate': array-like of size (N, M) with thermal CX rate in m^3.s^-1.

    :param repository_path: Path to the atomic data repository.
    """

    rates2update = RecursiveDict()
    rates2update[donor_element][donor_charge][receiver_element] = rate

    update_thermal_cx_rates(rates2update, repository_path)


def update_thermal_cx_rates(rates, repository_path=None):
    """
    Updates the thermal charge exchange rate files
    `/thermal_cx/<donor_element>/<donor_charge>/<receiver_element>.json`
    in the atomic data repository.

    File contains multiple rates, indexed by the ion charge state.

    :param rates: Dictionary in the form:

    |  { <donor_element>: { <donor_charge>: { <receiver_element>: { <donor_charge>: <rate> } } } }, where
    |      <donor_element> is the element donating the electron.
    |      <donor_charge> is the charge of the donating atom/ion.
    |      <receiver_element> is the element receiving the electron.
    |      <receiver_charge> is the charge of the receiving atom/ion.
    |      <rate> is the thermal CX rate dictionary containing the following entries:
    |          'ne': array-like of size (N) with electron density in m^-3,
    |          'te': array-like of size (M) with electron temperature in eV,
    |          'rate': array-like of size (N, M) with thermal CX rate in m^3.s^-1.

    :param repository_path: Path to the atomic data repository.
    """

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH

    for donor_element in rates.keys():
        for donor_charge in rates[donor_element].keys():
            for receiver_element, rate_data in rates[donor_element][donor_charge].items():

                # sanitise and validate arguments
                if not isinstance(receiver_element, Element):
                    raise TypeError('The receiver_element must be an Element object.')

                rate_path = 'thermal_cx/{0}/{1}/{2}.json'.format(donor_element.symbol.lower(),
                                                                 donor_charge, receiver_element.symbol.lower())
                path = os.path.join(repository_path, rate_path)

                _update_and_write_adf11(receiver_element, rate_data, path)


def _update_and_write_adf11(species, rate_data, path):

        # read in any existing rates
        try:
            with open(path, 'r') as f:
                content = RecursiveDict.from_dict(json.load(f))
        except FileNotFoundError:
            content = RecursiveDict()

        for charge, rates in rate_data.items():

            if not valid_charge(species, charge):
                raise ValueError('Charge state is larger than the number of protons in the specified species.')

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


def get_ionisation_rate(element, charge, repository_path=None):
    """
    Reads the ionisation rate for the given species and charge
    from the atomic data repository.

    :param element: Plasma species (Element/Isotope).
    :param charge: Charge of the plasma species.
    :param repository_path: Path to the atomic data repository.

    :return rate: Ionisation rate dictionary containing the following entries:

    |      'ne': 1D array of size (N) with electron density in m^-3,
    |      'te': 1D array of size (M) with electron temperature in eV,
    |      'rate': 2D array of size (N, M) with ionisation rate in m^3.s^-1.

    """

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH

    path = os.path.join(repository_path, 'ionisation/{}.json'.format(element.symbol.lower()))
    try:
        with open(path, 'r') as f:
            content = json.load(f)
        d = content[str(charge)]
    except (FileNotFoundError, KeyError):
        raise RuntimeError('Requested ionisation rate (element={}, charge={})'
                           ' is not available.'.format(element.symbol, charge))

    # convert to numpy arrays
    d['ne'] = np.array(d['ne'], np.float64)
    d['te'] = np.array(d['te'], np.float64)
    d['rate'] = np.array(d['rate'], np.float64)

    return d


def get_recombination_rate(element, charge, repository_path=None):
    """
    Reads the recombination rate for the given species and charge
    from the atomic data repository.

    :param element: Plasma species (Element/Isotope).
    :param charge: Charge of the plasma species.
    :param repository_path: Path to the atomic data repository.

    :return rate: Recombination rate dictionary containing the following entries:

    |      'ne': 1D array of size (N) with electron density in m^-3,
    |      'te': 1D array of size (M) with electron temperature in eV,
    |      'rate': 2D array of size (N, M) with recombination rate in m^3.s^-1.

    """

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH

    path = os.path.join(repository_path, 'recombination/{}.json'.format(element.symbol.lower()))
    try:
        with open(path, 'r') as f:
            content = json.load(f)
        d = content[str(charge)]
    except (FileNotFoundError, KeyError):
        raise RuntimeError('Requested recombination rate (element={}, charge={})'
                           ' is not available.'.format(element.symbol, charge))

    # convert to numpy arrays
    d['ne'] = np.array(d['ne'], np.float64)
    d['te'] = np.array(d['te'], np.float64)
    d['rate'] = np.array(d['rate'], np.float64)

    return d


def get_thermal_cx_rate(donor_element, donor_charge, receiver_element, receiver_charge, repository_path=None):
    """
    Reads the thermal charge exchange rate for the given species and charge
    from the atomic data repository.

    :param donor_element: Element donating the electron.
    :param donor_charge: Charge of the donating atom/ion.
    :param receiver_element: Element receiving the electron.
    :param receiver_charge: Charge of the receiving atom/ion.
    :param repository_path: Path to the atomic data repository.

    :return rate: Thermal CX rate dictionary containing the following entries:

    |      'ne': 1D array of size (N) with electron density in m^-3,
    |      'te': 1D array of size (M) with electron temperature in eV,
    |      'rate': 2D array of size (N, M) with thermal CX rate in m^3.s^-1.

    """

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH

    rate_path = 'thermal_cx/{0}/{1}/{2}.json'.format(donor_element.symbol.lower(), donor_charge,
                                                     receiver_element.symbol.lower())
    path = os.path.join(repository_path, rate_path)
    try:
        with open(path, 'r') as f:
            content = json.load(f)
        d = content[str(receiver_charge)]
    except (FileNotFoundError, KeyError):
        raise RuntimeError('Requested thermal charge-exchange rate (donor={}, donor charge={}, receiver={})'
                           ' is not available.'
                           ''.format(donor_element.symbol, donor_charge, receiver_element.symbol, receiver_charge))

    # convert to numpy arrays
    d['ne'] = np.array(d['ne'], np.float64)
    d['te'] = np.array(d['te'], np.float64)
    d['rate'] = np.array(d['rate'], np.float64)

    return d
