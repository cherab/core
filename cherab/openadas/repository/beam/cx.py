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
from cherab.core.utility import RecursiveDict
from cherab.core.atomic import Element
from ..utility import DEFAULT_REPOSITORY_PATH, valid_charge, encode_transition

"""
Utilities for managing the local rate repository - beam cx rate section.
"""


def add_beam_cx_rate(donor_ion, donor_metastable, receiver_ion, receiver_charge, transition, rate, repository_path=None):
    """
    Adds a single beam CX PEC to the repository.

    If adding multiple rate, consider using the update_beam_cx_rates() function
    instead. The update function avoid repeatedly opening and closing the rate
    files.

    :param donor_ion: Beam neutral atom (Element/Isotope) donating the electron.
    :param donor_metastable: Metastable/excited level of beam neutral atom.
    :param receiver_ion: Element/Isotope receiving the electron.
    :param receiver_charge: Charge of the receiving atom/ion.
    :param transition: Tuple containing (initial level, final level).
    :param rate: Beam CX PEC dictionary containing the following entries:

    |      'eb': array-like of size (N) with beam energy in eV/amu,
    |      'ti': array-like of size (M) with receiver ion temperature in eV,
    |      'ni': array-like of size (K) with plasma ion density in m^-3,
    |      'z': array-like of size (L) with plasma Z-effective,
    |      'b': array-like of size (J) with magnetic field strength in Tesla,
    |      'qeb': array-like of size (N) with CX PEC energy component in photon.m^3.s-1,
    |      'qti': array-like of size (M) with CX PEC temperature component in photon.m^3.s-1,
    |      'qni': array-like of size (K) with CX PEC density component in photon.m^3.s-1,
    |      'qz': array-like of size (L) with CX PEC Zeff component in photon.m^3.s-1,
    |      'qb': array-like of size (J) with CX PEC B-field component in photon.m^3.s-1,
    |      'qref': reference CX PEC in photon.m^3.s-1.
    |  The total beam CX PEC: q = qeb * qti * qni * qz * qb / qref^4.

    :param repository_path: Path to the atomic data repository.
    """

    update_beam_cx_rates({
        donor_ion: {
            receiver_ion: {
                receiver_charge: {
                    transition: {
                        donor_metastable: rate
                    }
                }
            }
        }
    }, repository_path)


def update_beam_cx_rates(rates, repository_path=None):
    """
    Updates the beam CX PEC files
    beam/cx/<donor_ion>/<receiver_ion>/<receiver_charge>.json
    in the atomic data repository.

    File contains multiple metastable-resolved rates, indexed by transition.

    :param rates: Dictionary in the form:

    |  { <donor_ion>: { <receiver_ion>: { <receiver_charge>: { <transition>: {<donor_metastable>: <rate>} } } } }, where
    |      <donor_ion> is the beam neutral atom (Element/Isotope) donating the electron.
    |      <donor_metastable> is the metastable/excited level of beam neutral atom.
    |      <receiver_ion> is the Element/Isotope receiving the electron.
    |      <receiver_charge> is the charge of the receiving atom/ion.
    |      <transition> is the tuple containing (initial level, final level).
    |      <rate> is the beam CX PEC dictionary containing the following entries:
    |          'eb': array-like of size (N) with beam energy in eV/amu,
    |          'ti': array-like of size (M) with receiver ion temperature in eV,
    |          'ni': array-like of size (K) with plasma ion density in m^-3,
    |          'z': array-like of size (L) with plasma Z-effective,
    |          'b': array-like of size (J) with magnetic field strength in Tesla,
    |          'qeb': array-like of size (N) with CX PEC energy component in photon.m^3.s-1,
    |          'qti': array-like of size (M) with CX PEC temperature component in photon.m^3.s-1,
    |          'qni': array-like of size (K) with CX PEC density component in photon.m^3.s-1,
    |          'qz': array-like of size (L) with CX PEC Zeff component in photon.m^3.s-1,
    |          'qb': array-like of size (J) with CX PEC B-field component in photon.m^3.s-1,
    |          'qref': reference CX PEC in photon.m^3.s-1.
    |      The total beam CX PEC: q = qeb * qti * qni * qz * qb / qref^4.

    :param repository_path: Path to the atomic data repository.
    """

    def sanitise_and_validate(data, x_key, x_name, y_key, y_name):
        """
        Sanitises and validates pairs of rate data arrays.

        Converts arrays to numpy arrays and check that the dimensions of the
        supplied data are consistent.

        Arrays are converted in place, the supplied dictionary is modified.

        :param data: Rate data dictionary.
        :param x_key: Key of independent variable data.
        :param x_name: Name of data array for error reporting.
        :param y_key: Key of dependent variable data.
        :param y_name: Name of data array for error reporting.
        """

        # convert to numpy arrays
        data[x_key] = np.array(data[x_key], np.float64)
        data[y_key] = np.array(data[y_key], np.float64)

        # check dimensions
        if data[x_key].ndim != 1:
            raise ValueError('The {} array must be a 1D array.'.format(x_name))

        if data[y_key].ndim != 1:
            raise ValueError('The {} array must be a 1D array.'.format(y_name))

        if data[x_key].shape != data[y_key].shape:
            raise ValueError('The {} and {} arrays have inconsistent lengths.'.format(x_name, y_name))

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH

    for donor, receivers in rates.items():
        for receiver, charge_states in receivers.items():
            for charge, transitions in charge_states.items():

                # sanitise and validate
                if not isinstance(donor, Element):
                    raise TypeError('The element must be an Element object.')

                if not isinstance(receiver, Element):
                    raise TypeError('The element must be an Element object.')

                if not valid_charge(receiver, charge):
                    raise ValueError('Charge state is larger than the number of protons in the element.')

                path = os.path.join(repository_path, 'beam/cx/{}/{}/{}.json'.format(donor.symbol.lower(), receiver.symbol.lower(), charge))

                # read in any existing rates
                try:
                    with open(path, 'r') as f:
                        content = RecursiveDict.from_dict(json.load(f))
                except FileNotFoundError:
                    content = RecursiveDict()

                # json keys are strings, must convert metastable key to integer
                for transition, metastables in content.items():
                    content[transition] = RecursiveDict({int(metastable): rate for metastable, rate in metastables.items()})

                # update content
                for transition, metastables in transitions.items():

                    # add/replace data for each metastable
                    transition_key = encode_transition(transition)
                    for metastable in metastables:

                        if not metastable >= 0:
                            raise ValueError('Donor metastable level cannot be less than zero.')

                        data = rates[donor][receiver][charge][transition][metastable]

                        # sanitise/validate data
                        data['qref'] = float(data['qref'])
                        sanitise_and_validate(data, 'eb', 'beam energy', 'qeb', 'beam energy effective rate')
                        sanitise_and_validate(data, 'ti', 'ion temperature', 'qti', 'ion temperature effective rate')
                        sanitise_and_validate(data, 'ni', 'ion density', 'qni', 'ion density effective rate')
                        sanitise_and_validate(data, 'z', 'Zeff', 'qz', 'Zeff effective rate')
                        sanitise_and_validate(data, 'b', 'B-field magnitude', 'qb', 'B-field magnitude effective rate')

                        content[transition_key][metastable] = {
                            'eb': data['eb'].tolist(),
                            'ti': data['ti'].tolist(),
                            'ni': data['ni'].tolist(),
                            'z': data['z'].tolist(),
                            'b': data['b'].tolist(),
                            'qref': data['qref'],
                            'qeb': data['qeb'].tolist(),
                            'qti': data['qti'].tolist(),
                            'qni': data['qni'].tolist(),
                            'qz': data['qz'].tolist(),
                            'qb': data['qb'].tolist(),
                        }

                # create directory structure if missing
                directory = os.path.dirname(path)
                if not os.path.isdir(directory):
                    os.makedirs(directory)

                # write new data
                with open(path, 'w') as f:
                    json.dump(content, f, indent=2, sort_keys=True)


def get_beam_cx_rates(donor_ion, receiver_ion, receiver_charge, transition, repository_path=None):
    """
    Reads a single beam CX PEC from the repository.

    :param donor_ion: Beam neutral atom (Element/Isotope) donating the electron.
    :param donor_metastable: Metastable/excited level of beam neutral atom.
    :param receiver_ion: Element/Isotope receiving the electron.
    :param receiver_charge: Charge of the receiving atom/ion.
    :param transition: Tuple containing (initial level, final level).
    :param repository_path: Path to the atomic data repository.

    :return rate: Beam CX PEC dictionary containing the following entries:

    |      'eb': 1D array of size (N) with beam energy in eV/amu,
    |      'ti': 1D array of size (M) with receiver ion temperature in eV,
    |      'ni': 1D array of size (K) with plasma ion density in m^-3,
    |      'z': 1D array of size (L) with plasma Z-effective,
    |      'b': 1D array of size (J) with magnetic field strength in Tesla,
    |      'qeb': 1D array of size (N) with CX PEC energy component in photon.m^3.s-1,
    |      'qti': 1D array of size (M) with CX PEC temperature component in photon.m^3.s-1,
    |      'qni': 1D array of size (K) with CX PEC density component in photon.m^3.s-1,
    |      'qz': 1D array of size (L) with CX PEC Zeff component in photon.m^3.s-1,
    |      'qb': 1D array of size (J) with CX PEC B-field component in photon.m^3.s-1,
    |      'qref': reference CX PEC in photon.m^3.s-1.
    |  The total beam CX PEC: q = qeb * qti * qni * qz * qb / qref^4.

    """

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH
    path = os.path.join(repository_path, 'beam/cx/{}/{}/{}.json'.format(donor_ion.symbol.lower(), receiver_ion.symbol.lower(), receiver_charge))
    try:
        with open(path, 'r') as f:
            content = json.load(f)
        rates = content[encode_transition(transition)]
    except (FileNotFoundError, KeyError):
        raise RuntimeError('Requested beam CX effective emission rates (donor={}, receiver={}, charge={}, transition={})'
                           ' are not available.'.format(donor_ion.symbol, receiver_ion.symbol, receiver_charge, transition))

    # sanitise data and convert to (more useful) numpy arrays rather than lists
    for rate in rates.values():
        rate['eb'] = np.array(rate['eb'], np.float64)
        rate['ti'] = np.array(rate['ti'], np.float64)
        rate['ni'] = np.array(rate['ni'], np.float64)
        rate['z'] = np.array(rate['z'], np.float64)
        rate['b'] = np.array(rate['b'], np.float64)
        rate['qref'] = float(rate['qref'])
        rate['qeb'] = np.array(rate['qeb'], np.float64)
        rate['qti'] = np.array(rate['qti'], np.float64)
        rate['qni'] = np.array(rate['qni'], np.float64)
        rate['qz'] = np.array(rate['qz'], np.float64)
        rate['qb'] = np.array(rate['qb'], np.float64)

    return [(int(metastable), rate) for metastable, rate in rates.items()]
