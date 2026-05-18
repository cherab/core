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
import numpy as np
from cherab.core.atomic import Element
from ..utility import DEFAULT_REPOSITORY_PATH, valid_charge

"""
Utilities for managing the local rate repository - beam population section.
"""


def add_beam_population_rate(beam_species, beam_metastable, target_ion, target_charge, rate, repository_path=None):
    """
    Adds a single beam population rate to the repository.

    :param beam_species: Beam neutral species (Element/Isotope).
    :param beam_metastable: Metastable level of beam neutral atom.
    :param target_ion: Target species (Element/Isotope).
    :param target_charge: Charge of the target species.
    :param rate: Beam population rate dictionary containing the following entries:

    |      'e': array-like of size (N) with interaction energy in eV/amu,
    |      'n': array-like of size (M) with target electron density in m^-3,
    |      't': array-like of size (K) with target electron temperature in eV,
    |      'sen': array-like of size (N, M) with dimensionless beam population rate energy component.
    |      'st': array-like of size (K) with dimensionless beam population rate temperature component.
    |      'eref': reference interaction energy in eV/amu,
    |      'nref': reference target electron density in m^-3,
    |      'tref': reference target electron temperature in eV,
    |      'sref': reference dimensionless beam population rate.
    |  The total beam population rate: s = sen * st / sref.

    :param repository_path: Path to the atomic data repository.
    """

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH

    # sanitise and validate arguments
    if not isinstance(beam_species, Element):
        raise TypeError('The beam_species must be an Element object.')

    if beam_metastable < 0:
        raise ValueError('Metastable level cannot be less than 0.')

    if not isinstance(target_ion, Element):
        raise TypeError('The beam_species must be an Element object.')

    if not valid_charge(target_ion, target_charge):
        raise ValueError('Charge state is larger than the number of protons in the target ion.')

    # sanitise and validate rate data
    e = np.array(rate['e'], np.float64)
    n = np.array(rate['n'], np.float64)
    t = np.array(rate['t'], np.float64)
    sen = np.array(rate['sen'], np.float64)
    st = np.array(rate['st'], np.float64)

    if e.ndim != 1:
        raise ValueError('Beam energy array must be a 1D array.')

    if n.ndim != 1:
        raise ValueError('Density array must be a 1D array.')

    if t.ndim != 1:
        raise ValueError('Temperature array must be a 1D array.')

    if (e.shape[0], n.shape[0]) != sen.shape:
        raise ValueError('Beam energy, density and combined rate data arrays have inconsistent sizes.')

    if t.shape != st.shape:
        raise ValueError('Temperature and temperature rate data arrays have inconsistent sizes.')

    # convert numpy arrays to lists for json export
    rate['e'] = e.tolist()
    rate['n'] = n.tolist()
    rate['t'] = t.tolist()
    rate['sen'] = sen.tolist()
    rate['st'] = st.tolist()

    rate['eref'] = float(rate['eref'])
    rate['nref'] = float(rate['nref'])
    rate['tref'] = float(rate['tref'])
    rate['sref'] = float(rate['sref'])

    path = os.path.join(repository_path, 'beam/population/{}/{}/{}/{}.json'.format(beam_species.symbol.lower(), beam_metastable, target_ion.symbol.lower(), target_charge))

    # create directory structure if missing
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    # write new data (simply overwrite any existing rate)
    with open(path, 'w') as f:
        json.dump(rate, f, indent=2, sort_keys=True)


def update_beam_population_rates(rates, repository_path=None):
    """
    Updates the beam population rate files
    /beam/population/<beam species>/<beam metastable>/<target ion>/<target_charge>.json
    in the atomic data repository.

    Each json file contains a single rate, so it can simply be replaced.

    :param rates: Dictionary in the form:

    |  { <beam_species>: { <beam_metastable>: { <target_ion>: {<target_charge>: <rate>} } } }, where
    |      <beam_species> is the beam neutral species (Element/Isotope)
    |      <beam_metastable> is the metastable level of beam neutral atom.
    |      <target_ion> is the target species (Element/Isotope).
    |      <target_charge> is the charge of the target species.
    |      <rate> is the beam population rate dictionary containing the following fields:
    |          'e': array-like of size (N) with interaction energy in eV/amu,
    |          'n': array-like of size (M) with target electron density in m^-3,
    |          't': array-like of size (K) with target electron temperature in eV,
    |          'sen': array-like of size (N, M) with dimensionless beam population rate energy component.
    |          'st': array-like of size (K) with dimensionless beam population rate temperature component.
    |          'eref': reference interaction energy in eV/amu,
    |          'nref': reference target electron density in m^-3,
    |          'tref': reference target electron temperature in eV,
    |          'sref': reference dimensionless beam population rate.
    |      The total beam population rate: s = sen * st / sref.

    :param repository_path: Path to the atomic data repository.
    """

    for beam_species, beam_metastables in rates.items():
        for beam_metastable, target_ions in beam_metastables.items():
            for target_ion, target_charge_states in target_ions.items():
                for target_charge, rate in target_charge_states.items():
                    add_beam_population_rate(beam_species, beam_metastable, target_ion, target_charge, rate, repository_path)


def get_beam_population_rate(beam_species, beam_metastable, target_ion, target_charge, repository_path=None):
    """
    Reads a single beam population rate from the repository.

    :param beam_species: Beam neutral species (Element/Isotope).
    :param beam_metastable: Metastable level of beam neutral atom.
    :param target_ion: Target species (Element/Isotope).
    :param target_charge: Charge of the target species.
    :param repository_path: Path to the atomic data repository.

    :return rate: Beam population rate dictionary containing the following entries:

    |      'e': 1D array of size (N) with interaction energy in eV/amu,
    |      'n': 1D array of size (M) with target electron density in m^-3,
    |      't': 1D array of size (K) with target electron temperature in eV,
    |      'sen': 2D array of size (N, M) with dimensionless beam population rate energy component.
    |      'st': 1D array of size (K) with dimensionless beam population rate temperature component.
    |      'eref': reference interaction energy in eV/amu,
    |      'nref': reference target electron density in m^-3,
    |      'tref': reference target electron temperature in eV,
    |      'sref': reference dimensionless beam population rate.
    |  The total beam population rate: s = sen * st / sref.

    """

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH
    path = os.path.join(repository_path, 'beam/population/{}/{}/{}/{}.json'.format(beam_species.symbol.lower(), beam_metastable, target_ion.symbol.lower(), target_charge))
    try:
        with open(path, 'r') as f:
            rate = json.load(f)
    except FileNotFoundError:
        raise RuntimeError('Requested beam population rate (beam species={}, beam metastable={}, target ion={}, target charge={})'
                           ' is not available.'.format(beam_species.symbol, beam_metastable, target_ion.symbol, target_charge))

    # convert lists to numpy arrays
    rate['e'] = np.array(rate['e'], np.float64)
    rate['n'] = np.array(rate['n'], np.float64)
    rate['t'] = np.array(rate['t'], np.float64)
    rate['sen'] = np.array(rate['sen'], np.float64)
    rate['st'] = np.array(rate['st'], np.float64)

    return rate
