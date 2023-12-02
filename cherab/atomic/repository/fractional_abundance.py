
# Copyright 2016-2023 Euratom
# Copyright 2016-2023 United Kingdom Atomic Energy Authority
# Copyright 2016-2023 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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


def add_fractional_abundance(species, charge, data, repository_path=None):
    """
    Adds fractional abundance for a single species to the repository.

    If adding multiple abundances, consider using the update_fractional_abundances()
    function instead. The update function avoids repeatedly opening and closing
    the data files.

    :param species: Plasma species (Element/Isotope).
    :param charge: Charge of the plasma species.
    :param data: Fractional abundance dictionary containing the following fields:

    |      'ne': array-like of size (N) with electron density in m^-3,
    |      'te': array-like of size (M) with electron temperature in eV,
    |      'fractional_abundance': array-like of size (N, M) with fractional abundance.
    |      'reference': Optional data reference string.

    :param repository_path: Path to the atomic data repository.
    """

    update_fractional_abundances({
        species: {
            charge: data
        }
    }, repository_path)


def update_fractional_abundances(data, repository_path=None):
    """
    Update the files for the fractional abundances:
    /fractional_abundance/<species>.json
    in the atomic data repository.

    File contains multiple fractional abundances, indexed by the ion's charge state.

    :param data: Dictionary in the form {<species>: {<charge>: <abundance>}}, where

    |      <species> is the plasma species (Element/Isotope),
    |      <charge> is the charge of the plasma species,
    |      <abundance> is the fractional abundance dictionary containing the following fields:
    |          'ne': array-like of size (N) with electron density in m^-3,
    |          'te': array-like of size (M) with electron temperature in eV,
    |          'fractional_abundance': array-like of size (N, M) with the fractional abundance.
    |          'reference': Optional data reference string.

    :param repository_path: Path to the atomic data repository.
    """

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH

    for species, abundances in data.items():

        # sanitise and validate arguments
        if not isinstance(species, Element):
            raise TypeError('The species must be an Element object.')

        path = os.path.join(repository_path, 'fractional_abundance/{}.json'.format(species.symbol.lower()))

        # read in any existing fractional abundances
        try:
            with open(path, 'r') as f:
                content = RecursiveDict.from_dict(json.load(f))
        except FileNotFoundError:
            content = RecursiveDict()

        for charge, abundance in abundances.items():

            if not valid_charge(species, charge):
                raise ValueError('The charge state is larger than the number of protons in the specified species.')

            # sanitise and validate abundance data
            te = np.array(abundance['te'], np.float64)
            ne = np.array(abundance['ne'], np.float64)
            fractional_abundance = np.array(abundance['fractional_abundance'], np.float64)

            if ne.ndim != 1:
                raise ValueError('Density array must be a 1D array.')

            if te.ndim != 1:
                raise ValueError('Temperature array must be a 1D array.')

            if (ne.shape[0], te.shape[0]) != fractional_abundance.shape:
                raise ValueError('Electron temperature, density and abundance data arrays have inconsistent sizes.')

            # update file content with new fractional abundance
            content[str(charge)] = {
                'te': te.tolist(),
                'ne': ne.tolist(),
                'fractional_abundance': fractional_abundance.tolist(),
            }
            if 'reference' in abundance:
                content[str(charge)]['reference'] = str(abundance['reference'])

        # create directory structure if missing
        directory = os.path.dirname(path)
        if not os.path.isdir(directory):
            os.makedirs(directory)

        # write new data
        with open(path, 'w') as f:
            json.dump(content, f, indent=2, sort_keys=True)


def get_fractional_abundance(element, charge, repository_path=None):
    """
    Reads fractional abundance for the given species and charge
    from the atomic data repository.

    :param element: Plasma species (Element/Isotope).
    :param charge: Charge of the plasma species.
    :param repository_path: Path to the atomic data repository.

    :return data: Fractional abundance dictionary containing the following fields:

    |      'ne': 1D array of size (N) with electron density in m^-3,
    |      'te': 1D array of size (M) with electron temperature in eV,
    |      'fractional_abundance': 2D array of size (N, M) with fractional abundance.
    |      'reference': Optional data reference string.
    """

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH

    path = os.path.join(repository_path, 'fractional_abundance/{}.json'.format(element.symbol.lower()))
    try:
        with open(path, 'r') as f:
            content = json.load(f)
        d = content[str(charge)]
    except (FileNotFoundError, KeyError):
        raise RuntimeError('Requested fractional abundance (element={}, charge={})'
                           ' is not available.'.format(element.symbol, charge))

    # convert to numpy arrays
    d['ne'] = np.array(d['ne'], np.float64)
    d['te'] = np.array(d['te'], np.float64)
    d['fractional_abundance'] = np.array(d['fractional_abundance'], np.float64)

    return d
