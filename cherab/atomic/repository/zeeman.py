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
from cherab.core.utility import RecursiveDict
from cherab.core.atomic import Element
from .utility import DEFAULT_REPOSITORY_PATH, valid_charge, encode_transition

"""
Utilities for managing the local atomic repository - Zeeman splitting section.
"""


def add_zeeman_structure(element, charge, transition, data, repository_path=None):
    r"""
    Adds a single Zeeman multiplet structure to the repository.

    If adding multiple structures, consider using the update_zeeman_structures() function
    instead. The update function avoid repeatedly opening and closing the Zeeman structure
    files.

    :param element: Plasma species (Element/Isotope).
    :param charge: Charge of the plasma species.
    :param transition: Tuple containing (initial level, final level).
    :param data: A dictionary containing the central wavelengths and relative intensities
        of the polarised Zeeman components with respect to the magnetic field strength.
        It has the following keys:
    |      'b': A 1D array of shape (N,) with magnetic field strength.
    |      'polarisation': A 1D array of shape (M,) with component polarisation
    |                      0 for :math:`\pi`-polarisation,
    |                      -1 for :math:`\sigma-`-polarisation,
    |                      1 for :math:`\sigma+`-polarisation.
    |      'wavelength': A 2D array of shape (M, N) with component wavelength as functions of
    |                    magnetic field strength.
    |      'ratio': A 2D array of shape (M, N) with component relative intensities
    |               as functions of magnetic field strength.
    |      'reference': Optional string containg the reference to the data source.

    :param repository_path: Path to the atomic data repository.
    """

    update_zeeman_structures({
        element: {
            charge: {
                transition: data
            }
        }
    }, repository_path)


def update_zeeman_structures(zeeman_structures, repository_path=None):
    r"""
    Updates the Zeeman multiplet structure files lineshape/zeeman/multiplet/<element>/<charge>.json
    in the atomic data repository.

    File contains multiple Zeeman structures, indexed by the transition.

    :param zeeman_structures: Dictionary in the form:

    |  { <element>: { <charge>: { <transition>: <data> } } }, where
    |      <element> is the plasma species (Element/Isotope).
    |      <charge> is the charge of the plasma species.
    |      <transition> is the tuple containing (initial level, final level).
    |      <data> is the dictionary containing the central wavelengths and relative intensities
    |        of the polaraised  Zeeman components with respect to the magnetic field strength.
    |        It has the following keys:
    |            'b': A 1D array of shape (N,) with magnetic field strength.
    |            'polarisation': A 1D array of shape (M,) with component polarisation
    |                0 for :math:`\pi`-polarisation,
    |                -1 for :math:`\sigma-`-polarisation,
    |                1 for :math:`\sigma+`-polarisation.
    |            'wavelength': A 2D array of shape (M, N) with component wavelength as
    |                functions of magnetic field strength.
    |            'ratio': A 2D array of shape (M, N) with component relative intensities
    |                as functions of magnetic field strength.
    |            'reference': Optional string containg the reference to the data source.

    :param repository_path: Path to the atomic data repository.
    """

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH

    for element, charge_states in zeeman_structures.items():
        for charge, transitions in charge_states.items():

            # sanitise and validate
            if not isinstance(element, Element):
                raise TypeError('The element must be an Element object.')

            if not valid_charge(element, charge):
                raise ValueError('Charge state is larger than the number of protons in the element.')

            path = os.path.join(repository_path, 'lineshape/zeeman/multiplet/{}/{}.json'.format(element.symbol.lower(), charge))

            # read in any existing zeeman structures
            try:
                with open(path, 'r') as f:
                    content = RecursiveDict.from_dict(json.load(f))
            except FileNotFoundError:
                content = RecursiveDict()

            # add/replace data for a transition
            for transition, data in transitions.items():
                key = encode_transition(transition)

                # sanitise/validate data
                data['b'] = np.array(data['b'], np.float64)
                data['polarisation'] = np.array(data['polarisation'], np.int32)
                data['wavelength'] = np.array(data['wavelength'], np.float64)
                data['ratio'] = np.array(data['ratio'], np.float64)

                if data['b'].ndim != 1:
                    raise ValueError('Magnetic field strength array must be a 1D array.')

                if data['polarisation'].ndim != 1:
                    raise ValueError('Polarisation array must be a 1D array.')

                if (data['polarisation'].shape[0], data['b'].shape[0]) != data['wavelength'].shape:
                    raise ValueError('Polarisation, magnetic field strength and wavelength data arrays have inconsistent sizes.')

                if data['wavelength'].shape != data['ratio'].shape:
                    raise ValueError('Wavelength and ratio data arrays have inconsistent sizes.')

                content[key] = {
                    'b': data['b'].tolist(),
                    'polarisation': data['polarisation'].tolist(),
                    'wavelength': data['wavelength'].tolist(),
                    'ratio': data['ratio'].tolist()
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


def get_zeeman_structure(element, charge, transition, repository_path=None):
    r"""
    Reads the Zeeman multiplet structure from the repository for the given
    element, charge and transition.

    :param element: Plasma species (Element/Isotope).
    :param charge: Charge of the plasma species.
    :param transition: Tuple containing (initial level, final level).
    :param repository_path: Path to the atomic data repository.

    :return data: A dictionary containing the central wavelengths and relative intensities
        of the polaraised Zeeman components with respect to the magnetic field strength.
        It has the following keys:
    |      'b': A 1D array of shape (N,) with magnetic field strength.
    |      'polarisation': A 1D array of shape (M,) with component polarisation
    |                      0 for :math:`\pi`-polarisation,
    |                      -1 for :math:`\sigma-`-polarisation,
    |                      1 for :math:`\sigma+`-polarisation.
    |      'wavelength': A 2D array of shape (M, N) with component wavelength as functions of
    |                    magnetic field strength.
    |      'ratio': A 2D array of shape (M, N) with component relative intensities
    |               as functions of magnetic field strength.
    |      'reference': Optional string containg the reference to the data source.
    """

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH
    path = os.path.join(repository_path, 'lineshape/zeeman/multiplet/{}/{}.json'.format(element.symbol.lower(), charge))
    try:
        with open(path, 'r') as f:
            content = json.load(f)
        data = content[encode_transition(transition)]
    except (FileNotFoundError, KeyError):
        raise RuntimeError('Requested Zeeman structure (element={}, charge={}, transition={})'
                           ' is not available.'.format(element.symbol, charge, transition))

    # convert to numpy arrays
    data['b'] = np.array(data['b'], np.float64)
    data['polarisation'] = np.array(data['polarisation'], np.int32)
    data['wavelength'] = np.array(data['wavelength'], np.float64)
    data['ratio'] = np.array(data['ratio'], np.float64)

    return data
