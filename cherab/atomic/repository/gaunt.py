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
from .utility import DEFAULT_REPOSITORY_PATH

"""
Utilities for managing the local atomic repository - Gaunt factor section.
"""


def update_free_free_gaunt_factor(data, repository_path=None):
    r"""
    Updates the free-free Gaunt factor in the repository.
    The Gaunt factor is defined in the space of parameters:
    :math:`u = h{\nu}/kT` and :math:`{\gamma}^{2} = Z^{2}Ry/kT`.
    See T.R. Carson, 1988, Astron. & Astrophys., 189,
    `319 <https://ui.adsabs.harvard.edu/#abs/1988A&A...189..319C/abstract>`_ for details.

    :param data: Dictionary containing the Gaunt factor data with the following keys:
    |      'u': A 1D array-like of size (N) of real values.
    |      'gamma2': A 1D array-like of size (M)  of real values.
    |      'gaunt_factor': 2D array of size (N, M) of real values storing the Gaunt factor values at u, gamma2.
           'reference': Optional data reference string.

    :param repository_path: Path to the atomic data repository.
    """

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH

    u = np.array(data['u'], np.float64)
    gamma2 = np.array(data['gamma2'], np.float64)
    gaunt_factor = np.array(data['gaunt_factor'], np.float64)

    if u.ndim != 1:
        raise ValueError('The "u" array must be a 1D array.')

    if gamma2.ndim != 1:
        raise ValueError('The "gamma2" array must be a 1D array')

    if (u.shape[0], gamma2.shape[0]) != gaunt_factor.shape:
        raise ValueError('The "u", "gamma2" and "gaunt factor" data arrays have inconsistent sizes.')

    content = {
        'u': u.tolist(),
        'gamma2': gamma2.tolist(),
        'gaunt_factor': gaunt_factor.tolist()
    }
    if 'reference' in data:
        content['reference'] = str(data['reference'])

    path = os.path.join(repository_path, 'gaunt/free_free_gaunt_factor.json')
    # create directory structure if missing
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    # write new data
    with open(path, 'w') as f:
        json.dump(content, f, indent=2, sort_keys=True)


def get_free_free_gaunt_factor(repository_path=None):
    r"""
    Reads the free-free Gaunt factor from the repository.
    The Gaunt factor is defined in the space of parameters:
    :math:`u = h{\nu}/kT` and :math:`{\gamma}^{2} = Z^{2}Ry/kT`.
    See T.R. Carson, 1988, Astron. & Astrophys., 189,
    `319 <https://ui.adsabs.harvard.edu/#abs/1988A&A...189..319C/abstract>`_ for details.

    :return data: Dictionary containing the Gaunt factor data with the following keys:

    |      'u': A 1D array of size (N) of real values.
    |      'gamma2': A 1D array of size (M) of real values.
    |      'gaunt_factor': 2D array of size (N, M) of real values storing the Gaunt factor values at u, gamma2.
    |      'reference': Optional data reference string.
    """

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH
    path = os.path.join(repository_path, 'gaunt/free_free_gaunt_factor.json')
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError):
        raise RuntimeError('Free-free Gaunt factor is missing in the atomic repository.')

    # convert to numpy arrays
    data['u'] = np.array(data['u'], np.float64)
    data['gamma2'] = np.array(data['gamma2'], np.float64)
    data['gaunt_factor'] = np.array(data['gaunt_factor'], np.float64)

    return data
