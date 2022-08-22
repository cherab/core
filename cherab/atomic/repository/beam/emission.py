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
from cherab.core.utility import RecursiveDict
from cherab.core.atomic import Element
from ..utility import DEFAULT_REPOSITORY_PATH, valid_charge, encode_transition

"""
Utilities for managing the local rate repository - PEC section.
"""


def add_beam_emission_rate(beam_species, target_ion, target_charge, transition, rate, repository_path=None):
    """
    Adds a single beam emission rate to the repository.

    If adding multiple rate, consider using the update_beam_emission_rates()
    function instead. The update function avoid repeatedly opening and closing
    the rate files.

    :param repository_path:
    :return:
    """

    update_beam_emission_rates({
        beam_species: {
            target_ion: {
                target_charge: {
                    transition: rate
                }
            }
        }
    }, repository_path)


def update_beam_emission_rates(rates, repository_path=None):
    """
    Beam emission rate file structure

    /beam/emission/<beam species>/<target ion>/<target_charge>.json

    File contains multiple rates, indexed by transition.
    """

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH

    for beam_species, target_ions in rates.items():
        for target_ion, target_charge_states in target_ions.items():
            for target_charge, transitions in target_charge_states.items():

                # sanitise and validate arguments
                if not isinstance(beam_species, Element):
                    raise TypeError('The beam_species must be an Element object.')

                if not isinstance(target_ion, Element):
                    raise TypeError('The beam_species must be an Element object.')

                if not valid_charge(target_ion, target_charge):
                    raise ValueError('Charge state is larger than the number of protons in the target ion.')

                path = os.path.join(repository_path, 'beam/emission/{}/{}/{}.json'.format(beam_species.symbol.lower(), target_ion.symbol.lower(), target_charge))

                # read in any existing rates
                try:
                    with open(path, 'r') as f:
                        content = RecursiveDict.from_dict(json.load(f))
                except FileNotFoundError:
                    content = RecursiveDict()

                # add/replace data for a transition
                for transition in transitions:
                    key = encode_transition(transition)
                    rate = rates[beam_species][target_ion][target_charge][transition]

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

                    # update file content with new rate
                    content[key] = {
                        'e': e.tolist(),
                        'n': n.tolist(),
                        't': t.tolist(),
                        'sen': sen.tolist(),
                        'st': st.tolist(),
                        'eref': float(rate['eref']),
                        'nref': float(rate['nref']),
                        'tref': float(rate['tref']),
                        'sref': float(rate['sref'])
                    }

                # create directory structure if missing
                directory = os.path.dirname(path)
                if not os.path.isdir(directory):
                    os.makedirs(directory)

                # write new data
                with open(path, 'w') as f:
                    json.dump(content, f, indent=2, sort_keys=True)


def get_beam_emission_rate(beam_species, target_ion, target_charge, transition, repository_path=None):

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH
    path = os.path.join(repository_path, 'beam/emission/{}/{}/{}.json'.format(beam_species.symbol.lower(), target_ion.symbol.lower(), target_charge))
    try:
        with open(path, 'r') as f:
            content = json.load(f)
        rate = content[encode_transition(transition)]
    except (FileNotFoundError, KeyError):
        raise RuntimeError('Requested beam emission rate (beam species={}, target ion={}, target charge={}, transition={})'
                           ' is not available.'.format(beam_species.symbol, target_ion.symbol, target_charge, transition))

    # convert lists to numpy arrays
    rate['e'] = np.array(rate['e'], np.float64)
    rate['n'] = np.array(rate['n'], np.float64)
    rate['t'] = np.array(rate['t'], np.float64)
    rate['sen'] = np.array(rate['sen'], np.float64)
    rate['st'] = np.array(rate['st'], np.float64)

    return rate