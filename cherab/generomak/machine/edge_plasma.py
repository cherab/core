# Copyright 2016-2021 Euratom
# Copyright 2016-2021 United Kingdom Atomic Energy Authority
# Copyright 2016-2021 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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
import pickle
from cherab.tools.plasmas import plasma_from_2d_profiles


def load_edge_plasma_profiles(file_path=None):
    """
    Loads Generomak edge plasma profiles defined on a triangular grid as numpy arrays.

    :param str file_path: Path to the pickle file with edge plasma profiles (optional).

    :return dict edge_plasma_profiles:
    """

    if file_path is None:
        generomak_folder = os.path.dirname(__file__)
        file_path = os.path.join(generomak_folder, "data/edge_plasma/generomak_edge_plasma.pkl")

    with open(file_path, "rb") as f:
        edge_plasma_profiles = pickle.load(f)

    return edge_plasma_profiles


def load_edge_plasma(parent=None, file_path=None):
    """ Loads Generomak edge plasma profiles as Plasma object.
        :param Node parent: The plasma's parent node in the scenegraph, e.g. a World object.
        :param str file_path: Path to the pickle file with edge plasma profiles (optional).

        :rtype: Plasma
    """

    edge_plasma_profiles = load_edge_plasma_profiles(file_path)
    plasma = plasma_from_2d_profiles(edge_plasma_profiles['vertex_coords'],
                                     edge_plasma_profiles['triangles'],
                                     edge_plasma_profiles['electron_density'],
                                     edge_plasma_profiles['electron_temperature'],
                                     edge_plasma_profiles['species_density'],
                                     edge_plasma_profiles['species_temperature'],
                                     parent=parent,
                                     name='Generomak edge plasma')

    return plasma
