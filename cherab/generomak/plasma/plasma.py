
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
import json
import numpy as np
from scipy.constants import atomic_mass, electron_mass

from raysect.core import Vector3D, translate
from raysect.core.math.function.float.function2d.interpolate import Discrete2DMesh
from raysect.primitive import Cylinder, Subtract

from cherab.core import AtomicData, Plasma, Maxwellian, Species
from cherab.core.atomic.elements import hydrogen, carbon, lookup_isotope, lookup_element
from cherab.core.utility import RecursiveDict
from cherab.core.math.mappers import AxisymmetricMapper, VectorAxisymmetricMapper

from cherab.openadas import OpenADAS

from cherab.generomak.equilibrium import load_equilibrium

def load_edge_profiles():
    """
    Loads Generomak edge plasma profiles

    Return a single dictionary with available edge and plasma species temperature and 
    density profiles. The profiles are saved on a 2D triangular mesh.

    :return: dictionary with mesh, electron and plasma composition profiles

    .. code-block:: pycon
       >>> # This example shows how to load data and create 2D edge interpolators
       >>>
       >>> from raysect.core.math.function.float.function2d.interpolate import Discrete2DMesh
       >>>
       >>>
       >>> data = load_edge_profiles()
       >>> 
       >>> # create electron temperature 2D mesh interpolator
       >>> te = Discrete2DMesh(data["mesh"]["vertex_coords"],
                               data["mesh"]["triangles"],
                               data["electron"]["temperature"], limit=False)

       >>> # create hydrogen 0+ density 2D mesh interpolator
       >>> n_h0 = Discrete2DMesh.instance(te, data["composition"]["hydrogen"][0]["temperature"])
    """
    profiles_dir = os.path.join(os.path.dirname(__file__), "/data/plasma/edge")

    edge_data = RecursiveDict()
    path = os.path.join(profiles_dir, "mesh.json")
    with open(path, "r") as fhl:
        edge_data["mesh"] = json.load(fhl)

    path = os.path.join(profiles_dir, "electrons.json")
    with open(path, "r") as fhl:
        edge_data["electron"] = json.load(fhl)

    saved_elements = (hydrogen, carbon)

    for element in saved_elements:
        for chrg in range(element.atomic_number + 1):
            path = os.path.join(profiles_dir, "{}{:d}.json".format(element.name, chrg))

            with open(path, "r") as fhl:
                file_data = json.load(fhl)
                element_name = file_data["element"]
                charge = file_data["charge"]

                edge_data["composition"][element_name][charge] = file_data

    return edge_data.freeze()

def get_edge_interpolators():
    """
    Provides Generomak edge profiles 2d interpolator

    :return: dictionary holding instances of Discrete2DMesh density
             and temperature interpolators for plasma species
    """

    profiles = load_edge_profiles()

    mesh_interp = RecursiveDict()

    te = Discrete2DMesh(profiles["mesh"]["vertex_coords"],
                        profiles["mesh"]["triangles"],
                        profiles["electron"]["temperature"], limit=False)
    ne = Discrete2DMesh.instance(te, profiles["electron"]["temperature"], limit=False)

    mesh_interp["electron"]["temperature"] = te
    mesh_interp["electron"]["density"] = ne

    for elem_name, elem_data in profiles["composition"].items():
        for stage, stage_data in elem_data.items():

            t = Discrete2DMesh.instance(te, stage_data["temperature"], limit=False)
            n = Discrete2DMesh.instance(te, stage_data["density"], limit=False)

            mesh_interp["composition"][elem_name][stage]["temperature"] = t
            mesh_interp["composition"][elem_name][stage]["density"] = n
            mesh_interp["composition"][elem_name][stage]["element"] = stage_data["element"]

    
    return mesh_interp.freeze()
    
def get_edge_distributions():
    """
    Provides Generomak edge Maxwellian distribution of plasma species

    :return: Dictionary holding instances of Maxwellian distributions for plasma species.
    """

    mesh_interp = get_edge_interpolators()

    zero_vector = Vector3D(0, 0, 0)

    dists = RecursiveDict()

    n3d = AxisymmetricMapper(mesh_interp["electron"]["density"])
    t3d = AxisymmetricMapper(mesh_interp["electron"]["temperature"])

    dists["electron"] = Maxwellian(n3d, t3d, zero_vector, electron_mass)

    for elem_name, elem_data in mesh_interp["composition"].items():
        for stage, stage_data in elem_data.items():

            # get element or isotope
            try:
                element = lookup_isotope(elem_name)
            except ValueError:
                element = lookup_element(elem_name)
                
            n3d = AxisymmetricMapper(stage_data["density"])
            t3d = AxisymmetricMapper(stage_data["temperature"])
            mass = element.atomic_weight * atomic_mass
            dists["composition"][elem_name][stage]["distribution"] = Maxwellian(n3d, t3d, zero_vector, mass)
            dists["composition"][elem_name][stage]["element"] = element

    return dists.freeze()

def get_edge_plasma(atomic_data=None, parent=None, name="Generomak edge plasma"):
    """
    Provides Generomak Edge plasma.

    :param atomic_data: Instance of AtomicData, default isOpenADAS()
    :param parent: parent of the plasma node, defaults None
    :param name: name of the plasma node, defaults "Generomak edge plasma"
    :return: populated Plasma object
    """
    
    # load Generomak equilibrium
    equilibrium = load_equilibrium()

    # create or check atomic_data
    if atomic_data is not None:
     if not isinstance(atomic_data, AtomicData):
         raise ValueError("atomic_data has to be of type AtomicData")   
    else:
        atomic_data = OpenADAS()

    # base plasma geometry on mesh vertices
    profiles_dir = os.path.join(os.path.dirname(__file__), "data/plasma/edge")
    path = os.path.join(profiles_dir, "mesh.json")
    with open(path, "r") as fhl:
        mesh = json.load(fhl)

    vertex_coords = np.asarray(mesh["vertex_coords"])
    r_range = (vertex_coords[:, 0].min(), vertex_coords[:, 0].max())
    z_range = (vertex_coords[:, 1].min(), vertex_coords[:, 1].max())
    plasma_height = z_range[1] - z_range[0]

    padding = 1e-3 #enlarge for safety

    outer_column = Cylinder(radius=r_range[1], height=plasma_height)
    inner_column = Cylinder(radius=r_range[0], height=plasma_height + 2 * padding)
    inner_column.transform = translate(0, 0, -padding)

    plasma_geometry = Subtract(outer_column, inner_column)
    geometry_transform = translate(0, 0, -outer_column.height / 2)
    
    # get distributions
    dists = get_edge_distributions()

    # create plasma composition list
    plasma_composition = []
    for elem_name, elem_data in dists["composition"].items():
        for stage, stage_data in elem_data.items():
            species = Species(stage_data["element"], stage, stage_data["distribution"])
            plasma_composition.append(species)

    # Populate plasma
    plasma = Plasma(parent=parent)
    plasma.name = name
    plasma.geometry = plasma_geometry
    plasma.atomic_data = atomic_data
    plasma.electron_distribution = dists["electron"]
    plasma.composition = plasma_composition
    plasma.geometry_transform = geometry_transform
    plasma.b_field = VectorAxisymmetricMapper(equilibrium.b_field)

    return plasma
