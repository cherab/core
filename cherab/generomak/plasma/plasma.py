
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
from raysect.core.math.function.float import Arg1D, Exp1D, Constant1D
from raysect.primitive import Cylinder, Subtract

from cherab.core import AtomicData, Plasma, Maxwellian, Species
from cherab.core.atomic.elements import hydrogen, carbon, lookup_isotope, lookup_element
from cherab.core.utility import RecursiveDict
from cherab.core.math.mappers import AxisymmetricMapper, VectorAxisymmetricMapper
from cherab.core.math.clamp import ClampInput1D

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
    profiles_dir = os.path.join(os.path.dirname(__file__), "data/edge")

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
    ne = Discrete2DMesh.instance(te, profiles["electron"]["density"], limit=False)

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
    profiles_dir = os.path.join(os.path.dirname(__file__), "data/edge")
    path = os.path.join(profiles_dir, "mesh.json")
    with open(path, "r") as fhl:
        mesh = json.load(fhl)

    vertex_coords = np.asarray(mesh["vertex_coords"])
    r_range = (vertex_coords[:, 0].min(), vertex_coords[:, 0].max())
    z_range = (vertex_coords[:, 1].min(), vertex_coords[:, 1].max())
    plasma_height = z_range[1] - z_range[0]

    padding = 1e-3  # enlarge for safety

    outer_column = Cylinder(radius=r_range[1], height=plasma_height)
    inner_column = Cylinder(radius=r_range[0], height=plasma_height + 2 * padding)
    inner_column.transform = translate(0, 0, -padding)

    plasma_geometry = Subtract(outer_column, inner_column)
    geometry_transform = translate(0, 0, z_range[0])

    # get distributions
    dists = get_edge_distributions()

    # create plasma composition list
    plasma_composition = []
    for elem_data in dists["composition"].values():
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


def get_double_parabola(v_min, v_max, convexity, concavity, xmin=0, xmax=1):
    """
    Returns a 1d double-quadratic Function1D

    The retuned Function1D is of the form

    .. math:: f(x) = ((v_{max} - v_{min}) * ((1 - ((1 - x_{norm}) ** convexity)) ** concavity) + v_min)

    where the :math: `x_norm` is calculated as

    .. math:: x_{norm} = (x - xmin) / (xmax - xmin).

    The returned function is decreasing and monotonous and its domain is [xmin, xmax].

    :param v_min: The minimum value of the profile at xmax.
    :param v_max: The maximum value of the profile at xmin.
    :param convexity: Controls the convexity of the profile in the lower values part of the profile.
    :param concavity: Controls the concavity of the profile in the higher values part of the profile.
    :param xmin: The lower edge of the function domain. Defaults to 0.
    :param xmax: The upper edge of the function domain Defaults to 1.
    :return: Function1D
    """

    x = Arg1D() #the free parameter

    # funciton for the normalised free variable
    x_norm = ClampInput1D((x - xmin) / (xmax - xmin), 0, 1)

    # profile function
    return (v_max - v_min) * ((1 - ((1 - x_norm) ** convexity)) ** concavity) + v_min


def get_exponential_growth(initial_value, growth_rate, initial_position=1):
    """
    returns exponentially growing Function1D

    The returned Function1D is of the form:

    ::math::
      v_0 \exp((x - x_0) * \lambda)

    where v_0 is the initial_value, x_0 is the initial_position and lambda is the growth_rate.

    :param initial_value: The value of the function at the initial position.
    :param growth_rate: Growth constant of the profile.
    :param initial_position: The initial position of the profile. Defaults to 1.
    :return: Function1D
    """

    x = Arg1D() #the free parameter
    return initial_value * Exp1D((x - initial_position) * growth_rate)


def get_maxwellian_distribution(equilibrium, f1d_density, f1d_temperature, f1d_vtor, f1d_vpol, f1d_vnorm, rest_mass):
    """ Returns Maxwellian distribution for equilibrium mapped 1d profiles

    :param equilibrium: Instance of EFITEquilibrium
    :param f1d_density: Function1D describing density profile.
    :param f1d_temperature: Function1D describing temperature profile.
    :param f1d_vtor: Function1D describing bulk toroidal rotation velocity profile.
    :param f1d_vpol: Function1D describing bulk poloidal rotation velocity profile.
    :param f1d_vnorm: Function1D describing bulk velocity normal to magnetic surfaces.
    :rest_mass: Rest mass of the distribution species.
    :return: Maxwellian distribution
    """


    # map profiles to 3D
    f3d_te = equilibrium.map3d(f1d_temperature)
    f3d_ne = equilibrium.map3d(f1d_density)
    f3d_v = equilibrium.map_vector3d(f1d_vtor, f1d_vpol, f1d_vnorm)

    # return Maxwellian distribution
    return Maxwellian(f3d_ne, f3d_te, f3d_v, rest_mass)


def get_edge_profile_values(r, z, edge_interpolators=None):
    """
    Evalueate edge plasma profiles at the position [r, z]

    :param r: Radial distance in cylindrical cordinates in m.
    :param z: Elevation in cylindrical coordinates in m.
    :param edge_interpolators: Dictionary with edge interpolators in the shape
           returned by the get_edge_interpolators function.
    :return: Dictionary of edge values at [R, Z]
    """
    # load edge interpolators if not passed as argument
    if edge_interpolators is None:
        edge_interp = get_edge_interpolators()
    else:
        edge_interp = edge_interpolators
    # create recursive dictionary to store profile values
    lcfs_values = RecursiveDict()

    # add electron values
    lcfs_values["electron"]["temperature"] = edge_interp["electron"]["temperature"](r, z)
    lcfs_values["electron"]["density"] = edge_interp["electron"]["density"](r, z)

    # add species values
    for spec, desc in edge_interp['composition'].items():
        for chrg, chrg_desc in desc.items():
            for prop, val in chrg_desc.items():
                if prop in ["temperature", "density"]:
                    lcfs_values["composition"][spec][chrg][prop] = val(r, z)
                else:
                    lcfs_values["composition"][spec][chrg][prop] = val

    return lcfs_values.freeze()


def get_core_profiles_arguments(**kwargs):
    """
    Returns dictionary with core profile arguments

    The function compares the passed keyword arguments with the list of core profile arguments (listed below).
    If there is a match, the default value is overwritten by th passed value, the default value is kept
    otherwise.

    List of core parameters, their meaning and default values
        ne_core: (default 5e19) core electron density
        ne_convexity: (default 1.09) (default ) convexity of the electron density profile
        ne_concavity: (default 0.24) concavity of the electron density profile
        te_core core: (default 3e3) electron temperature
        te_convexity: (default 2.35) convexity of the electron temperature profile
        te_concavity: (default 1.26) concavity of the electron temperature profile
        nh_core: (default 5e19) density of H1+
        nh_convexity: (default 1.09) convexity of H1+ density profile
        nh_concavity: (default 0.24) concavity of H1+ density profile
        th_core: (default 2.8e3) H1+ temperature
        th_convexity: (default 1) convexity of H1+ temperature profile
        th_concavity: (default 0.82) concavity of H1+ temperature profile
        th0_fraction: (default 0.8) H0 temperature factor
        nh0_decay decay: (default 20) rate of H0 density profile
        timp_core: (default 2.7e3) core impurity temperature
        timp_convexity: (default 1) convexity of impurity temperature profile
        timp_concavity: (default 0.82) concavity of impurity temperature profile
        nimp_core: (default 5e17) impurity density
        nimp_convexity: (default 1.09) convexity of impurity density profile
        nimp_concavity: (default 0.24) concavity of impurity density profile
        nimp_decay: (default 30) decay rate of impurity density profile (except bare nuclei)
        vtor_core: (default 1e5) toroidal rotation velocity m/s
        vtor_edge: (default 1e4) toroidal rotation velocity at the edge m/s
        vtor_convexity: (default 2) convexity of the toroidal rotation profile
        vtor_concavity: (default 4) concavity of the toroidal rotation profile
        vpol_lcfs: (default 2e4) Bulk poloidal rotation velocity in m/s
        vpol_decay: (default 0.08) Decay rate of poloidal rotation velocity

    :return: dictionary of profile arguments
    """
    
    core_args = {"ne_core": 5e19, "ne_convexity": 1.09,
                         "ne_concavity": 0.24, "te_core": 3e3,
                         "te_convexity": 2.35, "te_concavity": 1.26,
                         "nh_core": 5e19, "nh_convexity": 1.09,
                         "nh_concavity": 0.24, "th_core": 2.8e3,
                         "th_convexity": 1, "th_concavity": 0.82,
                         "th0_fraction": 0.8, "nh0_decay": 20,
                         "timp_core": 2.7e3, "timp_convexity": 1,
                         "timp_concavity": 0.82, "nimp_core": 5e17,
                         "nimp_convexity": 1.09, "nimp_concavity": 0.24,
                         "nimp_decay": 30,
                         "vtor_core": 1e5, "vtor_edge": 1e4,
                         "vtor_convexity": 2, "vtor_concavity": 4,
                         "vpol_lcfs": 2e4, "vpol_decay": 0.08}

    if not kwargs:
        return core_args

   # change passed values of core args
    for key, item in kwargs.items():
        core_args[key] = item

    return core_args


def get_core_profiles_description(lcfs_values=None, core_args=None):
    """
    Returns dictionary of core profile functions and species descriptions

    :param lcfs_values: Dictionary of profile values at the separatrix on outer midplane.
                        The dictionary has to have the same format as the one returned by
                        the function get_edge_profile_values. The default value is the
                        dictionary returned by the call get_edge_profile_values for r, z
                        on last closed flux surface on outer midplane.
    :param core_args: Dictionary with arguments describing the core profiles. The dictionary
                      has to have the same shape as the one returned by the funciton
                      get_core_profiles_description. The default value is the dictionary
                      returned by the get_core_profiles() call.
    :return: dictionary of Function1D profiles
    """
    if lcfs_values is None:
        # get edge profiles and calculate profile values at midplane outer lcfs
        equilibrium = load_equilibrium()
        r = equilibrium.psin_to_r(1)
        z = 0
        lcfs_values = get_edge_profile_values(r, z)

    if core_args is None:
        core_args = get_core_profiles_arguments()

    # toroidal rotation profile
    f1d_vtor = get_double_parabola(core_args["vtor_edge"], core_args["vtor_core"],
                                   core_args["vtor_convexity"], core_args["vtor_concavity"])

    # poloidal rotation profile
    f1d_vpol = get_exponential_growth(core_args["vpol_lcfs"], core_args["vpol_decay"]) 

    # velocity normal to magnetic surfaces
    f1d_vnorm = Constant1D(0)

    # construct dictionary with 1D profile functions
    profiles = RecursiveDict()

    # Setup electron profiles with double parabola shapes
    profiles["electron"]["f1d_temperature"] = get_double_parabola(lcfs_values["electron"]["temperature"],
                                                                  core_args["te_core"], core_args["te_convexity"],
                                                                  core_args["te_concavity"], xmin=1, xmax=0)
    profiles["electron"]["f1d_density"] = get_double_parabola(lcfs_values["electron"]["density"],
                                                              core_args["ne_core"], core_args["ne_convexity"],
                                                              core_args["ne_concavity"], xmin=1, xmax=0)
    profiles["electron"]["f1d_vtor"] = Constant1D(0)
    profiles["electron"]["f1d_vpol"] = Constant1D(0)
    profiles["electron"]["f1d_vnorm"] = Constant1D(0)

    # Setup H1+ profiles with double parabola shapes
    profiles["composition"]["hydrogen"][1]["f1d_temperature"] = get_double_parabola(lcfs_values["composition"]["hydrogen"][1]["temperature"],
                                                                                    core_args["th_core"], core_args["th_convexity"], core_args["th_concavity"],
                                                                                    xmin=1, xmax=0)
    profiles["composition"]["hydrogen"][1]["f1d_density"] = get_double_parabola(lcfs_values["composition"]["hydrogen"][1]["density"],
                                                                                core_args["nh_core"], core_args["nh_convexity"],
                                                                                core_args["nh_concavity"], xmin=1, xmax=0)
    profiles["composition"]["hydrogen"][1]["f1d_vtor"] = f1d_vtor
    profiles["composition"]["hydrogen"][1]["f1d_vpol"] = f1d_vpol
    profiles["composition"]["hydrogen"][1]["f1d_vnorm"] = f1d_vnorm

    # setup H0+ profile shapes with temperature as a fraction of H1+ and density with decaying exponential
    profiles["composition"]["hydrogen"][0]["f1d_temperature"] = core_args["th0_fraction"] * get_double_parabola(lcfs_values["composition"]["hydrogen"][0]["temperature"],
                                                                                                                core_args["th_core"], core_args["th_convexity"],
                                                                                                                core_args["th_concavity"], xmin=1, xmax=0)
    profiles["composition"]["hydrogen"][0]["f1d_density"] = get_exponential_growth(lcfs_values["composition"]["hydrogen"][0]["density"], core_args["nh0_decay"])
    profiles["composition"]["hydrogen"][0]["f1d_vtor"] = f1d_vtor
    profiles["composition"]["hydrogen"][0]["f1d_vpol"] = f1d_vpol
    profiles["composition"]["hydrogen"][0]["f1d_vnorm"] = f1d_vnorm

    # setup C6+ profile shapes with double parabolas
    profiles["composition"]["carbon"][6]["f1d_temperature"] = get_double_parabola(lcfs_values["composition"]["carbon"][6]["temperature"],
                                                                                  core_args["timp_core"], core_args["timp_convexity"], core_args["timp_concavity"],
                                                                                  xmin=1, xmax=0)
    profiles["composition"]["carbon"][6]["f1d_density"] = get_double_parabola(lcfs_values["composition"]["carbon"][6]["density"], core_args["nimp_core"],
                                                                              core_args["nimp_convexity"], core_args["nimp_concavity"], xmin=1, xmax=0) 
    profiles["composition"]["carbon"][6]["f1d_vtor"] = f1d_vtor
    profiles["composition"]["carbon"][6]["f1d_vpol"] = f1d_vpol
    profiles["composition"]["carbon"][6]["f1d_vnorm"] = f1d_vnorm

    # setup CX+ profile shapes with temperature as double parabolas and density with decaying exponentials
    for chrg in range(6):
        profiles["composition"]["carbon"][chrg]["f1d_temperature"] = get_double_parabola(lcfs_values["composition"]["carbon"][chrg]["temperature"],
                                                                                         core_args["timp_core"], core_args["timp_convexity"], core_args["timp_concavity"],
                                                                                         xmin=1, xmax=0)
        profiles["composition"]["carbon"][chrg]["f1d_density"] = get_exponential_growth(lcfs_values["composition"]["carbon"][chrg]["density"], core_args["nimp_decay"])
        profiles["composition"]["carbon"][chrg]["f1d_vtor"] = f1d_vtor
        profiles["composition"]["carbon"][chrg]["f1d_vpol"] = f1d_vpol
        profiles["composition"]["carbon"][chrg]["f1d_vnorm"] = f1d_vnorm

    return profiles.freeze()


def get_core_distributions(profiles=None, equilibrium=None):
    """
    Returns a dictionary of core plasma species Maxwellian distributions.

    :param profiles: Dictionary of core particle profiles.  The dictionary has to have the same form 
                     as the one returned by the function get_core_profiles_description. 
                     The default value is the value returned by the call get_core_profiles_description().
    :param equilibrium: an instance of EFITEquilibrium.
    :return:  dictionary of core plasma species with Maxwellian distribution
    """
    # get core profile data if not passed sa argument
    if profiles is None:
        profiles = get_core_profiles_description()

    # load plasma equilibrium if not passed as argument
    if equilibrium is None:
        equilibrium = load_equilibrium()

    # build a dictionary with Maxwellian distributions
    species = RecursiveDict()
    species["electron"] = get_maxwellian_distribution(equilibrium, rest_mass=electron_mass,
                                                            **profiles["electron"])
    for name, spec in profiles["composition"].items():
        spec_cherab = _get_cherab_element(name)
        for chrg, desc in spec.items():
            rest_mass = atomic_mass * spec_cherab.atomic_weight
            species["composition"][name][chrg] = get_maxwellian_distribution(equilibrium, rest_mass=rest_mass,  **desc)

    return species.freeze()


def get_core_plasma(distributions=None, atomic_data=None, parent=None, name="Generomak core plasma"):
    """
    Provides Generomak core plasma.

    :param distributions: A dictionary of plasma distributions. Has to have the same format as the
                          dictionary returned by get_core_distributions. The default value
                          is the value returned by the call get_core_distributions().
    :param atomic_data: Instance of AtomicData, default is OpenADAS()
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

    # construct plasma primitive shape 
    padding = 1e-3 #enlarge for safety
    plasma_height = equilibrium.z_range[1] - equilibrium.z_range[0]
    outer_column = Cylinder(radius=equilibrium.r_range[1], height=plasma_height)
    inner_column = Cylinder(radius=equilibrium.r_range[0], height=plasma_height + 2 * padding)
    inner_column.transform = translate(0, 0, -padding)
    plasma_geometry = Subtract(outer_column, inner_column)

    # coordinate transform of the plasma frame
    geometry_transform = translate(0, 0, equilibrium.z_range[0])

    # load core distributions if needed
    if distributions is None:
        dists = get_core_distributions()
    else:
        dists = distributions

    # create plasma composition list
    plasma_composition = []
    for elem_name, elem_data in dists["composition"].items():
        for stage, stage_data in elem_data.items():
            elem = _get_cherab_element(elem_name)
            species = Species(elem, stage, stage_data)
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


def _get_cherab_element(name):
    """Returns cherab element instance

    :param name: Name or label of the element Cherab has to know.
    :return: Cherab element
    """
    try:
        return lookup_isotope(name)
    except ValueError:
        try:
            return lookup_element(name)
        except ValueError:
            raise ValueError("Unknown element name '{}' by Cherab".format(name))
