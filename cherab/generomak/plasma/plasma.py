
# Copyright 2016-2022 Euratom
# Copyright 2016-2022 United Kingdom Atomic Energy Authority
# Copyright 2016-2022 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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
from raysect.core.math.function.float import Arg1D, Exp1D, Constant1D, Interpolator1DArray, Blend2D
from raysect.core.math.function.vector3d import Constant2D as ConstantVector2D, Blend2D as BlendVector2D
from raysect.primitive import Cylinder, Subtract

from cherab.core import AtomicData, Plasma, Maxwellian, Species
from cherab.core.atomic.elements import hydrogen, carbon, lookup_isotope, lookup_element
from cherab.core.utility import RecursiveDict
from cherab.core.math.mappers import AxisymmetricMapper, VectorAxisymmetricMapper
from cherab.core.math.clamp import ClampInput1D

from cherab.tools.plasmas.ionisation_balance import interpolators1d_from_elementdensity, interpolators1d_match_plasma_neutrality

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
    ve = ConstantVector2D(Vector3D(0, 1.e-10, 0))  # avoid zero-length vectors for blending

    mesh_interp["electron"]["temperature"] = te
    mesh_interp["electron"]["density"] = ne
    mesh_interp["electron"]["velocity"] = ve

    for elem_name, elem_data in profiles["composition"].items():
        for stage, stage_data in elem_data.items():

            t = Discrete2DMesh.instance(te, stage_data["temperature"], limit=False)
            n = Discrete2DMesh.instance(te, stage_data["density"], limit=False)
            v = ConstantVector2D(Vector3D(0, 1.e-10, 0))  # avoid zero-length vectors for blending

            mesh_interp["composition"][elem_name][stage]["temperature"] = t
            mesh_interp["composition"][elem_name][stage]["density"] = n
            mesh_interp["composition"][elem_name][stage]["velocity"] = v

    return mesh_interp.freeze()


def get_2d_distributions(profiles_2d=None):
    """
    Provides Generomak Maxwellian distribution of plasma species for 2d profiles

    :param profiles_2d: Dictionary with 2D profile interpolators in the shape
                        returned by the get_edge_interpolators() or get_full_profiles() functions.
                        If not specified, will use the value returned by get_edge_interpolators().
    :return: Dictionary holding instances of Maxwellian distributions for plasma species.
    """

    profiles_2d = profiles_2d or get_edge_interpolators()

    dists = RecursiveDict()

    n3d = AxisymmetricMapper(profiles_2d["electron"]["density"])
    t3d = AxisymmetricMapper(profiles_2d["electron"]["temperature"])
    v3d = VectorAxisymmetricMapper(profiles_2d["electron"]["velocity"])

    dists["electron"] = Maxwellian(n3d, t3d, v3d, electron_mass)

    for elem_name, elem_data in profiles_2d["composition"].items():
        for stage, stage_data in elem_data.items():

            spec_cherab = _get_cherab_element(elem_name)

            n3d = AxisymmetricMapper(stage_data["density"])
            t3d = AxisymmetricMapper(stage_data["temperature"])
            v3d = VectorAxisymmetricMapper(stage_data["velocity"])
            mass = spec_cherab.atomic_weight * atomic_mass
            dists["composition"][elem_name][stage] = Maxwellian(n3d, t3d, v3d, mass)

    return dists.freeze()


def get_edge_plasma(atomic_data=None, parent=None, name="Generomak edge plasma"):
    """
    Provides Generomak default edge plasma.

    :param atomic_data: Instance of AtomicData, default is OpenADAS()
    :param parent: parent of the plasma node, defaults None
    :param name: name of the plasma node, defaults "Generomak edge plasma"
    :return: populated Plasma object
    """

    # load Generomak equilibrium
    equilibrium = load_equilibrium()

    # get edge distributions
    distributions = get_2d_distributions()

    # base plasma geometry on mesh vertices
    profiles_dir = os.path.join(os.path.dirname(__file__), "data/edge")
    path = os.path.join(profiles_dir, "mesh.json")
    with open(path, "r") as fhl:
        mesh = json.load(fhl)

    vertex_coords = np.asarray(mesh["vertex_coords"])
    r_range = (vertex_coords[:, 0].min(), vertex_coords[:, 0].max())
    z_range = (vertex_coords[:, 1].min(), vertex_coords[:, 1].max())

    return get_plasma(equilibrium=equilibrium, distributions=distributions,
                      r_range=r_range, z_range=z_range, atomic_data=atomic_data,
                      parent=parent, name=name)


def load_core_profiles():
    """
    Loads Generomak default core plasma profiles.

    Return a single dictionary with available core plasma species temperature and
    density profiles on a magnetic surface coordinate grid.

    :return: dictionary with electron and plasma composition profiles
    """
    profiles_dir = os.path.join(os.path.dirname(__file__), "data/core")

    core_data = RecursiveDict()
    path = os.path.join(profiles_dir, "psi_norm.json")
    with open(path, "r") as fhl:
        core_data["psi_norm"] = json.load(fhl)["psi_norm"]

    path = os.path.join(profiles_dir, "electrons.json")
    with open(path, "r") as fhl:
        core_data["electron"] = json.load(fhl)

    saved_elements = (hydrogen, carbon)

    for element in saved_elements:
        for chrg in range(element.atomic_number + 1):
            path = os.path.join(profiles_dir, "{}{:d}.json".format(element.name, chrg))

            with open(path, "r") as fhl:
                file_data = json.load(fhl)
                element_name = file_data["element"]
                charge = file_data["charge"]

                core_data["composition"][element_name][charge] = file_data

    return core_data.freeze()


def get_core_interpolators():
    """
    Provides 1d interpolators for Generomak default core profiles.

    :return: dictionary holding 1D interpolators of density,
             temperature and velocity for plasma species
    """

    profiles = load_core_profiles()

    core_interp = RecursiveDict()

    te = Interpolator1DArray(profiles["psi_norm"], profiles["electron"]["temperature"], 'cubic', 'nearest', 1.e-5)
    ne = Interpolator1DArray(profiles["psi_norm"], profiles["electron"]["density"], 'cubic', 'nearest', 1.e-5)
    ve_tor = Interpolator1DArray(profiles["psi_norm"], profiles["electron"]["vtor"], 'cubic', 'nearest', 1.e-5)
    ve_pol = Interpolator1DArray(profiles["psi_norm"], profiles["electron"]["vpol"], 'cubic', 'nearest', 1.e-5)
    ve_norm = Interpolator1DArray(profiles["psi_norm"], profiles["electron"]["vnorm"], 'cubic', 'nearest', 1.e-5)

    core_interp["electron"]["f1d_temperature"] = te
    core_interp["electron"]["f1d_density"] = ne
    core_interp["electron"]["f1d_vtor"] = ve_tor
    core_interp["electron"]["f1d_vpol"] = ve_pol
    core_interp["electron"]["f1d_vnorm"] = ve_norm

    for elem_name, elem_data in profiles["composition"].items():
        for stage, stage_data in elem_data.items():

            t = Interpolator1DArray(profiles["psi_norm"], stage_data["temperature"], 'cubic', 'nearest', 1.e-5)
            n = Interpolator1DArray(profiles["psi_norm"], stage_data["density"], 'cubic', 'nearest', 1.e-5)
            vtor = Interpolator1DArray(profiles["psi_norm"], stage_data["vtor"], 'cubic', 'nearest', 1.e-5)
            vpol = Interpolator1DArray(profiles["psi_norm"], stage_data["vpol"], 'cubic', 'nearest', 1.e-5)
            vnorm = Interpolator1DArray(profiles["psi_norm"], stage_data["vnorm"], 'cubic', 'nearest', 1.e-5)

            core_interp["composition"][elem_name][stage]["f1d_temperature"] = t
            core_interp["composition"][elem_name][stage]["f1d_density"] = n
            core_interp["composition"][elem_name][stage]["f1d_vtor"] = vtor
            core_interp["composition"][elem_name][stage]["f1d_vpol"] = vpol
            core_interp["composition"][elem_name][stage]["f1d_vnorm"] = vnorm

    return core_interp.freeze()


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

    x = Arg1D()  # the free parameter

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

    x = Arg1D()  # the free parameter
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
    values = RecursiveDict()

    # add electron values
    values["electron"]["temperature"] = edge_interp["electron"]["temperature"](r, z)
    values["electron"]["density"] = edge_interp["electron"]["density"](r, z)

    # add species values
    for spec, desc in edge_interp['composition'].items():
        for chrg, chrg_desc in desc.items():
            for prop, val in chrg_desc.items():
                if prop in ["temperature", "density"]:
                    values["composition"][spec][chrg][prop] = val(r, z)
                else:
                    values["composition"][spec][chrg][prop] = val

    return values.freeze()


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
        th_core: (default 2.8e3) H1+ temperature
        th_convexity: (default 2) convexity of H1+ temperature profile
        th_concavity: (default 1.26) concavity of H1+ temperature profile
        th0_fraction: (default 0.8) H0 temperature factor
        timp_core: (default 2.8e3) core impurity temperature
        timp_convexity: (default 2) convexity of impurity temperature profile
        timp_concavity: (default 1.26) concavity of impurity temperature profile
        nimp_core: (default 5e17) impurity density
        nimp_convexity: (default 1.09) convexity of impurity density profile
        nimp_concavity: (default 0.24) concavity of impurity density profile
        vtor_core: (default 1e5) toroidal rotation velocity m/s
        vtor_edge: (default 1e4) toroidal rotation velocity at the edge m/s
        vtor_convexity: (default 2) convexity of the toroidal rotation profile
        vtor_concavity: (default 4) concavity of the toroidal rotation profile
        vpol_lcfs: (default 2e4) Bulk poloidal rotation velocity in m/s
        vpol_decay: (default 0.08) Decay rate of poloidal rotation velocity

    :return: dictionary of profile arguments
    """

    core_args = {"ne_core": 5e19, "ne_convexity": 1.09, "ne_concavity": 0.24,
                 "te_core": 3e3, "te_convexity": 2.35, "te_concavity": 1.26,
                 "th_core": 2.8e3, "th_convexity": 2, "th_concavity": 1.26,
                 "th0_fraction": 0.8,
                 "timp_core": 2.8e3, "timp_convexity": 2, "timp_concavity": 1.26,
                 "nimp_core": 5e17, "nimp_convexity": 1.09, "nimp_concavity": 0.24,
                 "vtor_core": 1e5, "vtor_edge": 1e4, "vtor_convexity": 2, "vtor_concavity": 4,
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

    # total carbon impurity density at lcfs
    nimp_lcfs = sum([value["density"] for _, value in lcfs_values["composition"]["carbon"].items()])

    if core_args is None:
        core_args = get_core_profiles_arguments()

    # toroidal rotation profile
    f1d_vtor = get_double_parabola(core_args["vtor_edge"], core_args["vtor_core"],
                                   core_args["vtor_convexity"], core_args["vtor_concavity"], xmin=1, xmax=0)

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
    profiles["electron"]["f1d_vtor"] = Constant1D(1.e-10)  # avoid zero-length vectors for blending
    profiles["electron"]["f1d_vpol"] = Constant1D(0)
    profiles["electron"]["f1d_vnorm"] = Constant1D(0)

    # total carbon density
    carbon_total_density = get_double_parabola(nimp_lcfs, core_args["nimp_core"],
                                               core_args["nimp_convexity"], core_args["nimp_concavity"], xmin=1, xmax=0)

    # solve ionisation balance
    openadas = OpenADAS(permit_extrapolation=True)
    psin_1d = np.append(1. - np.geomspace(1.e-4, 1, 1023)[::-1], [1.])  # density profiles are sharp near psin=1
    density_profiles = {}
    density_profiles["carbon"] = interpolators1d_from_elementdensity(openadas, carbon, psin_1d, carbon_total_density,
                                                                     profiles["electron"]["f1d_density"],
                                                                     profiles["electron"]["f1d_temperature"])

    density_profiles["hydrogen"] = interpolators1d_match_plasma_neutrality(openadas, hydrogen, psin_1d, [density_profiles["carbon"]],
                                                                           profiles["electron"]["f1d_density"],
                                                                           profiles["electron"]["f1d_temperature"])

    # Setup ion profiles
    for element, prefix in ((hydrogen, "h"), (carbon, "imp")):
        name = element.name
        for chrg in range(element.atomic_number + 1):
            profiles["composition"][name][chrg]["f1d_temperature"] = get_double_parabola(lcfs_values["composition"][name][chrg]["temperature"],
                                                                                         core_args["t{}_core".format(prefix)],
                                                                                         core_args["t{}_convexity".format(prefix)],
                                                                                         core_args["t{}_concavity".format(prefix)],
                                                                                         xmin=1, xmax=0)
            profiles["composition"][name][chrg]["f1d_density"] = density_profiles[name][chrg]
            profiles["composition"][name][chrg]["f1d_vtor"] = f1d_vtor
            profiles["composition"][name][chrg]["f1d_vpol"] = f1d_vpol
            profiles["composition"][name][chrg]["f1d_vnorm"] = f1d_vnorm

    # multiply H0 temperature by th0_fraction
    profiles["composition"]["hydrogen"][0]["f1d_temperature"] *= core_args["th0_fraction"]

    return profiles.freeze()


def get_core_distributions(profiles=None, equilibrium=None):
    """
    Returns a dictionary of core plasma species Maxwellian distributions.

    :param profiles: Dictionary with core interpolators. The dictionary has to have
                     the same form as the one returned by the function
                     get_core_profiles_description or get_core_interpolators.
                     The default value is the value returned by the call
                     get_core_interpolators().
    :param equilibrium: an instance of EFITEquilibrium.
    :return:  dictionary of core plasma species with Maxwellian distribution
    """
    # get core profile data if not passed sa argument
    if profiles is None:
        profiles = get_core_interpolators()

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
            species["composition"][name][chrg] = get_maxwellian_distribution(equilibrium, rest_mass=rest_mass, **desc)

    return species.freeze()


def get_core_plasma(atomic_data=None, parent=None, name="Generomak core plasma"):
    """
    Provides Generomak default core plasma.

    :param atomic_data: Instance of AtomicData, default is OpenADAS()
    :param parent: parent of the plasma node, defaults None
    :param name: name of the plasma node, defaults "Generomak edge plasma"
    :return: populated Plasma object
    """

    # load Generomak equilibrium
    equilibrium = load_equilibrium()

    # load core distributions
    distributions = get_core_distributions(equilibrium=equilibrium)

    return get_plasma(equilibrium=equilibrium, distributions=distributions,
                      atomic_data=atomic_data, parent=parent, name=name)


def get_full_profiles(equilibrium=None, core_profiles=None, edge_profiles=None, mask=None):
    """
    Blends core and edge profiles using the mask function as a modulator.

    :param equilibrium: an instance of EFITEquilibrium. The default value is the value returned by
                        load_equilibrium().
    :param core_profiles: Dictionary with core interpolators. The dictionary has to have
                          the same form as the one returned by the function
                          get_core_profiles_description or get_core_interpolators.
                          The default value is the value returned by the call
                          get_core_interpolators().
    :param edge_profiles: Dictionary with edge interpolators in the shape
                          returned by the get_edge_interpolators function.
                          If not specified, will use the value returned by
                          get_edge_interpolators().
    :param Function2D mask: Scalar 2D function returning a value in the range [0, 1].
                            If not specified, will use core profiles for psi_normal < 0.94,
                            the edge profiles for psi_normal > 1 and a weighted sum of core and
                            edge profiles for 0.94 < psi_normal < 1, with the edge profile weight
                            increasing from 0 to 1 linearly.

    :return: dictionary of blended plasma profiles with the sturcture identical to edge_profiles.
    """

    equilibrium = equilibrium or load_equilibrium()

    core_profiles = core_profiles or get_core_interpolators()

    edge_profiles = edge_profiles or get_edge_interpolators()

    mask = mask or equilibrium.map2d(Interpolator1DArray([0, 0.94, 1.0, 1.1], [1, 1, 0, 0], 'linear', 'none', 0))

    # blended core and edge profiles
    blended_profiles = RecursiveDict()

    # map core profiles to 2D using the equilibrium
    te_core = equilibrium.map2d(core_profiles["electron"]["f1d_temperature"])
    ne_core = equilibrium.map2d(core_profiles["electron"]["f1d_density"])
    ve_core = equilibrium.map_vector2d(core_profiles["electron"]["f1d_vtor"],
                                       core_profiles["electron"]["f1d_vpol"],
                                       core_profiles["electron"]["f1d_vnorm"])

    blended_profiles["electron"]["temperature"] = Blend2D(edge_profiles["electron"]["temperature"], te_core, mask)
    blended_profiles["electron"]["density"] = Blend2D(edge_profiles["electron"]["density"], ne_core, mask)
    blended_profiles["electron"]["velocity"] = BlendVector2D(edge_profiles["electron"]["velocity"], ve_core, mask)

    for element, states in core_profiles["composition"].items():
        for charge, state in states.items():
            t_core = equilibrium.map2d(state["f1d_temperature"])
            n_core = equilibrium.map2d(state["f1d_density"])
            v_core = equilibrium.map_vector2d(state["f1d_vtor"], state["f1d_vpol"], state["f1d_vnorm"])

            edge_state = edge_profiles["composition"][element][charge]

            blended_profiles["composition"][element][charge]["temperature"] = Blend2D(edge_state["temperature"], t_core, mask)
            blended_profiles["composition"][element][charge]["density"] = Blend2D(edge_state["density"], n_core, mask)
            blended_profiles["composition"][element][charge]["velocity"] = BlendVector2D(edge_state["velocity"], v_core, mask)

    return blended_profiles.freeze()


def get_plasma(equilibrium=None, distributions=None, r_range=None, z_range=None, atomic_data=None, parent=None, name="Generomak plasma"):
    """
    Provides Generomak plasma. The full (core + edge) plasma is returned by default.

    :param equilibrium: an instance of EFITEquilibrium. The default value is the value returned by load_equilibrium().
    :param distributions: A dictionary of plasma distributions. Has to have the same format as the
                          dictionary returned by get_core_distributions or get_2d_distributions.
                          The default value is the value returned by the call:
                          get_2d_distributions(get_full_profiles(equilibrium)).
    :param r_range: Plasma domain range (min, max) in R direction in meters.
    :param z_range: Plasma domain range (min, max) in Z direction in meters.
    :param atomic_data: Instance of AtomicData, default is OpenADAS()
    :param parent: parent of the plasma node, defaults None
    :param name: name of the plasma node, defaults "Generomak plasma"
    :return: populated Plasma object
    """

    equilibrium = equilibrium or load_equilibrium()

    distributions = distributions or get_2d_distributions(get_full_profiles(equilibrium=equilibrium))

    r_range = r_range or equilibrium.r_range
    z_range = z_range or equilibrium.z_range

    # create or check atomic_data
    if atomic_data is not None:
        if not isinstance(atomic_data, AtomicData):
            raise ValueError("atomic_data has to be of type AtomicData")
    else:
        atomic_data = OpenADAS()

    # construct plasma primitive shape
    padding = 1e-3  # enlarge for safety
    plasma_height = z_range[1] - z_range[0]
    outer_column = Cylinder(radius=r_range[1], height=plasma_height)
    inner_column = Cylinder(radius=r_range[0], height=plasma_height + 2 * padding)
    inner_column.transform = translate(0, 0, -padding)
    plasma_geometry = Subtract(outer_column, inner_column)

    # coordinate transform of the plasma frame
    geometry_transform = translate(0, 0, z_range[0])

    # create plasma composition list
    plasma_composition = []
    for elem_name, elem_data in distributions["composition"].items():
        for stage, stage_data in elem_data.items():
            elem = _get_cherab_element(elem_name)
            species = Species(elem, stage, stage_data)
            plasma_composition.append(species)

    # Populate plasma
    plasma = Plasma(parent=parent)
    plasma.name = name
    plasma.geometry = plasma_geometry
    plasma.atomic_data = atomic_data
    plasma.electron_distribution = distributions["electron"]
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
