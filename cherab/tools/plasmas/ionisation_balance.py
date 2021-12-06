
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


import numpy as np
from scipy.optimize import lsq_linear
from collections.abc import Iterable
from raysect.core.math.function.float import Function1D, Function2D, Interpolator1DArray, Interpolator2DArray

from cherab.core import AtomicData
from cherab.core.atomic import Element
from cherab.core.math import AxisymmetricMapper
from cherab.tools.equilibrium import EFITEquilibrium


def _parameters_to_numpy(*parameters, free_variable=None):
    """
    Check the consistency of parameters.

    Parameters can be scalar numbers, numpy arrays or dictionary of type {charge: rate}.

    :param parameters: List of parameters.
    :param free_variable: Free variable for the interpolating functions.
    :return: parameters formed into numpy array
    """

    # convert all input into numpy arrays and get their shape
    # interpolators are turned into arrays using free_variable
    arrays = []
    shapes = []

    # values of free variable has to be numpy arrays
    if free_variable is not None:
        if isinstance(free_variable, (list, tuple)):
            for index, value in enumerate(free_variable):
                if not isinstance(value, np.ndarray):
                    free_variable[index] = np.array([value])
        elif np.isscalar(free_variable):
            free_variable = np.array([free_variable])

    # take care of all possible input types
    for param in parameters:
        if np.isscalar(param) and not isinstance(param, str):
            arrays.append(np.array([param]))
        elif isinstance(param, dict):  # deal with dictionary
            # if first item is an interpolator, use shape of free_variable
            if isinstance(param[0], Function1D):
                array = np.zeros((len(param), *free_variable.shape))
            elif isinstance(param[0], Function2D):
                array = np.zeros((len(param), *free_variable[0].shape, *free_variable[1].shape))
            else:
                array = np.zeros((len(param), *param[0].shape))
            # convert items into numpy arrays
            for key, value in param.items():
                array[key, ...] = _parameters_to_numpy(value, free_variable=free_variable)[0]
            arrays.append(array)
        elif isinstance(param, Function1D):  # take care of Function1D input type
            array = np.zeros(free_variable.shape)
            for index, value in enumerate(free_variable):  # evaluate for free_variable
                array[index] = param(value)
            arrays.append(array)
        elif isinstance(param, Function2D):  # take care of Function2D input type
            if not isinstance(param, (tuple, list)) and not len(free_variable) == 2:
                raise ValueError(
                    "In case of 2d interpolator free variable has to be tupple of length 2 storing x and y coordinates")
            array = np.zeros((*free_variable[0].shape, *free_variable[1].shape))
            for xindex, xvalue in enumerate(free_variable[0]):
                for yindex, yvalue in enumerate(free_variable[1]):
                    array[xindex, yindex] = param(xvalue, yvalue)
            arrays.append(array)
        elif not isinstance(param, np.ndarray):  # well there are types which should not be treated
            raise ValueError(
                "Parameters can be Iterable, scalar, list, Function1D, Function2D or None, {0} passed".format(
                    type(param)))
        else:
            arrays.append(param)

        shapes.append(arrays[-1].shape)

    # test if all arrays have the same shape
    if not all(shapes[0] == shape for shape in shapes):
        raise ValueError("Profiles and free_variable have to have the same shape")

    return arrays


def _assign_donor_density(donor_density, major_profile, free_variable=None):
    """
    If donor density is none, it should be assigned a zeros numpy array of the shape matching free_variable or
    major_profile if free_variable is None. It is populated if donor_density is an interpolating function.

    :param donor_density: donor density value or interpolator
    :param free_variable: free_variable value
    :param major_profile: major_profile value
    :return: numpy array
    """
    # donor density should be array of the correct shape, if not assigned
    if donor_density is None:  # if none, it hsa to be replaced by an array of zeros of the correct shape
        if isinstance(major_profile, Function1D) and free_variable is not None:
            donor_density = np.zeros_like(free_variable)
        elif isinstance(major_profile, Function2D) and free_variable is not None:
            donor_density = np.zeros((*free_variable[0].shape, *free_variable[1].shape))
        elif isinstance(major_profile, (Function1D, Function2D)) and free_variable is None:
            raise ValueError("free_variable has to be passed along with an iterpolator")
        elif isinstance(major_profile, Iterable):
            donor_density = np.zeros_like(major_profile)
        elif np.isscalar(major_profile) or np.isscalar(free_variable):
            donor_density = np.zeros([1])
    elif isinstance(donor_density, (Function1D, Function2D)):
        donor_density = _parameters_to_numpy(donor_density, free_variable=free_variable)[0]

    if np.isscalar(donor_density) or donor_density.ndim == 0:
        donor_density = np.array([donor_density])

    return donor_density


def get_rates_ionisation(atomic_data: AtomicData, element: Element):
    """
    Returns recombination rate interpolators for individual ion charges of the specified
    element from the specified data source.

    :param atomic_data: Any cherab Element
    :param element: Any cherab AtomicData source
    :return: dictionary of the form {charge: Interpolator}
    """
    coef_ionis = {}
    for i in np.arange(0, element.atomic_number):
        coef_ionis[i] = atomic_data.ionisation_rate(element, int(i))

    return coef_ionis


def get_rates_recombination(atomic_data: AtomicData, element: Element):
    """
    Returns recombination rate interpolators for individual ion charges of
    the specified element from the specified data source.

    :param atomic_data: Any cherab Element
    :param element: Any cherab AtomicData source
    :return: dictionary of the form {charge: Interpolator}
    """
    coef_recom = {}
    for i in np.arange(1, element.atomic_number + 1):
        coef_recom[i] = atomic_data.recombination_rate(element, int(i))

    return coef_recom


def get_rates_tcx(atomic_data: AtomicData, donor: Element, donor_charge, receiver: Element):
    """
    Returns thermal charge-exchange rate interpolators for individual ion charges of the
    specified element and donor from the specified data source.

    :param atomic_data: Any cherab AtomicData source
    :param donor: Element donating the electron in the CX collision.
    :param donor_charge: Charge of the donating element.
    :param receiver: Element receiving electron in the collision.
    :return: dictionary of the form {charge: Interpolator}
    """
    coef_tcx = {}
    for i in np.arange(1, receiver.atomic_number + 1):
        coef_tcx[i] = atomic_data.thermal_cx_rate(donor, donor_charge, receiver, int(i))

    return coef_tcx


def _fractional_abundance_point(element: Element, n_e, t_e, coef_ion, coef_recom, coef_tcx=None, tcx_donor_density=0):
    """
    Calculate fractional abundance of charge states of the specified element, for the specified temperature and density using
    steady state ionization balance.

    If tcx_donor is specified, the balance equation will take into accout effects
    of charge exchage with specified donor. The results are returned as fractional abundances i.e. ratio of the individual
    ionic charge state density to the overall element density.

    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab Element
    :param n_e: Electron density in m^-3 to calculate the balance for
    :param t_e: Electron temperature in eV to calculate the balance for
    :param coef_ion: Dictionary with ionization rates
    :param coef_recom: Dictionary with recombination rates
    :param coef_tcx: Optional, dictionary with thermal cx rates
    :param tcx_donor: Optional, any cherab element. Specifies donating species in tcx collisions.
    :param tcx_donor_density: Optional, mandatory if tcx_donor parameter passed. Specifies density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :return: array with fractional abundances of ionic charges. Array indexes correspond to ion charge state.
    """

    # atomic number to determine ionisation matrix shape
    atomic_number = element.atomic_number

    matbal = np.zeros((atomic_number + 1, atomic_number + 1))  # create ionisation balance matrix

    # fill the 1st and last rows of the fractional abundance matrix
    matbal[0, 0] -= coef_ion[0](n_e, t_e)
    matbal[0, 1] += coef_recom[1](n_e, t_e)
    matbal[-1, -1] -= coef_recom[atomic_number](n_e, t_e)
    matbal[-1, -2] += coef_ion[atomic_number - 1](n_e, t_e)

    if coef_tcx is not None:
        matbal[0, 1] += tcx_donor_density / n_e * coef_tcx[1](n_e, t_e)
        matbal[-1, -1] -= tcx_donor_density / n_e * coef_tcx[atomic_number](n_e, t_e)

    # fill rest of the lines
    for i in range(1, atomic_number):
        matbal[i, i - 1] += coef_ion[i - 1](n_e, t_e)
        matbal[i, i] -= (coef_ion[i](n_e, t_e) + coef_recom[i](n_e, t_e))
        matbal[i, i + 1] += coef_recom[i + 1](n_e, t_e)
        if coef_tcx is not None:
            matbal[i, i] -= tcx_donor_density / n_e * coef_tcx[i](n_e, t_e)
            matbal[i, i + 1] += tcx_donor_density / n_e * coef_tcx[i + 1](n_e, t_e)

    # for some reason calculation of stage abundance seems to yield better results than calculation of fractional abun.
    matbal = matbal * n_e  # multiply by ne to calculate abundance instead of fractional abundance

    # add sum constraints. Sum of all stages should be equal to electron density
    matbal = np.concatenate((matbal, np.ones((1, matbal.shape[1]))), axis=0)

    # construct RHS of the balance steady-state equation
    rhs = np.zeros((matbal.shape[0]))
    rhs[-1] = n_e

    abundance = lsq_linear(matbal, rhs, bounds=(0, n_e))["x"]

    # normalize to ne to get fractional abundance
    frac_abundance = abundance / n_e

    return frac_abundance


def _from_element_density_point(atomic_data: AtomicData, element: Element, element_density, n_e, t_e,
                                tcx_donor: Element = None, tcx_donor_n=None, tcx_donor_charge=0,
                                coef_ion=None, coef_recom=None, coef_tcx=None):
    """
    Calculate density of charge states of the specified element, for the specified electron temperature,
    electron density and absolute element density using steady state ionization balance.

    If tcx_donor is specified, the balance equation will take into account effects of charge exchange
    with the specified donor. The results are returned as density in m^-3.

    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab Element
    :param element_density: Density of the element in m^-3
    :param n_e: Electron density in m^-3 to calculate the balance for
    :param t_e: Electron temperature in eV to calculate the balance for
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n: Optional, mandatory if tcx_donor parameter passed. Specifies density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :param coef_ion: Optional, ionization rates. If not passed rates will be loaded (slow).
    :param coef_recom: Optional, recombination rates. If not passed rates will be loaded (slow).
    :param coef_tcx: Optional, thermal cx rates. If not passed rates will be loaded (slow).
    :return: array with densities in m^-3 of ion charge states. Array indexes correspond to ion charge state.
    """

    # load atomic data for the element
    if coef_ion is None:
        coef_ion = get_rates_ionisation(atomic_data, element)  # get ionisation rate interpolators
    if coef_recom is None:
        coef_recom = get_rates_recombination(atomic_data, element)  # get recombination rate interpolators

    # get tcx rate interpolators if requested
    if tcx_donor is not None and coef_tcx is None:
        coef_tcx = get_rates_tcx(atomic_data, tcx_donor, tcx_donor_charge, element)
    else:
        coef_tcx = None

    # calculate fractional abundance for the element
    fractional_abundance = _fractional_abundance_point(element, n_e, t_e, coef_ion, coef_recom, coef_tcx,
                                                       tcx_donor_n)

    # convert fractional abundance to densities
    abundance = fractional_abundance * element_density

    # warn user if plasma neutrality is violated due to too low electron density for the specified element density
    n_e_fromions = np.sum(abundance)
    if n_e_fromions > n_e:
        print("Plasma neutrality violated, {0} density too large".format(element.name))

    return abundance


def _match_element_density_point(atomic_data: AtomicData, element: Element, n_species, n_e, t_e,
                                 tcx_donor: Element = None,
                                 tcx_donor_density=None, tcx_donor_charge=0, coef_ion=None, coef_recom=None,
                                 coef_tcx=None):
    """
    Calculate density of charge states of the specified element, for the specified
    electron temperature and density.

    Ratio of densities of ionization stages of the element follows the steady state
    balance calculation for given electron properties. The absolute density of the
    element is determined to match the plasma neutrality (electron density) together
    with the other (provided) ion species densities. It is useful for example to fill
    in the bulk (e.g. hydrogen isotope or even helium) plasma element once rest of
    the impurities are known.

    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element to calculate matching density for
    :param n_species: list of arrays or dictionaries with ion densities of the rest of the plasma elements
    :param n_e: electron density in m^-3
    :param t_e: electron temperature in eV
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_density: Optional, mandatory if tcx_donor parameter passed. Specifies density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :param coef_ion: Optional, ionization rates. If not passed rates will be loaded (slow).
    :param coef_recom: Optional, recombination rates. If not passed rates will be loaded (slow).
    :param coef_tcx: Optional, thermal cx rates. If not passed rates will be loaded (slow).
    :return: array with densities in m^-3 of ion charge states. Array indexes correspond to ion charge state.
    """

    # load atomic data for the element
    if coef_ion is None:
        coef_ion = get_rates_ionisation(atomic_data, element)  # get ionisation rate interpolators
    if coef_recom is None:
        coef_recom = get_rates_recombination(atomic_data, element)  # get recombination rate interpolators

    # get tcx rate interpolators if requested
    if tcx_donor is not None and coef_tcx is None:
        coef_tcx = get_rates_tcx(atomic_data, tcx_donor, tcx_donor_charge, element)
    else:
        coef_tcx = None
    # calculate fractional abundance for given electron properties
    fractional_abundance = _fractional_abundance_point(element, n_e, t_e, coef_ion, coef_recom, coef_tcx,
                                                       tcx_donor_density)

    # calculate contributions of other species to the electron density
    element_n_e = n_e
    for abundance in n_species:
        for index, value in enumerate(abundance):
            element_n_e -= index * value

    # avoid negative densities due to passed n_e being too small
    if element_n_e < 0:
        element_n_e = 0

    # calculate mean charge of the bulk element
    z_mean = 0
    for index, value in enumerate(fractional_abundance):
        z_mean += index * value

    # calculate elenent density and normalize the fractional abundance
    element_n_i = element_n_e / z_mean
    densities = fractional_abundance * element_n_i

    return densities


def _fractional_abundance(atomic_data: AtomicData, element: Element, n_e,
                          t_e, tcx_donor: Element = None, tcx_donor_n=None, tcx_donor_charge=0,
                          coef_ion=None, coef_recom=None, coef_tcx=None):
    """
    Calculate Fractional abundance of the specified element for the specified
    electron density and temperature.

    Returns values of fractional abundances of the charge states of the element
    for given plasma parameters.

    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param n_e: numpy ndarray of values of electron density in m$^{-3}$
    :param t_e: numpy ndarray of values of electron temperature in [ev]
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n: Optional, mandatory if tcx_donor parameter passed. Numpy ndarray of values of electron density
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :param coef_ion: Optional, ionization rates. If not passed rates will be loaded (slow).
    :param coef_recom: Optional, recombination rates. If not passed rates will be loaded (slow).
    :param coef_tcx: Optional, thermal cx rates. If not passed rates will be loaded (slow).
    :return: dim 0 corresponds to element charge state, dim > 0 correspond to dimensions of provided values.
    """

    # load atomic data for the element
    if coef_ion is None:
        coef_ion = get_rates_ionisation(atomic_data, element)  # get ionisation rate interpolators
    if coef_recom is None:
        coef_recom = get_rates_recombination(atomic_data, element)  # get recombination rate interpolators

    # get tcx rate interpolators if requested
    if tcx_donor is not None and coef_tcx is None:
        coef_tcx = get_rates_tcx(atomic_data, tcx_donor, tcx_donor_charge, element)
    else:
        coef_tcx = None

    density = np.zeros((element.atomic_number + 1, *n_e.shape))
    for index in np.ndindex(*n_e.shape):
        density[(Ellipsis, *index)] = _fractional_abundance_point(element, n_e[index], t_e[index],
                                                                  coef_ion, coef_recom, coef_tcx,
                                                                  tcx_donor_n[index])

    return density


def fractional_abundance(atomic_data: AtomicData, element: Element, n_e,
                         t_e, tcx_donor: Element = None, tcx_donor_n=None, tcx_donor_charge=0, free_variable=None):
    """
    Calculate Fractional abundance of the specified element for the specified electron density and temperature.

    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param n_e: Scalar, iterable or interpolating function of values of electron density in m$^{-3}$
    :param t_e: Scalar, iterable or interpolating function of values of electron temperature in [ev]
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n: Optional, mandatory if tcx_donor parameter passed. Scalar, iterable or interpolating function of values of electron density
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :param free_variable: Mantadory if n_e, t_e or tcx_donor_n is an interpolating function. If 2D interpolator is passed
     free_variable has to be list or tuple of 1D arrays with coordinates
    :return: Dictionary with values of fractional abundances in the form {charge: values}
    """

    # donor density should be array of the correct shape, if not assigned or if interpolating function passed
    tcx_donor_n = _assign_donor_density(tcx_donor_n, n_e, free_variable)

    # check consistency of parameters and transform them into numpy arrays to allow calculations of frac. abundance
    n_e, t_e, tcx_donor_n = _parameters_to_numpy(n_e, t_e, tcx_donor_n, free_variable=free_variable)

    # calculate fractional abundance
    fractional_abundance = _fractional_abundance(atomic_data, element, n_e, t_e, tcx_donor, tcx_donor_n,
                                                 tcx_donor_charge)

    # transform into dictionary
    fractional_abundance_dict = {}
    for index, value in enumerate(fractional_abundance):
        fractional_abundance_dict[index] = value

    return fractional_abundance_dict


def _from_elementdensity(atomic_data: AtomicData, element: Element, element_density, n_e_profile,
                         t_e_profile, tcx_donor: Element = None, tcx_donor_n_profile=None, tcx_donor_charge=0):
    """
    For given plasma parameters the function calulates charge state densities of the element.

    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param element_density: Density profile of the element in m^-3
    :param n_e: numpy ndarray of values of electron density in m$^{-3}$
    :param t_e: numpy ndarray of values of electron temperature in [ev]
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n: Optional, mandatory if tcx_donor parameter passed. numpy ndarray of values of electron density
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :return: dim 0 corresponds to element charge state, dim > 0 correspond to dimensions of provided values.
    """

    # load atomic data for the element
    coef_ion = get_rates_ionisation(atomic_data, element)  # get ionisation rate interpolators
    coef_recom = get_rates_recombination(atomic_data, element)  # get recombination rate interpolators

    # get tcx rate interpolators if requested
    if tcx_donor is not None:
        coef_tcx = get_rates_tcx(atomic_data, tcx_donor, tcx_donor_charge, element)
    else:
        coef_tcx = None

    density = np.zeros((element.atomic_number + 1, *n_e_profile.shape))
    for index in np.ndindex(*n_e_profile.shape):
        density[(Ellipsis, *index)] = _from_element_density_point(atomic_data, element, element_density[index],
                                                                  n_e_profile[index], t_e_profile[index], tcx_donor,
                                                                  tcx_donor_n_profile[index], tcx_donor_charge,
                                                                  coef_ion, coef_recom, coef_tcx)

    return density


def from_elementdensity(atomic_data: AtomicData, element: Element, element_density, n_e,
                        t_e, tcx_donor: Element = None, tcx_donor_n=None, tcx_donor_charge=0, free_variable=None):
    """
    For given plasma parameters the function calculates charge state densities of the element.

    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param element_density: Density profile of the element in m^-3
    :param n_e: Scalar or iterable of values of electron density in m$^{-3}$
    :param t_e: Scalar or iterable of values of electron temperature in [ev]
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n: Optional, mandatory if tcx_donor parameter passed. Scalar or iterable of values of donor density
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :param free_variable: Mantadory if n_e, t_e or tcx_donor_n is an interpolating function.If 2D interpolator is passed
     free_variable has to be list or tuple of 1D arrays with coordinates
    :return: Dictionary with density profiles of charge states of the element in the form {charge: profile}
    """

    # donor density should be array of the correct shape, if not assigned or if interpolating function passed
    tcx_donor_n_profile = _assign_donor_density(tcx_donor_n, n_e, free_variable)

    # check consistency of parameters and transform them into numpy arrays to allow calculations of frac. abundance
    element_density, n_e, t_e, tcx_donor_n = _parameters_to_numpy(element_density, n_e, t_e,
                                                                  tcx_donor_n_profile, free_variable=free_variable)

    # calculate density profiles
    densities = _from_elementdensity(atomic_data, element, element_density, n_e, t_e, tcx_donor,
                                     tcx_donor_n, tcx_donor_charge)

    # transform into dictionary
    densities_dict = {}
    for index, value in enumerate(densities):
        densities_dict[index] = value

    return densities_dict


def _match_plasma_neutrality(atomic_data: AtomicData, element: Element, n_species, n_e_profile, t_e_profile,
                             tcx_donor: Element = None, tcx_donor_n_profile=None, tcx_donor_charge=0):
    """
    For given profiles of plasma parameters the function calulates density profiles of charge states of the element.

    The density is normalized using n_species_profiles and n_e_profiles to reach plasma neutrality condition.

    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param element_density_profile: Density profile of the element in m^-3
    :param n_e: Scalar or iterable of values of electron density in m$^{-3}$
    :param t_e: Scalar or iterable of values of electron temperature in [ev]
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n: Optional, mandatory if tcx_donor parameter passed. Scalar or iterable of values of donor density
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :return: Density profiles of charge states of the element. Dim 0 corresponds to charge of charge states.
    """

    # load atomic data for the element
    coef_ion = get_rates_ionisation(atomic_data, element)  # get ionisation rate interpolators
    coef_recom = get_rates_recombination(atomic_data, element)  # get recombination rate interpolators

    # get tcx rate interpolators if requested
    if tcx_donor is not None:
        coef_tcx = get_rates_tcx(atomic_data, tcx_donor, tcx_donor_charge, element)
    else:
        coef_tcx = None

    number_chargestates = element.atomic_number + 1

    density = np.zeros((number_chargestates, *n_e_profile.shape))
    for index in np.ndindex(*n_e_profile.shape):
        spec_list = []
        for spec in n_species:
            spec_list.append(spec[(Ellipsis, *index)])
        density[(Ellipsis, *index)] = _match_element_density_point(atomic_data, element, spec_list, n_e_profile[index],
                                                                   t_e_profile[index], tcx_donor,
                                                                   tcx_donor_n_profile[index],
                                                                   tcx_donor_charge, coef_ion, coef_recom, coef_tcx)

    return density


def match_plasma_neutrality(atomic_data: AtomicData, element: Element, n_species, n_e, t_e,
                            tcx_donor: Element = None, tcx_donor_n=None, tcx_donor_charge=0, free_variable=None):
    """
    For given profiles of plasma parameters the function calulates density profiles of charge states of the element.

    The density is normalized using n_species_profiles and n_e_profiles to reach plasme neutrality condition.

    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param element_density_profile: Density profile of the element in m^-3
    :param n_e: 1d profile giving values of electron density for free_variable
    :param t_e: 1d profile giving values of electron density for free_variable
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n: Optional, mandatory if tcx_donor parameter passed. 1d profile giving density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :param free_variable: Mantadory if n_e, t_e or tcx_donor_n is an interpolating function. If 2D interpolator is passed
     free_variable has to be list or tuple of 1D arrays with coordinates
    :return: Dictionary with density profiles of charge states of the element in the form {charge: profile}
    """

    # donor density should be array of the correct shape, if not assigned or if interpolating function passed
    tcx_donor_n = _assign_donor_density(tcx_donor_n, n_e, free_variable=free_variable)

    # check consistency of parameters and transform them into numpy arrays to allow calculations of frac. abundance
    n_e, t_e, tcx_donor_n = _parameters_to_numpy(n_e, t_e, tcx_donor_n, free_variable=free_variable)

    n_species_arrays = []
    for spec in n_species:
        spec = _parameters_to_numpy(spec, free_variable=free_variable)[0]
        n_species_arrays.append(spec)

    # calculate density profiles
    densities = _match_plasma_neutrality(atomic_data, element, n_species_arrays, n_e, t_e, tcx_donor,
                                         tcx_donor_n, tcx_donor_charge)

    # transform into dictionary
    densities_dict = {}
    for index, value in enumerate(densities):
        densities_dict[index] = value

    return densities_dict


def interpolators1d_fractional(atomic_data: AtomicData, element: Element, free_variable, n_e,
                               t_e, tcx_donor: Element = None, tcx_donor_n=None, tcx_donor_charge=0):
    """
    Creates 1d linear interpolators of fractional abundance of the specified element
    for the specified electron densities and temperatures.

    For more information see _fractional_abundance function.

    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param free_variable: Free variable (coordinate) to calculate the 1d fractional abundance interpolators from.If 2D interpolator is passed
     free_variable has to be list or tuple of 1D arrays with coordinates
    :param n_e_interpolator: 1d iterable or interpolator giving values of electron density for free_variable
    :param t_e_interpolator: 1d iterable or  interpolator giving values of electron density for free_variable
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n_interpolator: Optional, mandatory if tcx_donor parameter passed. 1d iterable interpolator giving
     density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :return: dictionary with 1d interpolators of fractional abundance of charge states of the element in the form {charge: density}
    """

    fractional_profiles = fractional_abundance(atomic_data, element, n_e, t_e, tcx_donor, tcx_donor_n, tcx_donor_charge,
                                               free_variable=free_variable)

    # use profiles to create interpolators for profiles
    fractional_interpolators = {}
    for key, item in fractional_profiles.items():
        fractional_interpolators[key] = Interpolator1DArray(free_variable, item, 'linear', 'none', 0)

    return fractional_interpolators


def interpolators2d_fractional(atomic_data: AtomicData, element: Element, free_variable, n_e,
                               t_e, tcx_donor: Element = None, tcx_donor_n=None, tcx_donor_charge=0):
    """
    Creates 1d linear interpolators of fractional abundance of the specified element
    for the specified electron densities and temperatures.

    For more information see _fractional_abundance function.

    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param free_variable: Free variable (coordinate) to calculate the 1d fractional abundance interpolators from.If 2D interpolator is passed
     free_variable has to be list or tuple of 1D arrays with coordinates
    :param n_e_interpolator: 1d interpolator giving values of electron density for free_variable
    :param t_e_interpolator: 1d interpolator giving values of electron density for free_variable
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n_interpolator: Optional, mandatory if tcx_donor parameter passed. 1d interpolator giving density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :return: dictionary with 1d interpolators of fractional abundance of charge states of the element in the form {charge: density}
    """

    fractional_profiles = fractional_abundance(atomic_data, element, n_e, t_e, tcx_donor, tcx_donor_n, tcx_donor_charge,
                                               free_variable=free_variable)

    # use profiles to create interpolators for profiles
    fractional_interpolators = {}
    for key, item in fractional_profiles.items():
        fractional_interpolators[key] = Interpolator2DArray(*free_variable, item, 'linear', 'none', 0, 0)

    return fractional_interpolators


def interpolators1d_from_elementdensity(atomic_data: AtomicData, element: Element, free_variable,
                                        element_density, n_e, t_e, tcx_donor: Element = None,
                                        tcx_donor_n=None,
                                        tcx_donor_charge=0):
    """
    Creates 1d linear interpolators of density profiles of the specified element for
    the specified electron densities and temperatures.

    For more information see _from_element_density function.

    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param free_variable: Free variable (coordinate) to calculate the 1d fractional abundance interpolators from.
    :param element_density: 1d iterable or an interpolator giving values of element density for free_variable in m^-3
    :param n_e_interpolator: 1d interpolator giving values of electron density for free_variable
    :param t_e_interpolator: 1d interpolator giving values of electron density for free_variable
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n_interpolator: Optional, mandatory if tcx_donor parameter passed. 1d interpolator giving density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :return: dictionary with 1d interpolators of fractional abundance of charge states of the element in the form {charge: interpolator}
    """

    densities = from_elementdensity(atomic_data, element, element_density, n_e, t_e, tcx_donor, tcx_donor_n,
                                    tcx_donor_charge,
                                    free_variable=free_variable)

    density_interpolators = {}
    for key, value in densities.items():
        density_interpolators[key] = Interpolator1DArray(free_variable, value, 'linear', 'none', 0)

    return density_interpolators


def interpolators1d_match_plasma_neutrality(atomic_data: AtomicData, element: Element, free_variable,
                                            species_density,
                                            n_e, t_e, tcx_donor: Element = None,
                                            tcx_donor_n=None,
                                            tcx_donor_charge=0):
    """
    Creates 1d linear interpolators of density profiles of the specified
    element for the specified electron densities and temperatures.

    For more information see _match_element_density function.

    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param free_variable: Free variable (coordinate) to calculate the 1d fractional abundance interpolators from
    :param species_density: 1d interpolator giving values of the element density for free_variable
    :param n_e: 1d interpolator giving values of electron density for free_variable
    :param t_e: 1d interpolator giving values of electron density for free_variable
    :param tcx_donor: specifies donating species in tcx collisions.
    :param tcx_donor_n: Optional, mandatory if tcx_donor parameter passed. 1d interpolator giving density of donors in m^-3
    :param tcx_donor_charge:  Optional, specifies the charge of the donor. Default is 0.
    :return: dictionary with 1d interpolators of fractional abundance of charge states of the element in the form {charge: interpolator}
    """

    density_profiles = match_plasma_neutrality(atomic_data, element, species_density, n_e,
                                               t_e, tcx_donor, tcx_donor_n, tcx_donor_charge,
                                               free_variable=free_variable)

    # use profiles to create interpolators for profiles
    density_interpolators = {}
    for key, item in density_profiles.items():
        density_interpolators[key] = Interpolator1DArray(free_variable, item, 'linear', 'none', 0)

    return density_interpolators


def interpolators2d_from_elementdensity(atomic_data: AtomicData, element: Element, free_variable,
                                        element_density, n_e, t_e, tcx_donor: Element = None,
                                        tcx_donor_n=None, tcx_donor_charge=0):
    """
    Creates 1d linear interpolators of density profiles of the specified element
    for the specified electron densities and temperatures.

    For more information see _from_element_density function.

    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param free_variable: A tupple containing two 1D arrays of coordinate points
    :param element_density_interpolator: 1d interpolator giving values of element density for free_variable in m^-3
    :param n_e_interpolator: 1d interpolator giving values of electron density for free_variable
    :param t_e_interpolator: 1d interpolator giving values of electron density for free_variable
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n_interpolator: Optional, mandatory if tcx_donor parameter passed. 1d interpolator giving density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :return: dictionary with 1d interpolators of fractional abundance of charge states of the element in the form {charge: interpolator}
    """

    densities = from_elementdensity(atomic_data, element, element_density, n_e, t_e, tcx_donor, tcx_donor_n,
                                    tcx_donor_charge,
                                    free_variable=free_variable)

    density_interpolators = {}
    for key, value in densities.items():
        density_interpolators[key] = Interpolator2DArray(*free_variable, value, 'linear', 'none', 0, 0)

    return density_interpolators


def interpolators2d_match_plasma_neutrality(atomic_data: AtomicData, element: Element, free_variable,
                                            species_density,
                                            n_e, t_e, tcx_donor: Element = None,
                                            tcx_donor_n=None,
                                            tcx_donor_charge=0):
    """
    Creates 1d linear interpolators of density profiles of the specified element
    for the specified electron densities and temperatures.

    For more information see _match_element_density function.

    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param free_variable: A tupple containing two 1D arrays of coordinate poitns
    :param species_density: 1d interpolator giving values of the element density for free_variable
    :param n_e: 1d interpolator giving values of electron density for free_variable
    :param t_e: 1d interpolator giving values of electron density for free_variable
    :param tcx_donor: specifies donating species in tcx collisions.
    :param tcx_donor_n: Optional, mandatory if tcx_donor parameter passed. 1d interpolator giving density of donors in m^-3
    :param tcx_donor_charge:  Optional, specifies the charge of the donor. Default is 0.
    :return: dictionary with 1d interpolators of fractional abundance of charge states of the element in the form {charge: interpolator}
    """

    density_profiles = match_plasma_neutrality(atomic_data, element, species_density, n_e,
                                               t_e, tcx_donor, tcx_donor_n, tcx_donor_charge,
                                               free_variable=free_variable)

    # use profiles to create interpolators for profiles
    density_interpolators = {}
    for key, item in density_profiles.items():
        density_interpolators[key] = Interpolator2DArray(*free_variable, item, 'linear', 'none', 0, 0)

    return density_interpolators


def abundance_axisymmetric_mapper(abundance):
    """
    Convert 2d abundance interpolators into AxisymmetricMapper.

    :param abundance: Dictionary with 2d Abundace/fractional abundance interpolators
    """

    mappers = {}
    for key, item in abundance.items():
        mappers[key] = AxisymmetricMapper(item)

    return mappers


def equilibrium_map3d_fractional(atomic_data: AtomicData, element: Element, equilibrium: EFITEquilibrium, psin_1d,
                                 n_e_profile, t_e_profile, tcx_donor: Element = None,
                                 tcx_donor_n=None, tcx_donor_charge = 0):
    """
    Creates AxisymmetricMapper interpolator of fractional abundance of the specified
    element for the specified electron densities, temperatures and equilibrium by using
    the equilibrium.map3d function.

    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param equilibrium: EFITEquilibrium object
    :param psin_1d:1D array with normalized poloidal flux coordinates
    :param n_e_profile: 1d iterable or interpolator giving values of electron density
    :param t_e_profile: 1d iterable or interpolator giving values of electron temperature
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n: Optional, mandatory if tcx_donor parameter passed. 1d iterable interpolator giving
     density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    """

    fractional_profiles = interpolators1d_fractional(atomic_data, element, psin_1d, n_e_profile, t_e_profile,
                                                     tcx_donor, tcx_donor_n, tcx_donor_charge)

    mapped_3d = {}
    for key, item in fractional_profiles.items():
        mapped_3d[key] = equilibrium.map3d(item)

    return mapped_3d


def equilibrium_map3d_from_elementdensity(atomic_data: AtomicData, element: Element, equilibrium: EFITEquilibrium,
                                          psin_1d, n_element, n_e_profile, t_e_profile, tcx_donor: Element = None,
                                          tcx_donor_n=None, tcx_donor_charge=0):
    """
    Creates AxisymmetricMapper interpolator of fractional abundance of the
    specified element for the specified electron densities, temperatures,
    element densities and equilibrium by using the equilibrium.map3d function.

    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param equilibrium: EFITEquilibrium object
    :param psin_1d:1D array with normalized poloidal flux coordinates
    :param element_density: 1d iterable or an interpolator giving values of element density for free_variable in m^-3
    :param n_e_profile: 1d iterable or interpolator giving values of electron density
    :param t_e_profile: 1d iterable or interpolator giving values of electron temperature
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n: Optional, mandatory if tcx_donor parameter passed. 1d iterable interpolator giving
     density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    """

    abundance_profiles = interpolators1d_from_elementdensity(atomic_data, element, psin_1d, n_element, n_e_profile,
                                                             t_e_profile, tcx_donor, tcx_donor_n, tcx_donor_charge)

    mapped_3d = {}
    for key, item in abundance_profiles.items():
        mapped_3d[key] = equilibrium.map3d(item)

    return mapped_3d


def equilibrium_map3d_match_plasma_neutrality(atomic_data: AtomicData, element: Element, equilibrium: EFITEquilibrium,
                                              psin_1d, species_density, n_e_profile, t_e_profile, tcx_donor: Element = None,
                                              tcx_donor_n=None, tcx_donor_charge=0):
    """
    Creates AxisymmetricMapper interpolator of fractional abundance of the specified
    element for the specified electron densities, temperatures
    and equilibrium by using the equilibrium.map3d function.

    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param equilibrium: EFITEquilibrium object
    :param psin_1d:1D array with normalized poloidal flux coordinates
    :param species_density: list of 1d iterables interpolators giving values of element density for the values of psi in m^-3
    :param n_e_profile: 1d iterable or interpolator giving values of electron density
    :param t_e_profile: 1d iterable or interpolator giving values of electron temperature
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n: Optional, mandatory if tcx_donor parameter passed. 1d iterable interpolator giving
     density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    """

    abundance_profiles = match_plasma_neutrality(atomic_data, element, species_density, n_e_profile, t_e_profile, tcx_donor, tcx_donor_n,
                                               tcx_donor_charge, free_variable=psin_1d)

    mapped_3d = {}
    for key, item in abundance_profiles.items():
        mapped_3d[key] = equilibrium.map3d((psin_1d, item))

    return mapped_3d
