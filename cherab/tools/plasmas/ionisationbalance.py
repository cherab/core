import numpy as np
from scipy.optimize import lsq_linear
from cherab.core import AtomicData
from cherab.core.atomic import Element
from cherab.core.math import Interpolate1DLinear, Function1D, Function2D


def _evaluate_1d(free_variable, interpolator):
    #if interpolator passed, free variable has to be checked
    if free_variable is None:
        raise ValueError("free_variable has to be provided if one of the parametres is an interpolator.")
    else:
        if np.isscalar(free_variable):#minimum number of dimensions has to be 1
            free_variable = np.array([free_variable])
        elif isinstance(free_variable, str) or not isinstance(free_variable, (np.ndarray, list, tuple)): #check types
            raise ValueError("free variable parameter can be np.ndarray, scalar or None, {0} passed".format(type(free_variable)))

    #use the interpolating function to populate the array
    array = np.zeros_like(free_variable)
    for index in np.ndindex(*free_variable.shape):
        array[index] = interpolator(free_variable[index])

    return array

def _evaluate_2d(free_variable, interpolator):
    #if interpolator passed, free variable has to be checked
    if free_variable is None:
        raise ValueError("free_variable has to be provided if one of the parametres is an interpolator.")

    x = free_variable[0]
    y = free_variable[1]

    if np.isscalar(x):#minimum number of dimensions has to be 1
        x = np.array([x])
    elif isinstance(x, str) or not isinstance(x, (np.ndarray, list, tuple)): #check types
        raise ValueError("free variable parameter can be np.ndarray, scalar or None, {0} passed".format(type(free_variable)))


    if np.isscalar(y):#minimum number of dimensions has to be 1
        y = np.array([y])
    elif isinstance(y, str) or not isinstance(y, (np.ndarray, list, tuple)): #check types
        raise ValueError("free variable parameter can be np.ndarray, scalar or None, {0} passed".format(type(free_variable)))

    #use the interpolating function to populate the array
    array = np.zeros_like(x)
    for index in np.ndindex(*x.shape):
        array[index] = interpolator(x[index], y[index])

    return array

def _parametres_to_numpy(*parametres, free_variable = None):
    """
    Check the consistency of parametres. Parametres can be scalar numbers, numpy arrays or of type Function1D or Function2D.
    The shapes of parametres have to be same. Shape of free_variable has to be the same of that of the parametres if both arrays and functions are passed.
    If so, nunpy arrays are returned.

    :param parametres: List of parametres.
    :param free_variable: Free variable for the interpolating functions.
    :return: parametres
    """

    # convert all input into numpy arrays and get their shape
    # interpolators are turned into arrays using free_variable
    arrays = []
    shapes = []
    for param in parametres:
        if np.isscalar(param) and not isinstance(param, str):
            arrays.append(np.array([param]))
        #dict of interpolating functions or arrays
        elif isinstance(param, (Function1D, Function2D)):
            #if interpolator passed, free variable has to be checked
            if free_variable is None:
                raise ValueError("free_variable has to be provided if one of the parametres is an interpolator.")
            else:
                if np.isscalar(free_variable):#minimum number of dimensions has to be 1
                    free_variable = np.array([free_variable])
                elif isinstance(free_variable, str) or not isinstance(free_variable, (np.ndarray, list, tuple)): #check types
                    raise ValueError("free variable parameter can be np.ndarray, scalar or None, {0} passed".format(type(free_variable)))

            #use the interpolating function to populate the array
            arrays.append(np.zeros_like(free_variable))
            for index in np.ndindex(*free_variable.shape):
                arrays[-1][index] = param(free_variable[index])
        elif not isinstance(param, np.ndarray):
            raise ValueError("Parametres can be np.ndarray, scalar, Function1D, Function2D or None, {0} passed".format(type(param)))
        else:
            arrays.append(param)

        shapes.append(arrays[-1].shape)

    #test if all arrays have the same shape
    if not all(shapes[0] == shape for shape in shapes):
        raise ValueError("Profiles and free_variable have to have the same shape")

    return arrays


def _assign_donor_density(donor_density, free_variable, major_profile):
    """
    If donor density is none, it should be assigned a zeros numpy array of the shape matching free_variable or
    major_profile if free_variable is None. It is populated if donor_density is an interpolating function.
    :param donor_density: donor density value or interpolator
    :param free_variable: free_variable value
    :param major_profile: major_profile value
    :return: numpy array
    """
    # donor density should be array of the correct shape, if not assigned
    if donor_density is None:
        if free_variable is not None:
            if np.isscalar(free_variable):
                donor_density = np.array([0])
            else:
                donor_density = np.zeros_like(free_variable)
        else:
            if np.isscalar(major_profile):
                donor_density = np.array([0])
            else:
                donor_density = np.zeros_like(major_profile)
    elif isinstance(donor_density, (Function1D, Function2D)):
        donor_density = _parametres_to_numpy(donor_density, free_variable=free_variable)



    return donor_density


def get_rates_ionisation(atomic_data: AtomicData, element: Element):
    """
    Returns recombination rate interpolators for individual ion charges of the specified element from the specified data source.
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
    Returns recombination rate interpolators for individual ion charges of the specified element from the specified data source.
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
    Returns thermal charge-exchange rate interpolators for individual ion charges of the specified element and donor from the specified data source.
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


def _fractional_abundance_point(atomic_data: AtomicData, element: Element, n_e, t_e, tcx_donor: Element = None,
                                tcx_donor_density=0, tcx_donor_charge=0):
    """
    Calculate fractional abundance of charge states of the specified element, for the specified temperature and density using
    steady state ionization balance. If tcx_donor is specified, the balance equation will take into accout effects
    of charge exchage with specified donor. The results are returned as fractional abundances i.e. ratio of the individual
    ionic charge state density to the overall element density.
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab Element
    :param n_e: Electron density in m^-3 to calculate the balance for
    :param t_e: Electron temperature in eV to calculate the balance for
    :param tcx_donor: Optional, any cherab element. Specifies donating species in tcx collisions.
    :param tcx_donor_density: Optional, mandatory if tcx_donor parameter passed. Specifies density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :return: array with fractional abundances of ionic charges. Array indexes correspond to ion charge state.
    """

    coef_ion = get_rates_ionisation(atomic_data, element)  # get ionisation rate interpolators
    coef_recom = get_rates_recombination(atomic_data, element)  # get recombination rate interpolators

    # get tcx rate interpolators if requested
    if tcx_donor is not None and tcx_donor_density > 0:
        coef_tcx = get_rates_tcx(atomic_data, tcx_donor, tcx_donor_charge, element)

    # atomic number to determine ionisation matrix shape
    atomic_number = element.atomic_number

    matbal = np.zeros((atomic_number + 1, atomic_number + 1))  # create ionisation balance matrix

    # fill the 1st and last rows of the fractional abundance matrix
    matbal[0, 0] -= coef_ion[0](n_e, t_e)
    matbal[0, 1] += coef_recom[1](n_e, t_e)
    matbal[-1, -1] -= coef_recom[atomic_number](n_e, t_e)
    matbal[-1, -2] += coef_ion[atomic_number - 1](n_e, t_e)

    if tcx_donor is not None and tcx_donor_density > 0:
        matbal[0, 1] += tcx_donor_density / n_e * coef_tcx[1](n_e, t_e)
        matbal[-1, -1] -= tcx_donor_density / n_e * coef_tcx[atomic_number](n_e, t_e)

    # fill rest of the lines
    for i in range(1, atomic_number):
        matbal[i, i - 1] += coef_ion[i - 1](n_e, t_e)
        matbal[i, i] -= (coef_ion[i](n_e, t_e) + coef_recom[i](n_e, t_e))
        matbal[i, i + 1] += coef_recom[i + 1](n_e, t_e)
        if tcx_donor is not None and tcx_donor_density > 0:
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
                                tcx_donor: Element = None, tcx_donor_density=None, tcx_donor_charge=0):
    """
    Calculate density of charge states of the specified element, for the specified electron temperature, electron density
    and absolute element density using steady state ionization balance. If tcx_donor is specified, the balance equation will take into
    accout effects of charge exchage with specified donor. The results are returned as density in m^-3
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab Element
    :param element_density: Density of the element in m^-3
    :param n_e: Electron density in m^-3 to calculate the balance for
    :param t_e: Electron temperature in eV to calculate the balance for
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_density: Optional, mandatory if tcx_donor parameter passed. Specifies density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :param element_density: dictionary of the form {charge: density} with density in m^-3
    :return: array with densities in m^-3 of ion charge states. Array indexes correspond to ion charge state.
    """
    # calculate fractional abundance for the element
    fractional_abundance = _fractional_abundance_point(atomic_data, element, n_e, t_e, tcx_donor, tcx_donor_density,
                                                       tcx_donor_charge)

    # convert fractional abundance to densities
    abundance = fractional_abundance * element_density

    # warn user if plasma neutrality is violated due to too low electron density for the specified element density
    n_e_fromions = np.sum(abundance)
    if n_e_fromions > n_e:
        print("Plasma neutrality violated, {0} density too large".format(element.name))

    return abundance


def _match_element_density_point(atomic_data: AtomicData, element: Element, n_species, n_e, t_e, tcx_donor: Element = None,
                                 tcx_donor_density=None, tcx_donor_charge=0):
    """
    Calculate density of charge states of the specified element, for the specified electron temperature and density.
    Ratio of densities of ionization stages of the element follows the steady state balance calculation for given electron properties.
    The absolute density of the element is determined to match the plasma neutrality (electron density) together with the other (provided) ion species densities.
    It is useful for example to fill in the bulk (e.g. hydrogen isotope or even helium) plasma element once rest of the impurities are
    known.
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element to calculate matching density for
    :param n_species: list of arrays or dictionaries with ion densities of the rest of the plasma elements
    :param n_e: electron density in m^-3
    :param t_e: electron temperature in eV
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_density: Optional, mandatory if tcx_donor parameter passed. Specifies density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :return: array with densities in m^-3 of ion charge states. Array indexes correspond to ion charge state.
    """
    if not isinstance(n_species, list) and not isinstance(n_species, dict) and not isinstance(n_species, np.ndarray):
        raise TypeError(
            "abundances has to be dictionary or a numpy array holding information about densities of a single element in the form {charge:density} or list of such dictionaries.")
    elif isinstance(n_species, dict) or isinstance(n_species, np.ndarray):
        n_species = [n_species]

    # extract possible dicts into numpy array
    for i in range(len(n_species)):
        if isinstance(n_species[i], dict):
            abundance = n_species[i]
            n_species[i] = np.zeros((len(abundance)))
            for key, item in abundance.items():
                n_species[i][key] = item

    # calculate fractional abundance for given electron properties
    fractional_abundance = _fractional_abundance_point(atomic_data, element, n_e, t_e, tcx_donor, tcx_donor_density,
                                                       tcx_donor_charge)

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
                          free_variable = None):
    """
    Calculate Fractional abundance of the specified element for the specified electron density and temperature.
    For more information see _fractional_abundance function. Returns values of fractional abundances of the charge
    states of the element for given plasma parametres.
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param n_e: Scalar, iterable of interpolating function providing values of electron density in m$^{-3}$
    :param t_e: Scalar, iterable of interpolating function providing values of electron temperature in [ev]
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n: Optional, mandatory if tcx_donor parameter passed. Scalar, iterable of interpolating function
                        providing values of electron density
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :param free_variable: Mantadory if n_e, t_e or tcx_donor_n is an interpolating function
    :return: dim 0 corresponds to element charge state, dim > 0 correspond to dimensions of provided values.
    """

    # donor density should be array of the correct shape, if not assigned or if interpolating function passed
    tcx_donor_n = _assign_donor_density(tcx_donor_n, free_variable, n_e)

    #check consistency of parametres and transform them into numpy arrays to allow calculations of frac. abundance
    n_e, t_e, tcx_donor_n = _parametres_to_numpy(n_e, t_e, tcx_donor_n, free_variable=free_variable)


    density = np.zeros((element.atomic_number + 1, *n_e.shape))
    for index in np.ndindex(*n_e.shape):
        density[(Ellipsis, *index)] = _fractional_abundance_point(atomic_data, element, n_e[index], t_e[index], tcx_donor, tcx_donor_n[index], tcx_donor_charge)

    return density


def fractional_abundance(atomic_data: AtomicData, element: Element, n_e,
                         t_e, tcx_donor: Element = None, tcx_donor_n=None, tcx_donor_charge=0,
                         free_variable = None):
    """
    Calculate Fractional abundance of the specified element for the specified electron density and temperature.
    For more information see _fractional_abundance function.
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param n_e: Scalar, iterable of interpolating function providing values of electron density in m$^{-3}$
    :param t_e: Scalar, iterable of interpolating function providing values of electron temperature in [ev]
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n: Optional, mandatory if tcx_donor parameter passed. Scalar, iterable of interpolating function
                        providing values of electron density
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :param free_variable: Mantadory if n_e, t_e or tcx_donor_n is an interpolating function
    :return: Dictionary with values of fractional abundances in the form {charge: values}
    """

    #calculate fractional abundance
    fractional_abundance = _fractional_abundance(atomic_data, element, n_e, t_e, tcx_donor, tcx_donor_n, tcx_donor_charge,
                                                 free_variable)

        # transform into dictionary
    fractional_abundance_dict = {}
    for index, value in enumerate(fractional_abundance):
        fractional_abundance_dict[index] = value

    return fractional_abundance_dict


def _from_elementdensity(atomic_data: AtomicData, element: Element, element_density, n_e_profile,
                         t_e_profile, tcx_donor: Element = None, tcx_donor_n_profile=None, tcx_donor_charge=0,
                         free_variable=None):
    """
    For given plasma parametres the function calulates charge state densities of the element. For more
    information see _from_element_density function.
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param element_density: Density profile of the element in m^-3
    :param n_e: Scalar, iterable of interpolating function providing values of electron density in m$^{-3}$
    :param t_e: Scalar, iterable of interpolating function providing values of electron temperature in [ev]
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n: Optional, mandatory if tcx_donor parameter passed. Scalar, iterable of interpolating function
                        providing values of electron density
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :param free_variable: Mantadory if n_e, t_e or tcx_donor_n is an interpolating function
    :return: dim 0 corresponds to element charge state, dim > 0 correspond to dimensions of provided values.
    """


    # donor density should be array of the correct shape, if not assigned or if interpolating function passed
    tcx_donor_n_profile = _assign_donor_density(tcx_donor_n_profile, free_variable, n_e_profile)

    #check consistency of parametres and transform them into numpy arrays to allow calculations of frac. abundance
    element_density, n_e_profile, t_e_profile, tcx_donor_n_profile = _parametres_to_numpy(element_density, n_e_profile, t_e_profile,
                                                                         tcx_donor_n_profile, free_variable=free_variable)

    density = np.zeros((element.atomic_number + 1, *n_e_profile.shape))
    for index in np.ndindex(*n_e_profile.shape):
        density[(Ellipsis, *index)] = _from_element_density_point(atomic_data, element, element_density[index], n_e_profile[index], t_e_profile[index], tcx_donor, tcx_donor_n_profile[index], tcx_donor_charge)

    return density


def from_elementdensity(atomic_data: AtomicData, element: Element, element_density, n_e,
                        t_e, tcx_donor: Element = None, tcx_donor_n=None, tcx_donor_charge=0,
                        free_variable=None):
    """
    For given plasma parametres the function calulates charge state densities of the element. For more
    information see _from_element_density function.
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param element_density: Density profile of the element in m^-3
    :param n_e: Scalar, iterable of interpolating function providing values of electron density in m$^{-3}$
    :param t_e: Scalar, iterable of interpolating function providing values of electron temperature in [ev]
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n: Optional, mandatory if tcx_donor parameter passed. Scalar, iterable of interpolating function
                        providing values of electron density
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :param free_variable: Mantadory if n_e, t_e or tcx_donor_n is an interpolating function
    :return: Dictionary with density profiles of charge states of the element in the form {charge: profile}
    """


    #calculate density profiles
    densities = _from_elementdensity(atomic_data, element, element_density, n_e, t_e, tcx_donor,
                                     tcx_donor_n, tcx_donor_charge, free_variable)

    # transform into dictionary
    densities_dict = {}
    for index, value in enumerate(densities):
        densities_dict[index] = value

    return densities_dict


def _match_plasma_neutrality(atomic_data: AtomicData, element: Element, n_species, n_e_profile, t_e_profile,
                             tcx_donor: Element = None, tcx_donor_n_profile=None, tcx_donor_charge=0,
                             free_variable=None):
    """
    For given profiles of plasma parametres the function calulates density profiles of charge states of the element. The density
    is normalized using n_species_profiles and n_e_profiles to reach plasme neutrality condition.
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param element_density_profile: Density profile of the element in m^-3
    :param n_e_profile: 1d profile giving values of electron density for free_variable
    :param t_e_profile: 1d profile giving values of electron density for free_variable
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n_profile: Optional, mandatory if tcx_donor parameter passed. 1d profile giving density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :return: Density profiles of charge states of the element. Dim 0 corresponds to charge of charge states.
    """

    # donor density should be array of the correct shape, if not assigned or if interpolating function passed
    tcx_donor_n_profile = _assign_donor_density(tcx_donor_n_profile, free_variable, n_e_profile)

    #check consistency of parametres and transform them into numpy arrays to allow calculations of frac. abundance
    n_e_profile, t_e_profile, tcx_donor_n_profile = _parametres_to_numpy(n_e_profile, t_e_profile,
                                                                         tcx_donor_n_profile, free_variable=free_variable)

    number_chargestates = element.atomic_number + 1

    n_species_arrays = []
    #extract dictionaries into lists
    for spec in n_species:
        if isinstance(spec, dict):
            n_species_arrays.append([])
            for key, item in spec.items():
                n_species_arrays[-1].append(item)
        else:
            n_species_arrays.append(spec)

    #transfer lists into arrays with densities
    for i in range(len(n_species_arrays)):
        if isinstance(spec, list):
            tmp = _parametres_to_numpy(*n_species_arrays[i], free_variable=free_variable)
            n_species_arrays[i] = np.zeros((len(tmp), *n_e_profile.shape))
            for j in range(len(tmp)):
                n_species_arrays[i][j,...] = tmp[j]

    # check element profile consistency
    for spec in n_species:
        if not spec[0, ...].shape == n_e_profile.shape:
            raise ValueError("profile shapes have to be identical")

    density = np.zeros((number_chargestates, *n_e_profile.shape))
    for index in np.ndindex(*n_e_profile.shape):
        spec_list = []
        for spec in n_species:
            spec_list.append(spec[(Ellipsis, *index)])
        density[(Ellipsis, *index)] = _match_element_density_point(atomic_data, element, spec_list, n_e_profile[index], t_e_profile[index], tcx_donor, tcx_donor_n_profile[index], tcx_donor_charge)

    return density


def match_plasma_neutrality(atomic_data: AtomicData, element: Element, n_species_profile, n_e_profile, t_e_profile,
                            tcx_donor: Element = None, tcx_donor_n_profile=None, tcx_donor_charge=0):
    """
    For given profiles of plasma parametres the function calulates density profiles of charge states of the element. The density
    is normalized using n_species_profiles and n_e_profiles to reach plasme neutrality condition.
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param element_density_profile: Density profile of the element in m^-3
    :param n_e_profile: 1d profile giving values of electron density for free_variable
    :param t_e_profile: 1d profile giving values of electron density for free_variable
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n_profile: Optional, mandatory if tcx_donor parameter passed. 1d profile giving density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :return: Dictionary with density profiles of charge states of the element in the form {charge: profile}

    """

    #calculate density profiles
    densities = _from_elementdensity(atomic_data, element, n_e_profile, n_e_profile, t_e_profile, tcx_donor,
                                     tcx_donor_n_profile, tcx_donor_charge)

    # transform into dictionary
    densities_dict = {}
    for index, value in enumerate(densities):
        densities_dict[index] = value

    return densities_dict


def interpolators1d_fractional(atomic_data: AtomicData, element: Element, free_variable, n_e,
                               t_e, tcx_donor: Element = None, tcx_donor_n=None, tcx_donor_charge=0):
    """
    Creates 1d linear interpolators of fractional abundance of the specified element for the specified electron densities and temperatures.
    For more information see _fractional_abundance function.
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param free_variable: Free variable (coordinate) to calculate the 1d fractional abundance interpolators from
    :param n_e_interpolator: 1d interpolator giving values of electron density for free_variable
    :param t_e_interpolator: 1d interpolator giving values of electron density for free_variable
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n_interpolator: Optional, mandatory if tcx_donor parameter passed. 1d interpolator giving density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :return: dictionary with 1d interpolators of fractional abundance of charge states of the element in the form {charge: density}
    """

    # donor density should be array of the correct shape, if not assigned or if interpolating function passed
    tcx_donor_n = _assign_donor_density(tcx_donor_n, free_variable, n_e)

    #check consistency of parametres and transform them into numpy arrays to allow calculations of frac. abundance
    n_e_profile, t_e_profile, tcx_donor_n_profile = _parametres_to_numpy(n_e, t_e, tcx_donor_n, free_variable=free_variable)

    fractional_profiles = _fractional_abundance(atomic_data, element, n_e, t_e, tcx_donor, tcx_donor_n, tcx_donor_charge,
                                                free_variable=free_variable)

    # use profiles to create interpolators for profiles
    fractional_interpolators = {}
    for index, value in enumerate(fractional_profiles):
        fractional_interpolators[index] = Interpolate1DLinear(free_variable, value)

    return fractional_interpolators


def interpolators1d_from_elementdensity(atomic_data: AtomicData, element: Element, free_variable,
                                        element_density_interpolator,
                                        n_e_interpolator, t_e_interpolator, tcx_donor: Element = None,
                                        tcx_donor_n_interpolator=None,
                                        tcx_donor_charge=0):
    """
    Creates 1d linear interpolators of density profiles of the specified element for the specified electron densities and temperatures.
    For more information see _from_element_density function.
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param free_variable: Free variable (coordinate) to calculate the 1d fractional abundance interpolators from
    :param element_density_interpolator: 1d interpolator giving values of element density for free_variable in m^-3
    :param n_e_interpolator: 1d interpolator giving values of electron density for free_variable
    :param t_e_interpolator: 1d interpolator giving values of electron density for free_variable
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n_interpolator: Optional, mandatory if tcx_donor parameter passed. 1d interpolator giving density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :return: dictionary with 1d interpolators of fractional abundance of charge states of the element in the form {charge: interpolator}
    """

    # Function returning None instead of donor density interpolator to later reduce number of if statements
    if tcx_donor is None:
        tcx_donor_n_interpolator = lambda psin: 0

    # calculate fractional abundance profiles for element ionic stages
    atomic_chargestates = element.atomic_number + 1
    fractional_profile = {}
    for i in range(atomic_chargestates):
        fractional_profile[i] = np.zeros((len(free_variable)))

    n_e_profile = np.zeros_like(free_variable)
    t_e_profile = np.zeros_like(free_variable)
    element_density_profile = np.zeros_like(free_variable)
    tcx_donor_n_profile = np.zeros_like(free_variable)

    # construct profiles from free_variable and interpolators
    for index, value in enumerate(free_variable):
        n_e_profile[index] = n_e_interpolator(value)
        t_e_profile[index] = t_e_interpolator(value)
        element_density_profile[index] = element_density_interpolator(value)
        tcx_donor_n_profile[index] = tcx_donor_n_interpolator(value)

    # calculate fractional abundace profiles
    density_profiles = _from_elementdensity(atomic_data, element, element_density_profile, n_e_profile,
                                            t_e_profile, tcx_donor, tcx_donor_n_profile, tcx_donor_charge)

    # use profiles to create interpolators for profiles
    density_interpolators = {}

    for rownumber, row in enumerate(density_profiles):
        density_interpolators[rownumber] = Interpolate1DLinear(free_variable, row)

    return density_interpolators


def interpolators1d_match_plasma_neutrality(atomic_data: AtomicData, element: Element, free_variable,
                                            species_density_interpolators,
                                            n_e_interpolator, t_e_interpolator, tcx_donor: Element = None,
                                            tcx_donor_n_interpolator=None,
                                            tcx_donor_charge=0):
    """
    Creates 1d linear interpolators of density profiles of the specified element for the specified electron densities and temperatures.
    For more information see _match_element_density function.
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param free_variable: Free variable (coordinate) to calculate the 1d fractional abundance interpolators from
    :param species_density_interpolators: 1d interpolator giving values of the element density for free_variable
    :param n_e_interpolator: 1d interpolator giving values of electron density for free_variable
    :param t_e_interpolator: 1d interpolator giving values of electron density for free_variable
    :param tcx_donor: specifies donating species in tcx collisions.
    :param tcx_donor_n_interpolator: Optional, mandatory if tcx_donor parameter passed. 1d interpolator giving density of donors in m^-3
    :param tcx_donor_charge:  Optional, specifies the charge of the donor. Default is 0.
    :return: dictionary with 1d interpolators of fractional abundance of charge states of the element in the form {charge: interpolator}
    """

    if not isinstance(species_density_interpolators, list) and not isinstance(species_density_interpolators, dict):
        raise TypeError(
            "abundances has to be dictionary holding information about densities of a single element in the form {charge:density} or list of such dictionaries.")
    elif isinstance(species_density_interpolators, dict):
        species_density_interpolators = [species_density_interpolators]

    # Function returning None instead of donor density interpolator to later reduce number of if statements
    if tcx_donor is None:
        tcx_donor_n_interpolator = lambda psin: 0

    n_e_profile = np.zeros_like(free_variable)
    t_e_profile = np.zeros_like(free_variable)
    tcx_n_donor_profile = np.zeros_like(free_variable)

    density_spec_profile = []
    for spec in species_density_interpolators:
        density_spec_profile.append(np.zeros((len(spec), free_variable.shape[0])))

    for index, value in enumerate(free_variable):
        n_e_profile[index] = n_e_interpolator(value)
        t_e_profile[index] = t_e_interpolator(value)
        tcx_n_donor_profile[index] = tcx_donor_n_interpolator(value)

        for i in range(len(species_density_interpolators)):
            for key, item in species_density_interpolators[i].items():
                density_spec_profile[i][key, index] = item(value)

    density_profiles = _match_plasma_neutrality(atomic_data, element, density_spec_profile, n_e_profile, t_e_profile,
                                                tcx_donor,
                                                tcx_n_donor_profile, tcx_donor_charge)
    # use profiles to create interpolators for profiles
    density_interpolators = {}
    for rownumber, row in enumerate(density_profiles):
        density_interpolators[rownumber] = Interpolate1DLinear(free_variable, row)

    return density_interpolators
