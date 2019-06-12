import numpy as np
from scipy.optimize import lsq_linear
from cherab.core import AtomicData
from cherab.core.atomic import Element
from cherab.core.math import Interpolate1DLinear


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


def _fractional_abundance(atomic_data: AtomicData, element: Element, n_e, t_e, tcx_donor: Element = None,
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


def fractional_abundance(atomic_data: AtomicData, element: Element, n_e, t_e, tcx_donor: Element = None,
                         tcx_donor_density=None, tcx_donor_charge=0):
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
    :return: dictionary of the form {charge: fractional abundance}
    """

    # get fractional abundances for the specifies conditions
    frac_abundance = _fractional_abundance(atomic_data, element, n_e, t_e, tcx_donor, tcx_donor_density,
                                           tcx_donor_charge)

    # create the dictionary to return the results in
    abundance_dict = {}
    for key, item in enumerate(frac_abundance):
        abundance_dict[key] = item

    return abundance_dict


def _from_element_density(atomic_data: AtomicData, element: Element, element_density, n_e, t_e,
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
    fractional_abundance = _fractional_abundance(atomic_data, element, n_e, t_e, tcx_donor, tcx_donor_density,
                                                 tcx_donor_charge)

    # convert fractional abundance to densities
    abundance = fractional_abundance * element_density

    # warn user if plasma neutrality is violated due to too low electron density for the specified element density
    n_e_fromions = np.sum(abundance)
    if n_e_fromions > n_e:
        print("Plasma neutrality violated, {0} density too large".format(element.name))

    return abundance


def from_element_density(atomic_data: AtomicData, element: Element, element_density, n_e, t_e,
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
    :param element_density: dictionary of the form {charge: density}
    :return: dictionary of the form {charge: density}. Densities are in m^-3
    """
    # calculate fractional abundance for the element
    abundance = _from_element_density(atomic_data, element, element_density, n_e, t_e, tcx_donor, tcx_donor_density,
                                      tcx_donor_charge)

    abundance_dict = {}
    # convert fractional abundance to densities
    for key, value in enumerate(abundance):
        abundance_dict[key] = value

    return abundance_dict


def from_stage_density(atomic_data: AtomicData, element: Element, stage_charge, stage_density, n_e, t_e,
                       tcx_donor: Element = None, tcx_donor_density=None, tcx_donor_charge=0):
    """
    Calculate density of individual charge states for the specified element,electron temperature, electron density and density of a single charge state
    using steady state ionization balance. The total density of the element is calculated from a fractional abundance calculation for the specified electron
    population properties and from the density of the specified ion charge state. This is approach can lead to wrong results if the fraction of the
    specified charge state is low. If tcx_donor is specified, the balance equation will take into accout effects
    of charge exchage with specified donor. The results are returned as density in m^-3
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab Element
    :param stage_charge: Charge of the ionic stage to calculate densities from
    :param stage_density: Density of the ionic stage to calculate densities from in m^-3
    :param n_e: Electron density in m^-3 to calculate the balance for
    :param t_e: Electron temperature in eV to calculate the balance for
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_density: Optional, mandatory if tcx_donor parameter passed. Specifies density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :param element_density dictionary of the form {charge: density}
    :return:
    """

    # calculate fractional abundance for the element
    abundance_dict = _fractional_abundance(atomic_data, element, n_e, t_e, tcx_donor, tcx_donor_density,
                                           tcx_donor_charge)
    # calculate the element density
    element_density = stage_density / abundance_dict[stage_charge]

    # calculate densities
    for key, item in abundance_dict.items():
        abundance_dict[key] *= element_density

    return abundance_dict


def _match_element_density(atomic_data: AtomicData, element: Element, n_species, n_e, t_e, tcx_donor: Element = None,
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
    fractional_abundance = _fractional_abundance(atomic_data, element, n_e, t_e, tcx_donor, tcx_donor_density,
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


def match_element_density(atomic_data: AtomicData, element: Element, abundances, n_e, t_e, tcx_donor: Element = None,
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
    :return: dictionary of the form {charge: density}. Densities are in m^-3
    """

    # calculate charge state densities
    element_abundance = _match_element_density(atomic_data, element, abundances, n_e, t_e, tcx_donor, tcx_donor_density,
                                               tcx_donor_charge)

    # put it into a dictionary
    element_abundance_dict = {}
    for index, value in enumerate(element_abundance):
        element_abundance_dict[index] = value

    return element_abundance_dict


def _profile1d_fractional(atomic_data: AtomicData, element: Element, n_e_profile, t_e_profile,
                          tcx_donor: Element = None, tcx_donor_n_profile=None, tcx_donor_charge=0):
    """
    Calculate profiles of fractional abundance of the specified element for the specified electron density and temperature.
    For more information see _fractional_abundance function.
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param n_e_profile: 1d numpy array profile giving values of electron density for free_variable
    :param t_e_profile: 1d numpy array profile giving values of electron density for free_variable
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n_profile: Optional, mandatory if tcx_donor parameter passed. 1d numpy array profile giving density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :return: 1d profiles of fractional abundances of charge states of the element. Dim 0 corresponds to the ion charge,
    dim 1 holds the profiles.
    """
    # Function returning None instead of donor density interpolator to later reduce number of if statements
    if tcx_donor is None:
        tcx_donor_n_profile = np.zeros_like(n_e_profile)

    # crate the array for profiles
    number_chargestates = element.atomic_number + 1
    fractional_profiles = np.zeros((number_chargestates, n_e_profile.shape[0]))

    # calculate fractional abungances for provided electron profiles
    for index, (n_e, t_e, n_donor) in enumerate(zip(n_e_profile, t_e_profile, tcx_donor_n_profile)):
        fractional_profiles[:, index] = _fractional_abundance(atomic_data, element, n_e, t_e, tcx_donor, n_donor,
                                                              tcx_donor_charge)

    return fractional_profiles


def profile1d_fractional(atomic_data: AtomicData, element: Element, n_e_profile, t_e_profile,
                         tcx_donor: Element = None, tcx_donor_n_profile=None, tcx_donor_charge=0):
    """
    Calculate profiles of fractional abundance of the specified element for the specified electron density and temperature.
    For more information see _fractional_abundance function.
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param n_e_profile: 1d profile giving values of electron density for free_variable
    :param t_e_profile: 1d profile giving values of electron density for free_variable
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n_profile: Optional, mandatory if tcx_donor parameter passed. 1d profile giving density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :return: Dictionary containing 1d numpy arrays of profiles of fractional abundances of charge states of the element in the form {charge: profile}
    """

    # get density profiles of the ion charges
    fractional_profiles = _profile1d_fractional(atomic_data, element, n_e_profile, t_e_profile, tcx_donor,
                                                tcx_donor_n_profile, tcx_donor_charge)

    # transform into dictionary
    fractional_profiles_dict = {}
    for index, value in enumerate(fractional_profiles):
        fractional_profiles_dict[index] = value

    return fractional_profiles_dict


def _profile1d_from_elementdensity(atomic_data: AtomicData, element: Element, element_density_profile, n_e_profile,
                                   t_e_profile,
                                   tcx_donor: Element = None, tcx_donor_n_profile=None, tcx_donor_charge=0):
    """
    For given profiles of plasma parametres the function calulates density profiles of charge states of the element. For more
    information see _from_element_density function.
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param element_density_profile: Density profile of the element in m^-3
    :param n_e_profile: 1d profile giving values of electron density for free_variable
    :param t_e_profile: 1d profile giving values of electron density for free_variable
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n_profile: Optional, mandatory if tcx_donor parameter passed. 1d profile giving density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :return: 1d profiles of fractional abundances of charge states of the element. Dim 0 is the profile, Dim 2 are the charge states
    """

    # calculate fractional abundance profiles for element ionic stages
    fractional_profiles = _profile1d_fractional(atomic_data, element, n_e_profile, t_e_profile, tcx_donor,
                                                tcx_donor_n_profile, tcx_donor_charge)

    # normalize fractional abundance profiles with profile of element density
    density_profiles = fractional_profiles * element_density_profile[np.newaxis, :]

    return density_profiles


def profile1d_from_elementdensity(atomic_data: AtomicData, element: Element, element_density_profile, n_e_profile,
                                  t_e_profile,
                                  tcx_donor: Element = None, tcx_donor_n_profile=None, tcx_donor_charge=0):
    """
    For given profiles of plasma parametres the function calulates density profiles of charge states of the element. For more
    information see _from_element_density function.
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param element_density_profile: Density profile of the element in m^-3
    :param n_e_profile: 1d profile giving values of electron density for free_variable
    :param t_e_profile: 1d profile giving values of electron density for free_variable
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n_profile: Optional, mandatory if tcx_donor parameter passed. 1d profile giving density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :return: 1d profiles of fractional abundances of charge states of the element. Dim 0 is the profile, Dim 2 are the charge states
    """

    # get array with density profiles
    density_profiles = _profile1d_from_elementdensity(atomic_data, element, element_density_profile, n_e_profile,
                                                      t_e_profile,
                                                      tcx_donor, tcx_donor_n_profile, tcx_donor_charge)

    # transform into dictionary
    density_profiles_dict = {}
    for rownumber, row in enumerate(density_profiles):
        density_profiles_dict[rownumber] = row

    return density_profiles_dict


def _profile1d_match_density(atomic_data: AtomicData, element: Element, n_species_profile, n_e_profile, t_e_profile,
                             tcx_donor: Element = None, tcx_donor_n_profile=None, tcx_donor_charge=0):
    """
    For given profiles of plasma parametres the function calulates profile of fractional abunces of charge states of the element.
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param n_species_profile: Density profile of the other plasma species in m^-3
    :param n_e_profile: 1d profile giving values of electron density for free_variable
    :param t_e_profile: 1d profile giving values of electron density for free_variable
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n_profile: Optional, mandatory if tcx_donor parameter passed. 1d profile giving density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :return: 1d profile of fractional abundance of charge states of the element
    """

    if not isinstance(n_species_profile, list) and not isinstance(n_species_profile, dict) and not isinstance(
            n_species_profile, np.ndarray):
        raise TypeError(
            "abundances has to be dictionary or a numpy array holding information about densities of a single element in the form {charge:density} or list of such dictionaries.")
    elif isinstance(n_species_profile, dict) or isinstance(n_species_profile, np.ndarray):
        n_species_profile = [n_species_profile]

    number_chargestates = element.atomic_number + 1

    # extract possible dicts into numpy array
    for i in range(len(n_species_profile)):
        if isinstance(n_species_profile[i], dict):
            abundance = n_species_profile[i]
            n_species_profile[i] = np.zeros((n_e_profile.shape[0], number_chargestates))
            for key, item in abundance.items():
                n_species_profile[i][key] = item

    # calculate fractional abundance profiles
    fractional_abundances = _profile1d_fractional(atomic_data, element, n_e_profile, t_e_profile, tcx_donor,
                                                  tcx_donor_n_profile, tcx_donor_charge)

    # normalize fractional abundance profiles to match electron densities
    density_matched = np.zeros_like(fractional_abundances)
    for rownumber, row in enumerate(fractional_abundances.T):
        element_n_e = n_e_profile[rownumber]
        for spec in n_species_profile:
            for index, value in enumerate(spec[:, rownumber]):
                element_n_e -= index * value

        if element_n_e < 0:
            element_n_e = 0
        density_matched[:, rownumber] = row * element_n_e

    return density_matched


def profile1d_match_density(atomic_data: AtomicData, element: Element, species_density_profiles, n_e_profile,
                            t_e_profile,
                            tcx_donor: Element = None, tcx_donor_n_profile=None, tcx_donor_charge=0):
    """
    For given profiles of plasma parametres the function calulates profile of fractional abunces of charge states of the element.
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param element_density_profile: Density profile of the element in m^-3
    :param n_e_profile: 1d profile giving values of electron density for free_variable
    :param t_e_profile: 1d profile giving values of electron density for free_variable
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n_profile: Optional, mandatory if tcx_donor parameter passed. 1d profile giving density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :return: 1d profile of fractional abundance of charge states of the element
    """

    # calculate density profiles
    density_matched = _profile1d_match_density(atomic_data, element, species_density_profiles, n_e_profile, t_e_profile,
                                               tcx_donor,
                                               tcx_donor_n_profile, tcx_donor_charge)

    # transform into dictionary
    density_matched_dict = {}
    for rownumber, row in enumerate(density_matched):
        density_matched_dict[rownumber] = row

    return density_matched_dict


def interpolators1d_fractional(atomic_data: AtomicData, element: Element, free_variable, n_e_interpolator,
                               t_e_interpolator,
                               tcx_donor: Element = None, tcx_donor_n_interpolator=None, tcx_donor_charge=0):
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
    # Function returning None instead of donor density interpolator to later reduce number of if statements
    if tcx_donor is None:
        tcx_donor_n_interpolator = lambda psin: 0

    n_e_profile = np.zeros_like(free_variable)
    t_e_profile = np.zeros_like(free_variable)
    tcx_donor_n_profile = np.zeros_like(free_variable)

    # construct profiles from free_variable and interpolators
    for i, value in enumerate(free_variable):
        n_e_profile[i] = n_e_interpolator(value)
        t_e_profile[i] = t_e_interpolator(value)
        tcx_donor_n_profile[i] = tcx_donor_n_interpolator(value)

    # calculate fractional abundace profiles
    fractional_profiles = _profile1d_fractional(atomic_data, element, n_e_profile, t_e_profile, tcx_donor,
                                                tcx_donor_n_profile, tcx_donor_charge)

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
    density_profiles = _profile1d_from_elementdensity(atomic_data, element, element_density_profile, n_e_profile,
                                                      t_e_profile, tcx_donor, tcx_donor_n_profile, tcx_donor_charge)

    # use profiles to create interpolators for profiles
    density_interpolators = {}

    for rownumber, row in enumerate(density_profiles):
        density_interpolators[rownumber] = Interpolate1DLinear(free_variable, row)

    return density_interpolators


def interpolators1d_match_element_density(atomic_data: AtomicData, element: Element, free_variable,
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

    density_profiles = _profile1d_match_density(atomic_data, element, density_spec_profile, n_e_profile, t_e_profile,
                                                tcx_donor,
                                                tcx_n_donor_profile, tcx_donor_charge)
    # use profiles to create interpolators for profiles
    density_interpolators = {}
    for rownumber, row in enumerate(density_profiles):
        density_interpolators[rownumber] = Interpolate1DLinear(free_variable, row)

    return density_interpolators
