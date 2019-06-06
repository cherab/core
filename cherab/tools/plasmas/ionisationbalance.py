import numpy as np
from scipy.optimize import lsq_linear
from cherab.core import AtomicData
from cherab.core.atomic import Element
from cherab.core.math import Interpolate1DCubic

def get_rates_ionisation(atomic_data:AtomicData, element:Element):
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

def get_rates_recombination(atomic_data:AtomicData, element:Element):
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

def get_rates_tcx(atomic_data:AtomicData, donor:Element, donor_charge, receiver:Element):
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

def fractional(atomic_data:AtomicData, element:Element, n_e, t_e, tcx_donor:Element=None, tcx_donor_density=None, tcx_donor_charge=0):
    """
    Calculate fractional abundance of individual charge states for the specified element, temperature and density using steady state ionization balance. If tcx_donor is specified,
    the balance equation will take into accout effects of charge exchage with specified donor. The results are returned as
    fractional abundances i.e. ratio of the individual ionic charge state density to the overall element density.
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab Element
    :param n_e: Electron density in m^-3 to calculate the balance for
    :param t_e: Electron temperature in eV to calculate the balance for
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_density: Optional, mandatory if tcx_donor parameter passed. Specifies density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :return: dictionary of the form {charge: fractional abundance}
    """

    coef_ion = get_rates_ionisation(atomic_data, element)  # get ionisation rate interpolators
    coef_recom = get_rates_recombination(atomic_data, element)  # get recombination rate interpolators

    #get tcx rate interpolators if requested
    if tcx_donor is not None and tcx_donor_density is not None:
        coef_tcx = get_rates_tcx(atomic_data, tcx_donor, tcx_donor_charge, element)

    #atomic number to determine ionisation matrix shape
    atomic_number = element.atomic_number

    matbal = np.zeros((atomic_number + 1, atomic_number + 1)) #create ionisation balance matrix

    # fill the 1st and last rows of the fractional abundance matrix
    matbal[0, 0] -= coef_ion[0](n_e, t_e)
    matbal[0, 1] += coef_recom[1](n_e, t_e)
    matbal[-1, -1] -= coef_recom[atomic_number](n_e, t_e)
    matbal[-1, -2] += coef_ion[atomic_number - 1](n_e, t_e)

    if tcx_donor_density is not None:
        matbal[0, 1] += tcx_donor_density / n_e * coef_tcx[1](n_e, t_e)
        matbal[-1, -1] -= tcx_donor_density / n_e * coef_tcx[atomic_number](n_e, t_e)

    #fill rest of the lines
    for i in range(1, atomic_number):
        matbal[i, i - 1] += coef_ion[i - 1](n_e, t_e)
        matbal[i, i] -= (coef_ion[i](n_e, t_e) + coef_recom[i](n_e, t_e))
        matbal[i, i + 1] += coef_recom[i + 1](n_e, t_e)
        if tcx_donor_density is not None:
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

    #create the dictionary to return the results in
    abundance_dict = {}
    for key, item in enumerate(frac_abundance):
        abundance_dict[key] = item

    return abundance_dict

def from_element_density(atomic_data:AtomicData, element:Element, element_density, n_e, t_e, tcx_donor:Element=None, tcx_donor_density=None, tcx_donor_charge=0):
    """
    Calculate density of individual charge states for the specified element,electron temperature, electron density and element density
    using steady state ionization balance. If tcx_donor is specified, the balance equation will take into accout effects
    of charge exchage with specified donor. The results are returned as density in m^-3
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab Element
    :param element_density: Density of the element in m^-3
    :param n_e: Electron density in m^-3 to calculate the balance for
    :param t_e: Electron temperature in eV to calculate the balance for
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_density: Optional, mandatory if tcx_donor parameter passed. Specifies density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :param element_density: dictionary of the form {charge: density}
    :return:
    """
    #calculate fractional abundance for the element
    abundance_dict = fractional(atomic_data, element, n_e, t_e, tcx_donor, tcx_donor_density, tcx_donor_charge)

    n_e_fromions = 0
    #convert fractional abundance to densities
    for key, item in abundance_dict.items():
        abundance_dict[key] *= element_density
        n_e_fromions += key * abundance_dict[key]

    if n_e_fromions > n_e:
        print("Plasma neutrality violated, {0} density too large".format(element.name))

    return abundance_dict

def from_stage_density(atomic_data:AtomicData, element:Element, stage_charge, stage_density, n_e, t_e, tcx_donor:Element=None, tcx_donor_density=None, tcx_donor_charge=0):
    """
    Calculate density of individual charge states for the specified element,electron temperature, electron density and density of a single charge state
    using steady state ionization balance. If tcx_donor is specified, the balance equation will take into accout effects
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

    #calculate fractional abundance for the element
    abundance_dict = fractional(atomic_data, element, n_e, t_e, tcx_donor, tcx_donor_density, tcx_donor_charge)
    #calculate the element density
    element_density = stage_density / abundance_dict[stage_charge]

    #calculate densities
    for key, item in abundance_dict.items():
        abundance_dict[key] *= element_density


    return abundance_dict

def match_element_density(atomic_data:AtomicData, element:Element, abundances, n_e, t_e, tcx_donor:Element=None, tcx_donor_density=None, tcx_donor_charge=0):
    """
    Calculates densities of charge states of a plasma element species for specified impurity densities, electron density and
    electron temperature. Ratio of densities of ionization stages of the element follows the steady state balance
    and the overall density is determined to match the plasma neutrality (electron density) together with other species.
    It is useful for example to fill in the bulk (e.g. hydrogen isotope or even helium) plasma element once rest of the impurities are
    known.
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element to calculate matching density for
    :param abundances: Abundance dictionaries for impurity elements
    :param n_e: electron density in m^-3
    :param t_e: electron temperature in eV
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_density: Optional, mandatory if tcx_donor parameter passed. Specifies density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :return: element_density dictionary of the form {charge: density}
    """
    if not isinstance(abundances, list) and not isinstance(abundances, dict):
        raise TypeError("abundances has to be dictionary holding information about densities of a single element in the form {charge:density} or list of such dictionaries.")
    elif isinstance(abundances, dict):
        abundances = [abundances]

    elenent_balance = fractional(atomic_data, element, n_e, t_e, tcx_donor, tcx_donor_density, tcx_donor_charge)

    #calculate contribution of bulk ions to electron density
    element_n_e = n_e
    for abundance in abundances:
        for key, item in abundance.items():
            element_n_e -= key * item

    #avoid negative densities due to passed n_e being too small
    if element_n_e < 0:
        element_n_e = 0

    #calculate mean charge of the bulk element
    z_mean = 0
    for key, item in elenent_balance.items():
        z_mean += key * item

    bulk_n_i = element_n_e / z_mean

    bulk_density = {}

    for key, item in elenent_balance.items():
        bulk_density[key] = item * bulk_n_i

    return bulk_density

def interpolators1d_fractional(atomic_data:AtomicData, element:Element, free_variable, n_e_interpolator:Interpolate1DCubic, t_e_interpolator:Interpolate1DCubic,
                               tcx_donor:Element=None, tcx_donor_n_interpolator:Interpolate1DCubic=None, tcx_donor_charge=0):
    """
    For given profiles of plasma parametres the function calulates interpolators for fractional abunces of charge states of the element. The free variable specifies the
    points fractional abundance will be calculated for. The 1d interpolators for ionic fractional abundances are then calculated from these points.
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param free_variable: Free variable (coordinate) to calculate the 1d fractional abundance interpolators from
    :param n_e_interpolator: 1d interpolator giving values of electron density for free_variable
    :param t_e_interpolator: 1d interpolator giving values of electron density for free_variable
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n_interpolator: Optional, mandatory if tcx_donor parameter passed. 1d interpolator giving density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :return: 1d interpolators of fractional abundance of charge states of the element
    """
    #Function returning None instead of donor density interpolator to later reduce number of if statements
    if tcx_donor is None:
        tcx_donor_n_interpolator = lambda psin : None

    #calculate fractional abundance profiles for element ionic stages
    atomic_chargestates = element.atomic_number + 1
    fractional_profile = {}
    for i in range(atomic_chargestates):
        fractional_profile[i] = np.zeros((len(free_variable)))

    #calculate fractional abungances for provided electron profiles
    for i, value in enumerate(free_variable):
        #calculate fractional abundace for one profile point
        abundance = fractional(atomic_data, element, n_e_interpolator(value), t_e_interpolator(value), tcx_donor, tcx_donor_n_interpolator(value), tcx_donor_charge)
        #transfer results into the profiles
        for key, item in abundance.items():
            fractional_profile[key][i] = item

    #use profiles to create interpolators for profiles
    fractional_interpolators = {}
    for key, item in fractional_profile.items():
        fractional_interpolators[key] = Interpolate1DCubic(free_variable, item)

    return fractional_interpolators

def interpolators1d_from_elementdensity(atomic_data:AtomicData, element:Element, free_variable, element_density_interpolator:Interpolate1DCubic,
                                        n_e_interpolator:Interpolate1DCubic, t_e_interpolator:Interpolate1DCubic, tcx_donor:Element=None, tcx_donor_n_interpolator:Interpolate1DCubic=None,
                                        tcx_donor_charge=0):
    """
    For given profiles of plasma parametres the function calulates interpolators of densities of charge states of the element. The free variable specifies the
    points fractional abundance will be calculated for. The 1d charge state density interpolators are then calculated from these points.
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param free_variable: Free variable (coordinate) to calculate the 1d fractional abundance interpolators from
    :param n_e_interpolator: 1d interpolator giving values of electron density for free_variable
    :param t_e_interpolator: 1d interpolator giving values of electron density for free_variable
    :param tcx_donor: Optional, specifies donating species in tcx collisions.
    :param tcx_donor_n_interpolator: Optional, mandatory if tcx_donor parameter passed. 1d interpolator giving density of donors in m^-3
    :param tcx_donor_charge: Optional, specifies the charge of the donor. Default is 0.
    :return: 1d interpolators of fractional abundance of charge states of the element
    """

    #Function returning None instead of donor density interpolator to later reduce number of if statements
    if tcx_donor is None:
        tcx_donor_n_interpolator = lambda psin : None

    #calculate densities of ionic stages of the element
    atomic_chargestates = element.atomic_number + 1
    density_profile = {}
    for i in range(atomic_chargestates):
        density_profile[i] = np.zeros((len(free_variable)))

    #calculate fractional abungances for provided electron profiles
    for i, value in enumerate(free_variable):
        #calculate fractional abundace for one profile point
        abundance = from_element_density(atomic_data, element,element_density_interpolator(value), n_e_interpolator(value), t_e_interpolator(value), tcx_donor, tcx_donor_n_interpolator(value), tcx_donor_charge)
        #transfer results into the profiles
        for key, item in abundance.items():
            density_profile[key][i] = item

    #use profiles to create interpolators for profiles
    density_interpolators = {}
    for key, item in density_profile.items():
        density_interpolators[key] = Interpolate1DCubic(free_variable, item)


    return density_interpolators

def interpolators1d_match_element_density(atomic_data:AtomicData, element:Element, free_variable, species_density_interpolators,
                                        n_e_interpolator:Interpolate1DCubic, t_e_interpolator:Interpolate1DCubic, tcx_donor:Element=None, tcx_donor_n_interpolator:Interpolate1DCubic=None,
                                        tcx_donor_charge=0):
    """
    For given profiles of plasma parametres the function calulates interpolators of densities of charge states of the element. The free variable specifies the
    points fractional abundance will be calculated for. The 1d interpolators for ionic fractional abundances are then calculated from these points. The element densit is calculated based
    on electron profiles and density profiles of other elements to follow plasma neutrality.
    :param atomic_data: Any cherab AtomicData source
    :param element: Any cherab element
    :param free_variable: Free variable (coordinate) to calculate the 1d fractional abundance interpolators from
    :param species_density_interpolators: 1d interpolator giving values of the element density for free_variable
    :param n_e_interpolator: 1d interpolator giving values of electron density for free_variable
    :param t_e_interpolator: 1d interpolator giving values of electron density for free_variable
    :param tcx_donor: specifies donating species in tcx collisions.
    :param tcx_donor_n_interpolator: Optional, mandatory if tcx_donor parameter passed. 1d interpolator giving density of donors in m^-3
    :param tcx_donor_charge:  Optional, specifies the charge of the donor. Default is 0.
    :return: 1d interpolators of density of charge states of the element
    """

    if not isinstance(species_density_interpolators, list) and not isinstance(species_density_interpolators, dict):
        raise TypeError("abundances has to be dictionary holding information about densities of a single element in the form {charge:density} or list of such dictionaries.")
    elif isinstance(species_density_interpolators, dict):
        species_density_interpolators = [species_density_interpolators]

    #Function returning None instead of donor density interpolator to later reduce number of if statements
    if tcx_donor is None:
        tcx_donor_n_interpolator = lambda psin : None

    #calculate densities of ionic stages of the element
    atomic_chargestates = element.atomic_number + 1
    density_profile = {}
    for i in range(atomic_chargestates):
        density_profile[i] = np.zeros((len(free_variable)))

    #calculate fractional abungances for provided electron profiles
    for i, value in enumerate(free_variable):
        element_densities = []
        for spec in species_density_interpolators:
            element_densities.append({})
            for key, item in spec.items():
                element_densities[-1][key] = item(value)

        #calculate fractional abundace for one profile point
        abundance = match_element_density(atomic_data, element, element_densities, n_e_interpolator(value), t_e_interpolator(value), tcx_donor, tcx_donor_n_interpolator(value), tcx_donor_charge)
        #transfer results into the profiles
        for key, item in abundance.items():
            density_profile[key][i] = item

    #use profiles to create interpolators for profiles
    density_interpolators = {}
    for key, item in density_profile.items():
        density_interpolators[key] = Interpolate1DCubic(free_variable, item)


    return density_interpolators