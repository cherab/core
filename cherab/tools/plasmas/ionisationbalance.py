import numpy as np
from scipy.optimize import lsq_linear
from cherab.core import AtomicData
from cherab.core.atomic import Element

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

def fractional(atomic_data, element, n_e, t_e, tcx_donor=None, tcx_donor_density=None, tcx_donor_charge=0):
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
    rhs = np.zeroes((matbal.shape[0]))
    rhs[-1] = n_e

    abundance = lsq_linear(matbal, rhs, bounds=(0, n_e))["x"]

    # normalize to ne to get fractional abundance
    frac_abundance = abundance / n_e

    #create the dictionary to return the results in
    abundance_dict = {}
    for key, item in enumerate(frac_abundance):
        abundance_dict[key] = item

    return abundance_dict

def from_element_density(atomic_data, element, element_density, n_e, t_e, tcx_donor=None, tcx_donor_density=None, tcx_donor_charge=0):
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

    #convert fractional abundance to densities
    for key, item in abundance_dict.items():
        abundance_dict[key] *= element_density


    return abundance_dict

def from_stage_density(atomic_data, element, stage_charge, stage_density, n_e, t_e, tcx_donor=None, tcx_donor_density=None, tcx_donor_charge=0):
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

def match_bulk_element_density(atomic_data, bulk_element, abundances, n_e, t_e):
    """
    Calculates densities of charge states of bulk plasma element for specified impurity densities, electron density and
    electron temperature. Ratio of densities of ionization stages of the bulk element follows the steady state balance
    and the overall density is determined to match the plasma neutrality (electron density) together with impurities.
    :param atomic_data: Any cherab AtomicData source
    :param bulk_element: Any cherab element
    :param abundances: Abundance dictionaries for impurity elements
    :param n_e: electron density in m^-3
    :param t_e: electron temperature in eV
    :return: element_density dictionary of the form {charge: density}
    """
    if not isinstance(abundances, list) and not isinstance(abundances, dict):
        raise TypeError("abundances has to be dictionary holding information about densities of a single element in the form {charge:density} or list of such dictionaries.")
    elif isinstance(abundances, dict):
        abundances = [abundances]

    bulk_balance = fractional(atomic_data, bulk_element, n_e, t_e)

    #calculate contribution of bulk ions to electron density
    bulk_n_e = n_e
    for abundance in abundances:
        for key, item in abundance.items():
            bulk_n_e -= key * item

    #avoid negative densities due to passed n_e being too small
    if bulk_n_e < 0:
        bulk_n_e = 0

    #calculate mean charge of the bulk element
    z_mean = 0
    for key, item in bulk_balance.items():
        z_mean += key * item

    bulk_n_i = bulk_n_e / z_mean

    bulk_density = {}

    for key, item in bulk_balance.items():
        bulk_density[key] = item * bulk_n_i

    return bulk_density