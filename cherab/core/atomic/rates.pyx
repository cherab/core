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

import numpy as np
import matplotlib.pyplot as plt


cdef class _PECRate:
    """
    Photon emissivity coefficient base class.
    """

    def __call__(self, double density, double temperature):
        """Returns a photon emissivity coefficient at the specified plasma conditions.

        This function just wraps the cython evaluate() method.
        """
        return self.evaluate(density, temperature)

    cpdef double evaluate(self, double density, double temperature) except? -1e999:
        """Returns a photon emissivity coefficient at given conditions.

        :param temperature: Receiver ion temperature in eV.
        :param density: Receiver ion density in m^-3
        :return: The effective PEC rate in W/m^3.
        """
        raise NotImplementedError("The evaluate() virtual method must be implemented.")

    def plot_temperature(self, temp_low=1, temp_high=1000, num_points=100, dens=1E19):

        temp = [10**x for x in np.linspace(np.log10(temp_low), np.log10(temp_high), num=num_points)]
        rates = [self.evaluate(dens, te) for te in temp]
        plt.semilogx(temp, rates, '.-')
        plt.xlabel("Temperature (eV)")
        plt.ylabel("PEC")


cdef class ImpactExcitationRate(_PECRate):
    """
    Impact excitation rate coefficient.
    """
    pass


cdef class RecombinationRate(_PECRate):
    """
    Recombination rate coefficient.
    """
    pass


cdef class ThermalCXRate(_PECRate):
    """
    Thermal charge exchange rate coefficient.
    """
    pass


cdef class BeamCXRate:
    """:math:`q^{eff}_{n\rightarrow n'}` [:math:`W.m^{3}.s^{-1}.str^{-1}`]

    Effective emission coefficient (or rate) for a charge-exchange line corresponding to a
    transition :math:`n\rightarrow n'` of ion :math:`Z^{(\alpha+1)+}` with electron donor
    :math:`H^0` in metastable state :math:`m_{i}`. Equivalent to
    :math:`q^{eff}_{n\rightarrow n'}` in `adf12 <http://open.adas.ac.uk/adf12>_`.
    """

    def __call__(self, double energy, double temperature, double density, double z_effective, double b_field):
        """Evaluates the Beam CX rate at the given plasma conditions.

        This function just wraps the cython evaluate() method.
        """
        return self.evaluate(energy, temperature, density, z_effective, b_field)

    cpdef double evaluate(self, double energy, double temperature, double density, double z_effective, double b_field) except? -1e999:
        """Evaluates the Beam CX rate at the given plasma conditions.

        :param float energy: Interaction energy in eV/amu.
        :param float temperature: Receiver ion temperature in eV.
        :param float density: Receiver ion density in m^-3
        :param float z_effective: Plasma Z-effective.
        :param float b_field: Magnetic field magnitude in Tesla.
        :return: The effective rate
        """
        raise NotImplementedError("The evaluate() virtual method must be implemented.")


cdef class _BeamRate:
    """
    Beam coefficient base class.
    """

    cpdef double evaluate(self, double energy, double density, double temperature) except? -1e999:
        """
        Returns the beam coefficient for the supplied parameters.

        :param energy: Interaction energy in eV/amu.
        :param density: Target electron density in m^-3
        :param temperature: Target temperature in eV.
        :return: The beam coefficient
        """
        raise NotImplementedError("The evaluate() virtual method must be implemented.")

    def __call__(self, double energy, double density, double temperature):
        return self.evaluate(energy, density, temperature)


cdef class BeamStoppingRate(_BeamRate):
    """:math:`S^{e, i}_{CR}` [:math:`m^3.s^{-1}`]

    The effective collisional radiative stopping coefficient :math:`S^{e, i}_{CR}`
    [:math:`m^3.s^{-1}`] for neutral atom :math:`X^0` in a mono-energetic beam by
    fully stripped ions :math:`Y^i` and their electrons.

    Equivalent to :math:`S^{e, i}_{CR}` as defined in ADAS `adf21 <http://open.adas.ac.uk/adf21>`_.
    """
    pass


cdef class BeamPopulationRate(_BeamRate):
    """:math:`bmp(X^0(m_i))` [dimensionless]

    Relative beam population of excited state :math:`m_i` over ground state for atom :math:`X^0`, :math:`bmp(X^0(m_i))`.

    The rate :math:`bmp(X^0(m_i))` is equivalent to the :math:`BMP` rate as defined in
    `adf22 <http://open.adas.ac.uk/adf22>`_ and is dimensionless.
    """
    pass


cdef class BeamEmissionRate(_BeamRate):
    """:math:`bme(X^0(m_i))` [dimensionless]

    Relative beam population of excited state :math:`m_i` over ground state for atom :math:`X^0`, :math:`bme(X^0(m_i))`.

    The rate :math:`bme(X^0(m_i))` is equivalent to the :math:`BME` rate as defined in
    `adf22 <http://open.adas.ac.uk/adf22>`_ and is dimensionless.
    """
    pass


cdef class RadiatedPower:
    """
    Total radiated power for a given species and radiation type.

    Radiation type can be:
    - 'total' (line + recombination + bremsstrahlung + charge exchange)
    - 'line' radiation
    - 'continuum' (recombination + bremsstrahlung)
    - 'cx' charge exchange

    :param Element element: the radiating element
    :param str radiation_type: selects the type of radiation to be included in the total rate.
    :param str name: optional label identifying this rate
    """

    def __init__(self, element, radiation_type, name=''):

        self.name = name
        self.element = element

        if radiation_type not in ['total', 'line', 'continuum', 'cx']:
            raise ValueError("RadiatedPower() radiation type must be one of ['total', 'line', 'continuum', 'cx'].")
        self.radiation_type = radiation_type

    cdef double evaluate(self, double electron_density, double electron_temperature) except? -1e999:
        """
        Evaluate the total radiated power at given plasma conditions.

        :param float electron_density: electron density in m^-3
        :param float electron_temperature: electron temperature in eV
        """
        raise NotImplementedError("The evaluate() virtual method must be implemented.")

    def __call__(self, double electron_density, double electron_temperature):
        """
        Evaluate the total radiated power of this species at the given plasma conditions.

        :param float electron_density: electron density in m^-3
        :param float electron_temperature: electron temperature in eV
        """
        return self.evaluate(electron_density, electron_temperature)

    def plot_temperature(self, temp_low=1, temp_high=1000, num_points=100, dens=1E19, species_dens=1E19):

        temp = [10**x for x in np.linspace(np.log10(temp_low), np.log10(temp_high), num=num_points)]
        radiation = [self.evaluate(dens, te) * species_dens for te in temp]
        plt.loglog(temp, radiation, '.-', label='{} - {}'.format(self.element.symbol, self.radiation_type))


cdef class StageResolvedLineRadiation:
    """
    Total ionisation state resolved line radiated power rate.

    :param Element element: the radiating element
    :param int ionisation: the integer charge state for this ionisation stage
    :param str name: optional label identifying this rate
    """

    def __init__(self, element, ionisation, name=''):

        if ionisation < 0:
            raise ValueError("Charge state must be neutral or positive.")
        self.ionisation = ionisation

        self.name = name
        self.element = element

    cdef double evaluate(self, double electron_density, double electron_temperature) except? -1e999:
        """
        Evaluate the total radiated power at given plasma conditions.

        :param float electron_density: electron density in m^-3
        :param float electron_temperature: electron temperature in eV
        """
        raise NotImplementedError("The evaluate() virtual method must be implemented.")

    def __call__(self, double electron_density, double electron_temperature):
        """
        Evaluate the total radiated power of this species at the given plasma conditions.

        :param float electron_density: electron density in m^-3
        :param float electron_temperature: electron temperature in eV
        """
        return self.evaluate(electron_density, electron_temperature)

    def plot_temperature(self, temp_low=1, temp_high=1000, num_points=100, dens=1E19, species_dens=1E19):

        temp = [10**x for x in np.linspace(np.log10(temp_low), np.log10(temp_high), num=num_points)]
        radiation = [self.evaluate(dens, te) * species_dens for te in temp]
        plt.loglog(temp, radiation, '.-', label='{}{}'.format(self.element.symbol, self.ionisation))


cdef class FractionalAbundance:
    """
    Rate provider for fractional abundances in thermodynamic equilibrium.

    :param Element element: the radiating element
    :param int ionisation: the integer charge state for this ionisation stage
    :param str name: optional label identifying this rate
    """

    def __init__(self, element, ionisation, name=''):

        self.name = name
        self.element = element

        if ionisation < 0:
            raise ValueError("Charge state must be neutral or positive.")
        self.ionisation = ionisation

    cdef double evaluate(self, double electron_density, double electron_temperature) except? -1e999:
        """
        Evaluate the fractional abundance of this ionisation stage at the given plasma conditions.

        :param float electron_density: electron density in m^-3
        :param float electron_temperature: electron temperature in eV
        """
        raise NotImplementedError("The evaluate() virtual method must be implemented.")

    def __call__(self, double electron_density, double electron_temperature):
        """
        Evaluate the fractional abundance of this ionisation stage at the given plasma conditions.

        :param float electron_density: electron density in m^-3
        :param float electron_temperature: electron temperature in eV
        """
        return self.evaluate(electron_density, electron_temperature)

    def plot_temperature(self, temp_low=1, temp_high=1000, num_points=100, dens=1E19):

        temp = [10**x for x in np.linspace(np.log10(temp_low), np.log10(temp_high), num=num_points)]
        abundances = [self.evaluate(dens, te) for te in temp]
        plt.semilogx(temp, abundances, '.-', label='{}{}'.format(self.element.symbol, self.ionisation))
