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


cdef class IonisationRate:
    """
    Effective ionisation rate for a given ion.
    """

    def __call__(self, double density, double temperature):
        """Returns an effective ionisation rate coefficient at the specified plasma conditions.

        This function just wraps the cython evaluate() method.
        """
        return self.evaluate(density, temperature)

    cpdef double evaluate(self, double density, double temperature) except? -1e999:
        """Returns an effective ionisation rate coefficient at the specified plasma conditions.

        :param temperature: Electron temperature in eV.
        :param density: Electron density in m^-3
        :return: The effective ionisation rate in m^-3.
        """
        raise NotImplementedError("The evaluate() virtual method must be implemented.")


cdef class RecombinationRate:
    """
    Effective recombination rate for a given ion.
    """

    def __call__(self, double density, double temperature):
        """Returns an effective recombination rate coefficient at the specified plasma conditions.

        This function just wraps the cython evaluate() method.
        """
        return self.evaluate(density, temperature)

    cpdef double evaluate(self, double density, double temperature) except? -1e999:
        """Returns an effective recombination rate coefficient at the specified plasma conditions.

        :param temperature: Electron temperature in eV.
        :param density: Electron density in m^-3
        :return: The effective ionisation rate in m^-3.
        """
        raise NotImplementedError("The evaluate() virtual method must be implemented.")


cdef class ThermalCXRate:
    """
    Effective charge exchange rate between two ions.
    """

    def __call__(self, double density, double temperature):
        """Returns an effective charge exchange rate coefficient at the specified plasma conditions.

        This function just wraps the cython evaluate() method.
        """
        return self.evaluate(density, temperature)

    cpdef double evaluate(self, double density, double temperature) except? -1e999:
        """Returns an effective charge exchange rate coefficient at the specified plasma conditions.

        :param temperature: Electron temperature in eV.
        :param density: Electron density in m^-3
        :return: The effective charge exchange rate in m^-3.
        """
        raise NotImplementedError("The evaluate() virtual method must be implemented.")


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


cdef class ImpactExcitationPEC(_PECRate):
    """
    Impact excitation rate coefficient.
    """
    pass


cdef class RecombinationPEC(_PECRate):
    """
    Recombination rate coefficient.
    """
    pass


cdef class ThermalCXPEC(_PECRate):
    """
    Thermal charge exchange rate coefficient.
    """
    pass


cdef class BeamCXPEC:
    r""":math:`q^{eff}_{n\rightarrow n'}` [:math:`W.m^{3}`]

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


cdef class BeamEmissionPEC(_BeamRate):
    """:math:`bme(X^0(m_i))` [:math:`W.m^3`]

    The effective beam emission coefficient, :math:`bme(X^0(m_i))`.

    The rate :math:`bme(X^0(m_i))` is equivalent to the :math:`BME` rate as defined in
    `adf22 <http://open.adas.ac.uk/adf22>`_.
    """
    pass


cdef class TotalRadiatedPower():
    """The total radiated power in equilibrium conditions."""

    def __init__(self, Element element):

        self.element = element

    def __call__(self, double electron_density, double electron_temperature):
        """
        Evaluate the radiated power rate at the given plasma conditions.

        Calls the cython evaluate() method under the hood.

        :param float electron_density: electron density in m^-3
        :param float electron_temperature: electron temperature in eV
        """
        return self.evaluate(electron_density, electron_temperature)

    cdef double evaluate(self, double electron_density, double electron_temperature) except? -1e999:
        """
        Evaluate the radiated power at the given plasma conditions.

        :param float electron_density: electron density in m^-3
        :param float electron_temperature: electron temperature in eV
        """
        raise NotImplementedError("The evaluate() virtual method must be implemented.")


cdef class _RadiatedPower:
    """Base class for ionisation-resolved radiated powers."""

    def __init__(self, Element element, int charge):

        self.element = element
        self.charge = charge

    def __call__(self, double electron_density, double electron_temperature):
        """
        Evaluate the radiated power rate at the given plasma conditions.

        Calls the cython evaluate() method under the hood.

        :param float electron_density: electron density in m^-3
        :param float electron_temperature: electron temperature in eV
        """
        return self.evaluate(electron_density, electron_temperature)

    cdef double evaluate(self, double electron_density, double electron_temperature) except? -1e999:
        """
        Evaluate the radiated power at the given plasma conditions.

        :param float electron_density: electron density in m^-3
        :param float electron_temperature: electron temperature in eV
        """
        raise NotImplementedError("The evaluate() virtual method must be implemented.")


cdef class LineRadiationPower(_RadiatedPower):
    """
    The total line radiation power driven by excitation.

    Equivalent to the `ADF11 PLT <http://open.adas.ac.uk/adf11>`_ coefficient.
    """
    pass


cdef class ContinuumPower(_RadiatedPower):
    """
    The power radiated from continuum, line power driven by recombination and Bremsstrahlung.

    Equivalent to the `ADF11 PRB <http://open.adas.ac.uk/adf11>`_ coefficient.
    """
    pass


cdef class CXRadiationPower(_RadiatedPower):
    """
    Total line power radiated due to charge transfer from thermal neutral hydrogen.

    Equivalent to the `ADF11 PRC <http://open.adas.ac.uk/adf11>`_ coefficient.
    """
    pass


cdef class FractionalAbundance:
    """
    Rate provider for fractional abundances in thermodynamic equilibrium.

    :param Element element: the radiating element
    :param int charge: the integer charge state for this ionisation stage
    :param str name: optional label identifying this rate
    """

    def __init__(self, element, charge, name=''):

        self.name = name
        self.element = element

        if charge < 0:
            raise ValueError("Charge state must be neutral or positive.")
        self.charge = charge

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
        plt.semilogx(temp, abundances, '.-', label='{}{}'.format(self.element.symbol, self.charge))
