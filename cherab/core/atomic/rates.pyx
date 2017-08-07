# Copyright 2014-2017 United Kingdom Atomic Energy Authority
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

cdef class _PECRate:
    """
    Photon emissivity coefficient base class.
    """
    
    cpdef double evaluate(self, double density, double temperature):
        """
        Returns a rate at given conditions.

        :param temperature: Receiver ion temperature in eV.
        :param density: Receiver ion density in m^-3
        :return: The effective rate.
        """
        raise NotImplementedError("The evaluate() virtual method must be implemented.")

    def __call__(self, double density, double temperature):
        return self.evaluate(density, temperature)


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
    """
    Rate provider base class.
    """

    cpdef double evaluate(self, double energy, double temperature, double density, double z_effective, double b_field) except? -1e999:
        """
        Returns a rate at given conditions.

        :param energy: Interaction energy in eV/amu.
        :param temperature: Receiver ion temperature in eV.
        :param density: Receiver ion density in m^-3
        :param z_effective: Plasma Z-effective.
        :param b_field: Magnetic field magnitude in Tesla.
        :return: The effective rate
        """
        raise NotImplementedError("The evaluate() virtual method must be implemented.")

    def __call__(self, double energy, double temperature, double density, double z_effective, double b_field):
        return self.evaluate(energy, temperature, density, z_effective, b_field)


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
    """
    Beam stopping coefficient.
    """
    pass


cdef class BeamPopulationRate(_BeamRate):
    """
    Beam population coefficient.
    """
    pass


cdef class BeamEmissionRate(_BeamRate):
    """
    Beam emission coefficient.
    """
    pass