# cython: language_level=3

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


cimport cython
from raysect.core cimport Point3D, Vector3D
from cherab.core cimport Species, Plasma, Beam, Element, DistributionFunction, BeamEmissionPEC, Spectrum, AtomicData
from cherab.core.math.function cimport autowrap_function1d, autowrap_function2d
from cherab.core.atomic.elements import Isotope, hydrogen
from cherab.core.model.lineshape cimport BeamEmissionMultiplet
from cherab.core.utility.constants cimport RECIP_4_PI, ELEMENTARY_CHARGE, ATOMIC_MASS

cdef double RECIP_ELEMENTARY_CHARGE = 1 / ELEMENTARY_CHARGE
cdef double RECIP_ATOMIC_MASS = 1 / ATOMIC_MASS


# example statistical weights supplied by E. Delabie for JET like plasmas
                            # [    Sigma group   ][        Pi group            ]
# STARK_STATISTICAL_WEIGHTS = [0.586167, 0.206917, 0.153771, 0.489716, 0.356513]
SIGMA_TO_PI = 0.56
SIGMA1_TO_SIGMA0 = 0.7060001671878492  # s1*2/s0
PI2_TO_PI3 = 0.3140003593919741  # pi2/pi3
PI4_TO_PI3 = 0.7279994935840365  # pi4/pi3


# TODO - the sigma/pi line ratios should be moved to an atomic data source
cdef class BeamEmissionLine(BeamModel):
    """Calculates beam emission multiplets for a single beam component.

    :param Line line: the transition of interest.
    """

    def __init__(self, Line line not None, Beam beam=None, Plasma plasma=None, AtomicData atomic_data=None,
                 sigma_to_pi=SIGMA_TO_PI, sigma1_to_sigma0=SIGMA1_TO_SIGMA0,
                 pi2_to_pi3=PI2_TO_PI3, pi4_to_pi3=PI4_TO_PI3):

        super().__init__(beam, plasma, atomic_data)

        self._sigma_to_pi = autowrap_function2d(sigma_to_pi)
        self._sigma1_to_sigma0 = autowrap_function1d(sigma1_to_sigma0)
        self._pi2_to_pi3 = autowrap_function1d(pi2_to_pi3)
        self._pi4_to_pi3 = autowrap_function1d(pi4_to_pi3)

        self.line = line

        # ensure that cache is initialised
        self._change()

    @property
    def line(self):
        return self._line

    @line.setter
    def line(self, Line value not None):

        # extract element from isotope
        if isinstance(value.element, Isotope):
            element = value.element.element
        else:
            element = value.element

        if element != hydrogen or value.charge != 0 or value.transition != (3, 2):
            raise ValueError('The BeamEmissionLine model currently only support Balmer-Alpha as the line choice.')

        # the data cache depends on the line configuration
        self._line = value
        self._change()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef Spectrum emission(self, Point3D beam_point, Point3D plasma_point, Vector3D beam_direction,
                            Vector3D observation_direction, Spectrum spectrum):

        cdef:
            double x, y, z
            double beam_density
            tuple species_rates
            DistributionFunction distribution
            BeamEmissionPEC rate_func
            double target_density, target_temperature, rate, radiance

        # cache data on first run
        if self._rates_list is None:
            self._populate_cache()

        # obtain beam density from beam
        beam_density = self._beam.density(beam_point.x, beam_point.y, beam_point.z)

        # abort calculation if beam density is zero
        if beam_density == 0.0:
            return spectrum

        # extract for more compact code
        x = plasma_point.x
        y = plasma_point.y
        z = plasma_point.z

        # sum radiance contributions from each species [W/m^3/str]
        radiance = 0
        for distribution, rate_func in self._rates_list:
            target_density = distribution.density(x, y, z)
            target_temperature = distribution.effective_temperature(x, y, z)
            rate = rate_func.evaluate(self._beam.get_energy(), target_density, target_temperature)
            radiance += target_density * beam_density * rate
        radiance *= RECIP_4_PI

        return self._lineshape.add_line(radiance, beam_point, plasma_point,
                                        beam_direction, observation_direction, spectrum)

    cdef int _populate_cache(self) except -1:

        cdef:
            Element beam_element
            tuple transition
            Species species
            BeamEmissionPEC rate

        # sanity checks
        if self._beam is None:
            raise RuntimeError("The emission model is not connected to a beam object.")
        if self._plasma is None:
            raise RuntimeError("The emission model is not connected to a plasma object.")
        if self._atomic_data is None:
            raise RuntimeError("The emission model is not connected to an atomic data source.")
        if self._line is None:
            raise RuntimeError("The emission line has not been set.")

        # check specified emission line is consistent with attached beam object
        beam_element = self._beam.element
        transition = self._line.transition
        if beam_element != self._line.element:
            raise TypeError("The specified line element '{}' is incompatible with the attached neutral "
                            "beam element '{}'.".format(self._line.element.symbol, beam_element.symbol))
        if self._line.charge != 0:
            raise TypeError("The transition specified does not belong to a neutral atom.")

        # obtain wavelength for specified line
        self._wavelength = self._atomic_data.wavelength(beam_element, 0, transition)

        # obtain beam emission rates and associated density function for each species in plasma
        self._rates_list = []
        for species in self._plasma.composition:
            rate = self._atomic_data.beam_emission_pec(beam_element, species.element, species.charge, transition)
            self._rates_list.append((species.distribution, rate))

        # instance line shape renderer
        self._lineshape = BeamEmissionMultiplet(self._line, self._wavelength, self._beam, self._sigma_to_pi,
                                                self._sigma1_to_sigma0, self._pi2_to_pi3, self._pi4_to_pi3)

    def _change(self):

        # clear cache to force regeneration on first use
        self._wavelength = 0.0
        self._rates_list = None
        self._lineshape = None
