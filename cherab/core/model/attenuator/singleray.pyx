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

from scipy.integrate import cumtrapz
import numpy as np
cimport numpy as np

from raysect.optical cimport AffineMatrix3D, Point3D, Vector3D, new_point3d
from cherab.core.utility import EvAmuToMS, EvToJ
from cherab.core.atomic cimport BeamStoppingRate, AtomicData
from cherab.core.math cimport Interpolate1DLinear
from cherab.core.plasma cimport Plasma
from cherab.core.beam cimport Beam
from cherab.core.species cimport Species
from cherab.core.utility.constants cimport DEGREES_TO_RADIANS

from libc.math cimport exp, sqrt, tan, M_PI
cimport cython


# todo: attenuation calculation could be optimised further using memory views etc...
cdef class SingleRayAttenuator(BeamAttenuator):

    def __init__(self, double step=0.01, bint clamp_to_zero=False, double clamp_sigma=5.0, Beam beam=None, Plasma plasma=None, AtomicData atomic_data=None):

        super().__init__(beam, plasma, atomic_data)

        self._source_density = 0.
        self._density = None
        self._stopping_data = None

        # spacing of density sample points along the beam
        if step <= 0.0:
            raise ValueError("The step size must be greater than zero.")
        self._step = step

        # beam density clamping optimisation settings
        if clamp_sigma <= 0.0:
            raise ValueError("The value of clamp_sigma must be greater than zero.")
        self.clamp_to_zero = clamp_to_zero
        self._clamp_sigma_sqr = clamp_sigma**2

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        if value <= 0.0:
            raise ValueError("The step size must be greater than zero.")
        self._step = value

        # inform beam that the density values may have changed
        self.notifier.notify()

        # reset cache (belt and braces, the beam should really trigger this via the beam change notification
        # that chains from the attenuator change notification)
        self._change()

    @property
    def clamp_sigma(self):
        return sqrt(self._clamp_sigma_sqr)

    @clamp_sigma.setter
    def clamp_sigma(self, value):
        if value <= 0.0:
            raise ValueError("The value of clamp_sigma must be greater than zero.")
        self._clamp_sigma_sqr = value ** 2

    @cython.cdivision(True)
    cpdef double density(self, double x, double y, double z) except? -1e999:
        """
        Returns the beam density at the specified point.
        
        The point is specified in beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: Density in m^-3. 
        """

        cdef double sigma_x, sigma_y, norm_radius_sqr, gaussian_sample

        # use cached data if available
        if self._stopping_data is None:
            self._populate_stopping_data_cache()

        if self._density is None:
            self._calc_attenuation()

        # calculate beam width
        sigma_x = self._beam.get_sigma() + z * self._tanxdiv
        sigma_y = self._beam.get_sigma() + z * self._tanydiv

        # normalised radius squared
        norm_radius_sqr = ((x / sigma_x)**2 + (y / sigma_y)**2)

        # clamp low densities to zero (beam models can skip their calculation if density is zero)
        # comparison is done using the squared radius to avoid a costly square root
        if self.clamp_to_zero:
            if norm_radius_sqr > self._clamp_sigma_sqr:
                return 0.0

        # bi-variate Gaussian distribution (normalised)
        gaussian_sample = exp(-0.5 * norm_radius_sqr) / (2 * M_PI * sigma_x * sigma_y)

        return self._density.evaluate(z) * gaussian_sample

    cpdef calculate_attenuation(self):
        """
        Trigger beam attenuation calculation
        """

        if self._stopping_data is None:
            self._populate_stopping_data_cache()
        self._calc_attenuation()

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _calc_attenuation(self):
        """
        Attenuation is calculated along the beam axis and extrapolated across the beam.
        
        Fill attribute '_density' with a 1D function taking meters as argument
        and returning a linear density in m^-1, calculated along the beam axis.
        """

        cdef:
            AffineMatrix3D beam_to_plasma
            Vector3D direction
            int nbeam, i
            np.ndarray beam_z, xaxis, yaxis, zaxis, beam_density
            double bzv

        # calculate transform to plasma space
        beam_to_plasma = self._beam.to(self._plasma)
        direction = self._beam.BEAM_AXIS.transform(beam_to_plasma)

        # sample points along the beam
        nbeam = max(1 + int(np.ceil(self._beam.length / self._step)), 4)
        beam_z = np.linspace(0.0, self._beam.length, nbeam)

        xaxis = np.zeros(nbeam)
        yaxis = np.zeros(nbeam)
        zaxis = np.zeros(nbeam)

        for i, bzv in enumerate(beam_z):

            paxis = new_point3d(0.0, 0.0, bzv).transform(beam_to_plasma)
            xaxis[i] = paxis.x
            yaxis[i] = paxis.y
            zaxis[i] = paxis.z

        beam_density = self._beam_attenuation(beam_z, xaxis, yaxis, zaxis,
                                              self._beam.energy, self._beam.power,
                                              self._beam.element.atomic_weight, direction)

        self._tanxdiv = tan(DEGREES_TO_RADIANS * self._beam.divergence_x)
        self._tanydiv = tan(DEGREES_TO_RADIANS * self._beam.divergence_y)

        # a tiny degree of extrapolation is permitted to handle numerical accuracy issues with the end of the array
        self._density = Interpolate1DLinear(beam_z, beam_density, extrapolate=True, extrapolation_range=1e-9)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray _beam_attenuation(self, np.ndarray axis, np.ndarray x, np.ndarray y, np.ndarray z,
                                          double energy, double power, double mass, Vector3D direction):
        """
        axis has to be sorted

        :param axis:
        :param x: list of positions in meters
        :param y: list of positions in meters
        :param z: list of positions in meters
        :param energy: beam energy in eV/amu
        :param power: beam power in W
        :param mass: atomic mass in amu
        :param direction:
        :return: a list of linear densities in m^-1
        """

        cdef:
            np.ndarray stopping_coeff
            double speed, beam_particle_rate, beam_density
            Vector3D beam_velosity
            int i, naxis

        naxis = axis.size

        speed = EvAmuToMS.to(energy)
        beam_velocity = direction.normalise() * speed

        beam_particle_rate = power / EvToJ.to(energy * mass)
        beam_density = beam_particle_rate / speed
        self._source_density = beam_density

        stopping_coeff = np.zeros(naxis)
        for i in range(naxis):
            stopping_coeff[i] = self._beam_stopping(x[i], y[i], z[i], beam_velocity)

        return beam_density * np.exp(-cumtrapz(stopping_coeff, axis, initial=0.0) / speed)

    @cython.cdivision(True)
    cdef double _beam_stopping(self, double x, double y, double z, Vector3D beam_velocity):
        """

        :param x: position in meters
        :param y: position in meters
        :param z: position in meters
        :param beam_velocity: beam velocity in m/s
        :return: a stopping coefficient in s^-1
        """

        # see www.adas.ac.uk/man/chap3-04.pdf equation 4.4.7
        # note: we have access to ni for each species so we have done away with
        # the impurity fractions used in the above document

        cdef:
            double density_sum, stopping_coeff, target_ne, target_ti, interaction_speed, interaction_energy, target_equiv_ne
            Species species
            BeamStoppingRate coeff
            int target_z
            Vector3D target_velocity, interaction_velocity

        # z-weighted density sum
        density_sum = 0
        for species, _ in self._stopping_data:
            density_sum += species.charge**2 * species.distribution.density(x, y, z)

        # stopping coefficient
        stopping_coeff = 0
        for species, coeff in self._stopping_data:

            # sample species distribution
            target_z = species.charge
            target_ne = species.distribution.density(x, y, z) * target_z
            target_ti = species.distribution.effective_temperature(x, y, z)
            target_velocity = species.distribution.bulk_velocity(x, y, z)

            # calculate mean beam interaction energy
            interaction_velocity = beam_velocity - target_velocity
            interaction_speed = interaction_velocity.length
            interaction_energy = EvAmuToMS.inv(interaction_speed)

            # species equivalent electron density
            target_equiv_ne = density_sum / target_z

            stopping_coeff += target_ne * coeff.evaluate(interaction_energy, target_equiv_ne, target_ti)

        return stopping_coeff

    cdef int _populate_stopping_data_cache(self) except -1:
        """
        Obtain the beam stopping data from the atomic data source.

        If the user has not specified the species with which the beam is to interact then it is assumed the beam
        interacts will all the plasma species.
        """

        cdef:
            Species species
            BeamStoppingRate stopping_coeff

        # sanity checks
        if not self._beam:
            raise ValueError("The beam attenuator is not connected to a beam object.")

        if not self._plasma:
            raise ValueError("The beam attenuator is not connected to a plasma object.")

        if not self._atomic_data:
            raise ValueError("The beam attenuator does not have an atomic data source.")

        self._stopping_data = []
        for species in self._plasma.composition:
            stopping_coeff = self._atomic_data.beam_stopping_rate(self._beam.element, species.element, species.charge)
            self._stopping_data.append((species, stopping_coeff))

    def _change(self):

        # reset cached data
        self._density = None
        self._stopping_data = None
