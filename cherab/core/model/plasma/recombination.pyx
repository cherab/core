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

from raysect.optical cimport Spectrum, Point3D, Vector3D
from cherab.core cimport Plasma, AtomicData
from cherab.core.model.spectra cimport doppler_shift, thermal_broadening, add_gaussian_line
from cherab.core.utility.constants cimport RECIP_4_PI


cdef class RecombinationLine(PlasmaModel):

    def __init__(self, Line line, Plasma plasma=None, AtomicData atomic_data=None):

        super().__init__(plasma, atomic_data)
        self._line = line

    cpdef Spectrum emission(self, Point3D point, Vector3D direction, Spectrum spectrum):

        cdef double ne, te, ni, radiance, sigma
        cdef double natural_wavelength, central_wavelength
        cdef Vector3D ion_velocity

        # cache data on first run
        if self._target_species is None:
            self._populate_cache()

        ne = self._plasma.get_electron_distribution().density(point.x, point.y, point.z)
        if ne <= 0.0:
            return spectrum

        te = self._plasma.get_electron_distribution().effective_temperature(point.x, point.y, point.z)
        if te <= 0.0:
            return spectrum

        ni = self._target_species.distribution.density(point.x, point.y, point.z)
        if ni <= 0.0:
            return spectrum

        ion_velocity = self._target_species.distribution.bulk_velocity(point.x, point.y, point.z)

        # calculate emission line central wavelength, doppler shifted along observation direction
        natural_wavelength = self._wavelength
        central_wavelength = doppler_shift(natural_wavelength, direction, ion_velocity)

        # add emission line to spectrum
        radiance = RECIP_4_PI * self._rates.evaluate(ne, te) * ne * ni
        sigma = thermal_broadening(natural_wavelength, te, self._line.element.atomic_weight)
        return add_gaussian_line(radiance, central_wavelength, sigma, spectrum)

    cdef inline int _populate_cache(self) except -1:

        # sanity checks
        if self._plasma is None or self._atomic_data is None:
            raise RuntimeError("The emission model is not connected to a plasma object.")

        if self._line is None:
            raise RuntimeError("The emission line has not been set.")

        # identify wavelength
        self._wavelength = self._atomic_data.wavelength(self._line.element, self._line.ionisation, self._line.transition)

        # locate target species
        # note: the target species receives an electron during recombination so must have
        # an ionisation +1 relative to the ionisation state required for the emission line
        receiver_ionisation = self._line.ionisation + 1
        try:
            self._target_species = self._plasma.composition.get(self._line.element, receiver_ionisation)
        except ValueError:
            raise RuntimeError("The plasma object does not contain the ion species for the specified line "
                               "(element={}, ionisation={}).".format(self._line.element.symbol, receiver_ionisation))

        # obtain rate function
        self._rates = self._atomic_data.recombination_rate(self._line.element, self._line.ionisation, self._line.transition)

    def _change(self):

        # clear cache to force regeneration on first use
        self._target_species = None
        self._wavelength = 0.0
        self._rates = None