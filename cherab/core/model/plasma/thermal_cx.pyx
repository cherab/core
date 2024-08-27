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

from raysect.optical cimport Spectrum, Point3D, Vector3D
from cherab.core cimport Plasma, AtomicData
from cherab.core.atomic cimport ThermalCXPEC
from cherab.core.model.lineshape cimport GaussianLine, LineShapeModel
from cherab.core.utility.constants cimport RECIP_4_PI


cdef class ThermalCXLine(PlasmaModel):
    r"""
    Emitter that calculates spectral line emission from a plasma object
    as a result of thermal charge exchange of the target species with the donor species.

    .. math::
        \epsilon_{\mathrm{CX}}(\lambda) = \frac{1}{4 \pi} n_{Z_\mathrm{i} + 1}
        \sum_j{n_{Z_\mathrm{j}} \mathrm{PEC}_{\mathrm{cx}}(n_\mathrm{e}, T_\mathrm{e}, T_{Z_\mathrm{j}})}
        f(\lambda),

    where :math:`n_{Z_\mathrm{i} + 1}` is the receiver species density,
    :math:`n_{Z_\mathrm{j}}` is the donor species density,
    :math:`\mathrm{PEC}_{\mathrm{cx}}` is the thermal CX photon emission coefficient
    for the specified spectral line of the :math:`Z_\mathrm{i}` ion,
    :math:`T_{Z_\mathrm{j}}` is the donor species temperature,
    :math:`f(\lambda)` is the normalised spectral line shape,

    :param Line line: Spectroscopic emission line object.
    :param Plasma plasma: The plasma to which this emission model is attached. Default is None.
    :param AtomicData atomic_data: The atomic data provider for this model. Default is None.
    :param object lineshape: Line shape model class. Default is None (GaussianLine).
    :param object lineshape_args: A list of line shape model arguments. Default is None.
    :param object lineshape_kwargs: A dictionary of line shape model keyword arguments. Default is None.

    :ivar Plasma plasma: The plasma to which this emission model is attached.
    :ivar AtomicData atomic_data: The atomic data provider for this model.
    """

    def __init__(self, Line line, Plasma plasma=None, AtomicData atomic_data=None, object lineshape=None,
                 object lineshape_args=None, object lineshape_kwargs=None):

        super().__init__(plasma, atomic_data)

        self._line = line

        self._lineshape_class = lineshape or GaussianLine
        if not issubclass(self._lineshape_class, LineShapeModel):
            raise TypeError("The attribute lineshape must be a subclass of LineShapeModel.")

        if lineshape_args:
            self._lineshape_args = lineshape_args
        else:
            self._lineshape_args = []
        if lineshape_kwargs:
            self._lineshape_kwargs = lineshape_kwargs
        else:
            self._lineshape_kwargs = {}

        # ensure that cache is initialised
        self._change()

    def __repr__(self):
        return '<ThermalCXLine: element={}, charge={}, transition={}>'.format(self._line.element.name, self._line.charge, self._line.transition)

    cpdef Spectrum emission(self, Point3D point, Vector3D direction, Spectrum spectrum):

        cdef:
            double ne, te, receiver_density, donor_density, donor_temperature, weighted_rate, radiance
            Species species
            ThermalCXPEC rate

        # cache data on first run
        if self._target_species is None:
            self._populate_cache()

        ne = self._plasma.get_electron_distribution().density(point.x, point.y, point.z)
        if ne <= 0.0:
            return spectrum

        te = self._plasma.get_electron_distribution().effective_temperature(point.x, point.y, point.z)
        if te <= 0.0:
            return spectrum

        receiver_density = self._target_species.distribution.density(point.x, point.y, point.z)
        if receiver_density <= 0.0:
            return spectrum

        # obtain composite CX PEC by iterating over all possible CX donors
        weighted_rate = 0
        for species, rate in self._rates:
            donor_density = species.distribution.density(point.x, point.y, point.z)
            donor_temperature = species.distribution.effective_temperature(point.x, point.y, point.z)
            weighted_rate += donor_density * rate.evaluate(ne, te, donor_temperature)

        # add emission line to spectrum
        radiance = RECIP_4_PI * weighted_rate * receiver_density
        return self._lineshape.add_line(radiance, point, direction, spectrum)

    cdef int _populate_cache(self) except -1:

        cdef:
            int receiver_charge
            Species species
            ThermalCXPEC rate

        # sanity checks
        if self._plasma is None:
            raise RuntimeError("The emission model is not connected to a plasma object.")
        if self._atomic_data is None:
            raise RuntimeError("The emission model is not connected to an atomic data source.")

        if self._line is None:
            raise RuntimeError("The emission line has not been set.")

        # locate target species
        receiver_charge = self._line.charge + 1
        try:
            self._target_species = self._plasma.composition.get(self._line.element, receiver_charge)
        except ValueError:
            raise RuntimeError("The plasma object does not contain the ion species for the specified CX line "
                               "(element={}, ionisation={}).".format(self._line.element.symbol, receiver_charge))

        # obtain rate functions
        self._rates = []
        # iterate over all posible electron donors in plasma composition
        # and for each donor, cache the PEC rate function for the CX reaction with this receiver
        for species in self._plasma.composition:
            # exclude the receiver species from the list of donors and omit fully ionised species
            if species != self._target_species and species.charge < species.element.atomic_number:
                rate = self._atomic_data.thermal_cx_pec(species.element, species.charge,  # donor
                                                        self._line.element, receiver_charge,  # receiver
                                                        self._line.transition)
                self._rates.append((species, rate))

        # identify wavelength
        self._wavelength = self._atomic_data.wavelength(self._line.element, self._line.charge, self._line.transition)

        # instance line shape renderer
        self._lineshape = self._lineshape_class(self._line, self._wavelength, self._target_species, self._plasma,
                                                self._atomic_data, *self._lineshape_args, **self._lineshape_kwargs)

    def _change(self):

        # clear cache to force regeneration on first use
        self._target_species = None
        self._wavelength = 0.0
        self._rates = None
        self._lineshape = None
