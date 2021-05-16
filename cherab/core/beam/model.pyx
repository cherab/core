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

from cherab.core.utility import Notifier


cdef class BeamModel:

    def __init__(self, Beam beam=None, Plasma plasma=None, AtomicData atomic_data=None):

        self._beam = beam
        self._plasma = plasma
        self._atomic_data = atomic_data

        # setup property change notifications for plasma
        if self._plasma:
            self._plasma.notifier.add(self._change)

        # setup property change notifications for beam
        if self._beam:
            self._beam.notifier.add(self._change)

    @property
    def plasma(self):
        return self._plasma

    @plasma.setter
    def plasma(self, Plasma value not None):

        # disconnect from previous plasma's notifications
        if self._plasma:
            self._plasma.notifier.remove(self._change)

        # attach to plasma to inform model of changes to plasma properties
        self._plasma = value
        self._plasma.notifier.add(self._change)

        # inform model source data has changed
        self._change()

    @property
    def beam(self):
        return self._beam

    @beam.setter
    def beam(self, Beam value not None):

        # disconnect from previous beam's notifications
        if self._beam:
            self._beam.notifier.remove(self._change)

        # attach to beam to inform model of changes to beam properties
        self._beam = value
        self._beam.notifier.add(self._change)

        # inform model source data has changed
        self._change()

    @property
    def atomic_data(self):
        return self._atomic_data

    @atomic_data.setter
    def atomic_data(self, AtomicData value not None):

        self._atomic_data = value

        # inform model source data has changed
        self._change()

    cpdef Spectrum emission(self, Point3D beam_point, Point3D plasma_point, Vector3D beam_direction, Vector3D observation_direction, Spectrum spectrum):
        """
        Calculate the emission for a point on the beam in a specified direction.

        Models implementing this method must add their spectral response to the
        supplied spectrum object. The spectrum units are spectral radiance per
        meter (W/m^3/str/nm).

        :param beam_point: Point in beam space.
        :param plasma_point: Point in plasma space.
        :param beam_direction: Beam axis direction in plasma space.
        :param observation_direction: Observation direction in plasma space.
        :param spectrum: Spectrum to which emission should be added.
        :return: Updated Spectrum object.
        """

        raise NotImplementedError('Virtual method must be implemented in a sub-class.')

    def _change(self):
        """
        Called if the plasma, beam or the atomic data source properties change.

        If the model caches calculation data that would be invalidated if its
        source data changes then this method may be overridden to clear the
        cache.
        """

        pass


cdef class BeamAttenuator:

    def __init__(self, Beam beam=None, Plasma plasma=None, AtomicData atomic_data=None):

        # must notify beam if the attenuator properties change, affecting the density values
        self.notifier = Notifier()

        # configure
        self._beam = beam
        self._plasma = plasma
        self._atomic_data = atomic_data

        # setup property change notifications for plasma
        if self._plasma:
            self._plasma.notifier.add(self._change)

        # setup property change notifications for beam
        if self._beam:
            self._beam.notifier.add(self._change)

    @property
    def plasma(self):
        return self._plasma

    @plasma.setter
    def plasma(self, Plasma value not None):

        # disconnect from previous plasma's notifications
        if self._plasma:
            self._plasma.notifier.remove(self._change)

        # attach to plasma to inform model of changes to plasma properties
        self._plasma = value
        self._plasma.notifier.add(self._change)

        # inform model source data has changed
        self._change()

    @property
    def beam(self):
        return self._beam

    @beam.setter
    def beam(self, Beam value not None):

        # disconnect from previous beam's notifications
        if self._beam:
            self._beam.notifier.remove(self._change)

        # attach to beam to inform model of changes to beam properties
        self._beam = value
        self._beam.notifier.add(self._change)

        # inform model source data has changed
        self._change()

    @property
    def atomic_data(self):
        return self._atomic_data

    @atomic_data.setter
    def atomic_data(self, AtomicData value not None):

        self._atomic_data = value

        # inform model source data has changed
        self._change()

    cpdef double density(self, double x, double y, double z) except? -1e999:
        """
        Returns the beam density at the specified point.

        The point is specified in beam space.

        :param x: x coordinate in meters.
        :param y: y coordinate in meters.
        :param z: z coordinate in meters.
        :return: Density in m^-3.
        """
        raise NotImplementedError("Virtual function density not defined.")

    def _change(self):
        """
        Called if the plasma, beam or the atomic data source properties change.

        If the model caches calculation data that would be invalidated if its
        source data changes then this method may be overridden to clear the
        cache.
        """

        pass
