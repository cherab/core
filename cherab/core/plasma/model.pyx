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

cdef class PlasmaModel:
    """
    A plasma emission model.

    When attached to a plasma, a plasma emission model samples the plasma properties
    and atomic data it needs to calculate its emission. The emission is calculated
    for a particular point and viewing orientation in plasma space.

    A new emission model is implemented by inheriting from this class and specifying
    the emission() function.

    If it is necessary to cache data to speed up the emission
    calculation and there is a risk the cached data may be made stale by changes to the
    plasma, the _change() method must be implemented to reset the cache. The _change()
    function is automatically called when changes occur on the Plasma object.

    The plasma and atomic data provider attributes will be automatically populated
    when the PlasmaModel is attached to the Plasma object. In general these should
    not be set by the user directly.

    :ivar Plasma plasma: The plasma to which this emission model is attached.
    :ivar AtomicData atomic_data: The atomic data provider for this model.
    """

    def __init__(self, Plasma plasma=None, AtomicData atomic_data=None):

        self._plasma = plasma
        self._atomic_data = atomic_data

        # setup change notification is we have been given a plasma object
        if self._plasma:
            self._plasma.notifier.add(self._change)

    @property
    def plasma(self):
        return self._plasma

    @plasma.setter
    def plasma(self, value):

        # disconnect from previous plasma's notifications
        if self._plasma:
            self._plasma.notifier.remove(self._change)

        # attach to plasma to inform model of changes to plasma properties
        self._plasma = value
        self._plasma.notifier.add(self._change)

        # inform model source data has changed
        self._change()

    @property
    def atomic_data(self):
        return self._atomic_data

    @atomic_data.setter
    def atomic_data(self, value):

        self._atomic_data = value

        # inform model source data has changed
        self._change()

    cpdef Spectrum emission(self, Point3D point, Vector3D direction, Spectrum spectrum):
        """
        Calculate the emission for a point in the plasma in a specified direction.

        Models implementing this method must add their spectral response to the
        supplied spectrum object. The spectrum units are spectral radiance per
        meter (W/m^3/str/nm).
        
        If a model has a directional response, the model should pass through
        its own reference axis e.g. Thomsen scattering laser direction. 
                       
        :param point: Point in plasma space.
        :param direction: Direction in plasma space.
        :param spectrum: Spectrum to which emission should be added.
        :return: Updated Spectrum object.
        """

        raise NotImplementedError('Virtual method must be implemented in a sub-class.')

    def _change(self):
        """
        Called if the plasma properties or the atomic data source changes.

        If the model caches calculation data that would be invalidated if its
        source data changes then this method may be overridden to clear the
        cache.

        This method is triggered if the plasma notifies the model of a change
        or the atomic data source is changed.
        """

        pass

