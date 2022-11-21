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


from cherab.core.beam.node cimport Beam
from cherab.core.atomic cimport Element
from cherab.core.distribution cimport DistributionFunction, ZeroDistribution
from cherab.core.utility import Notifier


cdef class BeamDistribution(DistributionFunction):

    def __init__(self):

        super().__init__()
        self._beam = None

    @property
    def element(self):
        return self._element

    @element.setter
    def element(self, Element value not None):
        self._element = value
        self._element_changed()
        self.notifier.notify()

    cdef Element get_element(self):
        return self._element
    
    @property
    def beam(self):
        return self._beam
    
    @beam.setter
    def beam(self, Beam value):

        self._beam = value
        self._beam_changed()

        self.notifier.notify()
    
    cpdef Beam get_beam(self):
        return self._beam
    
    cpdef list get_geometry(self):
        """
        Get list of Primitives forming the beam geometry
        """
        raise NotImplementedError('Virtual method must be implemented in a sub-class.')
    
    def _beam_changed(self):
        """
        Reaction to _beam changes

        Virtual method call.
        """
        pass
    
    def _element_changed(self):
        """
        Reaction to _element change

        Virtual method call.
        """
        pass
    
    def _modified(self):
        """
        Called when distribution chages

        Virtual method call.
        """
        self.notifier.notify()
