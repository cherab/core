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

# todo: for future expansion the transition tuple should become an object with options such as NResolvedTransition and NLResolvedTransition
cdef class Line:

    def __init__(self, Element element, int ionisation, tuple transition):

        if ionisation > element.atomic_number - 1:
            raise ValueError("Ionisation level cannot be larger than one less than the atomic number.")

        if ionisation < 0:
            raise ValueError("Ionisation level cannot be less than zero.")

        # These validation tests don't work for more advanced transition tuples, only valid for hydrogen as currently implemented.
        # TODO - re-instate when we have a more advanced transition objects
        # if transition[0] <= 0 or transition[1] <= 0:
        #     raise ValueError("Transition energy levels cannot be <= 0.")
        #
        # if transition[0] <= transition[1]:
        #     raise ValueError("The initial energy level of a transition must be greater "
        #                      "than the resulting energy level.")

        self.element = element
        self.ionisation = ionisation
        self.transition = transition


