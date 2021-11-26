
# Copyright 2016-2021 Euratom
# Copyright 2016-2021 United Kingdom Atomic Energy Authority
# Copyright 2016-2021 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

from numpy import ndarray

from raysect.optical.observer import TargettedPixel
from .base import Observer0DGroup


class TargettedPixelGroup(Observer0DGroup):
    """
    A group of targetted pixel under a single scene-graph node.

    A scene-graph object regrouping a series of 'TargettedPixel'
    observers as a scene-graph parent. Allows combined observation and display
    control simultaneously.

    :ivar list x_width: Width of pixel along local x axis
    :ivar list y_width: Width of pixel along local y axis
    :ivar list targets: Targets for preferential sampling
    :ivar list targetted_path_prob: Probability of ray being casted at the target

    """

    @Observer0DGroup.observers.setter
    def observers(self, value):

        if not isinstance(value, (list, tuple)):
            raise TypeError("The observers attribute of TargettedPixelGroup must be a list or tuple of TargettedPixel.")

        for observer in value:
            if not isinstance(observer, TargettedPixel):
                raise TypeError("The observers attribute of TargettedPixelGroup must be a list or tuple of "
                                "TargettedPixel. Value {} is not a TargettedPixel.".format(observer))

        # Prevent external changes being made to this list
        for observer in value:
            observer.parent = self

        self._observers = tuple(value)

    def add_observer(self, observer):
        """
        Adds new line of sight to the group.

        :param TargettedPixel observer: Targetted pixel to add.
        """

        if not isinstance(observer, TargettedPixel):
            raise TypeError("The observer argument must be of type TargettedPixel.")

        observer.parent = self
        self._observers = self._observers + (observer,)

    @property
    def x_width(self):
        return [pixel.x_width for pixel in self._observers]

    @x_width.setter
    def x_width(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._observers):
                for pixel, v in zip(self._observers, value):
                    pixel.x_width = v
            else:
                raise ValueError("The length of 'x_width' ({}) "
                                 "mismatches the number of pixels ({}).".format(len(value), len(self._sight_lines)))
        else:
            for pixel in self._observers:
                pixel.x_width = value
    @property
    def y_width(self):
        return [pixel.y_width for pixel in self._observers]

    @y_width.setter
    def y_width(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._observers):
                for pixel, v in zip(self._observers, value):
                    pixel.y_width = v
            else:
                raise ValueError("The length of 'y_width' ({}) "
                                 "mismatches the number of pixels ({}).".format(len(value), len(self._sight_lines)))
        else:
            for pixel in self._observers:
                pixel.y_width = value

    @property
    def targets(self):
        """Targets for preferential sampling"""
        return [pixel.target for pixel in self._observers]

    @targets.setter
    def targets(self, value):
        if isinstance(value, (list, tuple)):
            if len(value) == len(self._observers):
                for pixel, v in zip(self._observers, value):
                    pixel.target = v
                    v.parent = self
            else:
                raise ValueError("The length of 'value' ({}) "
                                 "mismatches the number of pixels ({}).".format(len(value), len(self._sight_lines)))
        else:
            value.parent = self
            for pixel in self._observers:
                pixel.target = value

    @property
    def targetted_path_prob(self):
        return [pixel.targetted_path_prob for pixel in self._observers]
    
    @targetted_path_prob.setter
    def targetted_path_prob(self, value):
        if isinstance(value, (list, tuple)):
            if len(value) == len(self._observers):
                for pixel, v in zip(self._observers, value):
                    pixel.targetted_path_prob = v
            else:
                raise ValueError("The length of 'value' ({}) "
                                 "mismatches the number of pixels ({}).".format(len(value), len(self._sight_lines)))
        else:
            for pixel in self._observers:
                pixel.targetted_path_prob = value
