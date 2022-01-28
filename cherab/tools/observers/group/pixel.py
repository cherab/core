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
from raysect.optical.observer import Pixel

from .base import Observer0DGroup


class PixelGroup(Observer0DGroup):
    """
    A group of pixels under a single scene-graph node.

    A scene-graph object regrouping a series of 'Pixel'
    observers as a scene-graph parent. Allows combined observation and display
    control simultaneously.

    :ivar list x_width: Width of pixel along local x axis
    :ivar list y_width: Width of pixel along local y axis
    """
    _OBSERVER_TYPE = Pixel

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
                                 "mismatches the number of pixels ({}).".format(len(value), len(self._observers)))
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
                                 "mismatches the number of pixels ({}).".format(len(value), len(self._observers)))
        else:
            for pixel in self._observers:
                pixel.y_width = value
