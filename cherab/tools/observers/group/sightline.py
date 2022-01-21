
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

from raysect.optical.observer import SightLine
from .base import Observer0DGroup


class SightLineGroup(Observer0DGroup):
    """
    A group of spectroscopic sight-lines under a single scene-graph node.

    A scene-graph object regrouping a series of 'SightLine'
    observers as a scene-graph parent. Allows combined observation and display
    control simultaneously.

    :ivar list observers: A list of lines of sight (SightLine instances)
                            in this group.

    .. code-block:: pycon

       >>> from math import cos, sin, pi
       >>> from matplotlib import pyplot as plt
       >>> from raysect.optical import World
       >>> from raysect.optical.observer import SpectralRadiancePipeline0D, RadiancePipeline0D, SightLine
       >>> from raysect.core.math import Point3D, Vector3D
       >>> from cherab.tools.observers import SightLineGroup
       >>> from cherab.tools.observers.plotting import plot_group_total, plot_group_spectra
       >>>
       >>> world = World()
       ...
       >>> group = SightLineGroup(parent=world)
       >>> group.add_observer(SightLine(Point3D(3., 0, 0), Vector3D(-cos(pi/10), 0, sin(pi/10)), name="SightLine 1"))
       >>> group.add_observer(SightLine(Point3D(3., 0, 0), Vector3D(-1, 0, 0), name="SightLine 2"))
       >>> group.add_observer(SightLine(Point3D(3., 0, 0), Vector3D(-cos(pi/10), 0, -sin(pi/10)), name="SightLine 3"))
       >>> group.connect_pipelines([(SpectralRadiancePipeline0D, 'MySpectralPipeline', None),
                                    (RadiancePipeline0D, 'MyMonoPipeline', None)])  # add pipelines to all observers in the group
       >>> group.spectral_bins = 512  # same value for all observers in the group
       >>> group.pixel_samples = [2000, 1000, 2000]  # individual value for each observer in the group
       >>> group.display_progress = False  # control pipeline parameters through the group observer
       >>> group.observe()  # combined observation
       >>> 
       >>> plot_group_spectra(group, item='MySpectralPipeline', in_photons=True)  # plot the spectra
       >>> plot_group_total(group, item='MyMonoPipeline')  # plot the total signals
       >>> plt.show()
    """

    @Observer0DGroup.observers.setter
    def observers(self, value):

        if not isinstance(value, (list, tuple)):
            raise TypeError("The observers attribute of LineOfSightGroup must be a list or tuple of SightLines.")

        for observer in value:
            if not isinstance(observer, SightLine):
                raise TypeError("The observers attribute of LineOfSightGroup must be a list or tuple of "
                                "SightLines. Value {} is not a SightLine.".format(observer))

        # Prevent external changes being made to this list
        for observer in value:
            observer.parent = self

        self._observers = tuple(value)

    def add_observer(self, observer):
        """
        Adds new line of sight to the group.

        :param SightLine observer: observer to add.
        """

        if not isinstance(observer, SightLine):
            raise TypeError("The observer argument must be of type SightLine.")

        observer.parent = self
        self._observers = self._observers + (observer,)
