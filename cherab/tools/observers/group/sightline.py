
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

from cherab.tools.observers.spectroscopy import SpectroscopicSightLine
from .base import Observer0DGroup


class SightLineGroup(Observer0DGroup):
    """
    A group of spectroscopic sight-lines under a single scene-graph node.

    A scene-graph object regrouping a series of 'SpectroscopicSightLine'
    observers as a scene-graph parent. Allows combined observation and display
    control simultaneously.

    :ivar list sight_lines: A list of lines of sight (SpectroscopicSightLine instances)
                            in this group.

    .. code-block:: pycon

       >>> from math import cos, sin, pi
       >>> from matplotlib import pyplot as plt
       >>> from raysect.optical import World
       >>> from raysect.optical.observer import SpectralRadiancePipeline0D, RadiancePipeline0D
       >>> from raysect.core.math import Point3D, Vector3D
       >>> from cherab.tools.observers import SpectroscopicSightLine, SightLineGroup
       >>>
       >>> world = World()
       ...
       >>> group = SightLineGroup(parent=world)
       >>> group.add_sight_line(SpectroscopicSightLine(Point3D(3., 0, 0), Vector3D(-cos(pi/10), 0, sin(pi/10)), name="SightLine 1"))
       >>> group.add_sight_line(SpectroscopicSightLine(Point3D(3., 0, 0), Vector3D(-1, 0, 0), name="SightLine 2"))
       >>> group.add_sight_line(SpectroscopicSightLine(Point3D(3., 0, 0), Vector3D(-cos(pi/10), 0, -sin(pi/10)), name="SightLine 3"))
       >>> group.connect_pipelines([(SpectralRadiancePipeline0D, 'MySpectralPipeline', None),
                                    (RadiancePipeline0D, 'MyMonoPipeline', None)])  # add pipelines to all sight lines in the group
       >>> group.spectral_bins = 512  # same value for all sight lines in the group
       >>> group.pixel_samples = [2000, 1000, 2000]  # individual value for each sight line in the group
       >>> group.display_progress = False  # control pipeline parameters through the group observer
       >>> group.observe()  # combined observation
       >>> group.plot_spectra(item='MySpectralPipeline', in_photons=True)  # plot the spectra
       >>> group.plot_total_signal(item='MyMonoPipeline')  # plot the total signals
       >>> plt.show()
    """

    @property
    def sight_lines(self):
        return self._sight_lines

    @sight_lines.setter
    def sight_lines(self, value):

        if not isinstance(value, (list, tuple)):
            raise TypeError("The sight_lines attribute of LineOfSightGroup must be a list or tuple of SpectroscopicSightLines.")

        for sight_line in value:
            if not isinstance(sight_line, SpectroscopicSightLine):
                raise TypeError("The sight_lines attribute of LineOfSightGroup must be a list or tuple of "
                                "SpectroscopicSightLines. Value {} is not a SpectroscopicSightLine.".format(sight_line))

        # Prevent external changes being made to this list
        for sight_line in value:
            sight_line.parent = self

        self._sight_lines = tuple(value)

    def add_sight_line(self, sight_line):
        """
        Adds new line of sight to the group.

        :param SpectroscopicSightLine sight_line: Sight line to add.
        """

        if not isinstance(sight_line, SpectroscopicSightLine):
            raise TypeError("The sight_line argument must be of type SpectroscopicSightLine.")

        sight_line.parent = self
        self._sight_lines = self._sight_lines + (sight_line,)
