
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
from cherab.tools.observers.spectroscopy import SpectroscopicFibreOptic
from .base import Observer0DGroup


class FibreOpticGroup(Observer0DGroup):
    """
    A group of fibre optics under a single scene-graph node.

    A scene-graph object regrouping a series of 'SpectroscopicFibreOptic'
    observers as a scene-graph parent. Allows combined observation and display
    control simultaneously.

    :ivar list sight_lines: A list of fibre optics (SpectroscopicFibreOptic instances) in this
                            group.
    :ivar list/float acceptance_angle: The angle in degrees between the z axis and the cone
                                       surface which defines the fibres solid angle sampling
                                       area. The same value can be shared between all sight lines,
                                       or each sight line can be assigned with individual value.
    :ivar list/float radius: The radius of the fibre tip in metres. This radius defines a circular
                             area at the fibre tip which will be sampled over. The same value
                             can be shared between all sight lines, or each sight line can be
                             assigned with individual value.

    .. code-block:: pycon

       >>> from math import cos, sin, pi
       >>> from matplotlib import pyplot as plt
       >>> from raysect.optical import World
       >>> from raysect.optical.observer import SpectralPowerPipeline0D, PowerPipeline0D
       >>> from raysect.core.math import Point3D, Vector3D
       >>> from cherab.tools.observers import SpectroscopicFibreOptic, FibreOpticGroup
       >>>
       >>> world = World()
       ...
       >>> group = FibreOpticGroup(parent=world)
       >>> group.add_sight_line(SpectroscopicFibreOptic(Point3D(3., 0, 0), Vector3D(-cos(pi/10), 0, sin(pi/10)), name="Fibre 1"))
       >>> group.add_sight_line(SpectroscopicFibreOptic(Point3D(3., 0, 0), Vector3D(-1, 0, 0), name="Fibre 2"))
       >>> group.add_sight_line(SpectroscopicFibreOptic(Point3D(3., 0, 0), Vector3D(-cos(pi/10), 0, -sin(pi/10)), name="Fibre 3"))
       >>> group.connect_pipelines([(SpectralPowerPipeline0D, 'MySpectralPipeline', None),
                                    (PowerPipeline0D, 'MyMonoPipeline', None)])  # add pipelines to all fibres in the group
       >>> group.acceptance_angle = 2  # same value for all fibres in the group
       >>> group.radius = 2.e-3
       >>> group.spectral_bins = 512
       >>> group.pixel_samples = [2000, 1000, 2000]  # individual value for each fibre in the group
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
            raise TypeError("The sight_lines attribute of FibreOpticGroup must be a list or tuple of SpectroscopicFibreOptics.")

        for sight_line in value:
            if not isinstance(sight_line, SpectroscopicFibreOptic):
                raise TypeError("The sight_lines attribute of FibreOpticGroup must be a list or tuple of "
                                "SpectroscopicFibreOptics. Value {} is not a SpectroscopicFibreOptic.".format(sight_line))

        # Prevent external changes being made to this list
        for sight_line in value:
            sight_line.parent = self

        self._sight_lines = tuple(value)

    def add_sight_line(self, sight_line):
        """
        Adds new fibre optic to the group.

        :param SpectroscopicFibreOptic sight_line: Fibre optic to add.
        """

        if not isinstance(sight_line, SpectroscopicFibreOptic):
            raise TypeError("The sightline argument must be of type SpectroscopicFibreOptic.")

        sight_line.parent = self
        self._sight_lines = self._sight_lines + (sight_line,)

    @property
    def acceptance_angle(self):
        # The angle in degrees between the z axis and the cone surface which defines the fibres
        # solid angle sampling area.
        return [sight_line.acceptance_angle for sight_line in self._sight_lines]

    @acceptance_angle.setter
    def acceptance_angle(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.acceptance_angle = v
            else:
                raise ValueError("The length of 'acceptance_angle' ({}) "
                                 "mismatches the number of sight-lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.acceptance_angle = value

    @property
    def radius(self):
        # The radius of the fibre tip in metres. This radius defines a circular area at the fibre tip
        # which will be sampled over.
        return [sight_line.radius for sight_line in self._sight_lines]

    @radius.setter
    def radius(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.radius = v
            else:
                raise ValueError("The length of 'radius' ({}) "
                                 "mismatches the number of sight-lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.radius = value
