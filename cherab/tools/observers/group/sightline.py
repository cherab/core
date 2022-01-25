
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
    A group of sight-lines under a single scene-graph node.

    A scene-graph object regrouping a series of 'SightLine'
    observers as a scene-graph parent. Allows combined observation and display
    control simultaneously.

    :ivar list observers: A list of lines of sight (SightLine instances)
                            in this group.

    .. code-block:: pycon

       >>> from math import cos, sin, pi
       >>>
       >>> import matplotlib.pyplot as plt
       >>> from raysect.core import translate, rotate_basis, Point3D, Vector3D
       >>> from raysect.optical import World
       >>> from raysect.optical.observer import RadiancePipeline0D, SpectralRadiancePipeline0D, PowerPipeline0D, SpectralPowerPipeline0D, SightLine
       >>>
       >>> from cherab.tools.observers import SightLineGroup
       >>> from cherab.tools.observers.group.plotting import plot_group_total, plot_group_spectra
       >>>
       >>> world = World()
       >>>
       >>> transform1 = translate(3., 0, 0) * rotate_basis(Vector3D(-cos(pi/10), 0, sin(pi/10)), Vector3D(0, 1, 0))
       >>> sightline_1 = SightLine(transform=transform1, name="SightLine 1")
       >>> transform2 = translate(3, 0 ,0) * rotate_basis(Vector3D(-1, 0, 0), Vector3D(0, 1, 0))
       >>> sightline_2 = SightLine(transform=transform2, name="SightLine 2")
       >>> transform3 = translate(3, 0, 0) * rotate_basis(Vector3D(-cos(pi/10), 0, -sin(pi/10)), Vector3D(0, 1, 0))
       >>> sightline_3 = SightLine(transform=transform3, name="SightLine 3")
       >>>
       >>> group = SightLineGroup(name='MySightLineGroup', parent=world, observers=[sightline_1, sightline_2])
       >>> group.add_observer(sightline_3)
       >>> pipelines = [SpectralRadiancePipeline0D, RadiancePipeline0D]
       >>> keywords = [{'name': 'MySpectralPipeline'}, {'name': 'MyMonoPipeline'}]
       >>> group.connect_pipelines(pipelines, keywords)  # add pipelines to all observers in the group
       >>> group.acceptance_angle = 2  # same value for all sightline_s in the group
       >>> group.radius = 2.e-3
       >>> group.spectral_bins = 512
       >>> group.pixel_samples = [2000, 1000, 2000]  # individual value for each sightline_ in the group
       >>> group.observe()  # combined observation
       >>>
       >>> plot_group_spectra(group, item='MySpectralPipeline', in_photons=True)  # plot the spectra
       >>> plot_group_total(group, item='MyMonoPipeline')  # plot the total signals
       >>> plt.show()
    """

    @Observer0DGroup.observers.setter
    def observers(self, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError("The observers attribute of SightLineGroup must be a list or tuple of SightLines.")

        for observer in value:
            if not isinstance(observer, SightLine):
                raise TypeError("The observers attribute of SightLineGroup must be a list or tuple of "
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
