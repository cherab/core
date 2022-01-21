
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

from raysect.core import Point3D, Vector3D
from raysect.optical.observer import SightLine
from raysect.optical.observer import SpectralRadiancePipeline0D

from .base import _SpectroscopicObserver0DBase


class SpectroscopicSightLine(SightLine, _SpectroscopicObserver0DBase):

    """
    .. deprecated:: 1.4.0
       Use Raysect's SightLine observer instead.

    A simple line of sight observer.

    Multiple `SpectroscopicSightLine` observers can be combined into `SightLineGroup`.

    :param Point3D origin: The origin point for this sight-line. (optional)
    :param Vector3D direction: The observation direction for this sight-line. (optional)
    :param list pipelines: A list of pipelines that will process the resulting spectra
                           from this observer.
                           Default is [SpectralRadiancePipeline0D(accumulate=False)].

    .. code-block:: pycon

       >>> from matplotlib import pyplot as plt
       >>> from raysect.optical import World
       >>> from raysect.core.math import Point3D, Vector3D
       >>> from cherab.tools.observers import SpectroscopicSightLine
       >>>
       >>> world = World()
       ...
       >>> sightline = SpectroscopicSightLine(Point3D(3., 0, 0), Vector3D(-1, 0, 0), name="MySightLine", parent=world)
       >>> sightline.display_progress = False  # control pipeline parameters through the group observer
       >>> sightline.pixel_samples = 5000
       >>> sightline.observe()
       >>> sightline.plot_spectrum(in_photons=True)  # plot the spectrum
       >>> plt.show()
    """

    def __init__(self, origin=None, direction=None, pipelines=None, parent=None, name=None):

        pipelines = pipelines or [SpectralRadiancePipeline0D(accumulate=False)]

        super().__init__(pipelines=pipelines, parent=parent, name=name)

        if origin is not None:
            self.origin = origin

        if direction is not None:
            self.direction = direction
