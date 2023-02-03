
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
from raysect.optical.observer import FibreOptic
from raysect.optical.observer import SpectralRadiancePipeline0D

from .base import _SpectroscopicObserver0DBase


class SpectroscopicFibreOptic(FibreOptic, _SpectroscopicObserver0DBase):

    """
    .. deprecated:: 1.4.0
       Use Raysect's FibreOptic observer instead.

    An optic fibre spectroscopic observer with non-zero acceptance angle.

    Rays are sampled over a circular area at the fibre tip and a conical solid angle
    defined by the acceptance_angle parameter.

    Multiple `SpectroscopicFibreOptic` observers can be combined into `FibreOpticGroup`.

    :param Point3D origin: The origin point for this sight-line. (optional)
    :param Vector3D direction: The observation direction for this sight-line. (optional)
    :param list pipelines: A list of pipelines that will process the resulting spectra
                           from this observer.
                           Default is [SpectralRadiancePipeline0D(accumulate=False)].
    :param float acceptance_angle: The angle in degrees between the z axis and the cone surface
                                   which defines the fibres solid angle sampling area.
    :param float radius: The radius of the fibre tip in metres. This radius defines a circular
                         area at the fibre tip which will be sampled over.

    .. code-block:: pycon

       >>> from matplotlib import pyplot as plt
       >>> from raysect.optical import World
       >>> from raysect.core.math import Point3D, Vector3D
       >>> from cherab.tools.observers import SpectroscopicFibreOptic
       >>>
       >>> world = World()
       ...
       >>> fibreoptic = SpectroscopicFibreOptic(Point3D(3., 0, 0), Vector3D(-1, 0, 0), name="MyFibreOptic", parent=world)
       >>> fibreoptic.acceptance_angle = 5.
       >>> fibreoptic.radius = 2.e-3
       >>> fibreoptic.display_progress = False  # control pipeline parameters through the group observer
       >>> fibreoptic.pixel_samples = 5000
       >>> fibreoptic.observe()
       >>> fibreoptic.plot_spectrum(in_photons=True)  # plots the spectrum
       >>> plt.show()
    """

    def __init__(self, origin=None, direction=None, pipelines=None, acceptance_angle=None, radius=None, parent=None, name=None):

        pipelines = pipelines or [SpectralRadiancePipeline0D(accumulate=False)]

        super().__init__(pipelines=pipelines, parent=parent, name=name, acceptance_angle=acceptance_angle, radius=radius)

        if origin is not None:
            self.origin = origin

        if direction is not None:
            self.direction = direction
