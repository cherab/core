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

import matplotlib.pyplot as plt
from numpy import ndarray
from raysect.optical import Spectrum
from raysect.optical.observer import SpectralRadiancePipeline0D, SpectralPowerPipeline0D, RadiancePipeline0D

from .base import Observer0DGroup
from cherab.tools.observers.spectroscopy import SpectroscopicFibreOptic, SpectroscopicSightLine


class SpectroscopicObserver0DGroup(Observer0DGroup):
    """
    .. deprecated:: 1.4.0
       Use SightLineGroup or FibreOpticGroup based on Raysect's observers instead.
    
    A base class for a group of 0D spectroscopic observers under a single scene-graph node.

    A scene-graph object regrouping a series of observers as a scene-graph parent.
    Allows combined observation and display control simultaneously.
    Note that for any property except `names` and `pipelines`, the same value can be shared between
    all sight lines, or each sight line can be assigned with individual value.

    :ivar list/Point3D origin: The origin points for the sight lines.
    :ivar list/Vector3D direction: The observation directions for the sight lines.
    :ivar list/bool display_progress: Toggles the display of live render progress.
    :ivar list/bool accumulate: Toggles whether to accumulate samples with subsequent
                                observations.
    """

    def __init__(self, parent=None, transform=None, name=None, observers=None):
        super().__init__(parent=parent, transform=transform, name=name, observers=observers)

    @property
    def sight_lines(self):
        return self._observers

    @sight_lines.setter
    def sight_lines(self, value):
        self.observers = value

    def add_sight_line(self, sight_line):
        """
        Adds new fibre optic to the group.

        :param SpectroscopicFibreOptic sight_line: Fibre optic to add.
        """
        self.add_observer(sight_line)

    @property
    def origin(self):
        # The origin points for the sight lines.
        return [sight_line.origin for sight_line in self._observers]

    @origin.setter
    def origin(self, value):
        if isinstance(value, (list, tuple)):
            if len(value) == len(self._observers):
                for sight_line, v in zip(self._observers, value):
                    sight_line.origin = v
            else:
                raise ValueError("The length of 'origin' ({}) "
                                 "mismatches the number of sight-lines ({}).".format(len(value), len(self._observers)))
        else:
            for sight_line in self._observers:
                sight_line.origin = value

    @property
    def direction(self):
        # The observation directions for the sight lines.
        return [sight_line.direction for sight_line in self._observers]

    @direction.setter
    def direction(self, value):
        if isinstance(value, (list, tuple)):
            if len(value) == len(self._observers):
                for sight_line, v in zip(self._observers, value):
                    sight_line.direction = v
            else:
                raise ValueError("The length of 'direction' ({}) "
                                 "mismatches the number of sight-lines ({}).".format(len(value), len(self._observers)))
        else:
            for sight_line in self._observers:
                sight_line.direction = value
    

    @property
    def display_progress(self):
        # Toggles the display of live render progress.
        return [observer.display_progress for observer in self._observers]

    @display_progress.setter
    def display_progress(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._observers):
                for observer, v in zip(self._observers, value):
                    observer.display_progress = v
            else:
                raise ValueError("The length of 'display_progress' ({}) "
                                 "mismatches the number of sight-lines ({}).".format(len(value), len(self._observers)))
        else:
            for observer in self._observers:
                observer.display_progress = value

    @property
    def accumulate(self):
        # Toggles whether to accumulate samples with subsequent calls to observe().
        return [observer.accumulate for observer in self._observers]

    @accumulate.setter
    def accumulate(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._observers):
                for observer, v in zip(self._observers, value):
                    observer.accumulate = v
            else:
                raise ValueError("The length of 'accumulate' ({}) "
                                 "mismatches the number of sight-lines ({}).".format(len(value), len(self._observers)))
        else:
            for observer in self._observers:
                observer.accumulate = value


    def connect_pipelines(self, properties=[(SpectralRadiancePipeline0D, None, None)]):
        """
        Connects pipelines of given kinds and names to each sight-line in the group.
        Connected pipelines are non-accumulating by default.

        :param list properties: 3-tuple list of pipeline properties in order (class, name, filter).
                                Default is [(SpectralRadiancePipeline0D, None, None)].
                                The following pipeline classes are supported:
                                    SpectralRadiacnePipeline0D,
                                    SpectralPowerPipeline0D,
                                    RadiacnePipeline0D,
                                    PowerPipeline0D.
                                Filters are applied to the mono pipelines only, namely,
                                PowerPipeline0D or RadiacnePipeline0D. The values provided for spectral
                                pipelines will be ignored. The filter must be an instance of
                                SpectralFunction or None.

        """

        for sight_line in self._observers:
            sight_line.connect_pipelines(properties)

    def _get_same_pipelines(self, item):
        pipelines = []
        sight_lines = []
        for sight_line in self._observers:
            try:
                pipelines.append(sight_line.get_pipeline(item))
            except (ValueError, IndexError):
                continue
            else:
                sight_lines.append(sight_line)

        if len(pipelines) == 0:
            raise ValueError("Pipeline {} was not found for any sight-line in this {}.".format((item, self.__class__.__name__)))

        pipeline_types = set(type(pipeline) for pipeline in pipelines)
        if len(pipeline_types) > 1:
            raise ValueError("Pipelines {} have different types for different sight-lines.".format(item))

        return pipelines, sight_lines

    def plot_total_signal(self, item=0, ax=None):
        """
        Plots total (wavelength-integrated) signal for each sight line in the group.

        :param str/int item: The index or name of the pipeline. Default: 0.
        :param Axes ax: Existing matplotlib axes.

        """

        pipelines, sight_lines = self._get_same_pipelines(item)

        if ax is None:
            _, ax = plt.subplots(constrained_layout=True)

        signal = []
        tick_labels = []
        for pipeline, sight_line in zip(pipelines, sight_lines):
            if isinstance(pipeline, SpectralPowerPipeline0D):
                spectrum = Spectrum(pipeline.min_wavelength, pipeline.max_wavelength, pipeline.bins)
                spectrum.samples = pipeline.samples.mean
                signal.append(spectrum.total())
            else:
                signal.append(pipeline.value.mean)

            if sight_line.name and len(sight_line.name):
                tick_labels.append(sight_line.name)
            else:
                tick_labels.append(self._observers.index(sight_line))

        if isinstance(pipeline, (SpectralRadiancePipeline0D, RadiancePipeline0D)):
            ylabel = 'Radiance (W/m^2/str)'
        else:  # SpectralPowerPipeline0D or PowerPipeline0D
            ylabel = 'Power (W)'

        ax.bar(list(range(len(signal))), signal, tick_label=tick_labels, label=item)

        if isinstance(item, int):
            # check if pipelines share the same name
            if len(set(pipeline.name for pipeline in pipelines)) == 1 and pipelines[0].name and len(pipelines[0].name):
                ax.set_title('{}: {}'.format(self.name, pipelines[0].name))
            else:
                # pipelines have different names or name is not set
                ax.set_title('{}: pipeline {}'.format(self.name, item))
        elif isinstance(item, str):
            ax.set_title('{}: {}'.format(self.name, item))

        ax.set_ylabel(ylabel)
        ax.set_xlabel('Line of sight')

        return ax

    def plot_spectra(self, item=0, in_photons=False, ax=None):
        """
        Plot the spectra observed by each line of sight in the group for a given pipeline.

        :param str/int item: The index or name of the pipeline. Default: 0.
        :param bool in_photons: If True, plots the spectrum in photon/s/nm instead of W/nm.
                                Default is False.
        :param Axes ax: Existing matplotlib axes.
        """

        pipelines, sight_lines = self._get_same_pipelines(item)

        if ax is None:
            _, ax = plt.subplots(constrained_layout=True)

        for sight_line in sight_lines:
            sight_line.plot_spectrum(item=item, in_photons=in_photons, ax=ax, extras=False)

        if isinstance(pipelines[0], SpectralRadiancePipeline0D):
            ylabel = 'Spectral radiance (photon/s/m^2/str/nm)' if in_photons else 'Spectral radiance (W/m^2/str/nm)'
        else:  # SpectralPowerPipeline0D
            ylabel = 'Spectral power (photon/s/nm)' if in_photons else 'Spectral power (W/nm)'

        if isinstance(item, int):
            # check if pipelines share the same name
            if len(set(pipeline.name for pipeline in pipelines)) == 1 and pipelines[0].name and len(pipelines[0].name):
                ax.set_title('{}: {}'.format(self.name, pipelines[0].name))
            else:
                # pipelines have different names or name is not set
                ax.set_title('{}: pipeline {}'.format(self.name, item))
        elif isinstance(item, str):
            ax.set_title('{}: {}'.format(self.name, item))

        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel(ylabel)
        ax.legend()

        return ax


class SpectroscopicFibreOpticGroup(SpectroscopicObserver0DGroup):
    """
    .. deprecated:: 1.4.0
       Use FibreOpticGroup based on Raysect's FibreOptic observer instead.
    
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
    _OBSERVER_TYPE = SpectroscopicFibreOptic

    @property
    def acceptance_angle(self):
        # The angle in degrees between the z axis and the cone surface which defines the fibres
        # solid angle sampling area.
        return [sight_line.acceptance_angle for sight_line in self._observers]

    @acceptance_angle.setter
    def acceptance_angle(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._observers):
                for sight_line, v in zip(self._observers, value):
                    sight_line.acceptance_angle = v
            else:
                raise ValueError("The length of 'acceptance_angle' ({}) "
                                 "mismatches the number of sight-lines ({}).".format(len(value), len(self._observers)))
        else:
            for sight_line in self._observers:
                sight_line.acceptance_angle = value

    @property
    def radius(self):
        # The radius of the fibre tip in metres. This radius defines a circular area at the fibre tip
        # which will be sampled over.
        return [sight_line.radius for sight_line in self._observers]

    @radius.setter
    def radius(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._observers):
                for sight_line, v in zip(self._observers, value):
                    sight_line.radius = v
            else:
                raise ValueError("The length of 'radius' ({}) "
                                 "mismatches the number of sight-lines ({}).".format(len(value), len(self._observers)))
        else:
            for sight_line in self._observers:
                sight_line.radius = value


class SpectroscopicSightLineGroup(SpectroscopicObserver0DGroup):
    """
    .. deprecated:: 1.4.0
       Use SightLineGroup based on Raysect's SightLine observer instead.

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
    _OBSERVER_TYPE = SpectroscopicSightLine

    @property
    def sensitivity(self):
        # A list of observer sensitivities.
        return [observer.sensitivity for observer in self._observers]

    @sensitivity.setter
    def names(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._observers):
                for observer, v in zip(self._observers, value):
                    observer.sensitivity = v
            else:
                raise ValueError("The length of 'sensitivity' ({}) ".format(len(value)) +
                                 "mismatches the number of observers ({}).".format(len(self._observers)))
        else:
            for observer in self._observers:
                observer.sensitivity = value
