
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
import matplotlib.pyplot as plt
from raysect.core import Node
from raysect.core.workflow import RenderEngine
from raysect.optical import Spectrum
from raysect.optical.observer import SpectralRadiancePipeline0D, SpectralPowerPipeline0D, RadiancePipeline0D


class Observer0DGroup(Node):
    """
    A base class for a group of 0D spectroscopic observers under a single scene-graph node.

    A scene-graph object regrouping a series of observers as a scene-graph parent.
    Allows combined observation and display control simultaneously.
    Note that for any property except `names` and `pipelines`, the same value can be shared between
    all sight lines, or each sight line can be assigned with individual value.

    :ivar list names: A list of sight-line names.
    :ivar list pipelines: A list of all pipelines connected to each sight-line in the group.
    :ivar list/Point3D origin: The origin points for the sight lines.
    :ivar list/Vector3D direction: The observation directions for the sight lines.
    :ivar list/RenderEngine render_engine: Rendering engine used by the sight lines.
                                           Note that if the engine is shared, changing its
                                           parameters for one sight line in a group will affect
                                           all sight lines.
    :ivar list/bool display_progress: Toggles the display of live render progress.
    :ivar list/bool accumulate: Toggles whether to accumulate samples with subsequent
                                observations.
    :ivar list/float min_wavelength: Lower wavelength bound for sampled spectral range.
    :ivar list/float max_wavelength: Upper wavelength bound for sampled spectral range.
    :ivar list/int spectral_bins: The number of spectral samples over the wavelength range.
    :ivar list/float ray_extinction_prob: Probability of ray extinction after every material
                                          intersection.
    :ivar list/float ray_extinction_min_depth: Minimum number of paths before russian roulette
                                               style ray extinction.
    :ivar list/int ray_max_depth: Maximum number of Ray paths before terminating Ray.
    :ivar list/float ray_important_path_weight: Relative weight of important path sampling.
    :ivar list/int pixel_samples: The number of samples to take per pixel.
    :ivar list/int samples_per_task: Minimum number of samples to request per task.
    """

    def __init__(self, parent=None, transform=None, name=None):
        super().__init__(parent=parent, transform=transform, name=name)

        self._sight_lines = tuple()

    def __getitem__(self, item):

        if isinstance(item, int):
            try:
                return self._sight_lines[item]
            except IndexError:
                raise IndexError("Sight-line number {} not available in this {} "
                                 "with only {} sight-lines.".format(item, self.__class__.__name__, len(self._sight_lines)))
        elif isinstance(item, str):
            sightlines = [sight_line for sight_line in self._sight_lines if sight_line.name == item]
            if len(sightlines) == 1:
                return sightlines[0]

            if len(sightlines) == 0:
                raise ValueError("Sight-line '{}' was not found in this {}.".format(item, self.__class__.__name__))

            raise ValueError("Found {} sight-lines with name {} in this {}.".format(len(sightlines), item, self.__class__.__name__))
        else:
            raise TypeError("{} key must be of type int or str.".format(self.__class__.__name__))

    @property
    def names(self):
        # A list of sight-line names.
        return [sight_line.name for sight_line in self._sight_lines]

    @names.setter
    def names(self, value):
        if isinstance(value, (list, tuple)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.name = v
            else:
                raise ValueError("The length of 'names' ({}) "
                                 "mismatches the number of sight-lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            raise TypeError("The names attribute must be a list or tuple.")

    @property
    def origin(self):
        # The origin points for the sight lines.
        return [sight_line.origin for sight_line in self._sight_lines]

    @origin.setter
    def origin(self, value):
        if isinstance(value, (list, tuple)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.origin = v
            else:
                raise ValueError("The length of 'origin' ({}) "
                                 "mismatches the number of sight-lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.origin = value

    @property
    def direction(self):
        # The observation directions for the sight lines.
        return [sight_line.direction for sight_line in self._sight_lines]

    @direction.setter
    def direction(self, value):
        if isinstance(value, (list, tuple)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.direction = v
            else:
                raise ValueError("The length of 'direction' ({}) "
                                 "mismatches the number of sight-lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.direction = value

    @property
    def render_engine(self):
        # Rendering engine used by the sight lines.
        return [sight_line.render_engine for sight_line in self._sight_lines]

    @render_engine.setter
    def render_engine(self, value):
        if isinstance(value, (list, tuple)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    if isinstance(v, RenderEngine):
                        sight_line.render_engine = v
                    else:
                        raise TypeError("The list 'render_engine' must contain only RenderEngine instances.")
            else:
                raise ValueError("The length of 'render_engine' ({}) "
                                 "mismatches the number of sight-lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            if not isinstance(value, RenderEngine):
                raise TypeError("The list 'render_engine' must contain only RenderEngine instances.")
            for sight_line in self._sight_lines:
                sight_line.render_engine = value

    @property
    def display_progress(self):
        # Toggles the display of live render progress.
        return [sight_line.display_progress for sight_line in self._sight_lines]

    @display_progress.setter
    def display_progress(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.display_progress = v
            else:
                raise ValueError("The length of 'display_progress' ({}) "
                                 "mismatches the number of sight-lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.display_progress = value

    @property
    def accumulate(self):
        # Toggles whether to accumulate samples with subsequent calls to observe().
        return [sight_line.accumulate for sight_line in self._sight_lines]

    @accumulate.setter
    def accumulate(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.accumulate = v
            else:
                raise ValueError("The length of 'accumulate' ({}) "
                                 "mismatches the number of sight-lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.accumulate = value

    @property
    def min_wavelength(self):
        # Lower wavelength bound for sampled spectral range.
        return [sight_line.min_wavelength for sight_line in self._sight_lines]

    @min_wavelength.setter
    def min_wavelength(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.min_wavelength = v
            else:
                raise ValueError("The length of 'min_wavelength' ({}) "
                                 "mismatches the number of sight-lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.min_wavelength = value

    @property
    def max_wavelength(self):
        # Upper wavelength bound for sampled spectral range.
        return [sight_line.max_wavelength for sight_line in self._sight_lines]

    @max_wavelength.setter
    def max_wavelength(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.max_wavelength = v
            else:
                raise ValueError("The length of 'max_wavelength' ({}) "
                                 "mismatches the number of sight-lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.max_wavelength = value

    @property
    def spectral_bins(self):
        # The number of spectral samples over the wavelength range.
        return [sight_line.spectral_bins for sight_line in self._sight_lines]

    @spectral_bins.setter
    def spectral_bins(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.spectral_bins = v
            else:
                raise ValueError("The length of 'spectral_bins' ({}) "
                                 "mismatches the number of sight-lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.spectral_bins = value

    @property
    def ray_extinction_prob(self):
        # Probability of ray extinction after every material intersection.
        return [sight_line.ray_extinction_prob for sight_line in self._sight_lines]

    @ray_extinction_prob.setter
    def ray_extinction_prob(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.ray_extinction_prob = v
            else:
                raise ValueError("The length of 'ray_extinction_prob' ({}) "
                                 "mismatches the number of sight-lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.ray_extinction_prob = value

    @property
    def ray_extinction_min_depth(self):
        # Minimum number of paths before russian roulette style ray extinction.
        return [sight_line.ray_extinction_min_depth for sight_line in self._sight_lines]

    @ray_extinction_min_depth.setter
    def ray_extinction_min_depth(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.ray_extinction_min_depth = v
            else:
                raise ValueError("The length of 'ray_extinction_min_depth' ({}) "
                                 "mismatches the number of sight-lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.ray_extinction_min_depth = value

    @property
    def ray_max_depth(self):
        # Maximum number of Ray paths before terminating Ray.
        return [sight_line.ray_max_depth for sight_line in self._sight_lines]

    @ray_max_depth.setter
    def ray_max_depth(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.ray_max_depth = v
            else:
                raise ValueError("The length of 'ray_max_depth' ({}) "
                                 "mismatches the number of sight-lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.ray_max_depth = value

    @property
    def ray_important_path_weight(self):
        # Relative weight of important path sampling.
        return [sight_line.ray_important_path_weight for sight_line in self._sight_lines]

    @ray_important_path_weight.setter
    def ray_important_path_weight(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.ray_important_path_weight = v
            else:
                raise ValueError("The length of 'ray_important_path_weight' ({}) "
                                 "mismatches the number of sight-lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.ray_important_path_weight = value

    @property
    def pixel_samples(self):
        # The number of samples to take per pixel.
        return [sight_line.pixel_samples for sight_line in self._sight_lines]

    @pixel_samples.setter
    def pixel_samples(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.pixel_samples = v
            else:
                raise ValueError("The length of 'pixel_samples' ({}) "
                                 "mismatches the number of sight-lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.pixel_samples = value

    @property
    def samples_per_task(self):
        # Minimum number of samples to request per task.
        return [sight_line.samples_per_task for sight_line in self._sight_lines]

    @samples_per_task.setter
    def samples_per_task(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.samples_per_task = v
            else:
                raise ValueError("The length of 'samples_per_task' ({}) "
                                 "mismatches the number of sight-lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.samples_per_task = value

    @property
    def pipelines(self):
        # A list of all pipelines connected to each sight-line in the group.
        return [sight_line.pipelines for sight_line in self._sight_lines]

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

        for sight_line in self._sight_lines:
            sight_line.connect_pipelines(properties)

    def observe(self):
        """
        Starts the observation.
        """
        for sight_line in self._sight_lines:
            sight_line.observe()

    def _get_same_pipelines(self, item):
        pipelines = []
        sight_lines = []
        for sight_line in self._sight_lines:
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
                tick_labels.append(self._sight_lines.index(sight_line))

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
