
# Copyright 2016-2018 Euratom
# Copyright 2016-2018 United Kingdom Atomic Energy Authority
# Copyright 2016-2018 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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
from raysect.core import Node, translate, rotate_basis, Point3D, Vector3D
from raysect.optical import Spectrum
from raysect.optical.observer import FibreOptic, SightLine, SpectralRadiancePipeline0D


class Observer0DGroup(Node):
    """
    A base class for a group of 0D spectroscopic observers under a single scene-graph node.

    A scene-graph object regrouping a series of observers as a scene-graph parent.
    Allows combined observation and display control simultaneously.
    """

    def __init__(self, parent=None, transform=None, name=''):
        super().__init__(parent=parent, transform=transform, name=name)

        self._sight_lines = []

    def __getitem__(self, item):

        if isinstance(item, int):
            try:
                return self._sight_lines[item]
            except IndexError:
                raise IndexError("Sight-line number {} not available in this LineOfSightGroup.".format(item))
        elif isinstance(item, str):
            for sight_line in self._sight_lines:
                if sight_line.name == item:
                    return sight_line
            else:
                raise ValueError("Sightline '{}' was not found in this LineOfSightGroup.".format(item))
        else:
            raise TypeError("LineOfSightGroup key must be of type int or str.")

    @property
    def names(self):
        """
        A list of sight-line names.

        :rtype: list
        """
        return [sight_line.name for sight_line in self._sight_lines]

    @names.setter
    def names(self, value):
        if isinstance(value, (list, tuple)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.name = v
            else:
                raise ValueError("The length of 'names' ({}) "
                                 "mismatches the number of sight lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            raise TypeError("The names attribute must be a list or tuple.")

    @property
    def point(self):
        """
        The observation points for the sight lines.

        The same value can be shared between all sight lines,
        or each sight line can be assigned with individual value.

        :rtype: list
        """
        return [sight_line.point for sight_line in self._sight_lines]

    @point.setter
    def point(self, value):
        if isinstance(value, (list, tuple)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.point = v
            else:
                raise ValueError("The length of 'point' ({}) "
                                 "mismatches the number of sight lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.point = value

    @property
    def direction(self):
        """
        The observation directions for the sight lines.

        The same value can be shared between all sight lines,
        or each sight line can be assigned with individual value.

        :rtype: list
        """
        return [sight_line.direction for sight_line in self._sight_lines]

    @direction.setter
    def direction(self, value):
        if isinstance(value, (list, tuple)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.direction = v
            else:
                raise ValueError("The length of 'direction' ({}) "
                                 "mismatches the number of sight lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.direction = value

    @property
    def min_wavelength(self):
        """
        Lower wavelength bound for sampled spectral range.

        The same value can be shared between all sight lines,
        or each sight line can be assigned with individual value.

        :rtype: list
        """
        return [sight_line.min_wavelength for sight_line in self._sight_lines]

    @min_wavelength.setter
    def min_wavelength(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.min_wavelength = v
            else:
                raise ValueError("The length of 'min_wavelength' ({}) "
                                 "mismatches the number of sight lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.min_wavelength = value

    @property
    def max_wavelength(self):
        """
        Upper wavelength bound for sampled spectral range.

        The same value can be shared between all sight lines,
        or each sight line can be assigned with individual value.

        :rtype: list
        """
        return [sight_line.max_wavelength for sight_line in self._sight_lines]

    @max_wavelength.setter
    def max_wavelength(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.max_wavelength = v
            else:
                raise ValueError("The length of 'max_wavelength' ({}) "
                                 "mismatches the number of sight lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.max_wavelength = value

    @property
    def spectral_bins(self):
        """
        The number of spectral samples over the wavelength range.

        The same value can be shared between all sight lines,
        or each sight line can be assigned with individual value.

        :rtype: list
        """
        return [sight_line.spectral_bins for sight_line in self._sight_lines]

    @spectral_bins.setter
    def spectral_bins(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.spectral_bins = v
            else:
                raise ValueError("The length of 'spectral_bins' ({}) "
                                 "mismatches the number of sight lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.spectral_bins = value

    @property
    def ray_extinction_prob(self):
        """
        Probability of ray extinction after every material intersection.

        The same value can be shared between all sight lines,
        or each sight line can be assigned with individual value.

        :rtype: list
        """
        return [sight_line.ray_extinction_prob for sight_line in self._sight_lines]

    @ray_extinction_prob.setter
    def ray_extinction_prob(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.ray_extinction_prob = v
            else:
                raise ValueError("The length of 'ray_extinction_prob' ({}) "
                                 "mismatches the number of sight lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.ray_extinction_prob = value

    @property
    def ray_extinction_min_depth(self):
        """
        Minimum number of paths before russian roulette style ray extinction.

        The same value can be shared between all sight lines,
        or each sight line can be assigned with individual value.

        :rtype: list
        """
        return [sight_line.ray_extinction_min_depth for sight_line in self._sight_lines]

    @ray_extinction_min_depth.setter
    def ray_extinction_min_depth(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.ray_extinction_min_depth = v
            else:
                raise ValueError("The length of 'ray_extinction_min_depth' ({}) "
                                 "mismatches the number of sight lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.ray_extinction_min_depth = value

    @property
    def ray_max_depth(self):
        """
        Maximum number of Ray paths before terminating Ray.

        The same value can be shared between all sight lines,
        or each sight line can be assigned with individual value.

        :rtype: list
        """
        return [sight_line.ray_max_depth for sight_line in self._sight_lines]

    @ray_max_depth.setter
    def ray_max_depth(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.ray_max_depth = v
            else:
                raise ValueError("The length of 'ray_max_depth' ({}) "
                                 "mismatches the number of sight lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.ray_max_depth = value

    @property
    def ray_important_path_weight(self):
        """
        Relative weight of important path sampling.

        The same value can be shared between all sight lines,
        or each sight line can be assigned with individual value.

        :rtype: list
        """
        return [sight_line.ray_important_path_weight for sight_line in self._sight_lines]

    @ray_important_path_weight.setter
    def ray_important_path_weight(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.ray_important_path_weight = v
            else:
                raise ValueError("The length of 'ray_important_path_weight' ({}) "
                                 "mismatches the number of sight lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.ray_important_path_weight = value

    @property
    def pixel_samples(self):
        """
        The number of samples to take per pixel.

        The same value can be shared between all sight lines,
        or each sight line can be assigned with individual value.

        :rtype: list
        """
        return [sight_line.pixel_samples for sight_line in self._sight_lines]

    @pixel_samples.setter
    def pixel_samples(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.pixel_samples = v
            else:
                raise ValueError("The length of 'pixel_samples' ({}) "
                                 "mismatches the number of sight lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.pixel_samples = value

    @property
    def samples_per_task(self):
        """
        Minimum number of samples to request per task.

        The same value can be shared between all sight lines,
        or each sight line can be assigned with individual value.

        :rtype: list
        """
        return [sight_line.samples_per_task for sight_line in self._sight_lines]

    @samples_per_task.setter
    def samples_per_task(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.samples_per_task = v
            else:
                raise ValueError("The length of 'samples_per_task' ({}) "
                                 "mismatches the number of sight lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.samples_per_task = value

    def observe(self):
        for sight_line in self._sight_lines:
            sight_line.observe()

    def plot_spectra(self, unit='J', ymax=None):
        """
        Plot the spectra observed by each line of sight in the group.

        :param str unit: Plots the spectrum in J/s/m2/str/nm (units='J'),
                         or in ph/s/m2/str/nm (units='ph').
        :param float ymax: Upper limit of y-axis.
        """
        for sight_line in self.sight_lines:
            sight_line.plot_spectra(unit=unit, extras=False)

        if ymax is not None:
            plt.ylim(ymax=ymax)

        plt.title(self.name)
        plt.xlabel('wavelength (nm)')
        plt.ylabel('radiance ({}/s/m^2/str/nm)'.format(unit))
        plt.legend()


class LineOfSightGroup(Observer0DGroup):
    """
    A group of spectroscopic sight-lines under a single scene-graph node.

    A scene-graph object regrouping a series of 'SpectroscopicSightLine'
    observers as a scene-graph parent. Allows combined observation and display
    control simultaneously.
    """

    @property
    def sight_lines(self):
        return self._sight_lines

    @sight_lines.setter
    def sight_lines(self, value):

        if not isinstance(value, list):
            raise TypeError("The sightlines attribute of LineOfSightGroup must be a list of SpectroscopicSightLines.")

        for sight_line in value:
            if not isinstance(sight_line, SpectroscopicSightLine):
                raise TypeError("The sightlines attribute of LineOfSightGroup must be a list of "
                                "SpectroscopicSightLines. Value {} is not a SpectroscopicSightLine.".format(sight_line))

        # Prevent external changes being made to this list
        value = value.copy()
        for sight_line in value:
            sight_line.parent = self

        self._sight_lines = value

    def add_sight_line(self, sight_line):

        if not isinstance(sight_line, SpectroscopicSightLine):
            raise TypeError("The sightline argument must be of type SpectroscopicSightLine.")

        sight_line.parent = self
        self._sight_lines.append(sight_line)


class FibreOpticGroup(Observer0DGroup):
    """
    A group of fibre optics under a single scene-graph node.

    A scene-graph object regrouping a series of 'SpectroscopicFibreOptic'
    observers as a scene-graph parent. Allows combined observation and display
    control simultaneously.
    """

    @property
    def sight_lines(self):
        return self._sight_lines

    @sight_lines.setter
    def sight_lines(self, value):

        if not isinstance(value, list):
            raise TypeError("The sightlines attribute of LineOfSightGroup must be a list of SpectroscopicSightLines.")

        for sight_line in value:
            if not isinstance(sight_line, SpectroscopicFibreOptic):
                raise TypeError("The sightlines attribute of LineOfSightGroup must be a list of "
                                "SpectroscopicSightLines. Value {} is not a SpectroscopicFibreOptic.".format(sight_line))

        # Prevent external changes being made to this list
        value = value.copy()
        for sight_line in value:
            sight_line.parent = self

        self._sight_lines = value

    def add_sight_line(self, sight_line):

        if not isinstance(sight_line, SpectroscopicFibreOptic):
            raise TypeError("The sightline argument must be of type SpectroscopicFibreOptic.")

        sight_line.parent = self
        self._sight_lines.append(sight_line)

    @property
    def acceptance_angle(self):
        """
        The angle in degrees between the z axis and the cone surface which defines the fibres
        solid angle sampling area.

        The same value can be shared between all sight lines,
        or each sight line can be assigned with individual value.

        :rtype: list
        """
        return [sight_line.acceptance_angle for sight_line in self._sight_lines]

    @acceptance_angle.setter
    def acceptance_angle(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.acceptance_angle = v
            else:
                raise ValueError("The length of 'acceptance_angle' ({}) "
                                 "mismatches the number of sight lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.acceptance_angle = value

    @property
    def radius(self):
        """
        The radius of the fibre tip in metres. This radius defines a circular area at the fibre tip
        which will be sampled over.

        The same value can be shared between all sight lines,
        or each sight line can be assigned with individual value.

        :rtype: list
        """
        return [sight_line.radius for sight_line in self._sight_lines]

    @radius.setter
    def radius(self, value):
        if isinstance(value, (list, tuple, ndarray)):
            if len(value) == len(self._sight_lines):
                for sight_line, v in zip(self._sight_lines, value):
                    sight_line.radius = v
            else:
                raise ValueError("The length of 'radius' ({}) "
                                 "mismatches the number of sight lines ({}).".format(len(value), len(self._sight_lines)))
        else:
            for sight_line in self._sight_lines:
                sight_line.radius = value


class _SpectroscopicObserver0DBase:
    """A base class for spectroscopic 0D observers."""

    @property
    def point(self):
        return self._point

    @point.setter
    def point(self, value):
        if not isinstance(value, Point3D):
            raise TypeError("Attribute 'point' must be of type Point3D.")

        if self._direction.x != 0 or self._direction.y != 0 or self._direction.z != 1:
            up = Vector3D(0, 0, 1)
        else:
            up = Vector3D(1, 0, 0)
        self._point = value
        self.transform = translate(value.x, value.y, value.z) * rotate_basis(self._direction, up)

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, value):
        if not isinstance(value, Vector3D):
            raise TypeError("Attribute 'direction' must be of type Vector3D.")

        if value.x != 0 or value.y != 0 or value.z != 1:
            up = Vector3D(0, 0, 1)
        else:
            up = Vector3D(1, 0, 0)
        self._direction = value
        self.transform = translate(self._point.x, self._point.y, self._point.z) * rotate_basis(value, up)

    @property
    def observed_spectrum(self):
        """
        Returns observed spectrum.

        :rtype: Spectrum
        """

        pipeline = self.pipelines[0]
        if not pipeline.samples:
            raise ValueError("No spectrum has been observed.")
        spectrum = Spectrum(pipeline.min_wavelength, pipeline.max_wavelength, pipeline.bins)
        spectrum.samples = pipeline.samples.mean
        return spectrum

    def plot_spectra(self, unit='J', ymax=None, extras=True):
        """
        Plot the observed spectrum for selected pipeline.

        :param str unit: Plots the spectrum in J/s/m2/str/nm (units='J'),
                         or in ph/s/m2/str/nm (units='ph').
        :param float ymax: Upper limit of y-axis.
        :param bool extras: If True, set title, axis labels and ymax.
        """

        if unit == 'J':
            # Spectrum objects are already in J/s/m2/str/nm
            spectrum = self.observed_spectrum
        elif unit == 'ph':
            # turn the samples into ph/s/m2/str/nm
            spectrum = self.observed_spectrum.new_spectrum()
            spectrum.samples = self.observed_spectrum.to_photons()
        else:
            raise ValueError("unit must be 'J' or 'ph'.")

        if spectrum.samples.size > 1:
            plt.plot(spectrum.wavelengths, spectrum.samples, label=self.name)
        else:
            plt.plot(spectrum.wavelengths, spectrum.samples, marker='o', ls='none', label=self.name)

        if extras:
            if ymax is not None:
                plt.ylim(ymax=ymax)
            plt.title(self.name)
            plt.xlabel('wavelength (nm)')
            plt.ylabel('radiance ({}/s/m^2/str/nm)'.format(unit))


class SpectroscopicSightLine(SightLine, _SpectroscopicObserver0DBase):

    """
    A simple line of sight observer.

    :param Point3D point: The observation point for this sight-line.
    :param Vector3D direction: The observation direction for this sight-line.
    :param bool accumulate: Whether to accumulate samples with subsequent calls
                            to observe() (default=False).
    :param bool display_progress: Toggles the display of live render progress (default=False).
    """

    def __init__(self, point, direction, parent=None, name="", accumulate=False, display_progress=False):

        self._point = Point3D(0, 0, 0)
        self._direction = Vector3D(1, 0, 0)
        pipelines = [SpectralRadiancePipeline0D(accumulate=accumulate, display_progress=display_progress)]

        super().__init__(pipelines=pipelines, parent=parent, name=name)

        self.point = point
        self.direction = direction

    @property
    def pipelines(self):
        """
        A list of pipelines to process the output spectra of these observations.

        :rtype: list
        """
        return super().pipelines

    @pipelines.setter
    def pipelines(self, value):
        if len(value) != 1:
            raise ValueError("This observer supports only a single pipeline.")
        if not isinstance(value[0], SpectralRadiancePipeline0D):
            raise TypeError("Processing pipeline must be a SpectralRadiancePipeline0D instance.")
        # Cannot overwrite a private attribute of cythonised parent class, so use the setter from the parent class.
        SightLine.pipelines.__set__(self, value)


class SpectroscopicFibreOptic(FibreOptic, _SpectroscopicObserver0DBase):

    """
    An optic fibre spectroscopic observer with non-zero acceptance angle.

    Rays are sampled over a circular area at the fibre tip and a conical solid angle
    defined by the acceptance_angle parameter.

    :param Point3D point: The observation point for this sight-line.
    :param Vector3D direction: The observation direction for this sight-line.
    :param float acceptance_angle: The angle in degrees between the z axis and the cone surface which defines the fibres
                                   solid angle sampling area.
    :param float radius: The radius of the fibre tip in metres. This radius defines a circular area at the fibre tip
                         which will be sampled over.
    :param bool accumulate: Whether to accumulate samples with subsequent calls
                            to observe() (default=False).
    :param bool display_progress: Toggles the display of live render progress (default=False).
    """

    def __init__(self, point, direction, acceptance_angle=None, radius=None, parent=None, name="",
                 accumulate=False, display_progress=False):

        self._point = Point3D(0, 0, 0)
        self._direction = Vector3D(1, 0, 0)
        pipelines = [SpectralRadiancePipeline0D(accumulate=accumulate, display_progress=display_progress)]

        super().__init__(pipelines=pipelines, parent=parent, name=name, acceptance_angle=acceptance_angle, radius=radius)

        self.point = point
        self.direction = direction

    @property
    def pipelines(self):
        """
        A list of pipelines to process the output spectra of these observations.

        :rtype: list
        """
        return super().pipelines

    @pipelines.setter
    def pipelines(self, value):
        if len(value) != 1:
            raise ValueError("This observer supports only a single pipeline.")
        if not isinstance(value[0], SpectralRadiancePipeline0D):
            raise TypeError("Processing pipeline must be a SpectralRadiancePipeline0D instance.")
        # Cannot overwrite a private attribute of cythonised parent class, so use the setter from the parent class.
        FibreOptic.pipelines.__set__(self, value)
