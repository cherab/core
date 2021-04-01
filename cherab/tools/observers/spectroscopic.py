
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

import matplotlib.pyplot as plt
from raysect.core import Node, AffineMatrix3D, translate, rotate_basis, Point3D, Vector3D
from raysect.optical import Spectrum
from raysect.optical.observer import Observer0D, FibreOptic, SpectralRadiancePipeline0D, SightLine


class LineOfSightGroup(Node):
    """
    A group of spectroscopic sight-lines under a single scene-graph node.

    A scene-graph object regrouping a series of 'SpectroscopeLineOfSight'
    observers as a scene-graph parent. Allows combined observation and display
    control simultaneously.
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
    def sight_lines(self):
        return self._sight_lines

    @sight_lines.setter
    def sight_lines(self, value):

        if not isinstance(value, list):
            raise TypeError("The sightlines attribute of LineOfSightGroup must be a list of SpectroscopicObserver0D.")

        for sight_line in value:
            if not isinstance(sight_line, SpectroscopicObserver0D):
                raise TypeError("The sightlines attribute of LineOfSightGroup must be a list of "
                                "SpectroscopicObserver0D. Value {} is not a SpectroscopicObserver0D.".format(sight_line))

        # Prevent external changes being made to this list
        value = value.copy()
        for sight_line in value:
            sight_line.parent = self

        self._sight_lines = value

    def add_sight_line(self, sight_line):

        if not isinstance(sight_line, SpectroscopicObserver0D):
            raise TypeError("The sightline argument must be of type SpectroscopicObserver0D.")

        sight_line.parent = self
        self._sight_lines.append(sight_line)

    def observe(self):
        for sight_line in self._sight_lines:
            sight_line.observe()

    def plot_spectra(self, unit='J', ymax=None):

        for sight_line in self.sight_lines:
            sight_line.plot_spectra(unit=unit, extras=False)

        if ymax is not None:
            plt.ylim(ymax=ymax)

        plt.title(self.name)
        plt.xlabel('wavelength (nm)')
        plt.ylabel('radiance ({}/s/m^2/str/nm)'.format(unit))
        plt.legend()


class SpectroscopicObserver0D:

    """
    A 0D spectroscopic observer (line of sight, optic fibre, etc.).

    :param Observer0D observer: A 0D observer.
    :param Point3D point: The observation point for this sight-line.
    :param Vector3D direction: The observation direction for this sight-line.
    """

    def __init__(self, observer, point, direction, parent=None, name=""):

        if not isinstance(observer, Observer0D):
            raise TypeError("Argument 'observer' must be of type Observer0D.")

        if not isinstance(point, Point3D):
            raise TypeError("Argument 'point' must be of type Point3D.")

        if not isinstance(direction, Vector3D):
            raise TypeError("Argument 'direction' must be of type Vector3D.")

        self._point = Point3D(0, 0, 0)
        self._direction = Vector3D(1, 0, 0)
        self._transform = AffineMatrix3D()

        self._spectral_pipeline = SpectralRadiancePipeline0D(accumulate=False)
        # TODO - carry over wavelength range and resolution settings
        self._observer = observer

        self._observer.pipelines = [self._spectral_pipeline]
        self.parent = parent
        self.name = name
        self.point = point
        self.direction = direction

    @property
    def parent(self):
        return self._observer.parent

    @parent.setter
    def parent(self, value):
        self._observer.parent = value

    @property
    def name(self):
        return self._observer.name

    @name.setter
    def name(self, value):
        self._observer.name = value

    @property
    def point(self):
        return self._point

    @point.setter
    def point(self, value):
        if not (self._direction.x == 0 and self._direction.y == 0 and self._direction.z == 1):
            up = Vector3D(0, 0, 1)
        else:
            up = Vector3D(1, 0, 0)
        self._point = value
        self._observer.transform = translate(value.x, value.y, value.z) * rotate_basis(self._direction, up)

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, value):
        if value.x != 0 and value.y != 0 and value.z != 1:
            up = Vector3D(0, 0, 1)
        else:
            up = Vector3D(1, 0, 0)
        self._direction = value
        self._observer.transform = translate(self._point.x, self._point.y, self._point.z) * rotate_basis(value, up)

    @property
    def min_wavelength(self):
        return self._observer.min_wavelength

    @min_wavelength.setter
    def min_wavelength(self, value):
        self._observer.min_wavelength = value

    @property
    def max_wavelength(self):
        return self._observer.max_wavelength

    @max_wavelength.setter
    def max_wavelength(self, value):
        self._observer.max_wavelength = value

    @property
    def spectral_bins(self):
        return self._observer.spectral_bins

    @spectral_bins.setter
    def spectral_bins(self, value):
        self._observer.spectral_bins = value

    @property
    def observed_spectrum(self):
        # TODO - throw exception if no observed spectrum
        pipeline = self._spectral_pipeline
        spectrum = Spectrum(pipeline.min_wavelength, pipeline.max_wavelength, pipeline.bins)
        spectrum.samples = pipeline.samples.mean
        return spectrum

    def observe(self):
        """
        Ask this sight-line to observe its world.
        """

        self._observer.observe()

    def plot_spectra(self, unit='J', ymax=None, extras=True):
        """
        Plot the observed spectrum.
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

        plt.plot(spectrum.wavelengths, spectrum.samples)

        if extras:
            if ymax is not None:
                plt.ylim(ymax=ymax)
            plt.title(self.name)
            plt.xlabel('wavelength (nm)')
            plt.ylabel('radiance ({}/s/m^2/str/nm)'.format(unit))


class SpectroscopicSightLine(SpectroscopicObserver0D):

    """
    A simple line of sight observer.

    Fires a single ray oriented along the observer's z axis in world space.

    :param Point3D point: The observation point for this sight-line.
    :param Vector3D direction: The observation direction for this sight-line.
    """

    def __init__(self, point, direction, parent=None, name=""):

        observer = SightLine()
        super().__init__(observer, point, direction, parent=parent, name=name)


class SpectroscopicFibreOptic(SpectroscopicObserver0D):

    """
    An optic fibre observer with non-zero acceptance angle.

    Rays are sampled over a circular area at the fibre tip and a conical solid angle
    defined by the acceptance_angle parameter.

    :param Point3D point: The observation point for this sight-line.
    :param Vector3D direction: The observation direction for this sight-line.
    :param float acceptance_angle: The angle in degrees between the z axis and the cone surface which defines the fibres
                                   solid angle sampling area.
    :param float radius: The radius of the fibre tip in metres. This radius defines a circular area at the fibre tip
                         which will be sampled over.
    """

    def __init__(self, point, direction, acceptance_angle=None, radius=None, parent=None, name=""):

        observer = FibreOptic(acceptance_angle=acceptance_angle, radius=radius)
        super().__init__(observer, point, direction, parent=parent, name=name)

    @property
    def acceptance_angle(self):
        return self._observer.acceptance_angle

    @acceptance_angle.setter
    def acceptance_angle(self, value):
        self._observer.acceptance_angle = value

    @property
    def radius(self):
        return self._observer.radius

    @radius.setter
    def radius(self, value):
        self._observer.radius = value
