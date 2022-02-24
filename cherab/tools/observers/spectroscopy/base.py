
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
from raysect.core import translate, rotate_basis, Point3D, Vector3D
from raysect.optical import Spectrum
from raysect.optical.observer import SpectralRadiancePipeline0D, SpectralPowerPipeline0D, RadiancePipeline0D, PowerPipeline0D


class _SpectroscopicObserver0DBase:
    """
    .. deprecated:: 1.4.0
       Use Raysect's observer classes instead.
    
    A base class for spectroscopic 0D observers.

    The observer allows to control some of the pipeline properties
    without accessing the pipelines. It has a built-in plotting method.

    Multiple spectroscopic 0D observers can be combined into a group.

    :ivar Point3D origin: The origin point of the sight line.
    :ivar Vector3D direction: The observation direction of the sight line.
    :ivar bool display_progress: Toggles the display of live render progress.
    :ivar bool accumulate: Toggles whether to accumulate samples with subsequent
                           calls to observe().

    """

    @property
    def origin(self):
        # The origin point of the sight line.
        return Point3D(0, 0, 0).transform(self.transform)

    @origin.setter
    def origin(self, value):
        if not isinstance(value, Point3D):
            raise TypeError("Attribute 'origin' must be of type Point3D.")
        
        direction = self.direction
        if direction.x != 0 or direction.y != 0 or direction.z != 1:
            up = Vector3D(0, 0, 1)
        else:
            up = Vector3D(1, 0, 0)
        self.transform = translate(value.x, value.y, value.z) * rotate_basis(direction, up)

    @property
    def direction(self):
        # The observation direction of the sight line.
        return Vector3D(0, 0, 1).transform(self.transform)

    @direction.setter
    def direction(self, value):
        if not isinstance(value, Vector3D):
            raise TypeError("Attribute 'direction' must be of type Vector3D.")

        if value.x != 0 or value.y != 0 or value.z != 1:
            up = Vector3D(0, 0, 1)
        else:
            up = Vector3D(1, 0, 0)
        origin = self.origin
        self.transform = translate(origin.x, origin.y, origin.z) * rotate_basis(value, up)

    @property
    def display_progress(self):
        # Toggles the display of live render progress.
        display_progress_list = []
        for pipeline in self.pipelines:
            if isinstance(pipeline, SpectralPowerPipeline0D):
                display_progress_list.append(pipeline.display_progress)
            else:
                display_progress_list.append(None)
        return display_progress_list

    @display_progress.setter
    def display_progress(self, value):
        for pipeline in self.pipelines:
            if isinstance(pipeline, SpectralPowerPipeline0D):
                pipeline.display_progress = value

    @property
    def accumulate(self):
        # Toggles whether to accumulate samples with subsequent calls to observe().
        accumulate_list = []
        for pipeline in self.pipelines:
            if isinstance(pipeline, (PowerPipeline0D, SpectralPowerPipeline0D)):
                accumulate_list.append(pipeline.accumulate)
            else:
                accumulate_list.append(None)
        return accumulate_list

    @accumulate.setter
    def accumulate(self, value):
        for pipeline in self.pipelines:
            if isinstance(pipeline, (PowerPipeline0D, SpectralPowerPipeline0D)):
                pipeline.accumulate = value

    def get_pipeline(self, item=0):
        """
        Gets a pipeline by its name or index.

        :param str/int item: The name of the pipeline or its index in the list.

        :rtype: Pipeline0D
        """
        if isinstance(item, int):
            try:
                return self.pipelines[item]
            except IndexError:
                raise IndexError("Pipeline number {} not available in this {} "
                                 "with only {} pipelines.".format(item, self.__class__.__name__, len(self.pipelines)))
        elif isinstance(item, str):
            pipelines = [pipeline for pipeline in self.pipelines if pipeline.name == item]
            if len(pipelines) == 1:
                return pipelines[0]

            if len(pipelines) == 0:
                raise ValueError("Pipeline '{}' was not found in this {}.".format(item, self.__class__.__name__))

            raise ValueError("Found {} pipelines with name {} in this {}.".format(len(pipelines), item, self.__class__.__name__))
        else:
            raise TypeError("{} key must be of type int or str.".format(self.__class__.__name__))

    def connect_pipelines(self, properties=[(SpectralRadiancePipeline0D, None, None)]):
        """
        Connects pipelines of given kinds and names to this sight line.
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

        pipelines = []
        for PipelineClass, name, filter_func in properties:
            if PipelineClass in (SpectralRadiancePipeline0D, SpectralPowerPipeline0D):
                pipelines.append(PipelineClass(accumulate=False, display_progress=False, name=name))
            elif PipelineClass in (RadiancePipeline0D, PowerPipeline0D):
                pipelines.append(PipelineClass(filter=filter_func, accumulate=False, name=name))
            else:
                raise ValueError("Unsupported pipeline class: {}. "
                                 "Only the following pipeline types are supported: "
                                 "SpectralRadiancePipeline0D, SpectralPowerPipeline0D, "
                                 "RadiancePipeline0D, PowerPipeline0D.".format(PipelineClass.__name__))
        self.pipelines = pipelines

    def plot_spectrum(self, item=0, in_photons=False, ax=None, extras=True):
        """
        Plot the observed spectrum for a given spectral pipeline.

        :param str/int item: The index or name of the pipeline. Default: 0.
        :param bool in_photons: If True, plots the spectrum in photon/s/nm instead of W/nm.
                                Default is False.
        :param Axes ax: Existing matplotlib axes.
        :param bool extras: If True, set title and axis labels.

        :rtype: matplotlib.pyplot.axes
        """

        pipeline = self.get_pipeline(item)
        if not isinstance(pipeline, SpectralPowerPipeline0D):
            raise TypeError('Pipeline {} is not a spectral pipeline. '
                            'The plot_spectrum() method works only with spectral pipelines.'.format(item))

        spectrum_observed = Spectrum(pipeline.min_wavelength, pipeline.max_wavelength, pipeline.bins)
        spectrum_observed.samples[:] = pipeline.samples.mean
        if in_photons:
            # turn the samples into photon/s
            spectrum = spectrum_observed.new_spectrum()
            spectrum.samples[:] = spectrum_observed.to_photons()
            unit = 'photon/s'
        else:
            spectrum = spectrum_observed
            unit = 'W'

        if ax is None:
            _, ax = plt.subplots(constrained_layout=True)

        if spectrum.samples.size > 1:
            ax.plot(spectrum.wavelengths, spectrum.samples, label=self.name)
        else:
            ax.plot(spectrum.wavelengths, spectrum.samples, marker='o', ls='none', label=self.name)

        if extras:
            if isinstance(pipeline, SpectralRadiancePipeline0D):
                ylabel = 'Spectral radiance ({}/m^2/str/nm)'.format(unit)
            else:  # SpectralPowerPipeline0D
                ylabel = 'Spectral power ({}/nm)'.format(unit)

            if isinstance(item, int):
                if pipeline.name and len(pipeline.name):
                    ax.set_title('{}: {}'.format(self.name, pipeline.name))
                else:
                    # pipelines have different names or name is not set
                    ax.set_title('{}: pipeline {}'.format(self.name, item))
            elif isinstance(item, str):
                ax.set_title('{}: {}'.format(self.name, item))

            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel(ylabel)

        return ax
