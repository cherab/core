
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

from raysect.optical import Spectrum
from raysect.optical.observer import RadiancePipeline0D, PowerPipeline0D, SpectralRadiancePipeline0D, SpectralPowerPipeline0D


def select_pipelines(group, item):
    pipelines = []
    observers = []
    for observer in group._observers:
        if isinstance(item, int):
            if item < len(observer.pipelines):
                pipelines.append(observer.pipelines[item])
                observers.append(observer)
        elif isinstance(item, str):
            matching_pipelines = [pipeline for pipeline in observer.pipelines if pipeline.name == item]
            if len(matching_pipelines) == 1:
                pipelines.append(matching_pipelines[0])
                observers.append(observer)

    if len(pipelines) == 0:
        raise ValueError("Pipeline {} was not found for any observer in this {}.".format((item, group.__class__.__name__)))

    pipeline_types = set(type(pipeline) for pipeline in pipelines)
    if len(pipeline_types) > 1:
        raise ValueError("Pipelines {} have different types for different observers.".format(item))

    return pipelines, observers


def plot_group_total(group, item=0, ax=None):
    """
    Plots total (wavelength-integrated) signal for each observer in the group.
    
    :param str/int item: The index or name of the pipeline. Default: 0.
    :param Axes ax: Existing matplotlib axes.
    :return Axes ax: Matplotlib axes with plotted spectra
    """

    pipelines, observers = select_pipelines(group, item)

    if ax is None:
        _, ax = plt.subplots(constrained_layout=True)

    signal = []
    tick_labels = []
    for pipeline, sight_line in zip(pipelines, observers):
        if isinstance(pipeline, SpectralPowerPipeline0D):
            spectrum = Spectrum(pipeline.min_wavelength, pipeline.max_wavelength, pipeline.bins)
            spectrum.samples[:] = pipeline.samples.mean
            signal.append(spectrum.total())
        else:
            signal.append(pipeline.value.mean)

        if sight_line.name and len(sight_line.name):
            tick_labels.append(sight_line.name)
        else:
            tick_labels.append(group._sight_lines.index(sight_line))

    if isinstance(pipeline, (SpectralRadiancePipeline0D, RadiancePipeline0D)):
        ylabel = 'Radiance (W/m^2/str)'
    else:  # SpectralPowerPipeline0D or PowerPipeline0D
        ylabel = 'Power (W)'

    ax.bar(list(range(len(signal))), signal, tick_label=tick_labels, label=item)

    if isinstance(item, int):
        # check if pipelines share the same name
        if len(set(pipeline.name for pipeline in pipelines)) == 1 and pipelines[0].name and len(pipelines[0].name):
            ax.set_title('{}: {}'.format(group.name, pipelines[0].name))
        else:
            # pipelines have different names or name is not set
            ax.set_title('{}: pipeline {}'.format(group.name, item))
    elif isinstance(item, str):
        ax.set_title('{}: {}'.format(group.name, item))

    ax.set_ylabel(ylabel)
    ax.set_xlabel('Line of sight')

    return ax


def plot_group_spectra(group, item=0, in_photons=False, ax=None):
    """
    Plot the spectra observed by each observer in the group for a given pipeline.
    
    :param str/int item: The index or name of the pipeline. Default: 0.
    :param bool in_photons: If True, plots the spectrum in photon/s/nm instead of W/nm.
                            Default is False.
    :param Axes ax: Existing matplotlib axes.
    :return Axes ax: Matplotlib axes with plotted spectra
    """

    pipelines, observers = select_pipelines(group, item)

    if ax is None:
        _, ax = plt.subplots(constrained_layout=True)
    
    for pipeline, observer in zip(pipelines, observers):
        spectrum_observed = Spectrum(pipeline.min_wavelength, pipeline.max_wavelength, pipeline.bins)
        spectrum_observed.samples[:] = pipeline.samples.mean
        if in_photons:
            # turn the samples into photon/s
            spectrum = spectrum_observed.new_spectrum()
            spectrum.samples[:] = spectrum_observed.to_photons()
        else:
            spectrum = spectrum_observed

        if spectrum.samples.size > 1:
            ax.plot(spectrum.wavelengths, spectrum.samples, label=observer.name)
        else:
            ax.plot(spectrum.wavelengths, spectrum.samples, marker='o', ls='none', label=observer.name)
    if isinstance(pipelines[0], SpectralRadiancePipeline0D):
        ylabel = 'Spectral radiance (photon/s/m^2/str/nm)' if in_photons else 'Spectral radiance (W/m^2/str/nm)'
    else:  # SpectralPowerPipeline0D
        ylabel = 'Spectral power (photon/s/nm)' if in_photons else 'Spectral power (W/nm)'

    if isinstance(item, int):
        # check if pipelines share the same name
        if len(set(pipeline.name for pipeline in pipelines)) == 1 and pipelines[0].name and len(pipelines[0].name):
            ax.set_title('{}: {}'.format(group.name, pipelines[0].name))
        else:
            # pipelines have different names or name is not set
            ax.set_title('{}: pipeline {}'.format(group.name, item))
    elif isinstance(item, str):
        ax.set_title('{}: {}'.format(group.name, item))

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(ylabel)
    ax.legend()

    return ax