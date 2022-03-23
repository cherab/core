
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

class SpectroscopicInstrument:
    """
    Base class for spectroscopic instruments (spectrometers, polychromators, etc.).
    This is an abstract class.

    :param str name: Instrument name.

    :ivar list pipeline_classes: The list of pipeline classes used with this instrument.
    :ivar list pipeline_kwargs: The list of dicts with keywords passed to init methods of
                                pipeline classes used with this instrument.
    :ivar float min_wavelength: Lower wavelength bound for spectral range.
    :ivar float max_wavelength: Upper wavelength bound for spectral range.
    :ivar int spectral_bins: The number of spectral samples over the wavelength range.
    """

    def __init__(self, name=''):
        self._pipeline_classes = None
        self.name = name
        self._clear_spectral_settings()

    @property
    def name(self):
        # Instrument name.
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)
        self._pipeline_kwargs = None

    @property
    def pipeline_classes(self):
        # The list of pipeline classes used with this instrument.
        if self._pipeline_classes is None:
            self._update_pipeline_classes()

        return self._pipeline_classes

    @property
    def pipeline_kwargs(self):
        # The list of dicts with keywords passed to init methods of
        # pipeline classes used with this instrument.
        if self._pipeline_kwargs is None:
            self._update_pipeline_kwargs()

        return self._pipeline_kwargs

    def create_pipelines(self):
        """ Returns a list of new pipelines created according to `pipeline_classes`
            and keyword arguments."""
        if self._pipeline_classes is None:
            self._update_pipeline_classes()
        if self._pipeline_kwargs is None:
            self._update_pipeline_kwargs()

        pipelines = []
        for PipelineClass, kwargs in zip(self._pipeline_classes, self._pipeline_kwargs):
            pipeline = PipelineClass(**kwargs)
            pipelines.append(pipeline)

        return pipelines

    @property
    def min_wavelength(self):
        # Lower wavelength bound for spectral range.
        if self._min_wavelength is None:
            self._update_spectral_settings()

        return self._min_wavelength

    @property
    def max_wavelength(self):
        # Upper wavelength bound for spectral range.
        if self._max_wavelength is None:
            self._update_spectral_settings()

        return self._max_wavelength

    @property
    def spectral_bins(self):
        # The number of spectral samples over the wavelength range.
        if self._spectral_bins is None:
            self._update_spectral_settings()

        return self._spectral_bins

    def _clear_spectral_settings(self):
        self._min_wavelength = None
        self._max_wavelength = None
        self._spectral_bins = None

    def _update_spectral_settings(self):
        raise NotImplementedError("To be defined in subclass.")

    def _update_pipeline_classes(self):
        raise NotImplementedError("To be defined in subclass.")

    def _update_pipeline_kwargs(self):
        raise NotImplementedError("To be defined in subclass.")
