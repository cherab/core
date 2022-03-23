
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

import numpy as np
from raysect.optical import Spectrum
from raysect.optical.observer import SpectralRadiancePipeline0D

from .instrument import SpectroscopicInstrument


class Spectrometer(SpectroscopicInstrument):
    """
    Spectrometer that can accommodate multiple spectra.

    Spectrometer is initialized with a sequence of calibration arrays (one array per accommodated
    spectrum) containing the wavelengths of the pixel borders. Namely, the values
    :math:`w_{k}^{i}` and :math:`w_{k}^{i+1}` define the spectral range of the pixel :math:`p_i`
    of the `k`-th spectrum. After the spectrum is ray-traced, it can be recalibrated with
    `spectrometer.calibrate(spectrum)`.

    Note that Raysect cannot raytrace the spectra with non-constant spectral resolution.
    Thus, the actual number of spectral bins of raytraced spectrum is defined with
    `min_bins_per_pixel` attribute.

    :param tuple wavelength_to_pixel: Wavelength-to-pixel calibration arrays.
    :param int min_bins_per_pixel: Minimal number of spectral bins
                                   per pixel. Default is 1.
    :param str name: Spectrometer name.

    :ivar tuple wavelengths: Central wavelengths of the pixels.

    .. code-block:: pycon

       >>> from raysect.optical import World, Spectrum
       >>> from raysect.optical.observer import FibreOptic
       >>> from cherab.tools.spectroscopy import Spectrometer
       >>> from matplotlib import pyplot as plt
       >>>
       >>> wavelength_to_pixel = ([400., 400.5, 401.5, 402., 404.],
       >>>                        [600., 600.5, 601.5, 602., 604., 607.])
       >>> spectrometer = Spectrometer(wavelength_to_pixel, min_bins_per_pixel=5,
       >>>                             name='MySpectrometer')
       >>>
       >>> world = World()
       >>> fibreoptic = FibreOptic(name="MyFibreOptic", parent=world)
       >>> fibreoptic.min_wavelength = spectrometer.min_wavelength
       >>> fibreoptic.max_wavelength = spectrometer.max_wavelength
       >>> fibreoptic.spectral_bins = spectrometer.spectral_bins
       >>> fibreoptic.pipelines = spectrometer.create_pipelines()
       >>> ...
       >>> fibreoptic.observe()
       >>> spectrum = Spectrum(fibreoptic.min_wavelength, fibreoptic.max_wavelength, fibreoptic.spectral_bins)
       >>> spectrum.samples[:] = fibreoptic.pipelines[0].mean
       >>> calibrated_spectra = spectrometer.calibrate(spectrum)
       >>> wavelengths = spectrometer.wavelengths
       >>>
       >>> plt.plot(wavelengths[0], calibrated_spectra[0])
       >>> plt.show()
    """

    def __init__(self, wavelength_to_pixel, min_bins_per_pixel=1, name=''):

        self.min_bins_per_pixel = min_bins_per_pixel
        self.wavelength_to_pixel = wavelength_to_pixel
        super().__init__(name)

    @property
    def wavelength_to_pixel(self):
        # Wavelength-to-pixel calibration arrays.
        return self._wavelength_to_pixel

    @wavelength_to_pixel.setter
    def wavelength_to_pixel(self, value):
        _wavelength_to_pixel = []
        _wavelengths = []
        for wl2pix in value:
            wl2pix = np.array(wl2pix, dtype=float)
            if wl2pix.ndim != 1:
                raise ValueError('Attribute wavelength_to_pixel must only contain one-dimensional arrays.')
            if wl2pix.size < 2:
                raise ValueError('Attribute wavelength_to_pixel must only contain arrays of at least 2 elements.')
            if np.any(np.diff(wl2pix) <= 0):
                raise ValueError('Attribute wavelength_to_pixel must only contain monotonically increasing arrays.')
            wl2pix.flags.writeable = False
            _wavelength_to_pixel.append(wl2pix)
            wl_center = 0.5 * (wl2pix[1:] + wl2pix[:-1])
            wl_center.flags.writeable = False
            _wavelengths.append(wl_center)
        self._wavelength_to_pixel = tuple(_wavelength_to_pixel)
        self._wavelengths = tuple(_wavelengths)
        self._clear_spectral_settings()

    @property
    def wavelengths(self):
        # Central wavelengths of the pixels.
        return self._wavelengths

    @property
    def min_bins_per_pixel(self):
        # Minimal number of spectral bins per pixel.
        return self._min_bins_per_pixel

    @min_bins_per_pixel.setter
    def min_bins_per_pixel(self, value):
        value = int(value)
        if value <= 0:
            raise ValueError("Attribute 'min_bins_per_pixel' must be positive.")

        self._min_bins_per_pixel = value
        self._clear_spectral_settings()

    def _update_pipeline_classes(self):
        self._pipeline_classes = [SpectralRadiancePipeline0D]

    def _update_pipeline_kwargs(self):
        self._pipeline_kwargs = [{'name': self._name}]

    def _update_spectral_settings(self):
        self._min_wavelength = min(wl2pix[0] for wl2pix in self._wavelength_to_pixel)
        self._max_wavelength = max(wl2pix[-1] for wl2pix in self._wavelength_to_pixel)
        step = min(np.diff(wl2pix).min() for wl2pix in self._wavelength_to_pixel) / self._min_bins_per_pixel
        self._spectral_bins = int(np.ceil((self._max_wavelength - self._min_wavelength) / step))

    def calibrate(self, spectrum):
        """
        Calibrates the spectrum according to the `wavelength_to_pixel` arrays
        by averaging it over the pixel widths.

        :param Spectrum spectrum: Spectrum to calibrate.

        :returns: A tuple of calibrated spectra as ndarrays.
        """
        if not isinstance(spectrum, Spectrum):
            raise TypeError('Argument spectrum must be a Spectrum instance.')
        if spectrum.min_wavelength > self.min_wavelength or spectrum.max_wavelength < self.max_wavelength:
            raise ValueError('Unable to calibrate the spectrum. '
                             'The spectrum has narrower range ({}, {}) than the spectrometer ({}, {}).'.format(spectrum.min_wavelength,
                                                                                                               spectrum.max_wavelength,
                                                                                                               self.min_wavelength,
                                                                                                               self.max_wavelength))
        calibrated_spectra = []
        for wl2pix in self.wavelength_to_pixel:
            calibrated_spectrum = np.zeros(wl2pix.size - 1)
            for i in range(wl2pix.size - 1):
                calibrated_spectrum[i] = spectrum.integrate(wl2pix[i], wl2pix[i + 1]) / (wl2pix[i + 1] - wl2pix[i])
            calibrated_spectra.append(calibrated_spectrum)

        return calibrated_spectra


class CzernyTurnerSpectrometer(Spectrometer):
    """
    Czerny-Turner spectrometer.

    The Czerny-Turner spectrometer is initialized with the parameters of the diffraction scheme
    and a sequence of accommodated spectra, each of which is determined by the lower wavelength
    bound and the number of pixels.

    This spectrometer automatically fills the wavelength-to-pixel calibration arrays
    according to the parameters of the diffraction scheme.

    :param int diffraction_order: Diffraction order.
    :param float grating: Diffraction grating in nm-1.
    :param float focal_length: Focal length in nm.
    :param float pixel_spacing: Pixel to pixel spacing on CCD in nm.
    :param float diffraction_angle: Angle between incident and diffracted light in degrees.
    :param tuple accommodated_spectra: A sequence of (`min_wavelength`, `pixels`) pairs, specifying
                                       the lower wavelength bound and the number of pixels
                                       of accommodated spectra.
    :param int min_bins_per_pixel: Minimal number of spectral bins
                                   per pixel. Default is 1.
    :param str name: Spectrometer name.

    :ivar tuple wavelength_to_pixel: Wavelength-to-pixel calibration arrays.

    .. code-block:: pycon

       >>> from raysect.optical import World
       >>> from raysect.optical.observer import FibreOptic
       >>> from cherab.tools.spectroscopy import CzernyTurnerSpectrometer
       >>>
       >>> world = World()
       >>> hires_spectrometer = CzernyTurnerSpectrometer(1, 2.e-3, 1.e9, 2.e4, 10.,
       >>>                                               ((600., 512), (700., 128)),
       >>>                                               name='MySpectrometer')
       >>> fibreoptic = FibreOptic(name="MyFibreOptic", parent=world)
       >>> fibreoptic.min_wavelength = hires_spectrometer.min_wavelength
       >>> fibreoptic.max_wavelength = hires_spectrometer.max_wavelength
       >>> fibreoptic.spectral_bins = hires_spectrometer.spectral_bins
       >>> fibreoptic.pipelines = hires_spectrometer.create_pipelines()
    """

    def __init__(self, diffraction_order, grating, focal_length, pixel_spacing, diffraction_angle,
                 accommodated_spectra, min_bins_per_pixel=1, name=''):
        self._accommodated_spectra = None
        self.diffraction_order = diffraction_order
        self.grating = grating
        self.focal_length = focal_length
        self.pixel_spacing = pixel_spacing
        self.diffraction_angle = diffraction_angle
        self.accommodated_spectra = accommodated_spectra
        self.min_bins_per_pixel = min_bins_per_pixel
        self.name = name

    @property
    def diffraction_order(self):
        # Diffraction order.
        return self._diffraction_order

    @diffraction_order.setter
    def diffraction_order(self, value):
        value = int(value)
        if value <= 0:
            raise ValueError("Attribute 'diffraction_order' must be positive.")

        self._diffraction_order = value
        # resolution has changed, recalculating wavelength_to_pixel
        self._update_wavelength_to_pixel()

    @property
    def grating(self):
        # Diffraction grating in nm-1.
        return self._grating

    @grating.setter
    def grating(self, value):
        if value <= 0:
            raise ValueError("Attribute 'grating' must be positive.")

        self._grating = value
        # resolution has changed, recalculating wavelength_to_pixel
        self._update_wavelength_to_pixel()

    @property
    def focal_length(self):
        # Focal length in nm.
        return self._focal_length

    @focal_length.setter
    def focal_length(self, value):
        if value <= 0:
            raise ValueError("Attribute 'focal_length' must be positive.")

        self._focal_length = value
        # resolution has changed, recalculating wavelength_to_pixel
        self._update_wavelength_to_pixel()

    @property
    def pixel_spacing(self):
        # Pixel to pixel spacing on CCD in nm.
        return self._pixel_spacing

    @pixel_spacing.setter
    def pixel_spacing(self, value):
        if value <= 0:
            raise ValueError("Attribute 'pixel_spacing' must be positive.")

        self._pixel_spacing = value
        # resolution has changed, recalculating wavelength_to_pixel
        self._update_wavelength_to_pixel()

    @property
    def diffraction_angle(self):
        # Angle between incident and diffracted light in degrees.
        return np.rad2deg(self._diffraction_angle)

    @diffraction_angle.setter
    def diffraction_angle(self, value):
        if value <= 0:
            raise ValueError("Attribute 'diffraction_angle' must be positive.")

        self._diffraction_angle = np.deg2rad(value)
        # resolution has changed, recalculating wavelength_to_pixel
        self._update_wavelength_to_pixel()

    @property
    def accommodated_spectra(self):
        return self._accommodated_spectra

    @accommodated_spectra.setter
    def accommodated_spectra(self, value):
        for min_wavelength, pixels in value:
            if min_wavelength <= 0:
                raise ValueError('The value of min_wavelength in accommodated_spectra must be positive.')
            if pixels <= 0:
                raise ValueError('The value of pixels in accommodated_spectra must be positive.')
        self._accommodated_spectra = value
        self._update_wavelength_to_pixel()

    def _update_wavelength_to_pixel(self):

        if self._accommodated_spectra is None:
            return

        _wavelength_to_pixel = []
        _wavelengths = []
        for min_wavelength, pixels in self._accommodated_spectra:
            pixels = int(pixels)
            wl2pix = np.zeros(pixels + 1)
            wl2pix[0] = min_wavelength
            for i in range(1, pixels + 1):
                wl2pix[i] = wl2pix[i - 1] + self.resolution(wl2pix[i - 1])
            wl2pix.flags.writeable = False
            _wavelength_to_pixel.append(wl2pix)
            wl_center = 0.5 * (wl2pix[1:] + wl2pix[:-1])
            wl_center.flags.writeable = False
            _wavelengths.append(wl_center)
        self._wavelength_to_pixel = tuple(_wavelength_to_pixel)
        self._wavelengths = tuple(_wavelengths)

        self._clear_spectral_settings()

    @property
    def wavelength_to_pixel(self):
        # Wavelength-to-pixel calibration arrays.
        return self._wavelength_to_pixel

    def resolution(self, wavelength):
        """
        Calculates spectral resolution in nm for a given wavelength.

        :param wavelength: Wavelength in nm.

        :returns: Resolution in nm.
        """
        grating = self._grating
        m = self._diffraction_order
        dxdp = self._pixel_spacing
        angle = self._diffraction_angle
        fl = self._focal_length

        p = 0.5 * m * grating * wavelength
        _resolution = dxdp * (np.sqrt(np.cos(angle)**2 - p * p) - p * np.tan(angle)) / (m * fl * grating)

        return _resolution
