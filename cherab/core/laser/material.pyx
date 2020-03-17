from raysect.optical cimport World, Primitive, Ray, Spectrum, Point3D, Vector3D, AffineMatrix3D
from raysect.optical.material.emitter cimport InhomogeneousVolumeEmitter
from raysect.optical.material.emitter.inhomogeneous cimport VolumeIntegrator
from cherab.core.laser.node cimport Laser
from cherab.core.laser.scattering cimport ScatteringModel
from cherab.core.laser.models.model_base cimport LaserModel
from cherab.core.laser.models.laserspectrum_base cimport LaserSpectrum
from cherab.core cimport Plasma

cdef class LaserMaterial(InhomogeneousVolumeEmitter):

    def __init__(self, Laser laser not None, VolumeIntegrator integrator not None):

        super().__init__(laser._integrator)

        self.laser = laser

    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D to_local, AffineMatrix3D to_world):


        cdef:
            double ne, te, laser_power_density, laser_volumetric_power
            Point3D position_plasma
            Vector3D pointing_vector, polarization_vector
            Py_ssize_t index

        position_plasma = point.transform(self._laser_to_plasma)

        ne = self._plasma.get_electron_distribution().density(position_plasma.x, position_plasma.y, position_plasma.z)
        te = self._plasma.get_electron_distribution().effective_temperature(position_plasma.x, position_plasma.y, position_plasma.z)
        laser_volumetric_power = self._laser_model.get_power_density(point.x, point.y, point.z)

        if ne == 0 or te == 0:
            return spectrum

        for index in range(self._laser_bins):
            laser_power_density = self._laser_spectrum_power_mv[index] * laser_volumetric_power  

            if laser_power_density > 0:

                pointing_vector = self._laser_model.get_pointing(point.x, point.y, point.z)
                polarization_vector = self._laser_model.get_polarization(point.x, point.y, point.z)
                spectrum = self._scattering_model.emission(ne, te, laser_power_density, self._laser_wavelength_mv[index],
                                                           direction, pointing_vector, polarization_vector, spectrum)

        return spectrum

    @property
    def laser(self):
        return self._laser

    @laser.setter
    def laser(self, value):
        if not isinstance(value, Laser):
            raise ValueError("Value has to be of type Laser, but {0} passed".format(type(value)))
        # unregister callback
        if isinstance(self._laser, Laser):
            self._laser.notifier.remove(self._laser_changed)

        self._laser = value
        self._laser.notifier.add(self._laser_changed)
        self._laser_changed()

    @property
    def plasma(self):
        return self._plasma

    @plasma.setter
    def plasma(self, value):
        if not isinstance(value, Plasma):
            raise ValueError("Value has to be of type Plasma, but {0} passed".format(type(value)))

        self._plasma = value
        self._plasma_changed()

    @property
    def scattering_model(self):
        return self._scattering_model

    @scattering_model.setter
    def scattering_model(self, value):
        if not isinstance(value, ScatteringModel):
            raise ValueError("Value has to be of type ScatteringModel, but {0} passed".format(type(value)))

        self._scattering_model = value

    @property
    def laser_model(self):
        return self._laser_model

    @laser_model.setter
    def laser_model(self, value):
        if not isinstance(value, LaserModel):
            raise ValueError("Value has to be of type LaserModel, but {0} passed".format(type(value)))

        self._laser_model = value

    @property
    def laser_spectrum(self):
        return self._laser_spectrum

    @laser_spectrum.setter
    def laser_spectrum(self, value):
        if not isinstance(value, LaserSpectrum):
            raise ValueError("Value has to be of type LaserSpectrum, but {0} passed".format(type(value)))

        if isinstance(self._laser_spectrum, LaserSpectrum):
            self._laser_spectrum.notifier.remove(self._laser_spectrum_changed)

        self._laser_spectrum = value
        self._laser_spectrum.notifier.add(self._laser_spectrum_changed)

        self._laser_spectrum_changed()

    def _laser_spectrum_changed(self):

        self._laser_spectrum_power_mv = self._laser_spectrum._power  # power in spectral bins (PSD * delta wavelength)
        self._laser_wavelength_mv = self._laser_spectrum._wavelengths_mv
        self._laser_bins = self._laser_spectrum._bins

    def _laser_changed(self):
        if isinstance(self._laser._plasma, Plasma):
            self.plasma = self._laser._plasma
        if isinstance(self._laser._scattering_model, ScatteringModel):
            self.scattering_model = self._laser._scattering_model
        if isinstance(self._laser._laser_spectrum, LaserSpectrum):
            self.laser_spectrum = self._laser._laser_spectrum
        if isinstance(self._laser._laser_model, LaserModel):
            self.laser_model = self._laser._laser_model

    def _plasma_changed(self):
        self._laser_to_plasma = self._laser.to(self._plasma)
