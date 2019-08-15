from raysect.core.math cimport AffineMatrix3D

from raysect.optical cimport World, Primitive, Ray, Spectrum, SpectralFunction, Point3D, Vector3D, AffineMatrix3D
from raysect.optical.material.emitter cimport InhomogeneousVolumeEmitter
from raysect.optical.material.emitter.inhomogeneous cimport VolumeIntegrator
from cherab.core.laser.node cimport Laser
from cherab.core.laser.scattering cimport ScatteringModel

from cherab.core.utility import Notifier
cdef class LaserMaterial(InhomogeneousVolumeEmitter):

    def __init__(self, Laser laser not None, VolumeIntegrator integrator not None):

        super().__init__(laser._integrator)

        self._laser = laser
        self._plasma = laser._plasma
        self._laser_to_plasma = None
        self._scattering_model = laser._scattering_model
        self._laser_model = laser._laser_model

        # configure models
        #for model in models:
        #    model.laser = laser
        #    model.plasma = plasma

        #self._plasma.notifier.add(self._change())
        #self._laser.notifier.add(self._change())

        self._change()


    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D to_local, AffineMatrix3D to_world):

        cdef:
            Point3D position_plasma
            Vector3D pointing_vector, polarization_vector
            double ne, te, laser_power_density


        # transform points and directions
        # todo: cache this transform and rebuild if beam or plasma notifies
        position_plasma = point.transform(self._laser_to_plasma)

        #get plasma and laser properties to pass to scattering model
        #vectors have to be in a single frame of reference to get correct angles in scattering calculations
        ne = self._plasma.get_electron_distribution().density(position_plasma.x, position_plasma.y, position_plasma.z)
        te = self._plasma.get_electron_distribution().effective_temperature(position_plasma.x, position_plasma.y, position_plasma.z)
        laser_power_density = self._laser_model.power_density(point.x, point.y, point.z)
        pointing_vector = self._laser_model.pointing(point.x, point.y, point.z)
        polarization_vector = self._laser_model.polarization(point.x, point.y, point.z)

        spectrum = self._scattering_model.emission(ne, te, laser_power_density, pointing_vector, polarization_vector,
                                                   direction, self._laser_model._laser_spectrum, spectrum)

        return spectrum

    def _change(self):
        self._laser_to_plasma = self._laser.to(self._plasma)
