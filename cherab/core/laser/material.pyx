from raysect.optical cimport World, Primitive, Ray, Spectrum, Point3D, Vector3D, AffineMatrix3D
from raysect.optical.material.emitter cimport InhomogeneousVolumeEmitter
from raysect.optical.material.emitter.inhomogeneous cimport VolumeIntegrator
from cherab.core.laser.node cimport Laser
from cherab.core.laser.scattering cimport ScatteringModel

cdef class LaserMaterial(InhomogeneousVolumeEmitter):

    def __init__(self, Laser laser not None, VolumeIntegrator integrator not None):

        super().__init__(laser._integrator)

        self._laser = laser
        self._plasma = laser._plasma
        self._laser_to_plasma = None
        self._scattering_model = laser._scattering_model

        # configure models
        #for model in models:
        #    model.laser = laser
        #    model.plasma = plasma

        self._plasma.notifier.add(self._change)
        self._laser.notifier.add(self._change)

        self._change()

    def _set_scattering_model(self, ScatteringModel value):
        self._scattering_model = value

    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D to_local, AffineMatrix3D to_world):

        cdef:
            Point3D position_plasma
            Vector3D pointing_vector, polarization_vector
            double ne, te, laser_power_density
            
        # transform points and directions
        position_plasma = point.transform(self._laser_to_plasma)
        spectrum = self._scattering_model.emission(position_plasma, point, direction, spectrum)
        return spectrum

    def _change(self):
        self._laser_to_plasma = self._laser.to(self._plasma)
