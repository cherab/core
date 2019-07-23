from raysect.core.math cimport AffineMatrix3D

from raysect.optical cimport World, Primitive, Ray, Spectrum, SpectralFunction, Point3D, Vector3D, AffineMatrix3D
from raysect.optical.material.emitter cimport InhomogeneousVolumeEmitter
from raysect.optical.material.emitter.inhomogeneous cimport VolumeIntegrator
from cherab.core.laser.model cimport LaserModel
from cherab.core.laser.node cimport Laser

from cherab.core.utility import Notifier
cdef class LaserMaterial(InhomogeneousVolumeEmitter):

    def __init__(self, Laser laser not None, Plasma plasma not None,
                 list models not None, VolumeIntegrator integrator not None):

        super().__init__(integrator)

        self._laser = laser
        self._plasma = plasma
        self._laser_to_plasma = None

        # validate
        for model in models:
            if not isinstance(model, LaserModel):
                raise TypeError('Laser model supplied to BeamMaterial is not a BeamModel.')

        # configure models
        #for model in models:
        #    model.laser = laser
        #    model.plasma = plasma

        self._models = models

        #self._plasma.notifier.add(self._change())
        #self._laser.notifier.add(self._change())

        self._change()


    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D to_local, AffineMatrix3D to_world):

        cdef:
            LaserModel laser_model
            Point3D plasma_point

        # transform points and directions
        # todo: cache this transform and rebuild if beam or plasma notifies
        plasma_point = point.transform(self._laser_to_plasma)

        # call each model and accumulate spectrum
        for model in self._models:
            spectrum = model.emission(point, plasma_point, direction, spectrum)

        return spectrum

    def _change(self):
        self._laser_to_plasma = self._laser.to(self._plasma)