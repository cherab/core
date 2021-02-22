from raysect.core.scenegraph._nodebase cimport _NodeBase
from raysect.optical cimport World, Primitive, Ray, Spectrum, Point3D, Vector3D, AffineMatrix3D
from raysect.optical.material.emitter cimport InhomogeneousVolumeEmitter
from raysect.optical.material.emitter.inhomogeneous cimport VolumeIntegrator

from cherab.core.laser.node cimport Laser
from cherab.core.laser.scattering cimport LaserEmissionModel

cdef class LaserMaterial(InhomogeneousVolumeEmitter):

    def __init__(self, Laser laser not None, _NodeBase laser_segment not None, list models, VolumeIntegrator integrator not None):

        super().__init__(laser._integrator)

        self._laser_segment_to_laser_node = laser_segment.to(laser)
        self._laser_to_plasma = laser_segment.to(laser._plasma)
        self.importance = laser.importance
        
        #validate and set models
        for model in models:
            if not isinstance(model, LaserEmissionModel):
                raise TypeError("Model supplied to laser are not LaserMaterial is not LaserEmissionModel")
            model.plasma = laser.plasma
            model.laser_profle = laser.laser_profle
            model.laser_spectrum = laser.laser_spectrum

        self._models = models


    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D to_local, AffineMatrix3D to_world):

        cdef:
            Point3D point_plasma, point_laser
            Vector3D direction_plasma, direction_laser
            LaserEmissionModel model

        point_laser = point.transform(self._laser_segment_to_laser_node)
        direction_laser = direction.transform(self._laser_segment_to_laser_node) # observation vector in the laser frame
        point_plasma = point.transform(self._laser_to_plasma)
        direction_plasma = direction.transform(self._laser_to_plasma)

        for model in self._models:
            spectrum = model.emission(point_plasma, direction_plasma, point_laser, direction_laser, spectrum)

        return spectrum        
