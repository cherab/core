
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

import numpy as np

from raysect.core import Node, translate, rotate_basis, Point3D, Vector3D, Ray as CoreRay, Primitive, World
from raysect.core.math.sampler import TargettedHemisphereSampler, RectangleSampler3D
from raysect.primitive import Box, Cylinder, Subtract
from raysect.optical import ConstantSF
from raysect.optical.observer import PowerPipeline0D, RadiancePipeline0D, \
    SpectralPowerPipeline0D, SpectralRadiancePipeline0D, SightLine, TargettedPixel
from raysect.optical.material.material import NullMaterial
from raysect.optical.material import AbsorbingSurface, UniformVolumeEmitter

from cherab.tools.inversions.voxels import VoxelCollection


R_2_PI = 1 / (2 * np.pi)


class BolometerCamera(Node):
    """
    A group of bolometer sight-lines under a single scene-graph node.

    A scene-graph object regrouping a series of 'BolometerFoil'
    observers as a scene-graph parent. Allows combined observation and display
    control simultaneously.
    """

    def __init__(self, camera_geometry=None, parent=None, transform=None, name=''):
        super().__init__(parent=parent, transform=transform, name=name)

        self._foil_detectors = []
        self._slits = []

        if camera_geometry is not None:
            if not isinstance(camera_geometry, Primitive):
                raise TypeError("camera_geometry must be a primitive")
            camera_geometry.parent = self
        self._camera_geometry = camera_geometry

    def __len__(self):
        return len(self._foil_detectors)

    def __iter__(self):
        for detector in self._foil_detectors:
            yield detector

    def __getitem__(self, item):

        if isinstance(item, int):
            try:
                return self._foil_detectors[item]
            except IndexError:
                raise IndexError("BolometerFoil number {} not available in this BolometerCamera.".format(item))
        elif isinstance(item, str):
            for detector in self._foil_detectors:
                if detector.name == item:
                    return detector
            else:
                raise ValueError("BolometerFoil '{}' was not found in this BolometerCamera.".format(item))
        else:
            raise TypeError("BolometerCamera key must be of type int or str.")

    @property
    def slits(self):
        return self._slits

    @property
    def foil_detectors(self):
        return self._foil_detectors

    @foil_detectors.setter
    def foil_detectors(self, value):

        if not isinstance(value, list):
            raise TypeError("The foil_detectors attribute of LineOfSightGroup must be a list of BolometerFoils.")

        # Prevent external changes being made to this list
        value = value.copy()
        for foil_detector in value:
            if not isinstance(foil_detector, BolometerFoil):
                raise TypeError("The foil_detectors attribute of BolometerCamera must be a list of "
                                "BolometerFoil objects. Value {} is not a BolometerFoil.".format(foil_detector))
            if not foil_detector.slit in self._slits:
                self._slits.append(foil_detector.slit)
            foil_detector.parent = self

        self._foil_detectors = value

    def add_foil_detector(self, foil_detector):

        if not isinstance(foil_detector, BolometerFoil):
            raise TypeError("The foil_detector argument must be of type BolometerFoil.")

        if not foil_detector.slit in self._slits:
            self._slits.append(foil_detector.slit)

        foil_detector.parent = self
        self._foil_detectors.append(foil_detector)

    def observe(self):

        observations = []
        for foil_detector in self._foil_detectors:
            foil_detector.observe()
            observations.append(foil_detector.pipelines[0].value.mean)

        return observations


class BolometerSlit(Node):

    def __init__(self, slit_id, centre_point, basis_x, dx, basis_y, dy, dz=0.001, parent=None, csg_aperture=False):

        self._centre_point = centre_point
        self._basis_x = basis_x.normalise()
        self.dx = dx
        self._basis_y = basis_y.normalise()
        self.dy = dy
        self.dz = dz

        # NOTE - target primitive and aperture surface cannot be co-incident otherwise numerics will cause Raysect
        # to be blind to one of the two surfaces.
        slit_normal = basis_x.cross(basis_y)
        transform = translate(centre_point.x, centre_point.y, centre_point.z) * rotate_basis(slit_normal, basis_y)

        super().__init__(parent=parent, transform=transform, name=slit_id)

        self.target = Box(lower=Point3D(-dx/2*1.01, -dy/2*1.01, -dz/2), upper=Point3D(dx/2*1.01, dy/2*1.01, dz/2),
                          transform=None, material=NullMaterial(), parent=self, name=slit_id+' - target')

        self._csg_aperture = None
        self.csg_aperture = csg_aperture

    @property
    def centre_point(self):
        return Point3D(0, 0, 0).transform(self.to_root())

    @property
    def normal_vector(self):
        return Vector3D(0, 0, 1).transform(self.to_root())

    @property
    def basis_x(self):
        return Vector3D(1, 0, 0).transform(self.to_root())

    @property
    def basis_y(self):
        return Vector3D(0, 1, 0).transform(self.to_root())

    @property
    def csg_aperture(self):
        return self._csg_aperture

    @csg_aperture.setter
    def csg_aperture(self, value):

        if value is True:
            width = max(self.dx, self.dy)
            face = Box(Point3D(-width, -width, -self.dz/2), Point3D(width, width, self.dz/2))
            slit = Box(lower=Point3D(-self.dx/2, -self.dy/2, -self.dz/2 - self.dz*0.1),
                       upper=Point3D(self.dx/2, self.dy/2, self.dz/2 + self.dz*0.1))
            self._csg_aperture = Subtract(face, slit, parent=self,
                                          material=AbsorbingSurface(), name=self.name+' - CSG Aperture')

        else:
            if isinstance(self._csg_aperture, Primitive):
                self._csg_aperture.parent = None
            self._csg_aperture = None


class BolometerFoil(TargettedPixel):
    """
    A rectangular bolometer detector.

    Can be configured to sample a single ray or fan of rays oriented along the
    observer's z axis in world space.
    """

    def __init__(self, detector_id, centre_point, basis_x, dx, basis_y, dy, slit,
                 parent=None, units="Power", accumulate=False):

        # perform validation of input parameters

        if not isinstance(dx, (float, int)):
            raise TypeError("dx argument for BolometerFoil must be of type float/int.")
        if not dx > 0:
            raise ValueError("dx argument for BolometerFoil must be greater than zero.")

        if not isinstance(dy, (float, int)):
            raise TypeError("dy argument for BolometerFoil must be of type float/int.")
        if not dy > 0:
            raise ValueError("dy argument for BolometerFoil must be greater than zero.")

        if not isinstance(slit, BolometerSlit):
            raise TypeError("slit argument for BolometerFoil must be of type BolometerSlit.")
        self._slit = slit

        if not isinstance(centre_point, Point3D):
            raise TypeError("centre_point argument for BolometerFoil must be of type Point3D.")
        self._centre_point = centre_point

        if not isinstance(basis_x, Vector3D):
            raise TypeError("The basis vectors of BolometerFoil must be of type Vector3D.")
        if not isinstance(basis_y, Vector3D):
            raise TypeError("The basis vectors of BolometerFoil must be of type Vector3D.")

        # set basis vectors
        self._basis_x = basis_x.normalise()
        self._basis_y = basis_y.normalise()
        self._normal_vec = self._basis_x.cross(self._basis_y)
        self._foil_to_slit_vec = self._centre_point.vector_to(self._slit.centre_point).normalise()
        self.units = units

        # setup root bolometer foil transform
        translation = translate(self._centre_point.x, self._centre_point.y, self._centre_point.z)
        rotation = rotate_basis(self._normal_vec, self._basis_y)

        if self.units == "Power":
            pipeline = PowerPipeline0D(accumulate=accumulate)
        elif self.units == "Radiance":
            pipeline = RadiancePipeline0D(accumulate=accumulate)
        else:
            raise ValueError("The units argument of BolometerFoil must be one of 'Power' or 'Radiance'.")

        super().__init__([slit.target], targetted_path_prob=1.0,
                         pipelines=[pipeline],
                         pixel_samples=1000, x_width=dx, y_width=dy, spectral_bins=1, quiet=True,
                         parent=parent, transform=translation * rotation, name=detector_id)

    def __repr__(self):
        """Returns a string representation of this BolometerFoil object."""
        return "<BolometerFoil - " + self.name + ">"

    @property
    def centre_point(self):
        return Point3D(0, 0, 0).transform(self.to_root())

    @property
    def normal_vector(self):
        return Vector3D(0, 0, 1).transform(self.to_root())

    @property
    def basis_x(self):
        return Vector3D(1, 0, 0).transform(self.to_root())

    @property
    def basis_y(self):
        return Vector3D(0, 1, 0).transform(self.to_root())

    @property
    def sightline_vector(self):
        return self.centre_point.vector_to(self._slit.centre_point)

    @property
    def slit(self):
        return self._slit

    def as_sightline(self):

        if self.units == "Power":
            pipeline = PowerPipeline0D(accumulate=False)
        elif self.units == "Radiance":
            pipeline = RadiancePipeline0D(accumulate=False)
        else:
            raise ValueError("The units argument of BolometerFoil must be one of 'Power' or 'Radiance'.")

        los_observer = SightLine(pipelines=[pipeline], pixel_samples=1, quiet=True,
                                 parent=self, name=self.name, transform=self.transform)
        los_observer.render_engine = self.render_engine
        los_observer.spectral_bins = self.spectral_bins
        los_observer.min_wavelength = self.min_wavelength
        los_observer.max_wavelength = self.max_wavelength

        return los_observer

    def trace_sightline(self):

        if not isinstance(self.root, World):
            raise ValueError("This BolometerFoil is not connected to a valid World scenegraph object.")

        centre_point = self.centre_point

        while True:

            # Find the next intersection point of the ray with the world
            intersection = self.root.hit(CoreRay(centre_point, self.sightline_vector))

            if intersection is None:
                raise RuntimeError("No material intersection was found for this sightline.")

            elif isinstance(intersection.primitive.material, NullMaterial):
                # apply a small displacement to avoid infinite self collisions due to numerics
                ray_displacement = min(self.x_width, self.y_width) / 100
                centre_point += self.sightline_vector * ray_displacement
                continue

            else:
                hit_point = intersection.hit_point.transform(intersection.primitive_to_world)
                return self.centre_point, hit_point, intersection.primitive

    def calculate_sensitivity(self, voxel_collection, ray_count=10000):

        if not isinstance(voxel_collection, VoxelCollection):
            raise TypeError("voxel_collection must be of type VoxelCollection")

        if self.units == "Power":
            pipeline = SpectralPowerPipeline0D(display_progress=False)
        elif self.units == "Radiance":
            pipeline = SpectralRadiancePipeline0D(display_progress=False)
        else:
            raise ValueError("Sensitivity units can only be of type 'Power' or 'Radiance'.")

        voxel_collection.set_active("all")

        cached_max_wavelength = self.max_wavelength
        cached_min_wavelength = self.min_wavelength
        cached_bins = self.spectral_bins
        cached_pipelines = self.pipelines
        cached_ray_count = self.pixel_samples

        self.pipelines = [pipeline]
        self.min_wavelength = 1
        self.max_wavelength = voxel_collection.count + 1
        self.spectral_bins = voxel_collection.count
        self.pixel_samples = ray_count

        self.observe()

        self.max_wavelength = cached_max_wavelength
        self.min_wavelength = cached_min_wavelength
        self.spectral_bins = cached_bins
        self.pipelines = cached_pipelines
        self.pixel_samples = cached_ray_count

        return pipeline.samples.mean

    def calculate_etendue(self, ray_count=10000, batches=10):

        if batches < 5:
            raise ValueError("We enforce a minimum batch size of 5 to ensure reasonable statistics.")

        target = self.slit.target

        world = self.slit.root
        detector_transform = self.to_root()

        # generate bounding sphere and convert to local coordinate system
        sphere = target.bounding_sphere()
        spheres = [(sphere.centre.transform(self.to_local()), sphere.radius, 1.0)]
        # instance targetted pixel sampler
        targetted_sampler = TargettedHemisphereSampler(spheres)

        etendues = []
        for i in range(batches):

            # sample pixel origins
            point_sampler = RectangleSampler3D(width=self.x_width, height=self.y_width)
            origins = point_sampler(samples=ray_count)

            passed = 0.0
            for origin in origins:

                # obtain targetted vector sample
                direction, pdf = targetted_sampler(origin, pdf=True)
                path_weight = R_2_PI * direction.z/pdf

                origin = origin.transform(detector_transform)
                direction = direction.transform(detector_transform)

                while True:

                    # Find the next intersection point of the ray with the world
                    intersection = world.hit(CoreRay(origin, direction))

                    if intersection is None:
                        passed += 1 * path_weight
                        break

                    elif isinstance(intersection.primitive.material, NullMaterial):
                        hit_point = intersection.hit_point.transform(intersection.primitive_to_world)
                        # apply a small displacement to avoid infinite self collisions due to numerics
                        ray_displacement = min(self.x_width, self.y_width) / 100
                        origin = hit_point + direction * ray_displacement
                        continue

                    else:
                        break

            etendue_fraction = passed / ray_count

            etendues.append(self.sensitivity * etendue_fraction)

        etendue = np.mean(etendues)
        etendue_error = np.std(etendues)

        return etendue, etendue_error
