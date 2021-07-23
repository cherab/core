
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

from enum import Enum
import functools
import numpy as np

from raysect.core import Node, translate, rotate_basis, Point3D, Vector3D, Ray as CoreRay, Primitive, World
from raysect.core.math.sampler import TargettedHemisphereSampler, RectangleSampler3D
from raysect.primitive import Box, Cylinder, Subtract, Union
from raysect.optical.observer import PowerPipeline0D, RadiancePipeline0D, \
    SpectralPowerPipeline0D, SpectralRadiancePipeline0D, SightLine, TargettedPixel
from raysect.optical.observer import PowerPipeline2D, RadiancePipeline2D, \
    SpectralPowerPipeline2D, SpectralRadiancePipeline2D, TargettedCCDArray
from raysect.optical.material.material import NullMaterial
from raysect.optical.material import AbsorbingSurface

from cherab.tools.inversions.voxels import VoxelCollection


R_2_PI = 1 / (2 * np.pi)


class _Units(Enum):
    POWER = "power"
    RADIANCE = "radiance"


class BolometerCamera(Node):
    """
    A group of bolometer sight-lines under a single scenegraph node.

    A scenegraph object that manages a collection of :class:`BolometerFoil`
    objects. Allows combined observation and display control simultaneously.

    :param Primitive camera_geometry: A Raysect primitive to supply as the
      box/aperture geometry.
    :param Node parent: The parent node of this camera in the scenegraph, often
      an optical World object.
    :param AffineMatrix3D transform: The relative coordinate transform of this
      bolometer camera relative to the parent.
    :param str name: The name for this bolometer camera.

    :ivar list foil_detectors: A list of the foil detector objects that belong
      to this camera.
    :ivar list slits: A list of the bolometer slit objects that belong to
      this camera.

    .. code-block:: pycon

       >>> from raysect.optical import World
       >>> from cherab.tools.observers import BolometerCamera
       >>>
       >>> world = World()
       >>> camera = BolometerCamera(name="MyBolometer", parent=world)
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
        """Yields the number of detectors in this bolometer camera."""

        return len(self._foil_detectors)

    def __iter__(self):
        """
        Iterates over the foil detectors in this camera.

        .. code-block:: pycon

           >>> detector_a, detector_b, detector_c = bolometer_camera
        """
        for detector in self._foil_detectors:
            yield detector

    def __getitem__(self, item):
        """
        Returns the detectors by integer index or the detector name.

        .. code-block:: pycon

           >>> detector_2 = bolometer_camera[1]
           >>> detector_a = bolometer_camera["detector_a"]
        """

        if isinstance(item, int):
            try:
                return self._foil_detectors[item]
            except IndexError:
                raise IndexError("Bolometer number {} not available in this BolometerCamera.".format(item))
        elif isinstance(item, str):
            for detector in self._foil_detectors:
                if detector.name == item:
                    return detector
            raise ValueError("Bolometer '{}' was not found in this BolometerCamera.".format(item))
        else:
            raise TypeError("BolometerCamera key must be of type int or str.")

    @property
    def slits(self):
        return self._slits.copy()

    @property
    def foil_detectors(self):
        return self._foil_detectors.copy()

    @foil_detectors.setter
    def foil_detectors(self, value):

        if not isinstance(value, list):
            raise TypeError(
                "The foil_detectors attribute of BolometerCamera must be a list of "
                "BolometerFoils or BolometerIRVBs."
            )

        # Prevent external changes being made to this list
        value = value.copy()
        for foil_detector in value:
            if not isinstance(foil_detector, (BolometerFoil, BolometerIRVB)):
                raise TypeError(
                    "The foil_detectors attribute of BolometerCamera must be a list of "
                    "BolometerFoil or BolometerIRVB objects. Value {} is not a BolometerFoil "
                    "or BolometerIRVB.".format(foil_detector)
                    )
            if not foil_detector.slit in self._slits:
                self._slits.append(foil_detector.slit)
            foil_detector.parent = self

        self._foil_detectors = value

    def add_foil_detector(self, foil_detector):
        """
        Add the given detector to this camera.

        :param (BolometerFoil, BolometerIRVB) foil_detector: An instanced bolometer foil detector.

        .. code-block:: pycon

           >>> bolometer_camera.add_foil_detector(foil_detector)
        """

        if not isinstance(foil_detector, (BolometerFoil, BolometerIRVB)):
            raise TypeError(
                "The foil_detector argument must be of type BolometerFoil or BolometerIRVB."
            )

        if not foil_detector.slit in self._slits:
            self._slits.append(foil_detector.slit)

        foil_detector.parent = self
        self._foil_detectors.append(foil_detector)

    def observe(self):
        """
        Take an observation with this camera.

        Calls observe() on each foil detector and returns their power measurements.
        """

        observations = []
        for foil_detector in self._foil_detectors:
            foil_detector.observe()
            observations.append(foil_detector.pipelines[0].value.mean)

        return observations


class BolometerSlit(Node):
    """
    A rectangular bolometer slit.

    A single slit can be shared by multiple detectors in the parent camera. The slit
    geometry is specified in terms of its centre_point, basis vectors in the plane of
    the slit and their respective lengths. When instantiating a
    :class:`BolometerSlit` object these values are defined in the local coordinate
    system of the slit's parent, usually a :class:`BolometerCamera` object. Accessing
    these properties on an existing :class:`BolometerSlit` object returns them in the
    world's coordinate system.

    If an external mesh model has been loaded for ray occlusion evaluation then this
    object is only used for targeting rays on the slit. If no mesh has been supplied,
    this object can construct an effective slit primitive from CSG operations.

    .. warning::
       Be very careful when using a CSG aperture. The aperture geometry is slightly
       larger than the slit dx and dy, which can cause partial occlusion of
       nearby primitives. It also relies on no rays being launched with directions
       outside the solid angle of the aperture's bounding sphere: depending on the
       foil-slit distance and slit size, and also the foil's targetted_path_prob,
       this may not be guaranteed. Supplying a proper mesh geometry for the camera
       is recommended instead of using a CSG aperture.

    :param str slit_id: The name for this slit.
    :param Point3D centre_point: The centre point of the slit.
    :param Vector3D basis_x: The x basis vector for the slit.
    :param float dx: The width of the slit along the x basis vector.
    :param Vector3D basis_y: The y basis vector for the slit.
    :param float dy: The height of the slit along the y basis vector.
    :param float dz: The thickness of the slit along the z basis vector.
    :param Node parent: The parent scenegraph node to which this slit belongs.
      Typically a :class:`BolometerCamera` or an optical :class:`World` object.
    :param bool csg_aperture: Toggles whether an occluding surface should be
      constructed for this slit using CSG operations.
    :param float curvature_radius: Slits in real bolometer cameras may
      have curved corners due to machining limitations. This parameter species
      the corner radius.

    :ivar Vector3D normal_vector: The normal vector of the slit constructed from
      the cross product of the x and y basis vectors.

    .. code-block:: pycon

       >>> from raysect.core import Point3D, Vector3D
       >>> from raysect.optical import World
       >>> from cherab.tools.observers import BolometerSlit
       >>>
       >>> world = World()
       >>>
       >>> # construct basis vectors
       >>> basis_x = Vector3D(1, 0, 0)
       >>> basis_y = Vector3D(0, 1, 0)
       >>> basis_z = Vector3D(0, 0, 1)
       >>>
       >>> # specify the slit
       >>> dx = 0.0025
       >>> dy = 0.005
       >>> centre_point = Point3D(0, 0, 0)
       >>> slit = BolometerSlit("slit", centre_point, basis_x, dx, basis_y, dy, parent=camera)
    """

    def __init__(self, slit_id, centre_point, basis_x, dx, basis_y, dy, dz=0.001,
                 parent=None, csg_aperture=False, curvature_radius=0):

        # perform validation of input parameters

        if not isinstance(dx, (float, int)):
            raise TypeError("dx argument for BolometerSlit must be of type float/int.")
        if not dx > 0:
            raise ValueError("dx argument for BolometerSlit must be greater than zero.")

        if not isinstance(dy, (float, int)):
            raise TypeError("dy argument for BolometerSlit must be of type float/int.")
        if not dy > 0:
            raise ValueError("dy argument for BolometerSlit must be greater than zero.")

        if not isinstance(centre_point, Point3D):
            raise TypeError("centre_point argument for BolometerSlit must be of type Point3D.")

        if not isinstance(curvature_radius, (float, int)):
            raise TypeError("curvature_radius argument for BolometerSlit "
                            "must be of type float/int.")
        if curvature_radius < 0:
            raise ValueError("curvature_radius argument for BolometerSlit "
                             "must not be negative.")

        if not isinstance(basis_x, Vector3D):
            raise TypeError("The basis vectors of BolometerSlit must be of type Vector3D.")
        if not isinstance(basis_y, Vector3D):
            raise TypeError("The basis vectors of BolometerSlit must be of type Vector3D.")

        self._centre_point = centre_point
        self._basis_x = basis_x.normalise()
        self.dx = dx
        self._basis_y = basis_y.normalise()
        self.dy = dy
        self.dz = dz
        self._curvature_radius = curvature_radius

        # NOTE - target primitive and aperture surface cannot be co-incident otherwise numerics will cause Raysect
        # to be blind to one of the two surfaces.
        slit_normal = basis_x.cross(basis_y)
        transform = translate(centre_point.x, centre_point.y, centre_point.z) * rotate_basis(slit_normal, basis_y)

        super().__init__(parent=parent, transform=transform, name=slit_id)

        self.target = Box(lower=Point3D(-dx/2*1.01, -dy/2*1.01, -dz/2), upper=Point3D(dx/2*1.01, dy/2*1.01, dz/2),
                          transform=None, material=NullMaterial(), parent=self, name=slit_id+' - target')

        self._csg_aperture = None
        self.csg_aperture = csg_aperture

        # round off the detector corners, if applicable
        if self._curvature_radius > 0:
            mask_corners(self)

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

    @property
    def curvature_radius(self):
        return self._curvature_radius


class BolometerFoil(TargettedPixel):
    """
    A rectangular foil bolometer detector.

    When instantiating a detector, the position and orientation
    (i.e. centre_point, basis_x and basis_y) are given in the local coordinate
    system of the foil's parent, usually a :class:`BolometerCamera` instance.
    When these properties are accessed after instantiation, they are given in
    the coordinate system of the world.

    :param str detector_id: The name for this detector.
    :param Point3D centre_point: The centre point of the detector.
    :param Vector3D basis_x: The x basis vector for the detector.
    :param float dx: The width of the detector along the x basis vector.
    :param Vector3D basis_y: The y basis vector for the detector.
    :param float dy: The height of the detector along the y basis vector.
    :param Node parent: The parent scenegraph node to which this detector belongs.
      Typically a :class:`BolometerCamera` or an optical :class:`World` object.
    :param str units: The units in which to perform observations, can
      be ['Power', 'Radiance'].
    :param bool accumulate: Whether this observer should accumulate samples
      with multiple calls to observe.
    :param float curvature_radius: Detectors in real bolometer cameras typically
      have curved corners due to machining limitations. This parameter species
      the corner radius.

    :ivar Vector3D normal_vector: The normal vector of the detector constructed from
      the cross product of the x and y basis vectors.
    :ivar Vector3D sightline_vector: The vector that points from the centre of the foil
      detector to the centre of the slit. Defines the effective sightline vector of the
      detector.

    .. code-block:: pycon

       >>> from raysect.core import Point3D, Vector3D
       >>> from raysect.optical import World
       >>> from cherab.tools.observers import BolometerFoil
       >>>
       >>> world = World()
       >>>
       >>> # construct basis vectors
       >>> basis_x = Vector3D(1, 0, 0)
       >>> basis_y = Vector3D(0, 1, 0)
       >>> basis_z = Vector3D(0, 0, 1)
       >>>
       >>> # specify a detector, you need already created slit and camera objects
       >>> dx = 0.0025
       >>> dy = 0.005
       >>> centre_point = Point3D(0, 0, -0.08)
       >>> detector = BolometerFoil("ch#1", centre_point, basis_x, dx, basis_y, dy, slit, parent=camera)
    """

    def __init__(self, detector_id, centre_point, basis_x, dx, basis_y, dy, slit,
                 parent=None, units="Power", accumulate=False, curvature_radius=0):

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

        if not isinstance(centre_point, Point3D):
            raise TypeError("centre_point argument for BolometerFoil must be of type Point3D.")

        if not isinstance(curvature_radius, (float, int)):
            raise TypeError("curvature_radius argument for BolometerFoil "
                            "must be of type float/int.")
        if curvature_radius < 0:
            raise ValueError("curvature_radius argument for BolometerFoil "
                             "must not be negative.")

        if not isinstance(basis_x, Vector3D):
            raise TypeError("The basis vectors of BolometerFoil must be of type Vector3D.")
        if not isinstance(basis_y, Vector3D):
            raise TypeError("The basis vectors of BolometerFoil must be of type Vector3D.")

        basis_x = basis_x.normalise()
        basis_y = basis_y.normalise()
        normal_vec = basis_x.cross(basis_y)
        self._slit = slit
        self._curvature_radius = curvature_radius
        self._accumulate = accumulate

        # setup root bolometer foil transform
        translation = translate(centre_point.x, centre_point.y, centre_point.z)
        rotation = rotate_basis(normal_vec, basis_y)

        super().__init__([slit.target], targetted_path_prob=1.0,
                         pixel_samples=1000, x_width=dx, y_width=dy, spectral_bins=1, quiet=True,
                         parent=parent, transform=translation * rotation, name=detector_id)

        # Update pipeline based on units
        self.units = units

        # round off the detector corners, if applicable
        if self._curvature_radius > 0:
            mask_corners(self)

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

    @property
    def curvature_radius(self):
        return self._curvature_radius

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, units):
        if units == "Power":
            pipeline = PowerPipeline0D(accumulate=self.accumulate)
        elif units == "Radiance":
            pipeline = RadiancePipeline0D(accumulate=self.accumulate)
        else:
            raise ValueError("The units property of BolometerFoil must be one of 'Power' or 'Radiance'.")
        self._units = units
        self.pipelines = [pipeline]

    @property
    def accumulate(self):
        return self._accumulate

    @accumulate.setter
    def accumulate(self, value):
        for pipeline in self.pipelines:
            pipeline.accumulate = value
            # Discard any samples from previous accumulate behaviour
            pipeline.value.clear()

    def as_sightline(self):
        """
        Constructs a SightLine observer for this bolometer.

        :rtype: SightLine
        """

        if self.units == "Power":
            pipeline = PowerPipeline0D(accumulate=False)
        elif self.units == "Radiance":
            pipeline = RadiancePipeline0D(accumulate=False)
        else:
            raise ValueError("The units argument of BolometerFoil must be one of 'Power' or 'Radiance'.")

        los_observer = SightLine(pipelines=[pipeline], pixel_samples=1, quiet=True,
                                 parent=self, name=self.name)
        los_observer.render_engine = self.render_engine
        los_observer.spectral_bins = self.spectral_bins
        los_observer.min_wavelength = self.min_wavelength
        los_observer.max_wavelength = self.max_wavelength
        # The observer's Z axis should be aligned along the line of sight vector
        los_observer.transform = rotate_basis(
            self.sightline_vector.transform(self.to_local()), self.basis_y
        )

        return los_observer

    def trace_sightline(self):
        """
        Traces the central sightline through the detector to see where the sightline terminates.

        Raises a RuntimeError exception if no intersection was found.

        :return: A tuple containing the origin point, hit point and terminating surface
          primitive.
        """

        if not isinstance(self.root, World):
            raise ValueError("This BolometerFoil is not connected to a valid World scenegraph object.")

        origin = self.centre_point
        direction = self.sightline_vector

        while True:

            # Find the next intersection point of the ray with the world
            intersection = self.root.hit(CoreRay(origin, direction))

            if intersection is None:
                raise RuntimeError("No material intersection was found for this sightline.")

            elif isinstance(intersection.primitive.material, NullMaterial):
                # apply a small displacement to avoid infinite self collisions due to numerics
                hit_point = intersection.hit_point.transform(intersection.primitive_to_world)
                ray_displacement = min(self.x_width, self.y_width) / 100
                origin = hit_point + direction * ray_displacement
                continue

            else:
                hit_point = intersection.hit_point.transform(intersection.primitive_to_world)
                return self.centre_point, hit_point, intersection.primitive

    def calculate_sensitivity(self, voxel_collection, ray_count=10000):
        r"""
        Calculates a sensitivity vector for this detector on the specified voxel collection.

        This function is used for calculating sensitivity matrices which can be
        combined for multiple detectors into a sensitivity matrix
        :math:`\mathbf{W}`. If the :class:`BolometerFoil` has units of "Power", the
        returned sensitivity matrix has units of [m³ sr]. If the
        :class:`BolometerFoil` has units of "Radiance", the returned sensitivity
        matrix has units of [m sr].

        :param VoxelCollection voxel_collection: The voxel collection on which to calculate
          the sensitivities.
        :param int ray_count: The number of rays to use in the calculation. This should be
          at least >= 10000 for decent statistics.
        :return: A 1D array of sensitivities with length equal to the number of voxels
          in the collection.
        """
        # This method exploits ToroidalVoxelCollection.set_active("all"), which
        # makes each voxel emit a different wavelength of light. By observing
        # the voxel collection with a spectral pipeline we can thus distinguish
        # the amount of emission from each individual voxel.
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

    def calculate_etendue(self, ray_count=10000, batches=10, max_distance=1e999):
        """
        Calculates the etendue of this detector.

        This function calculates the detectors etendue by evaluating the fraction of rays that
        pass un-impeded through the detector's aperture.

        :param int ray_count: The number of rays used per batch.
        :param int batches: The number of batches used to estimate the error on the etendue calculation.
        :param float max_distance: The maximum distance from the detector to consider intersections.
            If a ray makes it further than this, it is assumed to have passed through the aperture,
            regardless of what it hits. Use this if there are other primitives present in the scene
            which do not form the aperture.
        :return: A tuple (etendue, etendue_error).
        """

        if batches < 5:
            raise ValueError("We enforce a minimum batch size of 5 to ensure reasonable statistics.")

        target = self.slit.target

        world = self.slit.root
        detector_transform = self.to_root()

        # generate bounding sphere and convert to local coordinate system
        sphere = target.bounding_sphere()
        spheres = [(sphere.centre.transform(self.to_local()), sphere.radius, 1.0)]
        # instance targetted pixel sampler to sample directions
        targetted_sampler = TargettedHemisphereSampler(spheres)
        # instance rectangle pixel sampler to sample origins
        point_sampler = RectangleSampler3D(width=self.x_width, height=self.y_width)

        def etendue_single_run(_):
            """Worker function to calculate the etendue: will be run <batches> times"""
            origins = point_sampler(samples=ray_count)
            passed = 0.0
            for origin in origins:
                # obtain targetted vector sample
                direction, pdf = targetted_sampler(origin, pdf=True)
                path_weight = R_2_PI * direction.z / pdf
                # Transform to world space
                origin = origin.transform(detector_transform)
                direction = direction.transform(detector_transform)
                while True:
                    # Find the next intersection point of the ray with the world
                    intersection = world.hit(CoreRay(origin, direction, max_distance))
                    if intersection is None:
                        passed += 1 * path_weight
                        break
                    if isinstance(intersection.primitive.material, NullMaterial):
                        hit_point = intersection.hit_point.transform(intersection.primitive_to_world)
                        # apply a small displacement to avoid infinite self collisions due to numerics
                        ray_displacement = min(self.x_width, self.y_width) / 100
                        origin = hit_point + direction * ray_displacement
                        continue
                    break
            etendue = (passed / ray_count) * self.sensitivity
            return etendue

        etendues = []
        self.render_engine.run(list(range(batches)), etendue_single_run, etendues.append)
        etendue = np.mean(etendues)
        etendue_error = np.std(etendues)

        return etendue, etendue_error


class BolometerIRVB(TargettedCCDArray):
    """
    A rectangular infra red video bolometer (IRVB).

    Can be configured to sample a single ray per pixel, or fan of rays
    oriented along the observer's z axis.

    :param str name: The name for this detector.
    :param float width: The width of the detector along the x basis vector.
    :param tuple pixels: The number of pixels to divide the foil into.
      Pixels are square, so the height of the foil is determined by the
      width of the foil and the number of rows and columns of pixels.
    :param BolometerSlit slit: The slit the IRVB views through.
    :param AffineMatrix3D transform: The foil's transform relative to its parent.
    :param Node parent: The parent scenegraph node to which this detector belongs.
      Typically a BolometerCamera() or an optical World() object.
    :param bool units: The units in which to perform observations, can
      be ['power', 'radiance'], defaults to 'power'.
    :param bool accumulate: Whether this observer should accumulate samples
      with multiple calls to observe. Defaults to False.
    :param float curvature_radius: Detectors in real bolometer cameras may
      have curved corners due to machining limitations. This parameter species
      the corner radius.

    :ivar Vector3D normal_vector: The normal vector of the detector constructed from
      the cross product of the x and y basis vectors.
    :ivar float width: The extent of the bolometer foil in the basis_x direction.
    :ivar float height: The extent of the bolometer foil in the basis_y direction.
    :ivar array pixels_as_foils: A 2D array of pixels as individual BolometerFoil objects,
      useful for calling BolometerFoil methods on each pixel (e.g. sightline tracing).
      The array is indexed by (column, row).
    :ivar Vector3D sightline_vectors: A 2D array of vectors that point from the centre
      of each pixel on the foil detector to the centre of the slit. Defines the
      effective sightline vectors of the pixels of the detector. The array is indexed by
      (column, row).

    .. code-block:: pycon

       >>> from raysect.core import Point3D, Vector3D, translate, rotate_basis
       >>> from raysect.optical import World
       >>> from cherab.tools.observers import BolometerIRVB
       >>>
       >>> world = World()
       >>>
       >>> # construct transform, relative to parent's transform
       >>> centre_point = Point3D(0, 0, -0.08)
       >>> basis_x = Vector3D(1, 0, 0)
       >>> basis_y = Vector3D(0, 1, 0)
       >>> normal = basis_x.cross(basis_y)
       >>> transform = translate(*centre_point) * rotate_basis(normal, basis_y)
       >>>
       >>> # specify a detector, you need already created slit and camera objects
       >>> width = 0.0025
       >>> pixels = (10, 20)
       >>> detector = BolometerIRVB("irvb", width, pixels, slit, transform, parent=camera)
    """

    _PIPELINES = {_Units.POWER: PowerPipeline2D,
                  _Units.RADIANCE: RadiancePipeline2D}
    _SPECTRAL_PIPELINES = {_Units.POWER: SpectralPowerPipeline2D,
                           _Units.RADIANCE: SpectralRadiancePipeline2D}

    def __init__(self, name, width, pixels, slit, transform, parent=None,
                 units="power", accumulate=False, curvature_radius=0):

        # perform validation of input parameters
        width = float(width)
        if width < 0:
            raise ValueError("width argument for BolometerIRVB must be greater than zero.")

        if not isinstance(slit, BolometerSlit):
            raise TypeError("slit argument for BolometerIRVB must be of type BolometerSlit.")

        curvature_radius = float(curvature_radius)
        if curvature_radius < 0:
            raise ValueError("curvature_radius argument for BolometerIRVB "
                             "must not be negative.")

        self._slit = slit
        self._curvature_radius = curvature_radius
        self._accumulate = None  # Will be set after pipeline is created.

        super().__init__([slit.target], pixels=pixels, width=width,
                         targetted_path_prob=0.99, parent=parent, pipelines=[],
                         transform=transform, name=name)
        self.pixel_samples = 1000
        self.spectral_bins = 1
        self.quiet = True

        # Update pipeline based on units and accumulate.
        self.units = units
        self.accumulate = accumulate

        # round off the detector corners, if applicable
        if self._curvature_radius > 0:
            mask_corners(self)

    def __repr__(self):
        """Returns a string representation of this BolometerIRVB object."""
        return "<BolometerIRVB - " + self.name + ">"

    @property
    def pixels_as_foils(self):
        XAXIS = Vector3D(1, 0, 0)
        YAXIS = Vector3D(0, 1, 0)
        nx, ny = self.pixels
        pixel_width = self.width / nx
        pixel_height = self.height / ny
        # Foil pixels are defined in the foil's local coordinate system
        foil_bottom_left = Point3D(-self.width / 2, -self.height / 2, 0)
        pixels = []
        for x in range(nx):
            pixel_column = []
            for y in range(ny):
                pixel_centre = (foil_bottom_left
                                + (x + 0.5) * XAXIS * pixel_width
                                + (y + 0.5) * YAXIS * pixel_height)
                pixel = BolometerFoil(
                    detector_id="IRVB pixel ({},{})".format(x + 1, y + 1),
                    centre_point=pixel_centre, basis_x=XAXIS, dx=pixel_width,
                    basis_y=YAXIS, dy=pixel_height, slit=self._slit,
                    units=self._units.value.capitalize(), accumulate=False, parent=self
                )
                pixel_column.append(pixel)
            pixels.append(pixel_column)
        return np.asarray(pixels, dtype='object')

    @property
    def height(self):
        return self.width * self.pixels[1] / self.pixels[0]

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
    def sightline_vectors(self):
        return np.asarray(
            [[pixel.centre_point.vector_to(self._slit.centre_point) for pixel in pixel_column]
             for pixel_column in self.pixels_as_foils],
            dtype='object'
        )

    @property
    def slit(self):
        return self._slit

    @property
    def curvature_radius(self):
        return self._curvature_radius

    @property
    def units(self):
        return self._units.value

    @units.setter
    def units(self, units):
        if units.lower() == _Units.POWER.value:
            self._units = _Units.POWER
        elif units.lower() == _Units.RADIANCE.value:
            self._units = _Units.RADIANCE
        else:
            raise ValueError(
                "The units property of BolometerIRVB must be one of {}"
                .format([member.value for member in _Units.__members__])
            )
        pipeline_class = self._PIPELINES[self._units]
        pipeline = pipeline_class(accumulate=self.accumulate)
        self.pipelines = [pipeline]

    @property
    def accumulate(self):
        return self._accumulate

    @accumulate.setter
    def accumulate(self, value):
        self._accumulate = value
        for pipeline in self.pipelines:
            pipeline.accumulate = value
            # Discard any samples from previous accumulate behaviour
            if pipeline.frame is not None:
                pipeline.frame.clear()

    def as_sightlines(self):
        """
        Constructs a SightLine observer for each pixel in this bolometer.

        :return: A 2D array of Sightline objects.
        """
        pixels = self.pixels_as_foils
        sightlines = [[pixel.as_sightline() for pixel in pixel_column] for pixel_column in pixels]
        return np.asarray(sightlines, dtype='object')

    def trace_sightlines(self):
        """
        Trace the central sightlines through each pixel in the detector
        to see where the sightline terminates.

        Raises a RuntimeError exception if no intersections were found.

        :return: A 2D array of tuples containing the origin point, hit
          point and terminating surface primitive for each pixel.
        """
        pixels = self.pixels_as_foils
        traces = [[pixel.trace_sightline() for pixel in pixel_column] for pixel_column in pixels]
        return np.asarray(traces, dtype='object')

    def calculate_sensitivity(self, voxel_collection, ray_count=None):
        r"""
        Calculates a sensitivity vector for this detector on the specified voxel collection.

        This function is used for calculating sensitivity matrices which can be combined for
        multiple detectors into a sensitivity matrix :math:`\mathbf{W}`.

        :param VoxelCollection voxel_collection: The voxel collection on which to calculate
          the sensitivities.
        :param int ray_count: The number of rays to use in the calculation. This should be
          at least >= 10000 for decent statistics. Default is 10000.
        :return: A 3D array of sensitivities (ncol, nrow, nvoxels)
        """
        ray_count = ray_count or 10000

        # This method exploits ToroidalVoxelCollection.set_active("all"), which
        # makes each voxel emit a different wavelength of light. By observing
        # the voxel collection with a spectral pipeline we can thus distinguish
        # the amount of emission from each individual voxel.
        if not isinstance(voxel_collection, VoxelCollection):
            raise TypeError("voxel_collection must be of type VoxelCollection")

        pipeline_class = self._SPECTRAL_PIPELINES[self._units]
        pipeline = pipeline_class(display_progress=False)

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

        return pipeline.frame.mean

    def calculate_etendue(self, ray_count=None, batches=None, max_distance=None):
        """
        Calculates the etendue of each pixel in this detector.

        This function calculates the detector etendue by evaluating the ratio of rays that
        pass un-impeded through the detector's aperture. For this method to work, the detector
        and its aperture structures should be the only primitives present in the scene. If any
        other primitives are present, the results may be misleading.

        :param int ray_count: The number of rays used per batch (default 10000).
        :param int batches: The number of batches used to estimate the error on the etendue
          calculation. Default is 10.
        :param float max_distance: The maximum distance from the detector to consider intersections.
            If a ray makes it further than this, it is assumed to have passed through the aperture,
            regardless of what it hits. Use this if there are other primitives present in the scene
            which do not form the aperture. Default is infinity (no max distance).
        :return: a tuple (etendue, etendue_error), each of which is a 2D
          array of size (ncol, nrow)
        """
        ray_count = ray_count or 10000
        batches = batches or 10
        max_distance = max_distance or 1e999
        # Calculate the etendue of each pixel as a separate task
        nx, ny = self.pixels
        pixels_flattened = self.pixels_as_foils.ravel()
        etendues_flattened = np.empty_like(pixels_flattened, dtype=float)
        etendue_errors_flattened = np.empty_like(etendues_flattened)

        def calculate(pixel_index):
            foil = pixels_flattened[pixel_index]
            foil.render_engine.processes = 1
            return pixel_index, foil.calculate_etendue(ray_count, batches, max_distance)

        def update(result):
            index, (etendue, etendue_error) = result
            etendues_flattened[index] = etendue
            etendue_errors_flattened[index] = etendue_error

        self.render_engine.run(list(range(nx * ny)), calculate, update)
        # Reshape back to pixel dimensions
        etendue = etendues_flattened.reshape(self.pixels)
        etendue_error = etendue_errors_flattened.reshape(self.pixels)
        return etendue, etendue_error


def mask_corners(element):
    """
    Support detectors with rounded corners, by producing a mask to cover
    the corners.

    The mask is produced by cutting a rounded rectangle, formed of the
    union of two smaller perpendicular rectangles and four cylinders,
    from a rectangle the same size as the detector.

    The curvature radius should be given in units of metres.
    """
    # Make the mask very (but not infinitely) thin, so that raysect can actually
    # detect that it's there. Work in the local coordinate system of the
    # element, with dx=width, dy=height, dz=depth.
    dz = 1e-6
    rc = element.curvature_radius  # Shorthand
    try:
        dx = element.x_width
        dy = element.y_width
    except AttributeError:
        try:
            dx = element.dx
            dy = element.dy
        except AttributeError:
            dx = element.width
            dy = element.height

    # Make the elements to cut out from the cover slightly thicker than the
    # cover, to guard against rounding errors
    long_box = Box(lower=Point3D(-dx/2 + rc, -dy/2, -0.5 * dz),
                   upper=Point3D(dx/2 - rc, dy/2, 1.5 * dz))
    shot_box = Box(lower=Point3D(-dx/2, -dy/2 + rc, -0.5 * dz),
                   upper=Point3D(dx/2, dy/2 - rc, 1.5 * dz))
    cylinder_template = Cylinder(radius=rc, height=2 * dz)
    top_left_cylinder = cylinder_template.instance()
    top_left_cylinder.transform = translate(-dx/2 + rc, dy/2 - rc, -dz/2)
    top_right_cylinder = cylinder_template.instance()
    top_right_cylinder.transform = translate(dx/2 - rc, dy/2 - rc, -dz/2)
    bottom_right_cylinder = cylinder_template.instance()
    bottom_right_cylinder.transform = translate(dx/2 - rc, -dy/2 + rc, -dz/2)
    bottom_left_cylinder = cylinder_template.instance()
    bottom_left_cylinder.transform = translate(-dx/2 + rc, -dy/2 + rc, -dz/2)
    cutout = functools.reduce(Union, (long_box, shot_box, top_left_cylinder,
                                      top_right_cylinder, bottom_right_cylinder,
                                      bottom_left_cylinder))
    cover = Box(lower=Point3D(-dx/2, -dy/2, 0), upper=Point3D(dx/2, dy/2, dz))
    mask = Subtract(cover, cutout)

    mask.material = AbsorbingSurface()
    mask.transform = translate(0, 0, dz)
    mask.name = element.name + ' - rounded edges mask'
    mask.parent = element
