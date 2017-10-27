
# Copyright 2014-2017 United Kingdom Atomic Energy Authority
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

from raysect.core import Node, AffineMatrix3D, translate, rotate_basis, Point3D, Vector3D
from raysect.optical.observer import PowerPipeline0D, SightLine, TargetedPixel


class BolometerCamera(Node):
    """
    A group of bolometer sight-lines under a single scene-graph node.

    A scene-graph object regrouping a series of 'BolometerFoil'
    observers as a scene-graph parent. Allows combined observation and display
    control simultaneously.
    """

    def __init__(self, parent=None, transform=None, name=''):
        super().__init__(parent=parent, transform=transform, name=name)

        self._foil_detectors = []
        self._slits = []

    def __getitem__(self, item):

        if isinstance(item, int):
            try:
                return self._sight_lines[item]
            except IndexError:
                raise IndexError("Sight-line number {} not available in this LineOfSightGroup.".format(item))
        elif isinstance(item, str):
            for sight_line in self._sight_lines:
                if sight_line.name == item:
                    return sight_line
            else:
                raise ValueError("Sightline '{}' was not found in this LineOfSightGroup.".format(item))
        else:
            raise TypeError("LineOfSightGroup key must be of type int or str.")

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
            foil_detector.parent = self

        self._foil_detectors = value

    def add_foil_detector(self, foil_detector):

        if not isinstance(foil_detector, BolometerFoil):
            raise TypeError("The foil_detector argument must be of type BolometerFoil.")

        foil_detector.parent = self
        self._foil_detectors.append(foil_detector)

    def observe(self):
        for foil_detector in self._foil_detectors:
            foil_detector.observe()


class BolometerSlit(Node):

    def __init__(self, centre_point, basis_x, dx, basis_y, dy, dz=0.001, parent=None, transform=None, name=''):

        self.name = name
        self.centre_point = centre_point
        self.basis_x = basis_x
        self.dx = dx
        self.basis_y = basis_y
        self.dy = dy
        self.dz = dz


class BolometerFoil:
    """
    A rectangular bolometer detector.

    Can be configured to sample a single ray or fan of rays oriented along the
    observer's z axis in world space.
    """

    def __init__(self, name, centre_point, normal_vec, basis_x, dx, dy, slit, ray_type="Targeted", parent=None):

        self._centre_point = Point3D(0, 0, 0)
        self._normal_vec = Vector3D(1, 0, 0)
        self._basis_x = Vector3D(0, 1, 0)
        self._basis_y = Vector3D(0, 0, 1)
        self._transform = AffineMatrix3D()
        self._power_pipeline = PowerPipeline0D(accumulate=False)

        self.name = name

        if not isinstance(slit, BolometerSlit):
            raise TypeError("slit argument for BolometerFoil must be of type BolometerSlit.")
        self._slit = slit

        if ray_type == "Sightline":
            self._observer = SightLine(pipelines=[self._power_pipeline],
                                       pixel_samples=1, spectral_bins=1, parent=parent, name=name)
        elif ray_type == "Targeted":
            self._observer = TargetedPixel(target=slit, pipelines=[self._power_pipeline],
                                           pixel_samples=250, spectral_bins=1, parent=parent, name=name)
        else:
            raise ValueError("ray_type argument for BolometerFoil must be in ['Sightline', 'Targeted'].")
        self.ray_type = ray_type

        if not isinstance(centre_point, Point3D):
            raise TypeError("centre_point argument for BolometerFoil must be of type Point3D.")
        self._centre_point = centre_point

        if not isinstance(normal_vec, Vector3D):
            raise TypeError("basis_vectors property of BolometerFoil must be a tuple of Vector3Ds.")
        if not isinstance(basis_x, Vector3D):
            raise TypeError("basis_vectors property of BolometerFoil must be a tuple of Vector3Ds.")

        if not normal_vec.dot(basis_x) == 0:
            raise ValueError("The normal and x basis vectors must be orthogonal to define a basis set.")

        # set basis vectors
        self._normal_vec = normal_vec.normalise()
        self._basis_x = basis_x.normalise()
        self._basis_y = normal_vec.cross(basis_x)

        # set observer transform
        translation = translate(self._centre_point.x, self._centre_point.y, self._centre_point.z)
        rotation = rotate_basis(normal_vec, basis_x)
        self._observer.transform = translation * rotation

        if not isinstance(dx, float):
            raise TypeError("dx argument for BolometerFoil must be of type float.")
        if not dx > 0:
            raise ValueError("dx argument for BolometerFoil must be greater than zero.")
        self.dx = dx

        if not isinstance(dy, float):
            raise TypeError("dy argument for BolometerFoil must be of type float.")
        if not dy > 0:
            raise ValueError("dy argument for BolometerFoil must be greater than zero.")
        self.dy = dy

    @property
    def centre_point(self):
        return self._centre_point

    @property
    def normal_vec(self):
        return self._normal_vec

    @property
    def basis_x(self):
        return self._basis_x

    @property
    def basis_y(self):
        return self._basis_y

    @property
    def slit(self):
        return self._slit

    @property
    def observed_power(self):
        if self._power_pipeline.value.samples <= 0:
            raise ValueError("This bolometer has not yet made any observations.")
        return self._power_pipeline.value.mean

    def observe(self):
        """
        Ask this bolometer foil to observe its world.
        """
        self._observer.observe()
