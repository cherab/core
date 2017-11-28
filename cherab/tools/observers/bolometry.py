
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

import os
import json
import pickle

from raysect.core import Node, AffineMatrix3D, translate, rotate_basis, Point3D, Vector3D
from raysect.primitive import Box, Cylinder, Subtract
from raysect.optical.observer import PowerPipeline0D, RadiancePipeline0D, SightLine, TargettedPixel
from raysect.optical.material.material import NullMaterial
from raysect.optical.material import AbsorbingSurface, UnityVolumeEmitter

from cherab.tools.observers.inversion_grid import SensitivityMatrix


# TODO - add support for CAD files as camera box geometry
class BolometerCamera(Node):
    """
    A group of bolometer sight-lines under a single scene-graph node.

    A scene-graph object regrouping a series of 'BolometerFoil'
    observers as a scene-graph parent. Allows combined observation and display
    control simultaneously.
    """

    def __init__(self, box_geometry=None, parent=None, transform=None, name=''):
        super().__init__(parent=parent, transform=transform, name=name)

        self._foil_detectors = []
        self._slits = []
        self._box_geometry = box_geometry

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
                if detector.detector_id == item:
                    return detector
            else:
                raise ValueError("BolometerFoil '{}' was not found in this BolometerCamera.".format(item))
        else:
            raise TypeError("BolometerCamera key must be of type int or str.")

    def __getstate__(self, serialisation_format=None):
        state = {
            'CHERAB_Object_Type': 'BolometerCamera',
            'Version': 1,
            'Camera_ID': self.name,
        }

        if self._box_geometry:
            pass
        else:
            state['box_geometry'] = False

        slit_list = []
        for slit in self._slits:
            slit_list.append(slit.__getstate__())
        state['slits'] = slit_list

        detector_list = []
        for detector in self._foil_detectors:
            detector_list.append(detector.__getstate__(serialisation_format=serialisation_format))
        state['foil_detectors'] = detector_list

        return state

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
        for foil_detector in self._foil_detectors:
            foil_detector.observe()

    def save(self, filename):

        name, extention = os.path.splitext(filename)

        if extention == '.json':
            file_handle = open(filename, 'w')
            json.dump(self.__getstate__(serialisation_format=extention), file_handle, indent=2, sort_keys=True)

        elif extention == '.pickle':
            file_handle = open(filename, 'wb')
            pickle.dump(self.__getstate__(serialisation_format=extention), file_handle)

        else:
            raise NotImplementedError("Invalid serialisation format - '{}'.".format(extention))


class BolometerSlit(Node):

    def __init__(self, slit_id, centre_point, basis_x, dx, basis_y, dy, dz=0.001, parent=None, csg_aperture=False):

        self.slit_id = slit_id
        self.centre_point = centre_point
        self.basis_x = basis_x
        self.dx = dx
        self.basis_y = basis_y
        self.dy = dy
        self.dz = dz

        # NOTE - target primitive and aperture surface cannot be co-incident otherwise numerics will cause Raysect
        # to be blind to one of the two surfaces.
        slit_normal = basis_x.cross(basis_y)
        transform = translate(centre_point.x, centre_point.y, centre_point.z) * rotate_basis(slit_normal, basis_x)

        super().__init__(parent=parent, transform=transform, name=self.slit_id)

        self.primitive = Box(lower=Point3D(-dx/2*1.01, -dy/2*1.01, -dz/2), upper=Point3D(dx/2*1.01, dy/2*1.01, dz/2),
                             transform=None, material=NullMaterial(), parent=self, name=slit_id+' - target')

        if csg_aperture:
            face = Box(Point3D(-dx, -dy, -dz/2), Point3D(dx, dy, dz/2))
            slit = Box(lower=Point3D(-dx/2, -dy/2, -dz/2 - dz*0.1), upper=Point3D(dx/2, dy/2, dz/2 + dz*0.1))
            self.csg_aperture = Subtract(face, slit, parent=self, material=AbsorbingSurface(), name=slit_id+' - CSG Aperture')
        else:
            self.csg_aperture = None

    def __getstate__(self):

        state = {
            'CHERAB_Object_Type': 'BolometerSlit',
            'Version': 1,
            'Slit_ID': self.slit_id,
            'centre_point': self.centre_point.__getstate__(),
            'basis_x': self.basis_x.__getstate__(),
            'basis_y': self.basis_y.__getstate__(),
            'dx': self.dx,
            'dy': self.dy,
            'dz': self.dz,
        }

        if self.csg_aperture:
            state['csg_aperture'] = True
        else:
            state['csg_aperture'] = False

        return state


class BolometerFoil(Node):
    """
    A rectangular bolometer detector.

    Can be configured to sample a single ray or fan of rays oriented along the
    observer's z axis in world space.
    """

    def __init__(self, detector_id, centre_point, basis_x, dx, basis_y, dy, slit, parent=None):

        self.detector_id = detector_id

        # perform validation of input parameters

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

        # setup root bolometer foil transform
        translation = translate(self._centre_point.x, self._centre_point.y, self._centre_point.z)
        rotation = rotate_basis(self._normal_vec, self._basis_x)

        super().__init__(parent=parent, transform=translation * rotation, name=self.detector_id)

        # setup the observers
        self._los_radiance_pipeline = RadiancePipeline0D(accumulate=False)
        self._los_observer = SightLine(pipelines=[self._los_radiance_pipeline], pixel_samples=1, spectral_bins=1,
                                       parent=self, name=detector_id, quiet=True)

        self._volume_power_pipeline = PowerPipeline0D(accumulate=False)
        self._volume_radiance_pipeline = RadiancePipeline0D(accumulate=False)
        self._volume_observer = TargettedPixel([slit.primitive], targetted_path_prob=1.0,
                                               pipelines=[self._volume_power_pipeline, self._volume_radiance_pipeline],
                                               pixel_samples=1000, x_width=dx, y_width=dy,
                                               spectral_bins=1, parent=self, name=detector_id, quiet=True)

        # NOTE - the los observer may not be normal to the surface, in which case calculate an extra relative transform
        if self._normal_vec.dot(self._foil_to_slit_vec) != 1.0:
            relative_foil_to_slit_vec = self._foil_to_slit_vec.transform(self.to_local())
            relative_rotation = rotate_basis(relative_foil_to_slit_vec, Vector3D(1, 0, 0))
            self._los_observer.transform = relative_rotation
            self._los_cos_theta = self._normal_vec.dot(self._foil_to_slit_vec)
        else:
            self._los_cos_theta = 1.0

        # initialise sensitivity values
        self._los_radiance_sensitivity = None
        self._volume_radiance_sensitivity = None
        self._volume_power_sensitivity = None
        self._etendue = None

    def __getstate__(self, serialisation_format=None):

        state = {
            'CHERAB_Object_Type': 'BolometerFoil',
            'Version': 1,
            'Detector_ID': self.detector_id,
            'centre_point': self.centre_point.__getstate__(),
            'basis_x': self._basis_x.__getstate__(),
            'basis_y': self._basis_y.__getstate__(),
            'dx': self.dx,
            'dy': self.dy,
            'slit_id': self._slit.slit_id,
        }

        # add extra data if saving as binary
        if serialisation_format == '.pickle' and self._los_radiance_sensitivity is not None:
            state['los_radiance_sensitivity'] = self._los_radiance_sensitivity.__getstate__()
            state['volume_radiance_sensitivity'] = self._volume_radiance_sensitivity.__getstate__()
            state['volume_power_sensitivity'] = self._volume_power_sensitivity.__getstate__()
            state['etendue'] = self._etendue

        return state

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
    def los_radiance_sensitivity(self):
        if self._los_radiance_sensitivity is None:
            raise ValueError("The sensitivity of this BolometerFoil has not yet been calculated.")
        return self._los_radiance_sensitivity

    @property
    def volume_power_sensitivity(self):
        if self._volume_power_sensitivity is None:
            raise ValueError("The sensitivity of this BolometerFoil has not yet been calculated.")
        return self._volume_power_sensitivity

    @property
    def volume_radiance_sensitivity(self):
        if self._volume_radiance_sensitivity is None:
            raise ValueError("The sensitivity of this BolometerFoil has not yet been calculated.")
        return self._volume_radiance_sensitivity

    @property
    def etendue(self):
        if self._etendue is None:
            raise ValueError("The sensitivity of this BolometerFoil has not yet been calculated.")
        return self._etendue

    def observe_los(self):
        """
        Ask this bolometer foil to observe its world.
        """
        self._los_observer.observe()
        return self._los_radiance_pipeline.value.mean * self._los_cos_theta

    def observe_volume_radiance(self):
        """
        Ask this bolometer foil to observe its world.
        """
        self._volume_observer.observe()
        return self._volume_radiance_pipeline.value.mean

    def observe_volume_power(self):
        """
        Ask this bolometer foil to observe its world.
        """
        self._volume_observer.observe()
        return self._volume_power_pipeline.value.mean

    def calculate_sensitivity(self, grid):

        world = self._los_observer.root

        self._los_radiance_sensitivity = SensitivityMatrix(grid, self.detector_id, 'los radiance')
        self._volume_radiance_sensitivity = SensitivityMatrix(grid, self.detector_id, 'Volume mean radiance sensitivity')
        self._volume_power_sensitivity = SensitivityMatrix(grid, self.detector_id, 'Volume power sensitivity')

        for i in range(grid.count):

            p1, p2, p3, p4 = grid[i]

            r_inner = p1.x
            r_outer = p3.x
            if r_inner > r_outer:
                t = r_inner
                r_inner = r_outer
                r_outer = t

            z_lower = p2.y
            z_upper = p1.y
            if z_lower > z_upper:
                t = z_lower
                z_lower = z_upper
                z_upper = t

            # TODO - switch to using CAD method such that reflections can be included automatically
            cylinder_height = z_upper - z_lower

            outer_cylinder = Cylinder(radius=r_outer, height=cylinder_height, transform=translate(0, 0, z_lower))
            inner_cylinder = Cylinder(radius=r_inner, height=cylinder_height, transform=translate(0, 0, z_lower))
            cell_emitter = Subtract(outer_cylinder, inner_cylinder, parent=world, material=UnityVolumeEmitter())

            self._los_observer.observe()
            self._los_radiance_sensitivity.sensitivity[i] = self._los_radiance_pipeline.value.mean

            self._volume_observer.observe()
            self._volume_radiance_sensitivity.sensitivity[i] = self._volume_radiance_pipeline.value.mean
            self._volume_power_sensitivity.sensitivity[i] = self._volume_power_pipeline.value.mean

            outer_cylinder.parent = None
            inner_cylinder.parent = None
            cell_emitter.parent = None


def load_bolometer_camera(filename, parent=None, inversion_grid=None):

    name, extention = os.path.splitext(filename)

    if extention == '.json':
        file_handle = open(filename, 'r')
        camera_state = json.load(file_handle)

    elif extention == '.pickle':
        file_handle = open(filename, 'rb')
        camera_state = pickle.load(file_handle)

    else:
        raise IOError("Unrecognised CHERAB object file format - '{}'.".format(extention))

    if not camera_state['CHERAB_Object_Type'] == 'BolometerCamera':
        raise ValueError("The selected json file does not contain a valid BolometerCamera description.")
    if not camera_state['Version'] == 1.0:
        raise ValueError("The BolometerCamera description in the selected json file is out of date, version = {}.".format(camera_state['Version']))

    camera = BolometerCamera(name=camera_state['Camera_ID'], parent=parent)

    slit_dict = {}

    for slit in camera_state['slits']:

        if not slit['CHERAB_Object_Type'] == 'BolometerSlit':
            raise ValueError("The selected json file does not contain a valid BolometerCamera description.")
        if not slit['Version'] == 1.0:
            raise ValueError("The BolometerSlit description in the selected json file is out of date, "
                             "version = {}.".format(slit['Version']))

        slid_id = slit['Slit_ID']
        centre_point = Point3D(slit['centre_point'][0], slit['centre_point'][1], slit['centre_point'][2])
        basis_x = Vector3D(slit['basis_x'][0], slit['basis_x'][1], slit['basis_x'][2])
        dx = slit['dx']
        basis_y = Vector3D(slit['basis_y'][0], slit['basis_y'][1], slit['basis_y'][2])
        dy = slit['dy']
        dz = slit['dz']
        csg_aperture = slit['csg_aperture']
        slit_dict[slid_id] = BolometerSlit(slid_id, centre_point, basis_x, dx, basis_y, dy,
                                           dz=dz, parent=camera, csg_aperture=csg_aperture)

    for detector in camera_state['foil_detectors']:

        if not detector['CHERAB_Object_Type'] == 'BolometerFoil':
            raise ValueError("The selected json file does not contain a valid BolometerCamera description.")
        if not detector['Version'] == 1.0:
            raise ValueError("The BolometerFoil description in the selected json file is out of date, "
                             "version = {}.".format(detector['Version']))

        # detector_id, centre_point, basis_x, dx, basis_y, dy, slit, ray_type="Targeted", parent=None

        detector_id = detector['Detector_ID']
        centre_point = Point3D(detector['centre_point'][0], detector['centre_point'][1], detector['centre_point'][2])
        basis_x = Vector3D(detector['basis_x'][0], detector['basis_x'][1], detector['basis_x'][2])
        dx = detector['dx']
        basis_y = Vector3D(detector['basis_y'][0], detector['basis_y'][1], detector['basis_y'][2])
        dy = detector['dy']
        slit = slit_dict[detector['slit_id']]

        bolometer_foil = BolometerFoil(detector_id, centre_point, basis_x, dx, basis_y, dy, slit, parent=camera)

        # add extra sensitivity data stored in binary
        if extention == '.pickle' and inversion_grid is not None:
            try:

                detector_uid = detector['los_radiance_sensitivity']['detector_uid']
                description = detector['los_radiance_sensitivity']['description']
                los_radiance_sensitivity = SensitivityMatrix(inversion_grid, detector_uid, description=description)
                los_radiance_sensitivity.sensitivity[:] = detector['los_radiance_sensitivity']['sensitivity']
                bolometer_foil._los_radiance_sensitivity = los_radiance_sensitivity

                detector_uid = detector['volume_radiance_sensitivity']['detector_uid']
                description = detector['volume_radiance_sensitivity']['description']
                volume_radiance_sensitivity = SensitivityMatrix(inversion_grid, detector_uid, description=description)
                volume_radiance_sensitivity.sensitivity[:] = detector['volume_radiance_sensitivity']['sensitivity']
                bolometer_foil._volume_radiance_sensitivity = volume_radiance_sensitivity

                detector_uid = detector['volume_power_sensitivity']['detector_uid']
                description = detector['volume_power_sensitivity']['description']
                volume_power_sensitivity = SensitivityMatrix(inversion_grid, detector_uid, description=description)
                volume_power_sensitivity.sensitivity[:] = detector['volume_power_sensitivity']['sensitivity']
                bolometer_foil._volume_power_sensitivity = volume_power_sensitivity

                bolometer_foil._etendue = detector['etendue']

            except IndexError:
                continue

        camera.add_foil_detector(bolometer_foil)

    return camera
