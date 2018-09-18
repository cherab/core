
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

import os
import json
import pickle
import gzip
import numpy as np

from raysect.core import Node, translate, rotate_basis, Point3D, Vector3D, Ray as CoreRay, Primitive
from raysect.core.math.sampler import TargettedHemisphereSampler
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
                if detector.name == item:
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

        elif extention == '.pgz':
            with gzip.open(filename, 'wb') as file_handle:
                pickle.dump(self.__getstate__(serialisation_format='.pickle'), file_handle)

        elif extention == '.sav':
            import idlbridge as idl
            idl.put("bolometer_camera_data", self.__getstate__(serialisation_format=extention))
            idl.execute("save, bolometer_camera_data, FILENAME='{}'".format(filename))

        else:
            raise NotImplementedError("Invalid serialisation format - '{}'.".format(extention))


class BolometerSlit(Node):

    def __init__(self, slit_id, centre_point, basis_x, dx, basis_y, dy, dz=0.001, parent=None, csg_aperture=False):

        self.slit_id = slit_id
        self.centre_point = centre_point
        self.basis_x = basis_x.normalise()
        self.dx = dx
        self.basis_y = basis_y.normalise()
        self.dy = dy
        self.dz = dz

        # NOTE - target primitive and aperture surface cannot be co-incident otherwise numerics will cause Raysect
        # to be blind to one of the two surfaces.
        slit_normal = basis_x.cross(basis_y)
        transform = translate(centre_point.x, centre_point.y, centre_point.z) * rotate_basis(slit_normal, basis_y)

        super().__init__(parent=parent, transform=transform, name=self.slit_id)

        self.primitive = Box(lower=Point3D(-dx/2*1.01, -dy/2*1.01, -dz/2), upper=Point3D(dx/2*1.01, dy/2*1.01, dz/2),
                             transform=None, material=NullMaterial(), parent=self, name=slit_id+' - target')

        self._csg_aperture = None
        self.csg_aperture = csg_aperture

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
                                          material=AbsorbingSurface(), name=self.slit_id+' - CSG Aperture')

        else:
            if isinstance(self._csg_aperture, Primitive):
                self._csg_aperture.parent = None
            self._csg_aperture = None

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


class BolometerFoil(TargettedPixel):
    """
    A rectangular bolometer detector.

    Can be configured to sample a single ray or fan of rays oriented along the
    observer's z axis in world space.
    """

    def __init__(self, detector_id, centre_point, basis_x, dx, basis_y, dy, slit,
                 parent=None, render_engine=None):

        # perform validation of input parameters

        if not isinstance(dx, float):
            raise TypeError("dx argument for BolometerFoil must be of type float.")
        if not dx > 0:
            raise ValueError("dx argument for BolometerFoil must be greater than zero.")

        if not isinstance(dy, float):
            raise TypeError("dy argument for BolometerFoil must be of type float.")
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

        # setup root bolometer foil transform
        translation = translate(self._centre_point.x, self._centre_point.y, self._centre_point.z)
        rotation = rotate_basis(self._normal_vec, self._basis_y)

        super().__init__([slit.primitive], targetted_path_prob=1.0,
                         pipelines=[PowerPipeline0D(accumulate=False)],
                         pixel_samples=1000, x_width=dx, y_width=dy, spectral_bins=1, quiet=True,
                         parent=parent, transform=translation * rotation, name=detector_id)

    def __repr__(self):
        """Returns a string representation of this BolometerFoil object."""
        return "<BolometerFoil - " + self.name + ">"

    def __getstate__(self, serialisation_format=None):

        state = {
            'CHERAB_Object_Type': 'BolometerFoil',
            'Version': 1,
            'Detector_ID': self.name,
            'centre_point': self.centre_point.__getstate__(),
            'basis_x': self._basis_x.__getstate__(),
            'basis_y': self._basis_y.__getstate__(),
            'x_width': self.x_width,
            'y_width': self.y_width,
            'slit_id': self._slit.slit_id,
        }

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

    def trace_sightline(self):
        raise NotImplementedError("Please implement me.")

    def calculate_sensitivity(self, voxel_collection, ray_count=10000, units="Power"):

        if not isinstance(voxel_collection, VoxelCollection):
            raise TypeError("voxel_collection must be of type VoxelCollection")

        if units == "Power":
            pipeline = SpectralPowerPipeline0D(display_progress=False)
        elif units == "Radiance":
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


def load_bolometer_camera(filename, parent=None, camera_dict=None, render_engine=None):

    if filename == 'machine_description':
        name, extention = '', ''
        camera_state = camera_dict

    else:
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

        bolometer_foil = BolometerFoil(
            detector_id, centre_point, basis_x, dx, basis_y, dy, slit,
            parent=camera, render_engine=render_engine
        )

        camera.add_foil_detector(bolometer_foil)

    return camera


def assemble_weight_matrix(cameras, excluded_detectors=None):

    if excluded_detectors is None:
        excluded_detectors = []

    detector_keys = []
    num_detectors = 0
    for camera in cameras:
        for detector in camera:
            if detector.detector_id not in excluded_detectors:
                num_detectors += 1
                detector_keys.append(detector.detector_id)

    num_sensitivities = len(detector._volume_radiance_sensitivity.sensitivity)

    los_pow_weight_matrix = np.zeros((num_detectors, num_sensitivities))
    vol_pow_weight_matrix = np.zeros((num_detectors, num_sensitivities))
    los_rad_weight_matrix = np.zeros((num_detectors, num_sensitivities))
    vol_rad_weight_matrix = np.zeros((num_detectors, num_sensitivities))

    detector_id = 0
    for camera in cameras:
        for detector in camera:
            if detector.detector_id not in excluded_detectors:
                los_radiance_sensitivity = detector._los_radiance_sensitivity.sensitivity
                vol_power_sensitivity = detector._volume_power_sensitivity.sensitivity
                vol_radiance_sensitivity = detector._volume_radiance_sensitivity.sensitivity

                l_los = los_radiance_sensitivity.sum()
                l_vol = vol_power_sensitivity.sum()
                los_to_vol_factor = l_vol / l_los
                los_pow_weight_matrix[detector_id, :] = los_radiance_sensitivity * los_to_vol_factor
                vol_pow_weight_matrix[detector_id, :] = vol_power_sensitivity[:]

                los_rad_weight_matrix[detector_id, :] = los_radiance_sensitivity
                vol_rad_weight_matrix[detector_id, :] = vol_radiance_sensitivity

                detector_id += 1

    return detector_keys, los_pow_weight_matrix, vol_pow_weight_matrix, los_rad_weight_matrix, vol_rad_weight_matrix
