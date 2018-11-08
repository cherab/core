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
cimport numpy as np
cimport cython

from raysect.optical cimport Vector3D, Point2D, new_vector3d
from cherab.core.math cimport Function1D, autowrap_function1d
from cherab.core.math cimport Function2D, autowrap_function2d
from cherab.core.math cimport VectorFunction2D
from cherab.core.math cimport Interpolate1DCubic, Interpolate2DCubic
from cherab.core.math cimport PolygonMask2D
from cherab.core.math cimport IsoMapper2D, AxisymmetricMapper
from cherab.core.math cimport Blend2D, Constant2D


cdef class EFITEquilibrium:
    """
    An object representing an EFIT equilibrium time-slice.

    The EFIT data is interpolated to produced continuous functions of the
    equilibrium attributes, such as the magnetic flux (psi) and magnetic
    field.

    Note: psin_to_r mapping only exists if the psi axis is monotonic.

    :param r: EFIT grid radius axis values (array).
    :param z: EFIT grid height axis values (array).
    :param psi_grid: EFIT psi grid values (array).
    :param float psi_axis: The psi value at the magnetic axis.
    :param float psi_lcfs: The psi value at the LCFS.
    :param Point2D magnetic_axis: The coordinates of the magnetic axis.
    :param Point2D x_points: The list of x-points.
    :param Point2D x_points: The list of strike-points.
    :param f_profile: The current flux profile on psin (2xN array).
    :param float b_vacuum_radius: Vacuum B-field reference radius (in meters).
    :param float b_vacuum_magnitude: Vacuum B-Field magnitude at the reference radius.
    :param lcfs_polygon: A 2xN array of [[x0, ...], [y0, ...]] vertices specifying the LCFS boundary.
    :param limiter_polygon: A 2xN array of [[x0, ...], [y0, ...]] vertices specifying the limiter.
    :param float time: The time stamp of the time-slice (in seconds).
    """

    cdef:
        readonly Function2D psi, psi_normalised
        readonly double psi_axis, psi_lcfs
        readonly tuple r_range, z_range
        readonly Point2D magnetic_axis
        readonly tuple x_points, strike_points
        readonly VectorFunction2D b_field
        readonly Function2D inside_lcfs, inside_limiter
        readonly Function1D psin_to_r
        readonly double time
        readonly np.ndarray lcfs_polygon, limiter_polygon
        readonly np.ndarray psi_data, r_data, z_data
        double _b_vacuum_magnitude, _b_vacuum_radius
        Function1D _f_profile
        Function2D _dpsidr, _dpsidz

    def __init__(self, object r, object z, object psi_grid, double psi_axis, double psi_lcfs,
                 Point2D magnetic_axis not None, object x_points, object strike_points,
                 object f_profile, double b_vacuum_radius, double b_vacuum_magnitude,
                 object lcfs_polygon, object limiter_polygon, double time):

        self.time = time

        # convert types (allows interface to accept any nd array convertible sequence type
        r = np.array(r, dtype=np.float64)
        z = np.array(z, dtype=np.float64)
        psi = np.array(psi_grid, dtype=np.float64)
        f_profile = np.array(f_profile, dtype=np.float64)

        # store raw data
        self.r_data = r
        self.z_data = z
        self.psi_data = psi

        # interpolate poloidal flux grid data
        self.psi = Interpolate2DCubic(r, z, psi_grid)
        self.psi_axis = psi_axis
        self.psi_lcfs = psi_lcfs
        self.psi_normalised = Interpolate2DCubic(r, z, (psi_grid - psi_axis) / (psi_lcfs - psi_axis))

        # store equilibrium attributes
        self.r_range = r.min(), r.max()
        self.z_range = z.min(), z.max()
        self._b_vacuum_magnitude = b_vacuum_magnitude
        self._b_vacuum_radius = b_vacuum_radius
        self._f_profile = Interpolate1DCubic(f_profile[0, :], f_profile[1, :])

        # populate points
        self._process_points(magnetic_axis, x_points, strike_points)

        # populate polygons and inside/outside functions
        self._process_polygons(lcfs_polygon, limiter_polygon, self.psi_normalised)

        # calculate b-field
        dpsi_dr, dpsi_dz = self._calculate_differentials(r, z, psi_grid)
        self.b_field = EFITMagneticField(self.psi_normalised, dpsi_dr, dpsi_dz, self._f_profile, b_vacuum_radius, b_vacuum_magnitude, self.inside_lcfs)

        # generate interpolator to map from psi normalised to outboard major radius
        self._generate_psin_to_r_mapping()

    cpdef object _process_points(self, Point2D magnetic_axis, object x_points, object strike_points):

        x_points = tuple(x_points)
        strike_points = tuple(strike_points)

        # validate x points and strike points
        for point in x_points:
            if not isinstance(point, Point2D):
                raise TypeError('The list of x-points must contain only Point2D objects.')

        for point in strike_points:
            if not isinstance(point, Point2D):
                raise TypeError('The list of strike-points must contain only Point2D objects.')

        self.magnetic_axis = magnetic_axis
        self.x_points = x_points
        self.strike_points = strike_points

    cpdef object _process_polygons(self, object lcfs_polygon, object limiter_polygon, Function2D psi_normalised):

        # lcfs polygon
        # polygon mask requires an Nx2 array and it must be c contiguous
        # transposing simply swaps the indexing, so need to re-instance
        lcfs_polygon = np.ascontiguousarray(lcfs_polygon.transpose())
        self.lcfs_polygon = lcfs_polygon
        self.inside_lcfs = EFITLCFSMask(lcfs_polygon, psi_normalised)

        # limiter polygon
        if limiter_polygon is None:
            self.limiter_polygon = None
            self.inside_limiter = None
        else:
            # polygon mask requires an Nx2 array and it must be c contiguous
            # transposing simply swaps the indexing, so need to re-instance
            limiter_polygon = np.ascontiguousarray(limiter_polygon.transpose())
            self.limiter_polygon = limiter_polygon
            self.inside_limiter = PolygonMask2D(limiter_polygon)

    cpdef tuple _calculate_differentials(self, np.ndarray r, np.ndarray z, np.ndarray psi_grid):

        # calculate differentials
        dpsi_dix, dpsi_diy = np.gradient(psi_grid, edge_order=2)
        dix_dr = 1.0 / np.gradient(r, edge_order=2)
        diy_dz = 1.0 / np.gradient(z, edge_order=2)

        dpsi_dr = Interpolate2DCubic(r, z, dpsi_dix * dix_dr[:, np.newaxis])
        dpsi_dz = Interpolate2DCubic(r, z, dpsi_diy * diy_dz[np.newaxis, :])

        return dpsi_dr, dpsi_dz

    def _generate_psin_to_r_mapping(self):

        SAMPLE_RESOLUTION = 0.005  # sample resolution in meters

        rmin = self.magnetic_axis.x
        rmax = self.r_range[1]
        z = self.magnetic_axis.y
        samples = int((rmax - rmin) / SAMPLE_RESOLUTION)

        # sample from magnetic axis along z=z axis to maximum radius of psi
        r = np.linspace(rmin, rmax, samples)
        psin = np.empty(samples)
        for i, ri in enumerate(r):
            psin[i] = self.psi_normalised(ri, z)

        # check for monotonicity
        if (psin[0] < psin[1] and (np.diff(psin) <= 0).any()) or (psin[0] >= psin[1] and (np.diff(psin) >= 0).any()):
            self.psin_to_r = None
            return

        # interpolate sampled data, allowing a small bit of extrapolation to cope with numerical sampling accuracy
        self.psin_to_r = Interpolate1DCubic(psin, r, extrapolate=True, extrapolation_range=SAMPLE_RESOLUTION, extrapolation_type='quadratic')

    def map2d(self, object profile, double value_outside_lcfs=0.0):
        """
        Maps a 1D profile onto the equilibrium to give a 2D profile.

        :param profile: A 1D function or 2xN array.
        :param value_outside_lcfs: Value returned if point requested outside the LCFS (default=0.0).
        :return: Function2D object.
        """

        # convert data to a 1d function if not already a function object
        if isinstance(profile, Function1D) or callable(profile):
            profile = autowrap_function1d(profile)
        else:
            profile = np.array(profile, np.float64)
            profile = Interpolate1DCubic(profile[0, :], profile[1, :])

        # map around equilibrium
        f = IsoMapper2D(self.psi_normalised, profile)

        # mask off values outside the lcfs
        return Blend2D(Constant2D(value_outside_lcfs), f, self.inside_lcfs)

    def map3d(self, object profile, double value_outside_lcfs=0.0):
        """
        Maps a 1D profile onto the equilibrium to give a 3D profile.

        :param profile: A 1D function or Nx2 array.
        :param value_outside_lcfs: Value returned if point requested outside the LCFS (default=0.0).
        :return: Function3D object.
        """

        return AxisymmetricMapper(self.map2d(profile))


cdef class EFITLCFSMask(Function2D):
    """
    A 2D function that identifies if a point lies inside or outside the plasma LCFS.

    This mask function returns a value of 1 if the requested point lies inside
    the Last Closed Flux Surface (LCFS). A value of 0.0 is returns outside the LCFS.

    :param lcfs_polygon: An Nx2 array of (x, y) vertices specifying the LCFS boundary.
    :param psi_normalised: A 2D function of normalised poloidal flux.
    """

    cdef:
        PolygonMask2D _lcfs_polygon
        Function2D _psi_normalised

    def __init__(self, object lcfs_polygon, object psi_normalised):

        self._lcfs_polygon = PolygonMask2D(lcfs_polygon)
        self._psi_normalised = autowrap_function2d(psi_normalised)

    cdef double evaluate(self, double r, double z) except? -1e999:

        # As the lcfs polygon is a low resolution representation of the plasma boundary,
        # it is possible for regions of the plasma where psin > 1 to be included as inside the lcfs.
        # These points are excluded by checking the psi function directly
        return self._lcfs_polygon.evaluate(r, z) > 0.0 and self._psi_normalised.evaluate(r, z) <= 1.0


cdef class EFITMagneticField(VectorFunction2D):
    """
    A 2D magnetic field vector function derived from EFIT data.

    :param psi_normalised: A 2D function of normalised poloidal flux.
    :param dpsi_dr: A 2D function of the radius differential of poloidal flux.
    :param dpsi_dz: A 2D function of the height differential of poloidal flux.
    :param f_profile: A 1D function containing a current flux profile.
    :param b_vacuum_radius: Vacuum B-field reference radius (in meters).
    :param b_vacuum_magnitude: Vacuum B-Field magnitude at the reference radius.
    :param inside_lcfs: A 2D mask function returning 1 if inside the LCFS and 0 otherwise.
    """

    cdef:
        Function2D _psi_normalised, _dpsi_dr, _dpsi_dz, _inside_lcfs
        Function1D _f_profile
        double _b_vacuum_radius, _b_vacuum_magnitude

    def __init__(self, object psi_normalised, object dpsi_dr, object dpsi_dz, object f_profile, double b_vacuum_radius, double b_vacuum_magnitude, object inside_lcfs):

        self._psi_normalised = autowrap_function2d(psi_normalised)
        self._dpsi_dr = autowrap_function2d(dpsi_dr)
        self._dpsi_dz = autowrap_function2d(dpsi_dz)
        self._f_profile = autowrap_function1d(f_profile)
        self._b_vacuum_radius = b_vacuum_radius
        self._b_vacuum_magnitude = b_vacuum_magnitude
        self._inside_lcfs = autowrap_function2d(inside_lcfs)

    @cython.cdivision(True)
    cdef Vector3D evaluate(self, double r, double z):

        cdef double br, bz, bt, psi_n

        # calculate poloidal components of magnetic field from poloidal flux
        br = -self._dpsi_dz.evaluate(r, z) / r
        bz = self._dpsi_dr.evaluate(r, z) / r

        # when outside the last closed flux surface the toroidal field is the vacuum field
        # inside the last closed flux surface the toroidal field is derived from the current flux function f.
        if self._inside_lcfs.evaluate(r, z):

            # calculate toroidal field from current flux
            psi_n = self._psi_normalised.evaluate(r, z)
            bt = self._f_profile.evaluate(psi_n) / r

        else:

            # note: this is an approximation used by EFIT
            # todo: replace with a more accurate vacuum field calculation (if data available)
            bt = self._b_vacuum_magnitude * self._b_vacuum_radius / r

        return new_vector3d(br, bt, bz)

