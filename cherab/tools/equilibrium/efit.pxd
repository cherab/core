
from raysect.optical cimport Vector3D, Point2D
from cherab.core.math cimport Function1D
from cherab.core.math cimport Function2D
from cherab.core.math cimport VectorFunction2D
from cherab.core.math cimport PolygonMask2D

cimport numpy as np


cdef class EFITEquilibrium:


    cdef:
        readonly Function2D psi, psi_normalised
        readonly double psi_axis, psi_lcfs
        readonly tuple r_range, z_range
        readonly Point2D magnetic_axis
        readonly tuple x_points, strike_points
        readonly VectorFunction2D b_field, toroidal_vector, poloidal_vector, surface_normal
        readonly Function2D inside_lcfs, inside_limiter
        readonly Function1D psin_to_r
        readonly double time
        readonly np.ndarray lcfs_polygon, limiter_polygon
        readonly np.ndarray psi_data, r_data, z_data
        readonly Function1D q
        double _b_vacuum_magnitude, _b_vacuum_radius
        Function1D _f_profile
        Function2D _dpsidr, _dpsidz


    cpdef object _process_points(self, Point2D magnetic_axis, object x_points, object strike_points)

    cpdef object _process_polygons(self, object lcfs_polygon, object limiter_polygon, Function2D psi_normalised)

    cpdef tuple _calculate_differentials(self, np.ndarray r, np.ndarray z, np.ndarray psi_grid)


cdef class EFITLCFSMask(Function2D):

    cdef:
        PolygonMask2D _lcfs_polygon
        Function2D _psi_normalised

cdef class MagneticField(VectorFunction2D):

    cdef:
        Function2D _psi_normalised, _dpsi_dr, _dpsi_dz, _inside_lcfs
        Function1D _f_profile
        double _b_vacuum_radius, _b_vacuum_magnitude

cdef class PoloidalFieldVector(VectorFunction2D):

   cdef VectorFunction2D _field

cdef class FluxSurfaceNormal(VectorFunction2D):

    cdef VectorFunction2D _field


cdef class FluxCoordToCartesian(VectorFunction2D):

    cdef:
        VectorFunction2D _field
        Function1D _toroidal, _poloidal, _normal
        Function2D _psin
        Vector3D _value_outside_lcfs

