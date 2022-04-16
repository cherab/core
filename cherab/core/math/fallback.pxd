from cherab.core.math.function cimport Function1D, Function2D, Function3D


cdef class OutofRangeFallback1D(Function1D):

    cdef:
        Function1D _f
        double _xmin, _xmax, _fallback

cdef class OutofRangeFallback2D(Function2D):

    cdef:
        Function2D _f
        double _xmin, _xmax, _ymin, _ymax, _fallback

cdef class OutofRangeFallback3D(Function3D):

    cdef:
        Function3D _f
        double _xmin, _xmax, _ymin, _ymax, _zmin, _zmax
        double _fallback