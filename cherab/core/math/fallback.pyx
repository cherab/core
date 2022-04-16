from libc.math cimport INFINITY

from cherab.core.math.function cimport autowrap_function1d, autowrap_function2d, autowrap_function3d

cdef class OutofRangeFallback1D(Function1D):
    """
    Sets function value outside of (xmin, xmax) interval to a fallback constant.

    Wraps a Function1D and returns constant fallback outside of the specified interval
    [xmin, xmax]. A single limit can be used.

    :param Function1D f: A 1D function to be wrapped
    :param double fallback: Function value returned outside [xmin, xmax] interval
    :param double xmin: Optional, the lower bound of the x interval
    :param double xmax: Optional, the upper bound of the x interval
    :return: function or fallback value

    .. code-block:: pycon

        >>> from cherab.core.math import OutofRangeFallback1D
        >>>
        >>> def f1d(x): return x
        >>> fallback = OutofRangeFallback1D(f1d, 333., -10, 10)
        >>> fallback(1)
        1
        >>> fallback(-11)
        333.
        >>> fallback(11)
        333.
    """
    def __init__(self,object f, double fallback, double xmin=-INFINITY,
                 double xmax=INFINITY):

        if xmin >= xmax:
            raise ValueError("min value has to be smaller then max value")

        self._fallback = fallback

        self._xmin = xmin
        self._xmax = xmax
        self._f = autowrap_function1d(f)
        
    cdef double evaluate(self, double x) except? -1e999:

        if not self._xmin <= x <= self._xmax:
            return self._fallback
        
        return self._f.evaluate(x)


cdef class OutofRangeFallback2D(Function2D):
    """
    Sets function value outside of (xmin, xmax) or (ymin, ymax) interval to a fallback constant.

    Wraps a Function2D and returns constant fallback outside of the specified interval
    [xmin, xmax] or [ymin, ymax]. A single limit can be used.

    :param Function2D f: A 2D function to be wrapped
    :param double fallback: Function value returned outside [xmin, xmax] or [ymin, ymax] interval
    :param double xmin: Optional, the lower bound of the x interval
    :param double xmax: Optional, the upper bound of the x interval
    :param double ymin: Optional, the lower bound of the y interval
    :param double ymax: Optional, the upper bound of the y interval
    :return: function or fallback value

    .. code-block:: pycon

        >>> from cherab.core.math import OutofRangeFallback2D
        >>>
        >>> def f2d(x, y): return x + y
        >>> fallback = OutofRangeFallback1D(f2d, 333., -10, 20, -32, 45)
        >>> fallback(1, 45)
        46
        >>> fallback(-11, 1)
        333.
        >>> fallback(1, 46)
        333.
    """
    def __init__(self,object f, double fallback, double xmin=-INFINITY, double xmax=INFINITY,
                 double ymin=-INFINITY, double ymax=INFINITY):

        if xmin >= xmax or ymin >= ymax:
            raise ValueError("min value has to be smaller then max value")

        self._fallback = fallback

        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax
        self._f = autowrap_function2d(f)

    cdef double evaluate(self, double x, double y) except? -1e999:

        if not self._xmin <= x <= self._xmax or not self._ymin <= y <= self._ymax:
            return self._fallback
        
        return self._f.evaluate(x, y)

cdef class OutofRangeFallback3D(Function3D):
    """
    Sets function value outside of (xmin, xmax), (ymin, ymax) or (zmin, zmax) interval
    to a fallback constant.

    Wraps a Function3D and returns constant fallback outside of the specified interval
    [xmin, xmax], [ymin, ymax], [zmin, zmax]. A single limit can be used.

    :param Function3D f: A 3D function to be wrapped
    :param double fallback: Function value returned outside [xmin, xmax], [ymin, ymax]
      or [zmin, zmax] interval
    :param double xmin: Optional, the lower bound of the x interval
    :param double xmax: Optional, the upper bound of the x interval
    :param double ymin: Optional, the lower bound of the y interval
    :param double ymax: Optional, the upper bound of the y interval
    :param double zmin: Optional, the lower bound of the z interval
    :param double zmax: Optional, the upper bound of the z interval
    :return: function or fallback value

    .. code-block:: pycon

        >>> from cherab.core.math import OutofRangeFallback3D
        >>>
        >>> def f3d(x, y, z): return x + y + z
        >>> fallback = OutofRangeFallback1D(f3d, 333., -10, 20, -32, 45, 20, 80)
        >>> fallback(1, 45, 22)
        68.
        >>> fallback(-11, 1, 79)
        333.
        >>> fallback(1, 46, 10)
        333.
    """
    def __init__(self,object f, double fallback, double xmin=-INFINITY, double xmax=INFINITY,
                 double ymin=-INFINITY, double ymax=INFINITY,
                 double zmin=-INFINITY, double zmax=INFINITY):

        if xmin >= xmax or ymin >= ymax or zmin >= zmax:
            raise ValueError("lower bound value has to be smaller then upper bound value")

        self._fallback = fallback

        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax
        self._zmin = zmin
        self._zmax = zmax
        self._f = autowrap_function3d(f)
        

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        
        if not self._xmin <= x <= self._xmax or not self._ymin <= y <= self._ymax or \
           not self._zmin <= z <= self._zmax:
           return self._fallback

        return self._f.evaluate(x, y, z)