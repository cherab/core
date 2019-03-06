

cimport cython
from libc.math cimport exp, sqrt, pow
from scipy.constants import electron_mass, atomic_mass

from raysect.primitive import Box
from raysect.optical import World, Point3D, Vector3D

from cherab.core import Species, Maxwellian, Plasma
from cherab.core.math import Constant3D, ConstantVector3D
from cherab.core.math.function cimport Function3D
from cherab.core.atomic import hydrogen
from cherab.openadas import OpenADAS


cdef class NeutralFunction(Function3D):
    """A neutral profile that is constant outside the plasma, then exponentially decays
       inside the plasma boundary."""

    cdef double peak, sigma, pedestal_top
    cdef double _constant
    cdef bint _cache
    cdef double _cache_x, _cache_y, _cache_z, _cache_v

    def __init__(self, double peak_value, double sigma, double pedestal_top=1):

        self.peak = peak_value
        self.sigma = sigma
        self.pedestal_top = pedestal_top
        self._constant = (2*self.sigma*self.sigma)

        # last value cache
        self._cache = False
        self._cache_x = 0.0
        self._cache_y = 0.0
        self._cache_z = 0.0
        self._cache_v = 0.0

    @cython.cdivision(True)
    cdef double evaluate(self, double x, double y, double z) except? -1e999:

        cdef double value

        if self._cache:
            if x == self._cache_x and y == self._cache_y and z == self._cache_z:
                return self._cache_v

        if x >= 0:
            value = self.peak * exp(-(x**2) / self._constant)
        else:
            value = self.peak

        self._cache = True
        self._cache_x = x
        self._cache_y = y
        self._cache_z = z
        self._cache_v = value

        return value


cdef class IonFunction(Function3D):
    """An approximate pedestal plasma profile that follows a double
       quadratic between the plasma boundary and the pedestal top."""

    cdef double t_core, t_lcfs, pedestal_top, p, q
    cdef bint _cache
    cdef double _cache_x, _cache_y, _cache_z, _cache_v

    def __init__(self, double t_core, double t_lcfs, double p=2, double q=2, double pedestal_top=1):

        self.t_core = t_core
        self.t_lcfs = t_lcfs
        self.p = p
        self.q = q
        self.pedestal_top = pedestal_top

        # last value cache
        self._cache = False
        self._cache_x = 0.0
        self._cache_y = 0.0
        self._cache_z = 0.0
        self._cache_v = 0.0

    @cython.cdivision(True)
    cdef double evaluate(self, double x, double y, double z) except? -1e999:

        cdef double value

        if self._cache:
            if x == self._cache_x and y == self._cache_y and z == self._cache_z:
                return self._cache_v

        if 0 <= x <= self.pedestal_top:
            value = ((self.t_core - self.t_lcfs) *
                    pow((1 - pow((1-x) / self.pedestal_top, self.p)), self.q) + self.t_lcfs)
        elif x >= self.pedestal_top:
            value = self.t_core
        else:
            value = 0.0

        self._cache = True
        self._cache_x = x
        self._cache_y = y
        self._cache_z = z
        self._cache_v = value

        return value


# TODO - replace with ionisation balance calculations
def build_slab_plasma(width=1, length=5, height=1, peak_density=1e19, peak_temperature=2500,
                      pedestal_top=1, neutral_temperature=0.5, impurities=None,
                      world=None, atomic_data=OpenADAS(permit_extrapolation=True)):

    plasma = Plasma(parent=world)
    plasma.atomic_data = atomic_data
    plasma.geometry = Box(Point3D(0, -width/2, -height/2), Point3D(length, width/2, height/2))

    species = []

    # No net velocity for any species
    zero_velocity = ConstantVector3D(Vector3D(0, 0, 0))

    # define neutral species distribution
    h0_density = NeutralFunction(peak_density, 0.1, pedestal_top=pedestal_top)
    h0_temperature = Constant3D(neutral_temperature)
    h0_distribution = Maxwellian(h0_density, h0_temperature, zero_velocity,
                                 hydrogen.atomic_weight * atomic_mass)
    species.append(Species(hydrogen, 0, h0_distribution))

    # define hydrogen ion species distribution
    h1_density = IonFunction(peak_density, 0, pedestal_top=pedestal_top)
    h1_temperature = IonFunction(peak_temperature, 0, pedestal_top=pedestal_top)
    h1_distribution = Maxwellian(h1_density, h1_temperature, zero_velocity,
                                 hydrogen.atomic_weight * atomic_mass)
    species.append(Species(hydrogen, 1, h1_distribution))

    # add impurities
    if impurities:
        for impurity, ionisation, concentration in impurities:
            imp_density = IonFunction(peak_density * concentration, 0, pedestal_top=pedestal_top)
            imp_temperature = IonFunction(peak_temperature, 0, pedestal_top=pedestal_top)
            imp_distribution = Maxwellian(imp_density, imp_temperature, zero_velocity,
                                         impurity.atomic_weight * atomic_mass)
            species.append(Species(impurity, ionisation, imp_distribution))

    # define the electron distribution
    e_density = IonFunction(peak_density, 0, pedestal_top=pedestal_top)
    e_temperature = IonFunction(peak_temperature, 0, pedestal_top=pedestal_top)
    e_distribution = Maxwellian(e_density, e_temperature, zero_velocity, electron_mass)

    # define species
    plasma.b_field = ConstantVector3D(Vector3D(0, 0, 0))
    plasma.electron_distribution = e_distribution
    plasma.composition = species

    return plasma
