

cimport cython
from libc.math cimport exp, sqrt, pow
from scipy.constants import electron_mass, atomic_mass

from raysect.primitive import Box
from raysect.optical import World, Point3D, Vector3D

from cherab.core import Species, Maxwellian, Plasma
from cherab.core.math.function cimport Function3D
from cherab.core.atomic import hydrogen


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

        cdef double value, x_norm

        if self._cache:
            if x == self._cache_x and y == self._cache_y and z == self._cache_z:
                return self._cache_v

        x_norm = x / self.pedestal_top
        if 0 <= x_norm <= 1:
            value = ((self.t_core - self.t_lcfs) *
                    pow((1 - pow((1 - x_norm), self.p)), self.q) + self.t_lcfs)
        elif x_norm >= 1:
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
def build_slab_plasma(length=5, width=1, height=1, peak_density=1e19, peak_temperature=2500,
                      pedestal_top=1, neutral_temperature=0.5, impurities=None,
                      parent=None):
    """
    Constructs a simple slab of plasma.

    The plasma is defined for positive x starting at x = 0, symmetric in y-z. The plasma
    parameters such as electron density and temperature evolve in 1 dimension according
    to the input parameters specified. The slab includes an optional pedestal.

    Raysect cannot handle infinite geometry, so overall spatial dimensions of the slab need
    to be set, [length, width, height]. These can be set very large to make an effectively
    infinite slab of plasma, although the numerical performance will degrade accordingly.
    The dimensions should be set appropriately with valid assumptions for your scenario.

    Impurity species can be included as a list of tuples, where each tuple specifies an
    impurity. The specification format is (species, charge, concentration). For example:

        >>> impurities=[(carbon, 6, 0.005)]

    :param float length: the overall length of the slab along x.
    :param float width: the y width of the slab.
    :param float height: the z height of the slab.
    :param float peak_density: the peak electron density at the pedestal top.
    :param float peak_temperature: the peak electron temperature at the pedestal top.
    :param float pedestal_top: the length of the pedestal top.
    :param float neutral_temperature: the background neutral temperature.
    :param list impurities: an optional list of impurities to include.
    :param parent: the Raysect scene-graph parent node.
    :param atomic_data: the atomic data provider to use for subsequent spectroscopic calculations,
      defaults to atomic_data=OpenADAS(permit_extrapolation=True).

    .. code-block:: pycon

       >>> from raysect.optical import World
       >>> from cherab.core.atomic import carbon
       >>> from cherab.tools.plasmas.slab import build_slab_plasma
       >>>
       >>> plasma = build_slab_plasma(peak_density=5e19, impurities=[(carbon, 6, 0.005)])
       >>> plasma.parent = World()
    """

    plasma = Plasma(parent=parent)
    plasma.geometry = Box(Point3D(0, -width/2, -height/2), Point3D(length, width/2, height/2))

    species = []

    # No net velocity for any species
    zero_velocity = Vector3D(0, 0, 0)

    # define neutral species distribution
    h0_density = NeutralFunction(peak_density, 0.1, pedestal_top=pedestal_top)
    h0_temperature = neutral_temperature
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
    plasma.b_field = Vector3D(0, 0, 0)
    plasma.electron_distribution = e_distribution
    plasma.composition = species

    return plasma


def build_constant_slab_plasma(length=5, width=1, height=1, electron_density=1e19, electron_temperature=2.5e3,
                               plasma_species=None, b_field=Vector3D(0, 0, 0), parent=None):
    """
    Constructs a simple slab of plasma with constant conditions.

    The plasma is defined for positive x starting at x = 0, symmetric in y-z. The plasma
    parameters such as electron density and temperature are constant over the plasma volume.

    Raysect cannot handle infinite geometry, so overall spatial dimensions of the slab need
    to be set, [length, width, height]. These can be set very large to make an effectively
    infinite slab of plasma, although the numerical performance will degrade accordingly.
    The dimensions should be set appropriately with valid assumptions for your scenario.

    Ion species can be included as a list of tuples, where each tuple specifies an
    impurity. The specification format is (species, charge, density, temperature, velocity). For example:

        >>> plasma_species = [(carbon, 6, 1e18, 3.4e3, Vector3D(1.0e3, 0, 0))]

    If omitted, hydrogen distribution with properties equal to electrons is used:
        >>> plasma_species = [(hydrogen, 1, electron_density, electron_temperature, Vector3D(0, 0, 0))]
    
    If an empty list is passed, plasma contains only electrons.

    :param float length: the overall length of the slab along x.
    :param float width: the y width of the slab.
    :param float height: the z height of the slab.
    :param float electron_density: the electron density in m^-3 .
    :param float electron_temperature: the electron temperature in eV.
    :param list plasma_species: an optional list of impurities to include.
    :param Vector3D b_field: vector giving the magnetic field
    :param parent: the Raysect scene-graph parent node.

    .. code-block:: pycon

       >>> from raysect.optical import World
       >>> from cherab.core.atomic import hydrogen, carbon
       >>> from cherab.tools.plasmas.slab import build_constant_slab_plasma
       >>>
       >>> plasma_species = [(hydrogen, 0, 1e19, 3.5e3, Vector3D(5e3, 0, 0)), (carbon, 5, 1e18, 3.4e3, Vector3D(1.0e3, 0, 0))]
       >>> plasma = build_constant_slab_plasma(0.2, 0.5, 0.5, electron_density = 1.19, electron_temperature=4e4, plasma_species=plasma_species, b_field=Vector3D(0, 5, 0))
       >>> plasma.parent = World()
    """

    if plasma_species is None:
        plasma_species = [(hydrogen, 1, electron_density, electron_temperature, Vector3D(0, 0, 0))]

    # create electron distribution
    e_distribution = Maxwellian(electron_density, electron_temperature, Vector3D(0, 0, 0), electron_mass)

    # create ion species
    species = []
    for element, ionisation, density, temperature, velocity in plasma_species:
        imp_distribution = Maxwellian(density, temperature, velocity, element.atomic_weight * atomic_mass)
        species.append(Species(element, ionisation, imp_distribution))

    # create plasma, add particles and magnetic field
    plasma = Plasma(parent=parent)
    plasma.geometry = Box(Point3D(0, -width/2, -height/2), Point3D(length, width/2, height/2))
    plasma.electron_distribution = e_distribution
    plasma.b_field = b_field
    plasma.composition = species

    return plasma
