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

# cython: language_level=3
from cherab.core.utility import Notifier

from cherab.core.species import SpeciesNotFound
from raysect.optical cimport AffineMatrix3D, Vector3D
from raysect.optical.material.emitter.inhomogeneous cimport NumericalIntegrator

from cherab.core.math cimport Function3D, autowrap_function3d
from cherab.core.math cimport VectorFunction3D, autowrap_vectorfunction3d
from cherab.core.distribution cimport DistributionFunction, ZeroDistribution
from cherab.core.plasma.material cimport PlasmaMaterial
cimport cython


cdef class Composition:
    """
    The plasma composition manager.

    Used to control the adding and removing of Species objects from the Plasma object.
    This is because there can only ever be one Species object instance for each plasma
    species of a given element and charge state. Users never instantiate this class
    directly. Its always used indirectly through an instantiated Plasma object.
    """

    def __init__(self):

        self._species = {}
        self.notifier = Notifier()

    def __len__(self):
        return len(self._species)

    def __iter__(self):
        """
        Used to iterate over all the Species objects in the parent plasma.

        .. code-block:: pycon

           >>> [species for species in plasma.composition]
           [<Species: element=deuterium, charge=0>,
            <Species: element=deuterium, charge=1>]
        """

        return iter(self._species.values())

    def __getitem__(self, tuple item):
        """
        Species objects can be indexed with a tuple specifying their element and charge state.

        .. code-block:: pycon

           >>> plasma.composition[(deuterium, 0)]
           <Species: element=deuterium, charge=0>
        """

        try:
            element, charge = item
        except ValueError:
            raise ValueError('An (element, charge) tuple is required containing the element and '
                             'charge state of the species.')
        return self.get(element, charge)

    cpdef object set(self, object species):
        """
        Replaces the species in the composition with a new list of species.

        If there are multiple species with the same element and charge state in
        the list, only the last species with that specification will be added
        to the composition.

        :param Species species: A list containing the new species.
        
        .. code-block:: pycon
        
           >>> d0_species = Species(deuterium, 0, d0_distribution)
           >>> d1_species = Species(deuterium, 1, d1_distribution)
           >>> plasma.composition.set([d0_species, d1_species])
           >>> [species for species in plasma.composition]
           [<Species: element=deuterium, charge=0>,
            <Species: element=deuterium, charge=1>]
        """

        # must be an iterable
        species = tuple(species)

        # check contents of list are species
        for item in species:
            if not isinstance(item, Species):
                raise TypeError('The composition list must consist of only Species objects.')

        # reset composition
        self._species = {}

        for item in species:
            self._species[(item.element, item.charge)] = item
        self.notifier.notify()

    cpdef object add(self, Species species):
        """
        Adds a species to the plasma composition.
        
        Replaces any existing species with the same element and charge
        state already in the composition.
        
        :param Species species: A Species object.
        
        .. code-block:: pycon
        
           >>> d1_species = Species(deuterium, 1, d1_distribution)
           >>> plasma.composition.add(d1_species)
        """

        if not species:
            raise ValueError('Species must not be None type.')

        self._species[(species.element, species.charge)] = species
        self.notifier.notify()

    cpdef Species get(self, Element element, int charge):
        """
        Get a specified plasma species.
        
        Raises a ValueError if the specified species is not found in the composition.
        
        :param Element element: The element object of the requested species.
        :param int charge: The charge state of the requested species.
        :return: The requested Species object.
        
        .. code-block:: pycon

           >>> plasma.composition.get(deuterium, 1)
           <Species: element=deuterium, charge=1>
        """

        try:
            return self._species[(element, charge)]
        except KeyError:
            raise ValueError("Could not find a species with the specified element '{}' and charge {}."
                             "".format(element.name, charge))

    cpdef object clear(self):
        """Removes all Species object instances from the parent plasma."""

        self._species = {}
        self.notifier.notify()


cdef class ModelManager:

    def __init__(self):
        self._models = []
        self.notifier = Notifier()

    def __iter__(self):
        return iter(self._models)

    cpdef object set(self, object models):

        # copy models and test it is an iterable
        models = list(models)

        # check contents of list are plasma models
        for model in models:
            if not isinstance(model, PlasmaModel):
                raise TypeError('The model list must consist of only PlasmaModel objects.')

        self._models = models
        self.notifier.notify()

    cpdef object add(self, PlasmaModel model):

        if not model:
            raise ValueError('Model must not be None type.')

        self._models.append(model)
        self.notifier.notify()

    cpdef object clear(self):
        self._models = []
        self.notifier.notify()


cdef class Plasma(Node):
    """
    A scene-graph object representing a plasma.

    The Cherab Plasma object holds all the properties and state of a plasma
    and can optionally have emission models attached to it.

    To define a Plasma object you need to define the plasma composition,
    magnetic field and electron distribution. The Plasma composition consists
    of a collection of Species objects that define the individual distribution
    functions of specific neutral atoms or ions in the plasma. Each individual
    species can only appear once in the composition. For more information see
    the related objects, Species, Composition, and DistributionFunction. To
    define the magnetic field you must provide a function that returns a
    magnetic field vector at the requested coordinate in the local plasma
    coordinate system.

    The Plasma object is a Raysect scene-graph Node and lives in it's own
    coordinate space. This coordinate space is defined relative to it's parent
    scene-graph object by an AffineTransform. The plasma parameters are defined
    in the Plasma object coordinate space. Models using the plasma object must
    convert any spatial coordinates into plasma space before requesting values
    from the Plasma object.

    While a Plasma object can be used to simply hold and sample plasma properties,
    it can also be used as an emitter in Raysect scenes by attaching geometry
    and emission models. To add emission models you first need to define a
    bounding geometry for the plasma. The geometry is described by a Raysect
    Primitive. The Primitive may be positioned relative to the plasma coordinate
    system by setting the geometry_transform attribute. If no geometry transform
    is set, the Primitive will share the same coordinate system as the
    plasma.

    Once geometry is defined, plasma emission models may be attached to the plasma
    object by either setting the full list of models or adding to the list of
    models. See the ModelManager for more information. The plasma emission models
    must be derived from the PlasmaModel base class.

    Any change to the plasma object including adding/removing of species or models
    will result in a automatic notification being sent to objects that register
    with the Plasma objects' Notifier. All Cherab models and associated scene-graph
    objects such as Beams automatically handle the notifications internally to clear
    cached data. If you need to keep track of plasma changes in your own classes,
    a callback can be registered with the plasma Notifier which will be called in
    the event of a change to the plasma object. See the Notifier documentation.

    :param Node parent: The parent node in the Raysect scene-graph.
      See the Raysect documentation for more guidance.
    :param AffineMatrix3D transform: The transform defining the spatial position
      and orientation of this plasma. See the Raysect documentation if you need
      guidance on how to use AffineMatrix3D transforms.
    :param str name: The name for this plasma.
    :param VolumeIntegrator integrator: The configurable method for doing
      volumetric integration through the plasma along a Ray's path. Defaults to
      a numerical integrator with 1mm step size, NumericalIntegrator(step=0.001).

    :ivar AtomicData atomic_data: The atomic data provider class for this plasma.
      All plasma emission from this plasma will be calculated with the same provider.
    :ivar VectorFunction3D b_field: A vector function in 3D space that returns the
      magnetic field vector at any requested point.
    :ivar Composition composition: The composition object manages all the atomic plasma
      species and provides access to their distribution functions.
    :ivar DistributionFunction electron_distribution: A distribution function object
      describing the electron species properties.
    :ivar Primitive geometry: The Raysect primitive that defines the geometric extent
      of this plasma.
    :ivar AffineMatrix3D geometry_transform: The relative difference between the plasmas'
      local coordinate system and the bounding geometries' local coordinate system. Defaults
      to a Null transform.
    :ivar ModelManager models: The manager class that sets and provides access to the
      emission models for this plasma.


    .. code-block:: pycon

       >>> # This example shows how to initialise and populate a basic plasma
       >>>
       >>> from scipy.constants import atomic_mass, electron_mass
       >>> from raysect.core.math import Vector3D
       >>> from raysect.primitive import Sphere
       >>> from raysect.optical import World
       >>>
       >>> from cherab.core import Plasma, Species, Maxwellian
       >>> from cherab.core.atomic import deuterium
       >>> from cherab.openadas import OpenADAS
       >>>
       >>>
       >>> world = World()
       >>>
       >>> # create atomic data source
       >>> adas = OpenADAS(permit_extrapolation=True)
       >>>
       >>>
       >>> # Setup basic distribution functions for the species
       >>> d0_density = 1E17
       >>> d0_temperature = 1
       >>> bulk_velocity = Vector3D(0, 0, 0)
       >>> d0_distribution = Maxwellian(d0_density, d0_temperature, bulk_velocity, deuterium.atomic_weight * atomic_mass)
       >>> d0_species = Species(deuterium, 0, d0_distribution)
       >>>
       >>> d1_density = 1E18
       >>> d1_temperature = 10
       >>> d1_distribution = Maxwellian(d1_density, d1_temperature, bulk_velocity, deuterium.atomic_weight * atomic_mass)
       >>> d1_species = Species(deuterium, 1, d1_distribution)
       >>>
       >>> e_distribution = Maxwellian(1E18, 9.0, bulk_velocity, electron_mass)
       >>>
       >>> # Initialise Plasma object and populate with species specifications
       >>> plasma = Plasma(parent=world)
       >>> plasma.atomic_data = adas
       >>> plasma.geometry = Sphere(2.0)
       >>> plasma.b_field = Vector3D(1.0, 1.0, 1.0)
       >>> plasma.composition = [d0_species, d1_species]
       >>> plasma.electron_distribution = e_distribution
    """

    def __init__(self, object parent=None, AffineMatrix3D transform=None, str name=None,
                 integrator=NumericalIntegrator(step=0.001)):

        super().__init__(parent, transform, name)

        # plasma modification notifier
        self.notifier = Notifier()

        # plasma properties
        self.b_field = None
        self.electron_distribution = None

        # setup plasma composition handler and pass through notifications
        self._composition = Composition()
        self._composition.notifier.add(self._modified)

        # atomic data source passed to emission models
        self._atomic_data = None

        # plasma geometry
        self._geometry = None
        self._geometry_transform = None

        # setup emission model handler and trigger geometry rebuilding if the models change
        self._models = ModelManager()
        self._models.notifier.add(self._configure_geometry)

        # emission model integrator
        self._integrator = integrator

    @property
    def b_field(self):
        return self._b_field

    @b_field.setter
    def b_field(self, object value):
        # assign Vector3D(0, 0, 0) if None is passed
        if value is None:
            self._b_field = autowrap_vectorfunction3d(Vector3D(0, 0, 0))
        else:
            self._b_field = autowrap_vectorfunction3d(value)

        self._modified()

    # cython fast access
    cdef VectorFunction3D get_b_field(self):
        return self._b_field

    @property
    def electron_distribution(self):
        return self._electron_distribution

    @electron_distribution.setter
    def electron_distribution(self, DistributionFunction value):
        # assign ZeroDistribution if None value passed
        if value is None:
            self._electron_distribution = ZeroDistribution()
        else:
            self._electron_distribution = value

        self._modified()

    # cython fast access
    cdef DistributionFunction get_electron_distribution(self):
        return self._electron_distribution

    @property
    def composition(self):
        return self._composition

    @composition.setter
    def composition(self, object values):
        self._composition.set(values)

    # cython fast access
    cdef Composition get_composition(self):
        return self._composition

    @cython.cdivision(True)
    cpdef double z_effective(self, double x, double y, double z) except -1:
        """
        Calculates the effective Z of the plasma.

        .. math::
            Z_{eff} = \sum_{j=1}^N n_{i(j)} Z_j^2 / \sum_{k=1}^N n_{i(k)} Z_k

        where n is the species density and Z is the ionisation of the species.

        :param x: x coordinate in meters.
        :param y: y coordinate in meters.
        :param z: z coordinate in meters.
        :return: Calculated Z effective.
        :raises ValueError: If plasma does not contain any ionised species.
        
        .. code-block:: pycon
           
           >>> # With an already initialised plasma object...
           >>> plasma.z_effective(1, 1, 1)
           1.0
        """

        cdef:
            double ion_density, sum_nz, sum_nz2
            Species species

        sum_nz = 0
        sum_nz2 = 0
        for species in self._composition:
            if species.charge > 0:
                density = species.distribution.density(x, y, z)
                sum_nz += density * species.charge
                sum_nz2 += density * species.charge * species.charge

        if sum_nz2 == 0:
            raise ValueError('Plasma does not contain any ionised species.')

        return sum_nz2 / sum_nz

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double ion_density(self, double x, double y, double z):
        """
        Calculates the total ion density of the plasma.

        .. math::
            n_I = \sum_{k=1}^N n_i(k)

        :param x: x coordinate in meters.
        :param y: y coordinate in meters.
        :param z: z coordinate in meters.
        :return: Total ion density in m^-3.
        
        .. code-block:: pycon
           
           >>> # With an already initialised plasma object...
           >>> plasma.ion_density(1, 1, 1)
           1.1e+18
        """

        cdef:
            double ion_density = 0.0
            Species species

        for species in self._composition:
            ion_density += species.distribution.density(x, y, z)
        return ion_density

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, Primitive value):
        self._geometry = value
        self._configure_geometry()

    @property
    def geometry_transform(self):
        return self._geometry_transform

    @geometry_transform.setter
    def geometry_transform(self, AffineMatrix3D value):
        self._geometry_transform = value
        self._configure_geometry()

    @property
    def integrator(self):
        return self._integrator

    @integrator.setter
    def integrator(self, VolumeIntegrator value):
        self._integrator = value
        self._configure_geometry()

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, object values):

        # setting the emission models causes ModelManager to notify the Plasma object to configure geometry
        # so no need to explicitly rebuild here
        self._models.set(values)

    @property
    def atomic_data(self):
        return self._atomic_data

    @atomic_data.setter
    def atomic_data(self, AtomicData value):
        self._atomic_data = value
        self._configure_geometry()

    def _configure_geometry(self):

        # detach existing geometry
        # take a copy of self.children as it will be modified when unparenting
        children = self.children.copy()
        for child in children:
            child.parent = None

        # no further work if there are no emission models
        if not list(self._models):
            return

        # check necessary data is available
        if not self._geometry:
            raise ValueError('The plasma must have a defined geometry to be used with an emission model.')

        if not self._atomic_data:
            raise ValueError('The plasma must have an atomic data source to be used with an emission model.')

        # attach geometry to plasma
        self._geometry.parent = self
        self._geometry.name = 'Plasma Geometry'

        # transform geometry if geometry transform present
        if self._geometry_transform:
            self._geometry.transform = self._geometry_transform
            local_to_plasma = self._geometry.to(self)
        else:
            self._geometry.transform = AffineMatrix3D()
            local_to_plasma = None

        # build plasma material
        self._geometry.material = PlasmaMaterial(self, self._atomic_data, list(self._models), self.integrator, local_to_plasma)

    def _modified(self):
        """
        Called when a scene-graph change occurs that modifies this Node's root
        transforms. This will occur if the Node's transform is modified, a
        parent node transform is modified or if the Node's section of scene-
        graph is re-parented.
        """

        # plasma section of the scene-graph has been modified, alert dependents
        self.notifier.notify()
