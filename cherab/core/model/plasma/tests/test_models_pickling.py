import pickle
import unittest

from raysect.optical import Point3D, Vector3D, World
from raysect.optical.ray import Ray

from cherab.core.atomic import Line
from cherab.core.atomic.elements import deuterium, helium
from cherab.core.model import (Bremsstrahlung, ExcitationLine,
                               MultipletLineShape, RecombinationLine,
                               TotalRadiatedPower)
from cherab.openadas import OpenADAS
from cherab.tools.plasmas.slab import build_constant_slab_plasma


class TestSpecies(unittest.TestCase):

    def setUp(self):

        # set up spectral lines and shape arguments
        self.line_d = Line(deuterium, 0, (3, 2))
        self.line_he = Line(helium, 0, ("1s1 3d1 1d2.0", "1s1 2p1 1p1.0"))
        self.lineshape_args = [[[668.0, 664.0, 660.0],
                               [0.5, 0.3, 0.2]]]

        # set up plasma population properties
        self.ti_d = 1e3
        self.ni_d = 1e19
        self.v_d = Vector3D(0, 0, 0)

        self.ti_he = 1e3
        self.ni_he = 5e19
        self.v_he = Vector3D(0, 0, 0)
        self.te = 10

        # set up radiation models
        self.models = [ExcitationLine(self.line_d),
                       RecombinationLine(self.line_d),
                       ExcitationLine(self.line_he, lineshape=MultipletLineShape,
                                      lineshape_args=self.lineshape_args),
                       RecombinationLine(self.line_he, lineshape=MultipletLineShape,
                                         lineshape_args=self.lineshape_args),
                       TotalRadiatedPower(helium, 0),
                       Bremsstrahlung()
                      ]

        # set up plasma species
        self.plasma_species = [(deuterium, 0, self.ni_d, self.ti_d, self.v_d),
                (deuterium, 1, self.ni_d, self.ti_d, self.v_d),
                (helium, 0, self.ni_he, self.ti_he, self.v_he),
                (helium, 1, self.ni_he, self.ti_he, self.v_he)
                ]

        # set up plasma
        self.world = World()
        self.atomic_data = OpenADAS()
        self.plasma = build_constant_slab_plasma(electron_temperature=self.te,
                                    plasma_species=self.plasma_species)
        self.plasma.atomic_data=self.atomic_data
        self.plasma.models = self.models
        self.plasma.parent = self.world

    def test_pickle_models(self):
        """
        Test pickling a copy of plasma radiation models.

        2 instances of World with 2 separately created plasmas are used. The 
        second plasma models are obtained through pickling of the first plasma's
        radiation models. After observation the pickling is asserted by same vaulues
        of observed spectra.
        """

        #set up rays for observation
        origin = Point3D(-1, 0, 0)
        direction = Vector3D(1, 0, 0)
        ray = Ray(origin, direction, min_wavelength=655, max_wavelength=670, bins=1e3)
        ray_pickled = ray.copy()

        # initialise separete scenegraph
        world = World()
        atomic_data = OpenADAS()

        plasma_species = [(deuterium, 0, self.ni_d, self.ti_d, self.v_d),
                        (deuterium, 1, self.ni_d, self.ti_d, self.v_d),
                        (helium, 0, self.ni_he, self.ti_he, self.v_he),
                        (helium, 1, self.ni_he, self.ti_he, self.v_he)
                        ]
        plasma = build_constant_slab_plasma(electron_temperature=self.te,
                                            plasma_species=plasma_species)
        plasma.atomic_data=atomic_data
        plasma.parent = world

        # use unpickled-pickled copy of the self.plasma.models
        dumps = pickle.dumps(list(self.plasma.models))
        plasma.models = pickle.loads(dumps)

        spect = ray.trace(self.world)
        spect_pickled = ray_pickled.trace(world)

        self.assertEqual((spect.samples - spect_pickled.samples).max(), 0,
                           msg="Observed spectra have to be equal.")