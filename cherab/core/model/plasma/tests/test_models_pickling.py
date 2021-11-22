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

        # set up spectral lines and shape arguments
        line_d = Line(deuterium, 0, (3, 2))
        line_he = Line(helium, 0, ("1s1 3d1 1d2.0", "1s1 2p1 1p1.0"))
        lineshape_args = [[[668.0, 664.0, 660.0],
                               [0.5, 0.3, 0.2]]]

        # set up plasma population properties
        ti_d = 1e3
        ni_d = 1e19
        v_d = Vector3D(0, 0, 0)

        ti_he = 1e3
        ni_he = 5e19
        v_he = Vector3D(0, 0, 0)
        te = 10

        # set up radiation models
        models = [ExcitationLine(line_d),
                       RecombinationLine(line_d),
                       ExcitationLine(line_he, lineshape=MultipletLineShape,
                                      lineshape_args=lineshape_args),
                       RecombinationLine(line_he, lineshape=MultipletLineShape,
                                         lineshape_args=lineshape_args),
                       TotalRadiatedPower(helium, 0),
                       Bremsstrahlung()
                      ]

        # set up plasma species
        plasma_species = [(deuterium, 0, ni_d, ti_d, v_d),
                (deuterium, 1, ni_d, ti_d, v_d),
                (helium, 0, ni_he, ti_he, v_he),
                (helium, 1, ni_he, ti_he, v_he)
                ]

        # set up plasma
        world = World()
        atomic_data = OpenADAS()
        plasma = build_constant_slab_plasma(electron_temperature=te,
                                    plasma_species=plasma_species)
        plasma.atomic_data = atomic_data
        plasma.models = models
        plasma.parent = world

        # pickle plasma's radiation models
        model_pickle = pickle.dumps(list(plasma.models))

        # remove objects to test weakrefs
        del world
        del plasma
        del atomic_data
        del plasma_species
        del models

        with self.assertRaises(UnboundLocalError, msg="not all objects were successfully removed"):
            world
            plasma
            atomic_data
            plasma_species
            models

        # initialise separete scenegraph
        world = World()
        atomic_data = OpenADAS()

        plasma_species = [(deuterium, 0, ni_d, ti_d, v_d),
                        (deuterium, 1, ni_d, ti_d, v_d),
                        (helium, 0, ni_he, ti_he, v_he),
                        (helium, 1, ni_he, ti_he, v_he)
                        ]
        plasma = build_constant_slab_plasma(electron_temperature=te,
                                            plasma_species=plasma_species)
        plasma.atomic_data=atomic_data
        plasma.parent = world

        # use unpickle copy of the self.plasma.models
        plasma.models = pickle.loads(model_pickle)

        spect = ray.trace(world)
        spect_pickled = ray_pickled.trace(world)

        self.assertEqual((spect.samples - spect_pickled.samples).max(), 0,
                           msg="Observed spectra have to be equal.")
                           