import unittest
from unittest.mock import patch

import numpy as np

from raysect.optical import Point3D, Vector3D, Spectrum

from cherab.core.model.plasma import (
    ExcitationLine,
    RecombinationLine,
    TotalRadiatedPower,
)
from cherab.core.atomic import Line, hydrogen
from cherab.core.model import GaussianLine
from cherab.tools.plasmas.slab import build_slab_plasma
from cherab.openadas import OpenADAS


class TestPlasmaModels(unittest.TestCase):
    # make a slab plasma

    plasma = build_slab_plasma(peak_density=5e19)
    plasma.atomic_data = OpenADAS(permit_extrapolation=True)
    balmer_alpha = Line(hydrogen, 0, (3, 2))

    def setUp(self):
        # setup mock to avoid reading the data from the repository
        self.patcher_pec = patch(
            "cherab.openadas.openadas.repository.get_pec_excitation_rate",
            return_value={
                "ne": np.linspace(1e18, 1e20, 10),
                "te": np.linspace(1, 1e3, 12),
                "rate": np.ones((10, 12)),
            },
        )
        self.mock_get_pec = self.patcher_pec.start()

        self.patcher_rec = patch(
            "cherab.openadas.openadas.repository.get_pec_recombination_rate",
            return_value={
                "ne": np.linspace(1e18, 1e20, 10),
                "te": np.linspace(1, 1e3, 12),
                "rate": np.ones((10, 12)),
            },
        )
        self.mock_get_rec = self.patcher_rec.start()

        self.patcher_wl = patch(
            "cherab.openadas.openadas.repository.get_wavelength", return_value=656.28
        )
        self.mock_get_wavelength = self.patcher_wl.start()

    def tearDown(self):
        # stop the mocks after a test is run
        self.patcher_pec.stop()
        self.patcher_rec.stop()
        self.patcher_wl.stop()

    def test_excitation(self):
        exc = ExcitationLine(self.balmer_alpha)
        self.plasma.models = [exc]

        # sample emission to trigger the caching mechanism
        exc.emission(Point3D(0, 0, 0), Vector3D(0, 0, 1), Spectrum(300, 1000, 1000))

        # check exc has the correct line
        self.assertEqual(exc.line, self.balmer_alpha)

        # check exc has the correct lineshape
        self.assertIsInstance(exc.lineshape, GaussianLine)

        # check the mock was called
        self.mock_get_pec.assert_called_once()
        self.assertEqual(self.mock_get_wavelength.call_count, 2)

    def test_recombination(self):
        rec = RecombinationLine(self.balmer_alpha)
        self.plasma.models = [rec]

        # sample emission to trigger the caching mechanism
        rec.emission(Point3D(0, 0, 0), Vector3D(0, 0, 1), Spectrum(300, 1000, 1000))

        # check rec has the correct line
        self.assertEqual(rec.line, self.balmer_alpha)

        # check rec has the correct lineshape
        self.assertIsInstance(rec.lineshape, GaussianLine)

        # check the mock was called
        self.mock_get_rec.assert_called_once()
        self.assertEqual(self.mock_get_wavelength.call_count, 2)

    def test_total_radiated_power(self):
        trp = TotalRadiatedPower(hydrogen, 0)
        self.plasma.models = [trp]

        # check initialisation
        with self.assertRaises(ValueError):
            TotalRadiatedPower(hydrogen, 2)
        with self.assertRaises(ValueError):
            TotalRadiatedPower(hydrogen, -1)

        # check trp has the correct element and charge
        self.assertEqual(trp.element, hydrogen)
        self.assertEqual(trp.charge, 0)
