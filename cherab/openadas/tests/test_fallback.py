import unittest

from cherab.core.atomic.elements import helium, hydrogen, carbon
from cherab.openadas import OpenADAS


class TestRateFallback(unittest.TestCase):

    def test_rates_fallback(self):

        atomic_data = OpenADAS(permit_extrapolation=True,
                               ionisation_rate_fallback_value=0)
        rate = atomic_data.ionisation_rate(helium, 1)
        self.assertEqual(rate(0.9 * rate.density_range[0], 0.9 * rate.temperature_range[0]), 0)

        atomic_data = OpenADAS(permit_extrapolation=True,
                               recombination_rate_fallback_value=0)
        rate = atomic_data.recombination_rate(helium, 1)
        self.assertEqual(rate(0.9 * rate.density_range[0], 0.9 * rate.temperature_range[0]), 0)

        atomic_data = OpenADAS(permit_extrapolation=True,
                               thermal_cx_rate_fallback_value=0)
        rate = atomic_data.thermal_cx_rate(hydrogen, 0, carbon, 6)
        self.assertEqual(rate(0.9 * rate.density_range[0], 0.9 * rate.temperature_range[0]), 0)

        atomic_data = OpenADAS(permit_extrapolation=True,
                               beam_cx_pec_fallback_value=0)
        rate = atomic_data.beam_cx_pec(hydrogen, carbon, 6, (8, 7))
        for rt in rate:
            self.assertEqual(rt(0.9 * rt.beam_energy_range[0],
                                0.9 * rt.temperature_range[0],
                                0.9 * rt.density_range[0],
                                0.9 * rt.zeff_range[0],
                                0.9 * rt.b_field_range[0]
                                ), 0)

        atomic_data = OpenADAS(permit_extrapolation=True,
                               beam_stopping_rate_fallback_value=0)
        rate = atomic_data.beam_stopping_rate(hydrogen, carbon, 6)
        self.assertEqual(rate(0.9 * rate.beam_energy_range[0],
                              0.9 * rate.density_range[0],
                              0.9 * rate.temperature_range[0]), 0)
        
        atomic_data = OpenADAS(permit_extrapolation=True,
                               beam_population_rate_fallback_value=0)
        rate = atomic_data.beam_population_rate(hydrogen, 2, carbon, 6)
        self.assertEqual(rate(0.9 * rate.beam_energy_range[0],
                              0.9 * rate.density_range[0],
                              0.9 * rate.temperature_range[0]), 0)

        atomic_data = OpenADAS(permit_extrapolation=True,
                               beam_emission_pec_fallback_value=0)
        rate = atomic_data.beam_emission_pec(hydrogen, carbon, 6, (3, 2))
        self.assertEqual(rate(0.9 * rate.beam_energy_range[0],
                              0.9 * rate.density_range[0],
                              0.9 * rate.temperature_range[0]), 0)

        atomic_data = OpenADAS(permit_extrapolation=True,
                               impact_excitation_pec_fallback_value=0)
        rate =  atomic_data.impact_excitation_pec(hydrogen, 0, (3, 2))
        self.assertEqual(rate(0.9 * rate.density_range[0], 0.9 * rate.temperature_range[0]), 0)

        atomic_data = OpenADAS(permit_extrapolation=True,
                               recombination_pec_fallback_value=0)
        rate = atomic_data.recombination_pec(hydrogen, 0, (3, 2))
        self.assertEqual(rate(0.9 * rate.density_range[0], 0.9 * rate.temperature_range[0]), 0)

        atomic_data = OpenADAS(permit_extrapolation=True,
                               line_radiated_power_fallback_value=0)
        rate = atomic_data.line_radiated_power_rate(helium, 1)
        self.assertEqual(rate(0.9 * rate.density_range[0], 0.9 * rate.temperature_range[0]), 0)

        atomic_data = OpenADAS(permit_extrapolation=True,
                               continuum_radiated_power_fallback_value=0)
        rate = atomic_data.continuum_radiated_power_rate(helium, 1)
        self.assertEqual(rate(0.9 * rate.density_range[0], 0.9 * rate.temperature_range[0]), 0)

        atomic_data = OpenADAS(permit_extrapolation=True,
                               cx_radiated_power_fallback_value=0)
        rate = atomic_data.cx_radiated_power_rate(helium, 1)
        self.assertEqual(rate(0.9 * rate.density_range[0], 0.9 * rate.temperature_range[0]), 0)
