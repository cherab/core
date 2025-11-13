import unittest

from cherab.core.atomic import Line, deuterium


class TestLine(unittest.TestCase):

    def test_initialisation(self):
        line = Line(deuterium, 0, (3, 2))
        self.assertEqual(line.element, deuterium)
        self.assertEqual(line.charge, 0)
        self.assertEqual(line.transition, (3, 2))

        # test invalid charge
        with self.assertRaises(ValueError):
            Line(deuterium, 2, (3, 2))
        with self.assertRaises(ValueError):
            Line(deuterium, -1, (3, 2))

    def test_properties(self):
        element = deuterium
        charge = 0
        transition = (3, 2)
        line = Line(element, charge, transition)
        self.assertEqual(line.element, element)
        self.assertEqual(line.charge, charge)
        self.assertEqual(line.transition, transition)