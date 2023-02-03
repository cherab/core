import unittest

from raysect.core.workflow import RenderEngine
from raysect.optical.observer import Observer0D, SightLine, FibreOptic, Pixel, TargettedPixel, PowerPipeline0D, SpectralPowerPipeline0D
from raysect.primitive import Sphere

from cherab.tools.observers.group.base import Observer0DGroup
from cherab.tools.observers.group import SightLineGroup, FibreOpticGroup, PixelGroup, TargettedPixelGroup
from cherab.tools.raytransfer import pipelines


class Observer0DGroupTestCase(unittest.TestCase):
    _GROUP_CLASS = Observer0DGroup
    _NUM = 3

    def setUp(self):
        ObserverClass = self._GROUP_CLASS._OBSERVER_TYPE
        self.observers = [ObserverClass(pipelines=[PowerPipeline0D()]) for _ in range(self._NUM)]

    def test_get_item(self):
        """Tests all inputs for the __get_item__ method"""
        group = self._GROUP_CLASS(observers=self.observers)
        names = ['zero', 'one', 'two']
        group.names = names

        idx = slice(1, 3, 1)
        for observer, input_observer in zip(group[idx], self.observers[idx]):
            self.assertIs(observer, input_observer)
        
        for i, name in enumerate(names):
            self.assertIs(group[name], self.observers[i])

        with self.assertRaises(IndexError):
            group[len(group)]

        with self.assertRaises(TypeError):
            group[1.2]

        with self.assertRaises(ValueError):
            group['fail']

        group.names = ['fail'] * len(group)
        with self.assertRaises(ValueError):
            group['fail']

    def test_assignments(self):
        """Test assignments of all supported attributes of Observer0DGroup"""
        group = self._GROUP_CLASS()
        group.observers = self.observers

        for grouped_observer, input_observer in zip(group.observers, self.observers):
            self.assertIs(grouped_observer, input_observer, msg='Observers do not match')

        with self.assertRaises(ValueError):
            group.observers = [Sphere()]

        with self.assertRaises(TypeError):
            group.observers = Sphere()

        # names
        names = ['zero', 'one', 'two']
        group.names = names
        for grouped_observer, input_name in zip(group.observers, names):
            self.assertEqual(grouped_observer.name, input_name, msg='Observer name do not match')
        with self.assertRaises(ValueError):
            group.names = ['fail']
        with self.assertRaises(TypeError):
            group.names = 'fail'

        # pipelines
        ppln_0 = PowerPipeline0D(name='pipeline zero, observer zero')
        ppln_1 = PowerPipeline0D(name='pipeline one, observer one')
        ppln_2 = PowerPipeline0D(name='pipeline two, observer two')
        ppln_3 = PowerPipeline0D(name='pipeline three, observer two')

        pipelist = [[ppln_0], [ppln_1], [ppln_2, ppln_3]]
        group.pipelines = pipelist
        self.assertIs(group[0].pipelines[0], ppln_0, 'non matching pipeline')
        self.assertIs(group[1].pipelines[0], ppln_1, 'non matching pipeline')
        self.assertIs(group[2].pipelines[0], ppln_2, 'non matching pipeline')
        self.assertIs(group[2].pipelines[1], ppln_3, 'non matching pipeline')

        with self.assertRaises(ValueError):
            group.pipelines = [ppln_0]

        # render_engine        
        engine = RenderEngine()
        group.render_engine = engine
        for group_engine in group.render_engine:
            self.assertIs(group_engine, engine)

        with self.assertRaises(TypeError):
            group.render_engine = Sphere()

        engines = [RenderEngine() for _ in group.observers]
        group.render_engine = engines
        for group_engine, input_engine in zip(group.render_engine, engines):
            self.assertIs(group_engine, input_engine)

        with self.assertRaises(TypeError):
            group.render_engine = [RenderEngine() for _ in range(len(group) - 1)] + [Sphere()]
        with self.assertRaises(ValueError):
            group.render_engine = [RenderEngine() for _ in range(len(group) - 1)]

        # wavelengths        
        wvl = 500
        group.min_wavelength = wvl - 100
        group.max_wavelength = wvl + 100
        self.assertListEqual(group.min_wavelength, [wvl - 100] * len(group))
        self.assertListEqual(group.max_wavelength, [wvl + 100] * len(group))

        min_wvls = [90 + 10*i for i in range(len(group))]
        max_wvls = [100 + 10*i for i in range(len(group))]
        group.min_wavelength = min_wvls
        group.max_wavelength = max_wvls
        self.assertListEqual(group.min_wavelength, min_wvls)
        self.assertListEqual(group.max_wavelength, max_wvls)

        with self.assertRaises(ValueError):
            group.max_wavelength = [100] * (len(group) - 1)
        with self.assertRaises(ValueError):
            group.min_wavelength = [90] * (len(group) - 1)

        # spectral
        bins = [200 + i*100 for i in range(len(group))]
        rays = [2] * len(group)
        group.spectral_bins = bins
        group.spectral_rays = rays
        self.assertListEqual(group.spectral_bins, bins)

        bins = 300
        rays = 1
        group.spectral_bins = bins
        group.spectral_rays = rays
        for observer in group.observers:
            self.assertEqual(observer.spectral_bins, bins)
            self.assertEqual(observer.spectral_rays, rays)

        with self.assertRaises(ValueError):
            group.spectral_bins = [1000] * (len(group) + 1)

        # quiet        
        quiet = [True] * len(group)
        group.quiet = quiet
        self.assertListEqual(group.quiet, quiet)

        quiet = False
        group.quiet = quiet
        for observer in group.observers:
            self.assertEqual(observer.quiet, quiet)

        with self.assertRaises(ValueError):
            group.quiet = [False] * (len(group) + 1)

        # rays        
        probs = [0.2 + i*0.1 for i in range(len(group))]
        max_depths = [5 + i for i in range(len(group))]
        min_depths = [2 + i for i in range(len(group))]
        sampling = [False] * len(group)
        weights = [0.5 + i * 0.1 for i in range(len(group))]
        group.ray_extinction_prob = probs
        group.ray_max_depth = max_depths
        group.ray_extinction_min_depth = min_depths
        group.ray_importance_sampling = sampling
        group.ray_important_path_weight = weights
        self.assertListEqual(group.ray_extinction_prob, probs)
        self.assertListEqual(group.ray_max_depth, max_depths)
        self.assertListEqual(group.ray_extinction_min_depth, min_depths)
        self.assertListEqual(group.ray_importance_sampling, sampling)
        self.assertListEqual(group.ray_important_path_weight, weights)

        probs = 0.3
        max_depths = 6
        min_depths = 3
        sampling = True
        weights = 0.7
        group.ray_extinction_prob = probs
        group.ray_max_depth = max_depths
        group.ray_extinction_min_depth = min_depths
        group.ray_importance_sampling = sampling
        group.ray_important_path_weight = weights
        for observer in group.observers:
            self.assertEqual(observer.ray_extinction_prob, probs)
            self.assertEqual(observer.ray_max_depth, max_depths)
            self.assertEqual(observer.ray_extinction_min_depth, min_depths)
            self.assertEqual(observer.ray_importance_sampling, sampling)
            self.assertEqual(observer.ray_important_path_weight, weights)

        with self.assertRaises(ValueError):
            group.ray_extinction_prob = [0.5] * (len(group) + 1)
        with self.assertRaises(ValueError):
            group.ray_max_depth = [8] * (len(group) + 1)
        with self.assertRaises(ValueError):
            group.ray_extinction_min_depth = [4] * (len(group) + 1)
        with self.assertRaises(ValueError):
            group.ray_importance_sampling = [False] * (len(group) + 1)
        with self.assertRaises(ValueError):
            group.ray_important_path_weight = [0.7] * (len(group) + 1)
        
        # samples
        pixel_samples = [2000 + i*500 for i in range(len(group))]
        per_task = [5000 + i*100 for i in range(len(group))]
        group.pixel_samples = pixel_samples
        group.samples_per_task = per_task
        self.assertListEqual(group.pixel_samples, pixel_samples)
        self.assertListEqual(group.samples_per_task, per_task)

        pixel_samples = 10000
        per_task = 30000
        group.pixel_samples = pixel_samples
        group.samples_per_task = per_task
        for observer in group.observers:
            self.assertEqual(observer.pixel_samples, pixel_samples)
            self.assertEqual(observer.samples_per_task, per_task)

        with self.assertRaises(ValueError):
            group.pixel_samples = [5000] * (len(group) + 1)
        with self.assertRaises(ValueError):
            group.samples_per_task = [4000] * (len(group) + 1)

    def test_add_observer(self):
        group = self._GROUP_CLASS()
        for i in range(len(group)):
            group.add_observer(observer=self.observers[i])
            self.assertIs(group.observers[i], self.observers[i], "Added observer is not the observer passed")

    def test_connect_pipelines(self):
        group = self._GROUP_CLASS(observers=self.observers)

        ppln_classes = [PowerPipeline0D, SpectralPowerPipeline0D]
        names = ['power', 'spectral']
        keywords = [
            dict(name=names[0]),
            dict(name=names[1], display_progress=True),
        ]

        with self.assertRaises(ValueError):
            group.connect_pipelines(ppln_classes, keywords_list=[{}])

        group.connect_pipelines(ppln_classes)
        for pipelines in group.pipelines:
            for i, pipeline in enumerate(pipelines):
                self.assertIsInstance(pipeline, ppln_classes[i])
            self.assertIs(pipelines[1].display_progress, False)

        group.connect_pipelines(pipeline_classes=ppln_classes, keywords_list=keywords, suppress_display_progress=True)
        for pipelines in group.pipelines:
            for i, pipeline in enumerate(pipelines):
                self.assertIsInstance(pipeline, ppln_classes[i])
                self.assertEqual(pipeline.name, names[i])
            self.assertIs(pipelines[1].display_progress, False)

        group.connect_pipelines(pipeline_classes=ppln_classes, keywords_list=keywords, suppress_display_progress=False)
        for pipelines in group.pipelines:
            for i, pipeline in enumerate(pipelines):
                self.assertIsInstance(pipeline, ppln_classes[i])
                self.assertEqual(pipeline.name, names[i])
            self.assertIs(pipelines[1].display_progress, True)

        keywords2 = [
            dict(name=names[0]),
            dict(name=names[1], display_progress=False),
        ]
        group.connect_pipelines(pipeline_classes=ppln_classes, keywords_list=keywords2, suppress_display_progress=False)
        for pipelines in group.pipelines:
            for i, pipeline in enumerate(pipelines):
                self.assertIsInstance(pipeline, ppln_classes[i])
                self.assertEqual(pipeline.name, names[i])
            self.assertIs(pipelines[1].display_progress, False)


class SightLineGroupTestCase(Observer0DGroupTestCase):
    _GROUP_CLASS = SightLineGroup

    def test_sensitivity(self):
        sensitivities = [0.9, 0.8, 0.7]

        group = SightLineGroup(observers=self.observers)
        group.sensitivity = sensitivities
        self.assertListEqual(group.sensitivity, sensitivities)

        group.sensitivity = 1
        for sightline in group.observers:
            self.assertEqual(sightline.sensitivity, 1)

        with self.assertRaises(ValueError):
            group.sensitivity = [1] * (len(group) + 1)


class FibreOpticTestCase(Observer0DGroupTestCase):
    _GROUP_CLASS = FibreOpticGroup

    def test_radius(self):
        group = self._GROUP_CLASS(observers=self.observers)

        radius = [1e-2 for _ in group.observers]
        group.radius = radius
        self.assertListEqual(group.radius, radius)

        radius = 1e-3
        group.radius = radius
        for group_radius in group.radius:
            self.assertEqual(group_radius, radius)

        with self.assertRaises(ValueError):
            group.radius = [1e-1 for _ in range(len(group) + 1)]

    def test_acceptance_angle(self):
        group = self._GROUP_CLASS(observers=self.observers)

        acceptance_angle = [10] * len(group)
        group.acceptance_angle = acceptance_angle
        self.assertListEqual(group.acceptance_angle, acceptance_angle)

        acceptance_angle = 11
        group.acceptance_angle = acceptance_angle
        for group_acceptance_angle in group.acceptance_angle:
            self.assertEqual(group_acceptance_angle, acceptance_angle)

        with self.assertRaises(ValueError):
            group.acceptance_angle = [12] * (len(group) + 1)


class PixelGroupTestCase(Observer0DGroupTestCase):
    _GROUP_CLASS = PixelGroup

    def test_widths(self):
        group = self._GROUP_CLASS(observers=self.observers)

        x_width = [1e-2 for _ in group.observers]
        y_width = [1e-2 for _ in group.observers]
        group.x_width = x_width
        group.y_width = y_width
        self.assertListEqual(group.x_width, x_width)
        self.assertListEqual(group.y_width, y_width)

        x_width = 1e-3
        y_width = 1e-3
        group.x_width = x_width
        group.y_width = y_width
        for observer in group.observers:
            self.assertEqual(observer.x_width, x_width)
            self.assertEqual(observer.y_width, y_width)

        with self.assertRaises(ValueError):
            group.x_width = [1e-1] * (len(group) + 1)
        with self.assertRaises(ValueError):
            group.y_width = [1e-1] * (len(group) + 1)


class TargettedPixelGroupTestCase(PixelGroupTestCase):
    _GROUP_CLASS = TargettedPixelGroup

    def setUp(self):
        self.observers = [TargettedPixel(targets=[Sphere()], pipelines=[PowerPipeline0D()]) for _ in range(self._NUM)]

    def test_targets(self):
        group = self._GROUP_CLASS(observers=self.observers)

        targets = [Sphere(), Sphere()]
        group.targets = targets
        for observer in group:
            self.assertEqual(len(targets), len(observer.targets))
            for observer_target, input_target in zip(observer.targets, targets):
                self.assertIs(observer_target, input_target)

        targets = [[Sphere()] for _ in group.observers]
        group.targets = targets
        for observer, input_targets in zip(group.observers, targets):
            for group_target, input_target in zip(observer.targets, input_targets):
                self.assertIs(group_target, input_target)

        targets = [[Sphere()] for _ in range(len(group) + 1)]
        with self.assertRaises(ValueError):
            group.targets = targets

        # targetted path prob
        prob = [0.9, 0.95, 1]
        group.targetted_path_prob = prob
        self.assertListEqual(group.targetted_path_prob, prob)

        prob = 0.8
        group.targetted_path_prob = prob
        for group_targetted_path_prob in group.targetted_path_prob:
            self.assertEqual(group_targetted_path_prob, prob)

        with self.assertRaises(ValueError):
            group.targetted_path_prob = [0.7] * (len(group) + 1)
