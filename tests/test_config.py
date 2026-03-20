import textwrap
import unittest
import yaml

from gerd.__main__ import config_to_dataset_params, _transform_kwargs


def parse(yaml_str):
    return yaml.safe_load(textwrap.dedent(yaml_str))


class TestTransformKwargs(unittest.TestCase):
    def test_defaults(self):
        start, velocity = _transform_kwargs(None, "uniform")
        self.assertEqual(start, "uniform")
        self.assertEqual(velocity, 0.0)

    def test_explicit(self):
        start, velocity = _transform_kwargs({"start": 0.5, "velocity": 0.1}, None)
        self.assertEqual(start, 0.5)
        self.assertEqual(velocity, 0.1)

    def test_list_to_tuple(self):
        start, velocity = _transform_kwargs({"start": [0.1, 0.2], "velocity": [0.3, 0.4]}, None)
        self.assertEqual(start, (0.1, 0.2))
        self.assertEqual(velocity, (0.3, 0.4))

    def test_null_start(self):
        start, _ = _transform_kwargs({"start": None}, None)
        self.assertIsNone(start)


class TestConfigToDatasetParams(unittest.TestCase):
    def test_minimal_config(self):
        config = parse("resolution: [300, 300]")
        p = config_to_dataset_params(config, event_density=1.0, max_velocity=0.5)
        self.assertEqual(p.resolution, (300, 300))
        self.assertEqual(p.event_density, 1.0)
        self.assertEqual(p.max_velocity, 0.5)
        self.assertEqual(p.translate_start, "uniform")
        self.assertEqual(p.translate_velocity, 0.0)
        self.assertIsNone(p.scale_start)
        self.assertIsNone(p.rotate_start)
        self.assertIsNone(p.shear_start)

    def test_example_yaml(self):
        config = parse("""
            resolution: [300, 300]
            event_densities: [1.0]
            max_velocities: [0.1, 0.5, 1.0]
            translate:
              start: uniform
              velocity: uniform
            scale:
              start: uniform
              velocity: 0.0
        """)
        p = config_to_dataset_params(config, event_density=1.0, max_velocity=0.1)
        self.assertEqual(p.translate_start, "uniform")
        self.assertEqual(p.translate_velocity, "uniform")
        self.assertEqual(p.scale_start, "uniform")
        self.assertEqual(p.scale_velocity, 0.0)
        self.assertIsNone(p.rotate_start)
        self.assertIsNone(p.shear_start)

    def test_all_transforms(self):
        config = parse("""
            resolution: [240, 180]
            translate:
              start: uniform
              velocity: uniform
            scale:
              start: 0.5
              velocity: 0.01
            rotate:
              start: uniform
              velocity: 0.0
            shear:
              start: 0.0
              velocity: uniform
        """)
        p = config_to_dataset_params(config, event_density=0.5, max_velocity=0.2)
        self.assertEqual(p.resolution, (240, 180))
        self.assertEqual(p.translate_start, "uniform")
        self.assertEqual(p.translate_velocity, "uniform")
        self.assertEqual(p.scale_start, 0.5)
        self.assertEqual(p.scale_velocity, 0.01)
        self.assertEqual(p.rotate_start, "uniform")
        self.assertEqual(p.rotate_velocity, 0.0)
        self.assertEqual(p.shear_start, 0.0)
        self.assertEqual(p.shear_velocity, "uniform")

    def test_optional_fields_defaults(self):
        config = parse("resolution: [100, 100]")
        p = config_to_dataset_params(config, event_density=1.0, max_velocity=0.2)
        self.assertEqual(p.length, 128)
        self.assertAlmostEqual(p.bg_density, 0.001)
        self.assertEqual(p.shape_density, 1.0)
        self.assertTrue(p.polarity)
        self.assertEqual(p.device, "cuda")
        self.assertEqual(p.upsampling_factor, 8)
        self.assertEqual(p.upsampling_cutoff, 0.5)

    def test_optional_fields_override(self):
        config = parse("""
            resolution: [64, 64]
            length: 64
            bg_density: 0.01
            shape_density: 0.8
            polarity: false
            device: cpu
            upsampling_factor: 4
            upsampling_cutoff: 0.25
        """)
        p = config_to_dataset_params(config, event_density=1.0, max_velocity=0.2)
        self.assertEqual(p.length, 64)
        self.assertAlmostEqual(p.bg_density, 0.01)
        self.assertEqual(p.shape_density, 0.8)
        self.assertFalse(p.polarity)
        self.assertEqual(p.device, "cpu")
        self.assertEqual(p.upsampling_factor, 4)
        self.assertEqual(p.upsampling_cutoff, 0.25)


if __name__ == "__main__":
    unittest.main()
