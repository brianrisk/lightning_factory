import unittest

from lightning_factory.enums import Hyper, LossFunction, Optimizer, ActivationFunction
from lightning_factory import LightningFactory


class TestMergeParametersWithDefault(unittest.TestCase):

    def setUp(self):
        self.lf = LightningFactory()

    def test_precedence_of_provided_parameters(self):
        # Provided params should override defaults
        provided_params = {'learning_rate': 0.005, 'max_epochs': 15}
        expected_params = {k.value: v for k, v in self.lf.defaults.items()}
        expected_params.update(provided_params)
        merged_params = self.lf.merge_parameters_with_default(provided_params)
        self.assertEqual(merged_params[Hyper.LEARNING_RATE], 0.005)
        self.assertEqual(merged_params[Hyper.MAX_EPOCHS], 15)

    def test_ignore_unexpected_parameters(self):
        # Only the expected parameters should be merged
        provided_params = {'unexpected_param': 42, **self.lf.defaults}
        merged_params = self.lf.merge_parameters_with_default(provided_params)
        self.assertNotIn('unexpected_param', merged_params)

    def test_handling_str_and_enum_keys(self):
        # Method should handle both string and enum keys
        provided_params = {'batch_size': 32, 'learning_rate': 0.01}
        expected_params = {k.value: v for k, v in self.lf.defaults.items()}
        expected_params.update(provided_params)
        merged_params = self.lf.merge_parameters_with_default(provided_params)
        self.assertEqual(merged_params[Hyper.BATCH_SIZE], 32)
        self.assertEqual(merged_params[Hyper.LEARNING_RATE], 0.01)

    def test_empty_and_none_provided_parameters(self):
        # Providing empty dict or None should return defaults
        expected_defaults = self.lf.defaults.copy()
        merged_params_empty = self.lf.merge_parameters_with_default({})
        merged_params_none = self.lf.merge_parameters_with_default(None)
        self.assertEqual(merged_params_empty, expected_defaults)
        self.assertEqual(merged_params_none, expected_defaults)


if __name__ == '__main__':
    unittest.main()
