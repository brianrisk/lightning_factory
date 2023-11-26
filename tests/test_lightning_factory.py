import unittest

import pytorch_lightning as L

from lightning_factory import Hyper, LossFunction
from lightning_factory import LightningFactory


class TestLightningFactory(unittest.TestCase):

    def test_defaults(self):
        """ Test default parameters are set correctly. """
        lf = LightningFactory()
        self.assertEqual(lf.defaults[Hyper.LEARNING_RATE], 0.001)
        self.assertEqual(lf.defaults[Hyper.MAX_EPOCHS], 8)

    def test_custom_defaults(self):
        """ Test custom defaults override the factory defaults. """
        custom_learning_rate = 0.01
        lf = LightningFactory(learning_rate=custom_learning_rate)
        self.assertEqual(lf.defaults[Hyper.LEARNING_RATE], custom_learning_rate)

    def test_merge_parameters_with_default(self):
        """ Test that provided parameters override defaults. """
        lf = LightningFactory()
        custom_params = {str(Hyper.LEARNING_RATE): 0.02, str(Hyper.MAX_EPOCHS): 10}
        merged_params = lf.merge_parameters_with_default(custom_params)
        self.assertEqual(merged_params[Hyper.LEARNING_RATE], 0.02)
        self.assertEqual(merged_params[Hyper.MAX_EPOCHS], 10)

    def test_ffnn_raises_without_layers(self):
        """ Test that ffnn raises an exception if 'layers' is not provided. """
        lf = LightningFactory()
        with self.assertRaises(ValueError):
            lf.ffnn()

    def test_ffnn_returns_model(self):
        """ Test that ffnn returns a model when provided with layers. """
        lf = LightningFactory()
        model = lf.ffnn(layers=[5, 3, 1])
        # Check that the model is a subclass of LightningModule
        self.assertTrue(issubclass(model, L.LightningModule))

if __name__ == '__main__':
    unittest.main()
