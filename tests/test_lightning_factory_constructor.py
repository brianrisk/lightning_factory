import unittest

from lightning_factory.enums import Hyper
from lightning_factory.LightningFactory import LightningFactory

class TestLightningFactoryConstructor(unittest.TestCase):

    def test_explicit_constructor_arguments(self):
        # Test explicit constructor arguments override defaults
        explicit_args = {
            'learning_rate': 0.01,
            'max_epochs': 20,
            'batch_size': 128
        }
        lf = LightningFactory(**explicit_args)
        self.assertEqual(lf.get(Hyper.LEARNING_RATE), 0.01)
        self.assertEqual(lf.get(Hyper.MAX_EPOCHS), 20)
        self.assertEqual(lf.get(Hyper.BATCH_SIZE), 128)

        # Ensure argument not provided defaults to predefined value
        self.assertEqual(lf.get(Hyper.LOSS_FUNCTION), lf.defaults[Hyper.LOSS_FUNCTION])


if __name__ == '__main__':
    unittest.main()
