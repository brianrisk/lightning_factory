# Test suite for the create_ffnn functionality within the Lightning Factory library.

import unittest
import torch
import pytorch_lightning as pl
from lightning_factory import LightningFactory


class TestCreateFFNN(unittest.TestCase):

    def setUp(self):
        """Prepare the LightningFactory object before each test."""
        self.lf = LightningFactory()

    def test_ffnn_correct_initialization(self):
        """FFNN should initialize correctly with a proper layer list."""
        layers = [10, 5, 2]
        model = self.lf.ffnn(layers=layers)
        self.assertTrue(issubclass(model, pl.LightningModule), 'Model should be subclass of LightningModule')
