from lightning_factory.get_ffnn import get_ffnn
from lightning_factory.enums import (
    Hyper,
    WeightInit as Wi,
    Optimizer as Opti,
    LossFunction as Lf,
    ActivationFunction as Af
)


class LightningFactory:

    def __init__(self,
                 layers: [int] = None,
                 learning_rate: float = None,
                 max_epochs: int = None,
                 batch_size: int = None,
                 loss_function: Lf = None,
                 activation_function: Af = None,
                 optimizer: Opti = None,
                 weight_initialization: Wi = None
                 ):

        self.defaults = {
            Hyper.LAYERS: layers if layers else None,
            Hyper.LEARNING_RATE: learning_rate if learning_rate else 0.001,
            Hyper.MAX_EPOCHS: max_epochs if max_epochs else 8,
            Hyper.BATCH_SIZE: batch_size if batch_size else 64,
            Hyper.LOSS_FUNCTION: loss_function if loss_function else Lf.BCE,
            Hyper.ACTIVATION_FUNCTION: activation_function if activation_function else Af.ReLU,
            Hyper.OPTIMIZER: optimizer if optimizer else Opti.ADAM,
            Hyper.DROPOUT: 0,
            Hyper.L1_REGULARIZATION: 0,
            Hyper.L2_REGULARIZATION: 0,
            Hyper.WEIGHT_INITIALIZATION: weight_initialization if weight_initialization else Wi.XAVIER_UNIFORM
        }

    def merge_parameters_with_default(self, **kwargs):
        """ Override defaults with any provided arguments """
        params = {**self.defaults, **kwargs}
        return params

    def ffnn(
            self,
            layers: [int] = None,
            learning_rate: float = None,
            max_epochs: int = None,
            batch_size: int = None,
            loss_function: Lf = None,
            activation_function: Af = None,
            optimizer: Opti = None,
            weight_initialization: Wi = None
    ):
        """Creating a Feed-Forward Neural Network"""
        # Create a dictionary of the provided arguments, excluding 'None' values
        provided_params = {k: v for k, v in locals().items() if v is not None and k != 'self'}
        # Merge provided parameters with defaults
        params = self.merge_parameters_with_default(**provided_params)

        # Check if layers is None and raise an exception if it is
        if params[Hyper.LAYERS] is None:
            raise ValueError("The 'layers' parameter is required but was not provided.")

        return get_ffnn(params)

