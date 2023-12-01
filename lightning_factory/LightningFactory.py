from lightning_factory.create_ffnn import create_ffnn
from lightning_factory.enums import (
    Hyper,
    WeightInit as Wi,
    Optimizer as Opti,
    LossFunction as Lf,
    ActivationFunction as Af
)


class LightningFactory:

    def __init__(self, **kwargs):
        # Default parameters
        self.defaults = {
            Hyper.LAYERS: None,
            Hyper.LEARNING_RATE: 0.001,
            Hyper.MAX_EPOCHS: 8,
            Hyper.BATCH_SIZE: 64,
            Hyper.LOSS_FUNCTION: Lf.BCE,
            Hyper.ACTIVATION_FUNCTION: Af.ReLU,
            Hyper.OPTIMIZER: Opti.ADAM,
            Hyper.DROPOUT: 0,
            Hyper.L1_REGULARIZATION: 0,
            Hyper.L2_REGULARIZATION: 0,
            Hyper.WEIGHT_INITIALIZATION: Wi.XAVIER_UNIFORM
        }

        # Merge provided parameters with defaults
        self.defaults = self.merge_parameters_with_default(kwargs)

    def merge_parameters_with_default(self, provided_params):
        """ Override defaults with any provided arguments """
        if provided_params is None or len(provided_params) == 0:
            return self.defaults

        # Create a dictionary of enums for the provided arguments
        enum_params = {Hyper.get_enum_from_string(k): v for k, v in provided_params.items()}
        params = {**self.defaults, **enum_params}
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
        params = self.merge_parameters_with_default(provided_params)

        # Check if layers is None and raise an exception if it is
        if params[Hyper.LAYERS] is None:
            raise ValueError("The 'layers' parameter is required but was not provided.")

        return create_ffnn(params)
