# LightningFactory

LightningFactory is a Python library designed to simplify the creation of PyTorch Lightning 
models for various types of neural networks, such as Feed-Forward Neural Networks (FFNN), 
Long Short-Term Memory networks (LSTM), and more. It allows users to specify custom 
configurations or use sensible defaults for quick prototyping.

## Usage

```bash
pip install lightning_factory
```

To create a feed-forward neural network model:

```python
from lightning_factory import LightningFactory

lf = LightningFactory()
model = lf.ffnn(layers=[5, 3, 3, 1])
```

## Features

- Easy instantiation of FFNN models with customizable layers and parameters.
- Sensible default parameters for quick setup.
- Extendable to various other neural network architectures.
- Parameter validation to ensure necessary configurations are provided.


## Parameters

### `LightningFactory` Initialization Parameters

- `layers` (list of int): Specify the number of neurons in each layer.
- `learning_rate` (float): The learning rate for the optimizer.
- `max_epochs` (int): The maximum number of epochs for training.
- `batch_size` (int): The batch size for training.
- `loss_function` (Lf): The loss function to be used.
- `activation_function` (Af): The activation function for the neurons.
- `optimizer` (Opti): The optimizer for training.
- `weight_initialization` (Wi): The method for weight initialization.

