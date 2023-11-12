
![Lightning Factory](https://raw.githubusercontent.com/brianrisk/lightning_factory/master/images/lightning-factory-social-preview.jpg?raw=true)


# LightningFactory

LightningFactory is a Python library designed to simplify the creation of PyTorch Lightning 
models for various types of neural networks. It follows the parameterized factory pattern
and allows users to specify custom configurations or use common defaults for quick prototyping.

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
Easily define the layer structure:

![layers example](https://raw.githubusercontent.com/brianrisk/lightning_factory/master/images/lf-layers-example.jpg?raw=true)

Set default parameters when constructing the factory:
```python
from lightning_factory import LightningFactory
from lightning_factory import LossFunction

lf = LightningFactory(
    loss_function=LossFunction.MSE,
    batch_size=32
)
```

The `LightningFactory` class uses the following defaults when making a class:
```python
default = {
    'layers': None,
    'learning_rate': 0.001,
    'max_epochs': 8,
    'batch_size': 64,
    'loss_function': 'BCE',
    'activation_function': 'ReLU',
    'optimizer': 'Adam',
    'dropout': 0,
    'l1_regularization': 0,
    'l2_regularization': 0,
    'weight_initialization': 'xavier_uniform'
}
```