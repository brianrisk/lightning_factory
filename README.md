
![Lightning Factory](https://raw.githubusercontent.com/brianrisk/lightning_factory/master/images/lightning-factory-social-preview-dark.jpg?raw=true)


# Lightning Factory

Lightning Factory is a Python library designed to simplify the creation of PyTorch Lightning 
models for various types of neural networks. It follows the parameterized factory pattern
and allows users to specify custom configurations or use common defaults for quick prototyping.

## Usage

```bash
pip install lightning_factory
```

To create a feed-forward neural network model:

```python
import lightning_factory as lf

model = lf.ffnn(layers=[5, 3, 3, 1])
```
Easily define the layer structure:

![layers example](https://raw.githubusercontent.com/brianrisk/lightning_factory/master/images/lf-layers-example.jpg?raw=true)

Set default parameters when constructing the factory:
```python
from lightning_factory import LightningFactory
from lightning_factory import LossFunction
from lightning_factory import ActivationFunction

lf = LightningFactory(
    loss_function=LossFunction.MSE,
    batch_size=32,
    activation_function=ActivationFunction.Softplus
)
model1 = lf.ffnn(layers=[5,3,3,1])
model2 = lf.ffnn(layers=[5,8,4,2,1], activation_function=ActivationFunction.Tanh)
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

## Testing Coverage

| Name                                   |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| lightning\_factory/LightningFactory.py |       18 |        0 |        8 |        0 |    100% |           |
| lightning\_factory/\_\_init\_\_.py     |        3 |        0 |        0 |        0 |    100% |           |
| lightning\_factory/create\_ffnn.py     |       41 |       29 |       14 |        0 |     22% |19-30, 34-36, 40-46, 50-59, 63-71 |
| lightning\_factory/enums.py            |      121 |        0 |       18 |        0 |    100% |           |
| lightning\_factory/functions.py        |        4 |        2 |        0 |        0 |     50% |       6-7 |
|                              **TOTAL** |  **187** |   **31** |   **40** |    **0** | **80%** |           |

To run a coverage report
```shell
coverage report --format=markdown
```
