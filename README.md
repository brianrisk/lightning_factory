
![Lightning Factory](https://raw.githubusercontent.com/brianrisk/lightning_factory/master/images/lightning-factory-social-preview-dark.jpg?raw=true)


# Lightning Factory

PyTorch Lightning is great, but model building can be a bit...verbose.  
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

# setting the defaults.  These values will always be used unless otherwise specified
lf = LightningFactory(
    loss_function=LossFunction.MSE,
    batch_size=32,
    activation_function=ActivationFunction.Softplus
)
model1 = lf.ffnn(layers=[5,3,3,1])
model2 = lf.ffnn(layers=[5,8,4,2,1], activation_function=ActivationFunction.Tanh)
```

## Full example of building, training, testing and predicting

### Video tutorial

[![Lightning Factory Video Tutorial](https://raw.githubusercontent.com/brianrisk/lightning_factory/master/images/lightning-factory-video-thumbnail.jpg?raw=true)](https://youtu.be/7zqJZopgQSs?si=rxJkMyNF0o7VITd5)

### Full Example Code
```python
import lightning_factory as lf
from lightning_factory import d_at

# Loading stock data built for NNs by D.AT
# download sample data at: https://d.at/example-data
d_at.load_data(
    'data/train.csv',  # Training data
    'data/test.csv',   # Data time-separated from training; used to get precision, accuracy, etc
    'data/latest.csv'  # The most recent data.  The model will be predicting the labels
)

# creating our model
model = lf.ffnn(layers=[30, 3, 3, 1])

# training with our model
d_at.train(model)

# precision, accuracy, p-value of precision, confusion matrix
d_at.print_statistics()

# stocks ordered by which are most likely to have a `true` label
d_at.print_predictions()

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


### History
This code is born from the hyper-parameter tuning portion of [Stock Predictin Neural Netowrk and Machine Learning Examples](https://github.com/D-dot-AT/Stock-Prediction-Neural-Network-and-Machine-Learning-Examples)


### Future Work
This library is in early stages.  Future work involves adding factory methods for LSTMs, RNNs and more.  

If you have ideas for this, please fork and contribute!
