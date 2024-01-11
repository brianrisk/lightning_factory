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