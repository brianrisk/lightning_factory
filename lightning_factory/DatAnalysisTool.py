import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from scipy.stats import fisher_exact
from torch.utils.data import DataLoader
import pytorch_lightning as L

from lightning_factory import LightningFactory, Hyper


def calculate_precision_p_value(tp: int, fp: int, fn: int, tn: int):
    # If the prediction precision is lower than the overall rate, return 1.0
    if (tp / (tp + fp)) < ((fn + tp) / (tp + fp + tn + fn)):
        return 1.0
    contingency_table = [[tp, tp + fp], [fn + tp, tp + fp + tn + fn]]
    _, p_value = fisher_exact(contingency_table)
    return p_value


class DataAnalysisTool:
    """
    A Utility class to work with stock data from D.AT
    Loads, scales, trains, runs predictions, prints statistics
    """
    def __init__(self):
        # Model
        self.model = None

        # Data
        self.train_dataset = None
        self.input_feature_size = None
        self.y_vector_test = None
        self.x_vector_test_scaled = None
        self.tickers = None
        self.x_vector_latest = None

    def load_data(self, train_file, test_file, latest_file):
        # Training data
        train_data = pd.read_csv(train_file, header=None)
        x_vector_train = train_data.iloc[:, :-1].values
        y_vector_train = train_data.iloc[:, -1].values
        scaler = StandardScaler()
        x_vector_train_scaled = scaler.fit_transform(x_vector_train)
        self.train_dataset = TensorDataset(torch.tensor(x_vector_train_scaled), torch.tensor(y_vector_train))
        self.input_feature_size = x_vector_train_scaled.shape[1]

        # Testing data
        test_data = pd.read_csv(test_file, header=None)
        x_vector_test = test_data.iloc[:, :-1].values
        self.y_vector_test = test_data.iloc[:, -1].values
        self.x_vector_test_scaled = scaler.transform(x_vector_test)

        # Latest data
        latest_data = pd.read_csv(latest_file)
        self.tickers = latest_data.iloc[:, 0].values
        self.x_vector_latest = scaler.transform(latest_data.iloc[:, 1:].values)

    def train(self, model, lf: LightningFactory = LightningFactory()):
        # Train the model using the loaded data
        self.model = model
        # Initializing a trainer and training the model
        train_loader = DataLoader(self.train_dataset, batch_size=lf.get(Hyper.BATCH_SIZE))
        trainer = L.Trainer(max_epochs=lf.get(Hyper.MAX_EPOCHS), logger=False)
        trainer.fit(model, train_loader)

        model.eval()

    def print_statistics(self):
        print("printing statistics...")
        # Making predictions on test data
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(torch.tensor(self.x_vector_test_scaled)).numpy()

        # Binarizing predictions
        predictions_bin = (predictions > 0.5).astype(int)

        # Statistical analysis
        tn, fp, fn, tp = confusion_matrix(self.y_vector_test, predictions_bin).ravel()
        overall_positive_rate = float(fn + tp) / (tp + fp + tn + fn)
        precision = float(tp) / (tp + fp)
        accuracy = float(tp + tn) / (tp + fp + tn + fn)
        p_value = calculate_precision_p_value(tp=tp, fp=fp, fn=fn, tn=tn)

        # Step 6: Output
        # Print the following information:
        print(f'TN: {tn}')
        print(f'TP: {tp}')
        print(f'FN: {fn}')
        print(f'FP: {fp}')
        print(f'Overall positive rate: {overall_positive_rate}')
        print(f'Precision: {precision}')
        print(f'Accuracy: {accuracy}')
        print(f'P-value of precision: {p_value}')

    def print_predictions(self):
        scores = self.model(torch.Tensor(self.x_vector_latest)).detach().numpy()
        top_5_indices = scores.flatten().argsort()[-5:][::-1]
        print("Stocks with highest score for positive outcome:")
        for idx in top_5_indices:
            print(f'{self.tickers[idx]}: {scores[idx][0] * 100:.2f}%')

# Create an instance of the DataAnalysisTool class
d_at = DataAnalysisTool()
