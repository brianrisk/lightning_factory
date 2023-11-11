from enum import Enum, unique

import torch.optim as optim
from torch import nn


class StringEnum(Enum):

    def __str__(self):
        return self.value

    @classmethod
    def get_enum_from_string(cls, value):
        """Return the enum member matching the string value, or None if no match."""
        for name, member in cls.__members__.items():
            if member.value == value:
                return member
        return None


# Define an enum for hyperparameter keys to improve readability
# Note: string values must match parameter names in LightningFactory
@unique
class Hyper(StringEnum):
    NETWORK_TYPE = 'network_type'
    LEARNING_RATE = 'learning_rate'
    MAX_EPOCHS = 'max_epochs'
    BATCH_SIZE = 'batch_size'
    LAYERS = 'layers'
    LOSS_FUNCTION = 'loss_function'
    ACTIVATION_FUNCTION = 'activation_function'
    OPTIMIZER = 'optimizer'
    DROPOUT = 'dropout'
    L1_REGULARIZATION = 'l1_regularization'
    L2_REGULARIZATION = 'l2_regularization'
    WEIGHT_INITIALIZATION = 'weight_initialization'
    NUMBER_OF_HIDDEN_LAYERS = 'number_of_hidden_layers'
    HIDDEN_LAYER_SIZE = 'hidden_layer_size'


# Define an enum for optimizer keys
@unique
class Optimizer(StringEnum):
    ADAM = 'Adam'
    SGD = 'SGD'
    RMSPROP = 'RMSprop'
    ADAMW = 'AdamW'
    ADAGRAD = 'Adagrad'
    ADADELTA = 'Adadelta'
    ADAMAX = 'Adamax'
    LBFGS = 'LBFGS'
    RPROP = 'Rprop'
    ASGD = 'ASGD'
    FTRL = 'FTRL'


@unique
class NetworkType(StringEnum):
    FFNN = "FFNN"  # Feedforward Neural Network
    CNN = "CNN"    # Convolutional Neural Network
    RNN = "RNN"    # Recurrent Neural Network
    LSTM = "LSTM"  # Long Short-Term Memory network

# Optimizer function lookup
OPTIMIZER_CLASSES = {
    Optimizer.ADAM: optim.Adam,
    Optimizer.SGD: optim.SGD,
    Optimizer.RMSPROP: optim.RMSprop,
    Optimizer.ADAMW: optim.AdamW,
    Optimizer.ADAGRAD: optim.Adagrad,
    Optimizer.ADADELTA: optim.Adadelta,
    Optimizer.ADAMAX: optim.Adamax,
    Optimizer.LBFGS: optim.LBFGS,
    Optimizer.RPROP: optim.Rprop,
    Optimizer.ASGD: optim.ASGD,
}


@unique
class ActivationFunction(StringEnum):
    ELU = "ELU"
    Hardshrink = "Hardshrink"
    Hardsigmoid = "Hardsigmoid"
    Hardtanh = "Hardtanh"
    Hardswish = "Hardswish"
    LeakyReLU = "LeakyReLU"
    LogSigmoid = "LogSigmoid"
    MultiheadAttention = "MultiheadAttention"
    PReLU = "PReLU"
    ReLU = "ReLU"
    ReLU6 = "ReLU6"
    RReLU = "RReLU"
    SELU = "SELU"
    CELU = "CELU"
    GELU = "GELU"
    Sigmoid = "Sigmoid"
    SiLU = "SiLU"
    Mish = "Mish"
    Softplus = "Softplus"
    Softshrink = "Softshrink"
    Softsign = "Softsign"
    Tanh = "Tanh"
    Tanhshrink = "Tanhshrink"
    Threshold = "Threshold"
    GLU = "GLU"
    Softmin = "Softmin"
    Softmax = "Softmax"
    Softmax2d = "Softmax2d"
    LogSoftmax = "LogSoftmax"
    AdaptiveLogSoftmaxWithLoss = "AdaptiveLogSoftmaxWithLoss"


ACTIVATION_FUNCTIONS = {
    ActivationFunction.ELU: nn.ELU,
    ActivationFunction.Hardshrink: nn.Hardshrink,
    ActivationFunction.Hardsigmoid: nn.Hardsigmoid,
    ActivationFunction.Hardtanh: nn.Hardtanh,
    ActivationFunction.Hardswish: nn.Hardswish,
    ActivationFunction.LeakyReLU: nn.LeakyReLU,
    ActivationFunction.LogSigmoid: nn.LogSigmoid,
    ActivationFunction.MultiheadAttention: nn.MultiheadAttention,
    ActivationFunction.PReLU: nn.PReLU,
    ActivationFunction.ReLU: nn.ReLU,
    ActivationFunction.ReLU6: nn.ReLU6,
    ActivationFunction.RReLU: nn.RReLU,
    ActivationFunction.SELU: nn.SELU,
    ActivationFunction.CELU: nn.CELU,
    ActivationFunction.GELU: nn.GELU,
    ActivationFunction.Sigmoid: nn.Sigmoid,
    ActivationFunction.SiLU: nn.SiLU,
    ActivationFunction.Mish: nn.Mish,
    ActivationFunction.Softplus: nn.Softplus,
    ActivationFunction.Softshrink: nn.Softshrink,
    ActivationFunction.Softsign: nn.Softsign,
    ActivationFunction.Tanh: nn.Tanh,
    ActivationFunction.Tanhshrink: nn.Tanhshrink,
    ActivationFunction.Threshold: nn.Threshold,
    ActivationFunction.GLU: nn.GLU,
    ActivationFunction.Softmin: nn.Softmin,
    ActivationFunction.Softmax: nn.Softmax,
    ActivationFunction.Softmax2d: nn.Softmax2d,
    ActivationFunction.LogSoftmax: nn.LogSoftmax,
    ActivationFunction.AdaptiveLogSoftmaxWithLoss: nn.AdaptiveLogSoftmaxWithLoss
}


# Define an enum for weight initialization keys
@unique
class WeightInit(StringEnum):
    XAVIER_UNIFORM = 'xavier_uniform'
    XAVIER_NORMAL = 'xavier_normal'
    KAIMING_UNIFORM = 'kaiming_uniform'
    KAIMING_NORMAL = 'kaiming_normal'
    ORTHOGONAL = 'orthogonal'
    ZEROS = 'zeros'
    ONES = 'ones'
    CONSTANT = 'constant'
    EYE = 'eye'
    SPARSE = 'sparse'
    NORMAL = 'normal'
    UNIFORM = 'uniform'
    DIRAC = 'dirac'


# Weight initialization function lookup
WEIGHT_INITIALIZATIONS = {
    WeightInit.XAVIER_UNIFORM: nn.init.xavier_uniform_,
    WeightInit.XAVIER_NORMAL: nn.init.xavier_normal_,
    WeightInit.KAIMING_UNIFORM: nn.init.kaiming_uniform_,
    WeightInit.KAIMING_NORMAL: nn.init.kaiming_normal_,
    WeightInit.ORTHOGONAL: nn.init.orthogonal_,
    WeightInit.ZEROS: nn.init.zeros_,
    WeightInit.ONES: nn.init.ones_,
    WeightInit.CONSTANT: nn.init.constant_,
    WeightInit.EYE: nn.init.eye_,
    WeightInit.SPARSE: nn.init.sparse_,
    WeightInit.NORMAL: nn.init.normal_,
    WeightInit.UNIFORM: nn.init.uniform_,
    WeightInit.DIRAC: nn.init.dirac_,
}


@unique
class LossFunction(StringEnum):
    L1 = 'L1'
    MSE = 'MSE'
    CROSS_ENTROPY = 'Cross Entropy'
    CTC = 'CTC'
    NLL = 'NLL'
    POISSON_NLL = 'Poisson NLL'
    GAUSSIAN_NLL = 'Gaussian NLL'
    KLDIV = 'KL Div'
    BCE = 'BCE'
    BCE_WITH_LOGITS = 'BCE With Logits'
    MARGIN_RANKING = 'Margin Ranking'
    HINGE_EMBEDDING = 'Hinge Embedding'
    MULTI_LABEL_MARGIN = 'Multi Label Margin'
    HUBER = 'Huber'
    SMOOTH_L1 = 'Smooth L1'
    SOFT_MARGIN = 'Soft Margin'
    MULTI_LABEL_SOFT_MARGIN = 'Multi-Label Soft Margin'
    COSINE_EMBEDDING = 'Cosine Embedding'
    MULTI_MARGIN = 'Multi Margin'
    TRIPLET_MARGIN = 'Triplet Margin'
    TRIPLET_MARGIN_WITH_DISTANCE = 'Triplet Margin With Distance'


LOSS_FUNCTIONS = {
    LossFunction.L1: nn.L1Loss,
    LossFunction.MSE: nn.MSELoss,
    LossFunction.CROSS_ENTROPY: nn.CrossEntropyLoss,
    LossFunction.CTC: nn.CTCLoss,
    LossFunction.NLL: nn.NLLLoss,
    LossFunction.POISSON_NLL: nn.PoissonNLLLoss,
    LossFunction.GAUSSIAN_NLL: nn.GaussianNLLLoss,
    LossFunction.KLDIV: nn.KLDivLoss,
    LossFunction.BCE: nn.BCELoss,
    LossFunction.BCE_WITH_LOGITS: nn.BCEWithLogitsLoss,
    LossFunction.MARGIN_RANKING: nn.MarginRankingLoss,
    LossFunction.HINGE_EMBEDDING: nn.HingeEmbeddingLoss,
    LossFunction.MULTI_LABEL_MARGIN: nn.MultiLabelMarginLoss,
    LossFunction.HUBER: nn.HuberLoss,
    LossFunction.SMOOTH_L1: nn.SmoothL1Loss,
    LossFunction.SOFT_MARGIN: nn.SoftMarginLoss,
    LossFunction.MULTI_LABEL_SOFT_MARGIN: nn.MultiLabelSoftMarginLoss,
    LossFunction.COSINE_EMBEDDING: nn.CosineEmbeddingLoss,
    LossFunction.MULTI_MARGIN: nn.MultiMarginLoss,
    LossFunction.TRIPLET_MARGIN: nn.TripletMarginLoss,
    LossFunction.TRIPLET_MARGIN_WITH_DISTANCE: nn.TripletMarginWithDistanceLoss
}
