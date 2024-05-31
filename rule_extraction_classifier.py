import copy

import numpy as np
import torch
import torch.nn as nn
from _ckmeans_1d_dp import ckmeans
from pandas import DataFrame, Series
from sklearn.exceptions import NotFittedError
from torch import Tensor
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader

from bool_formula import Activation
from models.model_collection import ModelFactory, NNSpec
from my_logging.loggers import LogMetrics, Tracker
from q_neuron import QNG_from_QNN, QuantizedLayer
from rule_set import RuleSetGraph
from train_model import training_loop
from utilities import set_seed


class RuleExtractionClassifier:
    def __init__(
        self, lr: float, k: int, n_layer: int, l1: float, epochs: int, wd: float
    ):
        self.lr = lr
        self.k = k
        self.n_layer = n_layer
        self.l1 = l1
        self.device = "cpu"
        self.batch_size = 64
        self.epochs = 5000
        self.wd = wd

    # convert a df to tensor to be used in pytorch
    def df_to_tensor(self, df) -> Tensor:
        return torch.from_numpy(df.values).float().to(self.device)

    def convert_to_rule_set(self, model: nn.Sequential, dl: DataLoader):
        pass

    def train_q_model(self, X: DataFrame, y: Series) -> nn.Sequential:
        _, n_features = X.shape
        spec: NNSpec = [
            (self.k - i, self.k - i - 1, Activation.TANH) for i in range(self.n_layer)
        ]
        spec.append(((self.k - self.n_layer, 1, Activation.SIGMOID)))
        spec.pop(0)
        spec.insert(0, (n_features, self.k - 1, Activation.TANH))

        model = ModelFactory.get_model_by_spec(spec)
        X_tensor = self.df_to_tensor(X)
        y_tensor = self.df_to_tensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        metrics = self.train_mlp(dl, model, 1, self.lr, self.epochs, self.l1, self.wd)
        q_model = copy.deepcopy(model)
        while any(isinstance(l, nn.Linear) for l in q_model):
            q_model = quantize_first_lin_layer(q_model, dl)
        return q_model

    def fit(self, X: DataFrame, y: Series):
        q_model = self.train_q_model(X, y)
        q_ng = QNG_from_QNN(q_model, list(X.columns))
        self.bool_graph = RuleSetGraph.from_q_neuron_graph(q_ng)
        return self

    def predict(self, X):
        if not hasattr(self, "bool_graph"):
            raise NotFittedError
        data = {key: np.array(X[key], dtype=bool) for key in X.columns}
        return self.bool_graph(data)

    def train_mlp(
        self,
        dl: DataLoader,
        model: nn.Sequential,
        seed: int,
        lr: float,
        epochs: int,
        l1: float,
        wd: float,
    ):
        seed = set_seed(seed)
        loss_fn = nn.BCELoss()
        optim = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=wd,
        )
        tracker = Tracker(epochs=epochs)
        tracker.add_logger(
            LogMetrics(
                ["timestamp", "epoch", "train_loss", "train_acc"],
            )
        )

        return training_loop(
            model,
            loss_fn,
            optim,
            dl,
            dl,
            epochs=epochs,
            lambda1=l1,
            tracker=tracker,
            device="cpu",
        )


def quantize_first_lin_layer(model: nn.Sequential, dl: DataLoader) -> nn.Sequential:
    lin_layer_indices = [
        i for i in range(len(model)) if isinstance(model[i], nn.Linear)
    ]
    lin_layer_idx = lin_layer_indices[0]
    assert lin_layer_idx >= 0

    q_layer = quantize_layer(model, lin_layer_idx, len(lin_layer_indices) == 1, dl)
    model = nn.Sequential(
        *[model[i] for i in range(lin_layer_idx)],
        q_layer,
        *[model[i] for i in range(2 + lin_layer_idx, len(model))],
    )
    return model


def quantize_layer(
    model: nn.Sequential, lin_layer_idx: int, is_last: bool, dl: DataLoader
) -> QuantizedLayer:
    X, _ = next(iter(dl))
    lin_layer: nn.Linear = model[lin_layer_idx]  # type: ignore
    assert isinstance(lin_layer, nn.Linear)
    act = model[lin_layer_idx + 1]
    lin_layer.requires_grad_(False)

    if is_last:
        assert isinstance(act, nn.Sigmoid)
        return QuantizedLayer(lin_layer, torch.tensor(0.0), torch.tensor(1.0))
    for i in range(lin_layer_idx):
        X = model[i](X)

    y_hat: Tensor = act(lin_layer(X)).detach().numpy()
    x_thrs = Tensor(lin_layer.out_features)
    y_low = Tensor(lin_layer.out_features)
    y_high = Tensor(lin_layer.out_features)
    assert isinstance(act, nn.Tanh)
    for i in range(lin_layer.out_features):
        ck_ans = ckmeans(y_hat[:, i], (2))
        cluster = ck_ans.cluster
        max_0 = np.max(y_hat[:, i][cluster == 0])
        min_1 = np.min(y_hat[:, i][cluster == 1])
        y_thr = (max_0 + min_1) / 2
        x_thr = np.arctanh(y_thr)
        x_thrs[i] = x_thr
        y_low[i] = ck_ans.centers[0]
        y_high[i] = ck_ans.centers[1]
    with torch.no_grad():
        lin_layer.bias -= x_thrs  # type: ignore
    return QuantizedLayer(lin_layer, y_low, y_high)


def first_linear_layer(model: nn.Sequential) -> int:
    for idx, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            return idx
    return -1
