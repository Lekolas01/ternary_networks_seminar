import copy
import os
import timeit
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from ckmeans_1d_dp import ckmeans
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from torch import Tensor
from torch.nn import Sequential
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader

from bool_formula import Activation
from models.model_collection import ModelFactory, NNSpec, SteepTanh
from my_logging.loggers import LogMetrics, Tracker
from q_neuron import QNG_from_QNN, QuantizedLayer
from rule_set import RuleSetGraph
from train_model import training_loop
from utilities import set_seed


class RuleExtractionClassifier(BaseEstimator):
    def __init__(
        self,
        lr: float,
        k: int,
        n_layer: int,
        l1: float,
        epochs: int,
        wd: float,
        steepness=2,
        delay=100,
    ):
        self.lr = lr
        self.k = k
        self.n_layer = n_layer
        self.l1 = l1
        self.device = "cpu"
        self.batch_size = 64
        self.epochs = epochs
        self.wd = wd
        self.steepness = steepness
        self.delay = delay

    # convert a df to tensor to be used in pytorch
    def df_to_tensor(self, df) -> Tensor:
        return torch.from_numpy(df.values).float().to(self.device)

    def convert_to_rule_set(self, model: Sequential, dl: DataLoader):
        pass

    def train_qnn(self, X: Tensor, y: Tensor) -> tuple[Sequential, Sequential]:
        _, n_features = X.shape
        spec: NNSpec = [
            (self.k - i, self.k - i - 1, Activation.TANH) for i in range(self.n_layer)
        ]
        spec.append(((self.k - self.n_layer, 1, Activation.SIGMOID)))
        spec.pop(0)
        spec.insert(0, (n_features, self.k - 1, Activation.TANH))

        model = ModelFactory.get_model_by_spec(spec, steepness=self.steepness)
        if torch.any(model[0].weight.isnan()):
            print(spec)
        assert not torch.any(model[0].weight.isnan())
        # print(model)

        dataset = TensorDataset(X, y)
        # model_path = f"temp/testing_model.pth"
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        assert not torch.any(model[0].weight.isnan())
        metrics = self.train_mlp(
            dl, model, 1, self.lr, self.epochs, self.l1, self.wd, self.delay, False
        )
        #            torch.save(model, model_path)
        #            print(f"Saved model to {model_path}")
        #        else:
        #            model = torch.load(model_path)
        #            print(f"Load model from {model_path}")
        q_model = copy.deepcopy(model)
        nn_out = model(X)
        # qnn_out = q_model(X)
        # nn_pred = np.array(np.round(nn_out.detach().numpy()), dtype=bool)
        # qnn_pred = np.array(np.round(qnn_out.detach().numpy()), dtype=bool)
        # y_arr = np.array(y.detach().numpy(), dtype=bool)
        # nn_acc = np.mean(nn_pred == y_arr)
        # qnn_acc = np.mean(qnn_pred == y_arr)
        # fidelity = np.mean(nn_pred == qnn_pred)
        # print("Before first quantization:")
        # print(f"{nn_acc = } | {qnn_acc = } | {fidelity = }")

        while any(isinstance(l, nn.Linear) for l in q_model):
            assert torch.all(torch.isclose(model(X), nn_out))
            q_model = quantize_first_lin_layer(q_model, X)
            # qnn_out = q_model(X)
            # nn_pred = np.array(np.round(nn_out.detach().numpy()), dtype=bool)
            # qnn_pred = np.array(np.round(qnn_out.detach().numpy()), dtype=bool)
            # y_arr = np.array(y.detach().numpy(), dtype=bool)
            # nn_acc = np.mean(nn_pred == y_arr)
            # qnn_acc = np.mean(qnn_pred == y_arr)
            # fidelity = np.mean(nn_pred == qnn_pred)
            # print(f"{nn_acc = } | {qnn_acc = } | {fidelity = }")
        return q_model, model

    def fit(self, X: DataFrame, y: Series):
        start = timer()

        X_tensor = self.df_to_tensor(X)
        y_tensor = self.df_to_tensor(y)
        qnn, nn = self.train_qnn(X_tensor, y_tensor)

        q_ng = QNG_from_QNN(qnn, list(X.columns))
        self.bool_graph = RuleSetGraph.from_QNG(q_ng)
        nn_out = nn(X_tensor)
        nn_pred = np.array(np.round(nn_out.detach().numpy()), dtype=bool)
        qnn_pred = np.array(qnn(X_tensor))
        q_ng_pred = q_ng(X)
        bg_pred = self.bool_graph(X)
        nn_acc = np.mean(nn_pred == y)
        bg_acc = np.mean(bg_pred == y)
        fid_qnn = np.mean(nn_pred == qnn_pred)
        fid_qng = np.mean(nn_pred == q_ng_pred)
        fid_rule_set = np.mean(nn_pred == bg_pred)
        end = timer()
        print(
            f"{nn_acc = } | {bg_acc = } | fid(nn, qnn) = {fid_qnn} | fid(nn, q_ng) = {fid_qng} | fid(nn, rule_set) = {fid_rule_set} | compl. = {self.bool_graph.complexity()} | seconds = {end - start}"
        )
        return self

    def predict(self, X: DataFrame):
        if not hasattr(self, "bool_graph"):
            raise NotFittedError
        ans = self.bool_graph(X)
        return ans

    def __str__(self):
        return f"RuleExtractionClassifier(lr={self.lr}, k={self.k}, n_layer={self.n_layer}, l1={self.l1})"

    def train_mlp(
        self,
        dl: DataLoader,
        model: nn.Sequential,
        seed: int,
        lr: float,
        epochs: int,
        l1: float,
        wd: float,
        delay: int,
        log_metrics: bool,
    ):
        assert not torch.any(model[0].weight.isnan())
        seed = set_seed(seed)
        loss_fn = nn.BCELoss()
        assert not torch.any(model[0].weight.isnan())
        optim = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=wd,
        )
        assert not torch.any(model[0].weight.isnan())
        tracker = Tracker(epochs=epochs, delay=delay)
        assert not torch.any(model[0].weight.isnan())
        if log_metrics:
            assert not torch.any(model[0].weight.isnan())
            tracker.add_logger(
                LogMetrics(
                    ["timestamp", "epoch", "train_loss", "train_acc"],
                )
            )
            assert not torch.any(model[0].weight.isnan())
        assert not torch.any(model[0].weight.isnan())
        return training_loop(model, loss_fn, optim, dl, dl, epochs, l1, "cpu", tracker)


def quantize_first_lin_layer(model: nn.Sequential, X: Tensor) -> nn.Sequential:
    lin_layer_indices = [
        i for i in range(len(model)) if isinstance(model[i], nn.Linear)
    ]
    lin_layer_idx = lin_layer_indices[0]
    assert lin_layer_idx >= 0

    q_layer = quantize_layer(model, lin_layer_idx, len(lin_layer_indices) == 1, X)
    model = nn.Sequential(
        *[model[i] for i in range(lin_layer_idx)],
        q_layer,
        *[model[i] for i in range(2 + lin_layer_idx, len(model))],
    )
    return model


def quantize_layer(
    model: nn.Sequential, lin_layer_idx: int, is_last: bool, X: Tensor
) -> QuantizedLayer:
    lin_layer: nn.Linear = model[lin_layer_idx]  # type: ignore
    assert isinstance(lin_layer, nn.Linear)
    act = model[lin_layer_idx + 1]
    lin_layer.requires_grad_(False)
    n_samples = X.shape[0]

    if is_last:
        assert isinstance(act, nn.Sigmoid)
        return QuantizedLayer(lin_layer, torch.tensor(0.0), torch.tensor(1.0))
    for i in range(lin_layer_idx):
        X = model[i](X)
        # ERROR? don't you need to call act(X) as well?

    y_hat: Tensor = act(lin_layer(X))
    y_hat_arr = y_hat.detach().numpy().astype(np.float64)
    x_thrs = Tensor(lin_layer.out_features)
    y_low = Tensor(lin_layer.out_features)
    y_high = Tensor(lin_layer.out_features)
    cks = []
    assert isinstance(act, SteepTanh)
    for i in range(lin_layer.out_features):
        ck_ans = ckmeans(y_hat_arr[:, i], (2))
        cks.append(ck_ans)
        cluster = ck_ans.cluster
        if np.unique(cluster).shape[0] == 1:
            mean = ck_ans.centers[ck_ans.cluster][0]
            x_thrs[i] = 0.0
            y_low[i] = mean
            y_high[i] = mean
            continue

        max_0 = np.max(y_hat_arr[:, i][cluster == 0])
        try:
            min_1 = np.min(y_hat_arr[:, i][cluster == 1])
        except:
            temp = 0

        y_thr = (max_0 + min_1) / 2
        # the inverse of the activation function changes depending on the function's steepness
        x_thr = np.arctanh(y_thr) * 2 / act.k

        x_thrs[i] = x_thr
        y_low[i] = ck_ans.centers[0]
        y_high[i] = ck_ans.centers[1]
    with torch.no_grad():
        lin_layer.bias -= x_thrs  # type: ignore
    ans = QuantizedLayer(lin_layer, y_low, y_high)
    ll_out = y_hat
    ql_out = ans(X)
    # print(f"Avg. weight amplitude: {torch.mean(torch.abs(lin_layer.weight))}")
    # print(
    #     f"mean squared distance from cluster:{np.mean([ck.tot_withinss/n_samples for ck in cks])}"
    # )
    # print(f"srqd dist normalized:{np.mean([ck.tot_withinss /ck.totss for ck in cks])}")
    # print(type(ll_out))
    # print(type(ql_out))
    # print(f"mean abs dist in predictions: {torch.mean(torch.abs(ll_out - ql_out))}")
    return ans


def first_linear_layer(model: nn.Sequential) -> int:
    for idx, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            return idx
    return -1
