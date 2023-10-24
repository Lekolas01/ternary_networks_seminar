import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import glob, os
import torch
import torch.nn as nn
import utilities
from dataloading import DataloaderFactory


def get_indices_epochs(base_model_paths):
    indices = [int(p[6:8]) for p in base_model_paths]
    epochs = [int(p[14:17]) for p in base_model_paths]
    return indices, epochs


def get_stats_of_dir_path(dir_path):
    errors_path = dir_path / "errors.csv"
    configs_path = dir_path / "configs.csv"
    models_path = dir_path / "best"
    errors = pd.read_csv(errors_path)

    # Inspect best models
    model_paths = glob.glob(models_path.__str__() + "/*")
    base_model_paths = [os.path.basename(p) for p in model_paths]

    indices, epochs = get_indices_epochs(base_model_paths)
    best_models = [torch.load(p) for p in model_paths]
    best_quantized_models = [m.quantized(prune=True) for m in best_models]
    results = get_result(errors, indices, epochs)
    results = results.sort_values("simple_complexity", axis=0)
    return errors, results, best_quantized_models


def inspect(dir_path):
    errors, results, best_quantized_models = get_stats_of_dir_path(dir_path)
    sorted_results = results.sort_values("complexity", axis=0)

    plt.plot(sorted_results["simple_complexity"], sorted_results["quantized_valid_acc"])
    plt.xlabel("complexity of quantized/simplified NN")
    plt.ylabel("Accuracy")
    plt.title(os.path.basename(dir_path))
    plt.show()
    valid_loader = DataloaderFactory(
        ds="mushroom", train=False, shuffle=True, batch_size=128
    )
    names = valid_loader.dataset.df.columns[1:]

    for i, qm in enumerate(best_quantized_models):
        curr_result = results.iloc[i]
        weights = np.tanh(utilities.get_all_weights(qm).detach().cpu().numpy())
        print(
            f"Model {i}: Accuracy = {curr_result['quantized_valid_acc']} | Simple Complexity = {curr_result['simple_complexity']}"
        )
        pretty_print_classifier(qm.classifier, names)


def inspect_fronts(dir_paths):
    fronts = []
    plt.xlabel("simple complexity")
    plt.ylabel("quantized accuracy")
    plt.title("Comparison of multiple Fronts")
    for dir_path in dir_paths:
        errors, results, best_quantized_models = get_stats_of_dir_path(dir_path)
        sorted_results = results.sort_values("simple_complexity")
        plt.plot(
            sorted_results["simple_complexity"],
            sorted_results["quantized_valid_acc"],
            label=dir_path,
        )
    plt.legend()
    plt.show()


def pretty_print_classifier(classifier, names, print_first_layer=True):
    def pretty_print_linear_layer(w, bias, names=None):
        n_rows, n_cols = w.shape
        for i in range(n_rows):
            if np.count_nonzero(w[i]) == 0 and bias[i] == 0:
                continue
            print(f"\t\tNode {i}: ", end="")
            for j in range(n_cols):
                val = names[j] if print_first_layer and names is not None else j
                if w[i, j] == -1:
                    print(f"- {val} ", end="")
                elif w[i, j] == 1:
                    print(f"+ {val} ", end="")
            if bias is not None:
                if bias[i] == -1:
                    print(f"- b", end="")
                if bias[i] == 1:
                    print(f"+ b", end="")
            print()

    for idx, layer in enumerate(classifier):
        if isinstance(layer, nn.Linear):
            print(f"\tLayer {idx}:")
            w = layer.weight.detach().cpu().numpy().astype(np.int8)
            b = (
                layer.bias.detach().cpu().numpy().astype(np.int8)
                if layer.bias is not None
                else None
            )
            pretty_print_linear_layer(w, b, names if idx == 0 else None)
        print()


def get_result(errors: pd.DataFrame, indices, epochs):
    frames = [
        errors[(errors["idx"] == indices[i]) & (errors["epoch"] == epochs[i])]
        for i in range(len(indices))
    ]
    return pd.concat(frames, axis=0)


if __name__ == "__main__":
    base_path = Path("runs") / "mushroom" / "complexity"
    runs = [
        "sigmoid",
        "small_sigmoid",
    ]
    dir_paths = [base_path / r for r in runs]
    dir_path = Path("runs") / "mushroom" / "complexity" / "small_sigmoid"
    # inspect_fronts(dir_paths)
    inspect(dir_path)
